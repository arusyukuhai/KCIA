"""Local batch production and PT/SFT/PPO gradient accumulation.

Network transport, signed commits, room enrollment and GUI ownership are
outside this module. The public cli.py training functions are re-exported.
"""
from __future__ import annotations
import itertools
import math
from pathlib import Path
import torch
from safetensors.torch import load_file
from .model_loading_v110 import input_device


def causal_labels_from_attention(input_ids, attention_mask):
    """Preserve *real* EOS tokens when a causal LM uses EOS as padding.

    DataCollatorForLanguageModeling compares token ID with pad_token_id;
    for Llama/Qwen models whose PAD is EOS this silently drops every
    genuine EOS supervision target. Attention masks distinguish them.
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError('input IDs and attention masks differ in shape')
    labels=input_ids.clone()
    labels.masked_fill_(attention_mask == 0, -100)
    return labels


def training_batches(args, *, batch_file_loader=load_file):
    # A specialized SFT/PPO peer is permitted to omit a pretraining corpus.
    # Never fetch HF data for an objective with an explicit zero coefficient.
    if (getattr(args,'pt_weight',1.) == 0 and
            (getattr(args,'sft_jsonl',None) or getattr(args,'ppo_jsonl',None) or getattr(args,'sft_hf_repo','') or getattr(args,'ppo_hf_repo',''))):
        while True:
            yield {}  # run_step checks pt_weight=0 before invoking the model
    from .data_sampling import peer_sampling_seed, shuffled_cycle
    from .room_manifest import load_manifest
    manifest = (load_manifest(args.room_manifest) if getattr(args, 'room_manifest', None)
                and not (args.batches_dir or args.text_file or args.jsonl) else None)
    # Same room seed, but independent deterministic stream per persistent peer.
    root_seed = manifest['datasets']['seed'] if manifest is not None else 42
    data_seed = peer_sampling_seed(getattr(args, 'peer_id', None), root_seed)
    print(f'DATA SAMPLING peer={getattr(args, "peer_id", None) or "standalone"} '
          f'seed={data_seed} source={"huggingface" if manifest else "local"}', flush=True)
    if manifest is not None:
        from .hf_datasets import batches_from_room
        yield from batches_from_room(args,manifest)
        return
    if args.batches_dir:
        files = sorted(Path(args.batches_dir).glob('*.safetensors'))
        if not files:
            raise ValueError('No .safetensors batches in --batches-dir')
        for file in shuffled_cycle(files, data_seed):
            # A batch is exactly the named tensor arguments accepted by model(**batch).
            yield batch_file_loader(file)
    else:
        if args.hf_task not in ('causal-lm', 'masked-lm', 'seq2seq-lm', 'sequence-classification'):
            raise ValueError('Use --batches-dir for this task; no generic text loss is defined')
        from .hf_load_v85 import load_hf_tokenizer
        from transformers import DataCollatorForLanguageModeling
        tokenizer = load_hf_tokenizer(args.model, revision=args.revision,
                                                   trust_remote_code=bool(getattr(args,'trust_remote_code',False)),
                                                   local_files_only=bool(getattr(args,'trust_remote_code',False)))
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError('Tokenizer needs a PAD or EOS token')
            tokenizer.pad_token = tokenizer.eos_token
        if args.hf_task in ('causal-lm', 'masked-lm'):
            if not args.text_file:
                raise ValueError('--text-file required for text LM training')
            lines = [line.strip() for line in Path(args.text_file).read_text(encoding='utf8').splitlines() if line.strip()]
            if not lines:
                raise ValueError('empty text dataset')
            mlm = args.hf_task == 'masked-lm'
            collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15) if mlm else None
            local_rows=shuffled_cycle(lines, data_seed)
            for step in itertools.count():
                selected = list(itertools.islice(local_rows, args.batch_size))
                encoded = tokenizer(selected, truncation=True, max_length=args.max_length,
                                    padding='max_length', return_tensors='pt',
                                    **({'return_token_type_ids':False} if not mlm else {}))
                # Collator masks -100 on pad tokens for CLM, random masks for MLM.
                if mlm:
                    yield collator([{k:v[j] for k,v in encoded.items()} for j in range(args.batch_size)])
                else:
                    encoded['labels'] = causal_labels_from_attention(encoded['input_ids'], encoded['attention_mask'])
                    yield encoded
        else:
            if not args.jsonl:
                raise ValueError('--jsonl required for sequence/classification training')
            examples = [json.loads(line) for line in Path(args.jsonl).read_text(encoding='utf8').splitlines() if line.strip()]
            if not examples:
                raise ValueError('empty JSONL')
            local_rows=shuffled_cycle(examples, data_seed)
            for step in itertools.count():
                items = list(itertools.islice(local_rows, args.batch_size))
                enc = tokenizer([x['input'] if args.hf_task=='seq2seq-lm' else x['text'] for x in items],
                                return_tensors='pt',padding=True,truncation=True,max_length=args.max_length,
                                **({'return_token_type_ids':False} if args.hf_task=='seq2seq-lm' else {}))
                if args.hf_task == 'seq2seq-lm':
                    lab = tokenizer(text_target=[x['target'] for x in items],
                                    return_tensors='pt',padding=True,truncation=True,max_length=args.max_length)
                    labels = lab['input_ids']
                    labels[labels == tokenizer.pad_token_id] = -100
                    enc['labels'] = labels
                else:
                    enc['labels'] = torch.tensor([int(x['label']) for x in items], dtype=torch.long)
                yield enc


def run_step(model, adapter, optimizer, batch, args):
    with adapter.lock:
        model.train()
        target = input_device(model, args.device)
        batch = {k: v.to(target, non_blocking=(target.type == 'cuda')) if isinstance(v,torch.Tensor) else v
                 for k, v in batch.items()}
        from .model_contract_v64 import invoke_model, training_loss
        from .mixed_learning_v65 import enabled as mixed_enabled, build as build_mixer
        mixed=mixed_enabled(args)
        if mixed:
            previous=getattr(args,'_v65_mixer',None)
            old_adapter=getattr(args,'_v65_mixer_adapter',None)
            if previous is None or getattr(previous,'_closed',False) or (old_adapter is not None and old_adapter is not adapter):
                if previous is not None:
                    previous.close()  # epoch switch: discard old-policy PPO rollouts
                # A new signed generation invalidates the old Actor/Critic
                # pairing. Do not silently load a pre-v75 standalone Critic
                # checkpoint (same room id, but stale Actor baseline).
                skip_legacy=bool(getattr(args,'_v75_paired_active',False) and
                                 old_adapter is not None and old_adapter is not adapter)
                if skip_legacy:args._v75_skip_legacy_critic=True
                try:
                    args._v65_mixer=build_mixer(args)
                finally:
                    if skip_legacy:args._v75_skip_legacy_critic=False
                if getattr(args,'_v75_paired_active',False) and args._v65_mixer is not None:
                    args._v65_mixer._v75_paired=True
            args._v65_mixer_adapter=adapter
        mixer=getattr(args,'_v65_mixer',None) if mixed else None
        if mixer is not None and float(args.pt_weight)==0:
            loss=None  # intentional: no pretraining forward when its weight is zero
            outputs=None
        else:
            outputs = invoke_model(model, batch, input_map=getattr(args,'input_map',None))
            loss = training_loss(outputs,batch,task=getattr(args,'hf_task','causal-lm'),
                                 policy=getattr(args,'loss_policy','auto'))
        if mixer is not None:
            total=0.0
            if loss is not None:
                if not bool(torch.isfinite(loss.detach()).all()):
                    raise ValueError('nonfinite PT objective; refusing mixed optimizer update')
                (loss * args.pt_weight / args.grad_accum).backward()
                total += float(loss.detach())*args.pt_weight
                mixer.counts['pt']+=1
            if mixer.sft is not None and args.sft_weight>0:
                sft_loss=mixer.sft_loss(model,target)
                if not bool(torch.isfinite(sft_loss.detach()).all()):
                    raise ValueError('nonfinite SFT objective; refusing mixed optimizer update')
                (sft_loss*args.sft_weight/args.grad_accum).backward()
                total+=float(sft_loss.detach())*args.sft_weight
            if mixer.ppo is not None and args.ppo_weight>0:
                mixer.critic_optim.zero_grad(set_to_none=True)
                ppo_actor,ppo_critic,metrics=mixer.ppo_loss(model,target)
                if not bool(torch.isfinite(ppo_actor.detach()).all() and
                            torch.isfinite(ppo_critic.detach()).all()):
                    raise ValueError('nonfinite PPO objective; refusing mixed optimizer update')
                (ppo_actor*args.ppo_weight/args.grad_accum).backward()
                # Observable actor gradient signal; distinguish zero-reward
                # no-op runs from Critic updates that do not train the Actor.
                # Reduce on the Actor device and synchronize only ONCE;
                # .item() per adapter tensor severely stalls Apple MPS.
                actor_grad_norm_sq=None
                for parameter in adapter.trainable_parameters():
                    if parameter.grad is not None:
                        term=parameter.grad.detach().float().square().sum()
                        actor_grad_norm_sq=(term if actor_grad_norm_sq is None
                                            else actor_grad_norm_sq+term)
                metrics['actor_grad_norm']=(float(actor_grad_norm_sq.sqrt().detach())
                                            if actor_grad_norm_sq is not None else 0.0)
                if not math.isfinite(metrics['actor_grad_norm']):
                    raise ValueError('nonfinite PPO Actor gradient; refusing optimizer update')
                if (float(getattr(args,'pt_weight',1.))==0 and
                        float(getattr(args,'sft_weight',1.))==0 and
                        metrics['actor_grad_norm']<1e-12 and
                        not getattr(args,'_v100_zero_gradient_warned',False)):
                    print('KAGAMI PPO WARNING: Actor gradient is zero; sparse PPO reward may provide no learning signal. '
                          'For grounded JSONL reference rewards, try --ppo-reference-aux-coef 0.1 in a NEW room.',flush=True)
                    args._v100_zero_gradient_warned=True
                (ppo_critic*args.ppo_weight).backward()
                # AdamW on nan/inf gradients can permanently corrupt the
                # locally persisted Critic even when the actor step is refused.
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                       for p in mixer.critic.parameters() if p.requires_grad):
                    mixer.critic_optim.zero_grad(set_to_none=True)
                    raise ValueError('nonfinite PPO Critic gradient; refusing optimizer update')
                if (getattr(args,'distributed_critic',False) and not getattr(args,'critic_rollout_gossip',False)
                        and getattr(args,'mode',None)=='elastic'):
                    # CPU copy under the SAME lock as the value update. SQLite
                    # commits and RTC fanout happen on the event loop, never
                    # from this model-training worker thread.
                    from .distributed_critic_v78 import critic_params
                    before=[p.detach().float().cpu().clone() for p in critic_params(mixer)]
                else:
                    before=None
                critic_bound=float(getattr(args,'ppo_critic_grad_norm',0.))
                if critic_bound:
                    params=(mixer.critic.trainable_parameters() if getattr(args,'critic_kind','mlp')=='hf'
                            else mixer.critic.parameters())
                    torch.nn.utils.clip_grad_norm_(list(params),critic_bound,error_if_nonfinite=True)
                mixer.critic_optim.step()
                mixer.critic_steps+=1
                if before is not None:
                    args._v77_critic_before=before
                if (getattr(args,'critic_rollout_gossip',False) and getattr(args,'mode',None)=='elastic'
                        and mixer.reuses==1):
                    args._v80_rollout_pending=(mixer.export_rollout_v80(),mixer.cached['behavior_checkpoint_v82'])
                if mixer.critic_steps%10==0 and not getattr(mixer,'_v75_paired',False):
                    mixer.save_critic()
                total+=float(ppo_actor.detach())*args.ppo_weight
                args._v65_ppo_metrics=metrics
            # Independent PT/SFT accuracies; never mix different objectives
            # into one claimed 'accuracy'. Both are local training metrics.
            if (outputs is not None and getattr(args,'telemetry_file',None)
                    and getattr(args,'_metric_step',0)%10==0):
                from .model_contract_v64 import _loss_or_logits
                from .objective_metrics_v107 import token_top1
                with torch.no_grad():
                    try:
                        _,pt_logits=_loss_or_logits(outputs)
                        labels=batch.get('labels')
                        pt_accuracy=token_top1(pt_logits,labels,
                            causal=(getattr(args,'hf_task','causal-lm')=='causal-lm'))
                        args._v107_pt_accuracy=pt_accuracy
                        if pt_accuracy is not None:
                            args._latest_accuracy=pt_accuracy
                    except (RuntimeError,TypeError,ValueError,AttributeError):
                        args._v107_pt_accuracy=None
            args._metric_step=getattr(args,'_metric_step',0)+1
            if not math.isfinite(total):
                raise ValueError('nonfinite mixed objective')
            return total
        # Exact local minibatch Top-1, including dict-style HF outputs.
        # For causal LM it is shifted by one token; classification is unshifted.
        if getattr(args, 'telemetry_file', None) and getattr(args, '_metric_step', 0) % 10 == 0:
            from .model_contract_v64 import _loss_or_logits
            from .objective_metrics_v107 import token_top1
            with torch.no_grad():
                try:
                    _, logits = _loss_or_logits(outputs)
                    value = token_top1(logits, batch.get('labels'),
                        causal=getattr(args, 'hf_task', 'causal-lm')=='causal-lm')
                    args._v107_pt_accuracy = value
                    if value is not None:
                        args._latest_accuracy = value
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    args._v107_pt_accuracy = None
        args._metric_step = getattr(args, '_metric_step', 0) + 1
        (loss / args.grad_accum).backward()
        return float(loss.detach().cpu())

