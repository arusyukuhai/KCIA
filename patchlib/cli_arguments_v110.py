"""CLI argument groups; no training/model-loading side effects."""
from __future__ import annotations
from pathlib import Path
from .model_loading_v110 import AUTO_CLASSES


def add_model_args(p):
    p.add_argument('--model', required=True, help='HF model ID or local pretrained directory')
    p.add_argument('--model-id', help='Stable canonical identifier; must match peers')
    p.add_argument('--factory-file', help='Untrusted custom Python build_model() factory; offline container required')
    p.add_argument('--trust-remote-code', action='store_true',help='Explicit opt-in, only in sandbox with local pinned HF snapshot')
    p.add_argument('--state-base',help='Apply verified full-state base before attaching any patches')
    p.add_argument('--sharded-base-manifest',type=Path,
                   help='Load existing model state directly from verified .fmanifest shards (no .fstate assembly)')
    p.add_argument('--sharded-base-store',type=Path,
                   help='Content-addressed shard store for --sharded-base-manifest')
    p.add_argument('--initialization',choices=('pretrained','random','config'),default='pretrained',
        help='random creates official TinyStories-1M config weights from a fixed seed (no pretrained weights)')
    p.add_argument('--initial-seed',type=int,default=42,
        help='Shared random model genesis seed; MUST match all room peers')
    p.add_argument('--max-state-gb',type=int,default=32,help='Full state file limit in GiB')
    p.add_argument('--hf-task', choices=tuple(AUTO_CLASSES), default='causal-lm')
    p.add_argument('--auto-class', help='Override with a packaged Transformers AutoModelFor* class')
    p.add_argument('--revision', default='main', help='Prefer pinned Hub commit SHA')
    p.add_argument('--kind', choices=('lora','loha'), default='loha')
    p.add_argument('--rank', type=int, default=16,help='SHARED room LoHa rank; do not auto-change between peers')
    p.add_argument('--auto-rank',action='store_true',
        help='Local train only: device-based rank default; prohibited for P2P genesis')
    p.add_argument('--local-loha-rank', default='auto',
        choices=('auto','full','1','2','4','8','16','32','64','128','256'),
        help='Per-client physical LoHA factor rank (auto by device budget). '
             'Reduces adapter, optimizer and signed event bytes. Pinned per journal.')
    p.add_argument('--resource-report',type=Path,
        help='Write device resource/rank/expert recommendations to JSON')
    p.add_argument('--disk-linear-dir',type=Path,
        help='Opt-in frozen nn.Linear disk-mapped weights; per-layer forward/backward I/O')
    p.add_argument('--live-disk-linear-max-gb',type=int,default=32,
        help='Per live model mutable base on-disk limit in GiB (not a RAM guarantee)')
    p.add_argument('--disk-linear-min-mib',type=float,default=0.,
        help='Only offload Linear weights >= this size in MiB')
    p.add_argument('--wide-lora-rank',type=int,default=4,
        help='Mixed adapter: LoRA rank for automatically detected embeddings, output heads and very rectangular matrices (default 4)')
    p.add_argument('--target-scope',choices=('all','gpt_neo_linear','gpt_neo_full','auto','linear','universal'),default='all',
        help='universal: all supported differentiable Parameters, including 1-D/N-D custom kernels. Explicit opt-in creates a NEW P2P adapter layout')
    p.add_argument('--init-style',choices=('legacy','balanced'),default='balanced',
        help='LoHa initialization: balanced avoids near-zero initial gradients')
    p.add_argument('--optimizer',choices=('adamw','lora-muon'),default='adamw',
        help='Continual adapter optimizer; LoRA-Muon requires --kind lora')
    p.add_argument('--muon-momentum',type=float,default=.9)
    p.add_argument('--muon-weight-decay',type=float,default=0.)
    p.add_argument('--moe-plan',type=Path,help='Shared, deterministic explicit ModuleList expert plan JSON (experimental)')
    p.add_argument('--alpha', type=float, default=None)
    p.add_argument('--max-peers', type=int, default=8)
    p.add_argument('--device', default='cpu', help='cpu / cuda / mps')
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--allow-legacy-bin', action='store_true', help='Opt in to non-safetensors legacy checkpoint files')
    p.add_argument('--streaming-base', action='store_true',
        help='Create HF model on meta and mmap local safetensors; CPU streaming execution only')
    p.add_argument('--streaming-lazy-write',action='store_true',
        help='Experimental train-elastic: no up-front full mutable base copy; materialize each modified tensor once into private scratch')
    p.add_argument('--streaming-factor-ledger',action='store_true',
                   help='EXPERIMENTAL: retain verified before/after LoHA/LoRA factors as virtual base; compact exactly after bounded events')
    p.add_argument('--stream-factor-max-events',type=int,default=32,
                   help='maximum signed contributions per group before exact dense compaction (1..256)')
    p.add_argument('--streaming-page-cow',action='store_true',
        help='Experimental: retain only modified tensor pages on disk; no v2 auto epoch export')
    p.add_argument('--stream-page-kib',type=int,default=64,choices=(4,16,64,256,1024),
        help='Per-tensor sparse COW page size in KiB')
    p.add_argument('--streaming-expert-path',
        help='Expert ModuleList/Dict whose entirely frozen experts run on demand from CPU mappings')
    p.add_argument('--stream-lazy-limit-gb',type=int,default=0,
        help='Maximum lazy writable tensor scratch GiB (0=no hard cap; monitor disk usage)')
    p.add_argument('--streaming-scratch-dir',type=Path,
        help='Fast disk for temporary writable pages; live restarts reconstruct from immutable genesis')
    p.add_argument('--stream-chunk-mib',type=int,default=8,
        help='Max bytes per copied tensor slice during writable streaming initialization')
    p.add_argument('--disk-cache-fraction',type=float,default=.6,
        help='Fraction of free VRAM devoted to opportunistic LRU frozen-weight cache (max .85)')
    p.add_argument('--disk-cache-mib',type=int,default=0,
        help='Optional hard cache cap in MiB; 0 uses detected available memory')
    p.add_argument('--disk-tile-mib',type=int,default=16,
        help='Max frozen-weight transfer tile size in MiB for large projections')


def add_train_args(p):
    add_model_args(p)
    p.add_argument('--peer-id', help='Legacy peer identifier (auto-generated for --room-id)')
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument('--batches-dir', help='Task-neutral pretokenized .safetensors batches')
    src.add_argument('--text-file', help='Raw text for causal/MLM')
    src.add_argument('--jsonl', help='JSONL with input/target or text/label')
    p.add_argument('--room-manifest',type=Path,help='Room v2 Hugging Face dataset collection; no local text file required')
    p.add_argument('--loss-policy',choices=('auto','provided','causal','token','classification'),default='auto',
                   help='Model-neutral loss contract; auto uses returned scalar loss or known task logits')
    p.add_argument('--input-map',default='',
                   help='JSON mapping, e.g. {"input_ids":"tokens","attention_mask":null}; pin consistently across peers')
    # v65: one model, simultaneous gradient contributions from PT, SFT and PPO.
    p.add_argument('--mix-policy',type=Path,
                   help='Optional shared immutable JSON objective protocol: permits PT-only, SFT, or PPO specialized peers in one signed room')
    p.add_argument('--sft-jsonl',type=Path,help='Local JSONL {prompt,response}; alternatively use --sft-hf-repo and related flags')
    p.add_argument('--ppo-jsonl',type=Path,help='Local JSONL {prompt,reference}; alternatively use --ppo-hf-repo and related flags')
    p.add_argument('--sft-hf-repo',default='',help='Pinned Hub SFT repo')
    p.add_argument('--sft-hf-revision',default='',help='Pinned Hub SFT revision')
    p.add_argument('--sft-hf-config',default='',help='Pinned Hub SFT config')
    p.add_argument('--sft-hf-split',default='train',help='Pinned Hub SFT split')
    p.add_argument('--sft-hf-prompt-column',default='prompt',help='Pinned Hub SFT prompt_column')
    p.add_argument('--sft-hf-response-column',default='response',help='Pinned Hub SFT response_column')
    p.add_argument('--sft-hf-rejected-column',default='rejected',help='Pinned Hub SFT rejected column for preference optimization')
    p.add_argument('--ppo-hf-repo',default='',help='Pinned Hub PPO repo')
    p.add_argument('--ppo-hf-revision',default='',help='Pinned Hub PPO revision')
    p.add_argument('--ppo-hf-config',default='',help='Pinned Hub PPO config')
    p.add_argument('--ppo-hf-split',default='train',help='Pinned Hub PPO split')
    p.add_argument('--ppo-hf-prompt-column',default='prompt',help='Pinned Hub PPO prompt_column')
    p.add_argument('--ppo-hf-reference-column',default='reference',help='Pinned Hub PPO reference_column')
    p.add_argument('--pt-weight',type=float,default=1.,help='Pretraining objective gradient coefficient')
    p.add_argument('--sft-weight',type=float,default=1.,help='Response-only SFT objective coefficient')
    p.add_argument('--ppo-weight',type=float,default=.2,help='Clipped PPO objective coefficient')
    p.add_argument('--ppo-clip',type=float,default=.2,help='PPO actor and value clipping epsilon')
    p.add_argument('--ppo-kl-coef',type=float,default=.02,help='Penalty relative to frozen rollout behavior policy')
    p.add_argument('--ppo-value-coef',type=float,default=.5,help='Auxiliary local value estimator coefficient')
    p.add_argument('--ppo-gamma',type=float,default=.99)
    p.add_argument('--ppo-gae-lambda',type=float,default=.95)
    p.add_argument('--ppo-max-tokens',type=int,default=8,help='Max autoregressive PPO rollout tokens per mixed step')
    p.add_argument('--ppo-reuse',type=int,default=2,help='Max clipped PPO updates per sampled trajectory')
    # v99: opt-in stability and reference-free preference learning. Nondefault
    # settings are committed in the mixed-room genesis policy.
    p.add_argument('--ppo-adv-norm',choices=('legacy','rms','none'),default='legacy',
                   help='RMS preserves sign for short/single PPO trajectories')
    p.add_argument('--ppo-target-kl',type=float,default=0.,help='Recollect stale rollouts when per-token proxy KL exceeds this; 0 disables')
    p.add_argument('--ppo-adaptive-kl',action='store_true',help='Adapt KL penalty against the target using a bounded local controller')
    p.add_argument('--ppo-kl-horizon',type=int,default=128,help='Adaptive KL controller horizon in PPO calls')
    p.add_argument('--ppo-entropy-coef',type=float,default=0.,help='Entropy bonus in the PPO actor objective')
    p.add_argument('--ppo-critic-grad-norm',type=float,default=0.,help='Clip local PPO value-head gradients; 0 disables')
    p.add_argument('--ppo-reference-aux-coef',type=float,default=0.,
                   help='Opt-in grounded teacher-forced reference CE for sparse PPO rewards; pinned in new room protocol')
    p.add_argument('--sft-label-smoothing',type=float,default=0.,help='Optional response-only SFT label smoothing')
    p.add_argument('--sft-pref-coef',type=float,default=0.,help='SimPO-style reference-free preference coefficient for SFT JSONL prompt/response/rejected')
    p.add_argument('--sft-pref-beta',type=float,default=2.,help='Scale length-normalized chosen vs rejected log-probability margin')
    p.add_argument('--sft-pref-margin',type=float,default=0.,help='Preference logit margin')
    p.add_argument('--mixed-actor-grad-norm',type=float,default=0.,help='Global norm cap after all PT/SFT/PPO gradients accumulate; 0 disables')
    p.add_argument('--ppo-kv-cache',choices=('auto','off'),default='auto',
                   help='Use causal KV cache during no-gradient PPO rollout if forward explicitly supports it; never for PPO gradients')
    p.add_argument('--critic-kind',choices=('mlp','linear','hf'),default='mlp',help='PPO value critic: local MLP/Linear or pinned Hugging Face scalar value head')
    p.add_argument('--critic-hidden',type=int,default=32,help='Local value critic hidden width (8..512)')
    p.add_argument('--critic-lr',type=float,default=1e-3,help='Local PPO value critic learning rate')
    p.add_argument('--distributed-critic',action='store_true',help='Opt-in independent signed P2P Critic value-head training (new room; PPO peers only)')
    p.add_argument('--critic-rollout-gossip',action='store_true',
        help='EXPERIMENTAL v80: signed replay of peer PPO trajectories with receiver-side IS Critic updates (instead of sharing Critic parameter deltas)')
    p.add_argument('--critic-offpolicy',choices=('auto','gae','trajectory_is'),default='auto',
                   help='Critic value target: auto=unclipped trajectory IS in P2P mode, legacy GAE locally. No clipping; high variance possible')
    p.add_argument('--critic-hf-repo',default='',help='Pinned HF scalar value-model repo (organization/name)')
    p.add_argument('--critic-hf-revision',default='',help='Immutable 40-hex HF value critic revision')
    p.add_argument('--critic-hf-loader',choices=('sequence_classification','token_classification'),default='sequence_classification')
    p.add_argument('--critic-hf-device',choices=('cpu','cuda','mps'),default='cpu')
    p.add_argument('--critic-hf-max-length',type=int,default=512)
    p.add_argument('--critic-hf-head-init',choices=('pretrained','new'),default='pretrained',
        help='PRETRAINED requires a trained num_labels=1 scalar head. NEW constructs/zeros a fresh 1-output head (not pretrained), then trains it with PPO; new room required.')
    p.add_argument('--critic-init',type=Path,help='Optional KCIA safetensors value-critic initial weights')
    p.add_argument('--critic-state-dir',type=Path,help='Private local value-critic optimizer+weights directory')
    p.add_argument('--ppo-reward',choices=('token_match','exact','hf_model'),default='token_match',
                   help='Verifiable ground-truth reward; never model-judged')
    p.add_argument('--reward-model-id',default='',help='Hugging Face scalar sequence-classification Reward Model repo')
    p.add_argument('--reward-model-revision',default='',help='Pinned 40-hex commit SHA of Reward Model')
    p.add_argument('--reward-model-device',choices=('cpu','cuda','mps'),default='cpu',help='Inference device, independent from actor')
    p.add_argument('--reward-model-idle-offload',action='store_true',
                   help='Opt-in move frozen HF Reward Model from accelerator to CPU between PPO scoring calls (saves resident VRAM, transfer overhead)')
    p.add_argument('--reward-model-loader',choices=('sequence_classification','auto_model'),default='sequence_classification')
    p.add_argument('--reward-model-format',choices=('plain','chat'),default='plain')
    p.add_argument('--reward-model-output',choices=('logits','score','end_scores','rewards'),default='logits')
    p.add_argument('--reward-model-index',type=int,default=0,help='Reward column (index 0 for scalar)')
    p.add_argument('--reward-model-scale',type=float,default=1.0)
    p.add_argument('--reward-model-max-length',type=int,default=512)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--disk-optimizer-dir',type=Path,
        help='Opt-in disk-mapped AdamW moments; elastic live uses journal-tip-bound working snapshots')
    p.add_argument('--disk-optimizer-max-gb',type=int,default=32,
        help='Per-instance disk AdamW moment budget in GiB (elastic mode)')
    p.add_argument('--resume-disk-optimizer',action='store_true',
        help='Resume external AdamW moments; requires matching factor layout')
    p.add_argument('--log-steps', type=int, default=10)
    p.add_argument('--resume', help='Previous local .fpatch with exactly matching model')
    p.add_argument('--wire-dtype', choices=['float32','float16'], default='float16')
    p.add_argument('--output', default='local.fpatch')
    p.add_argument('--full-state-output',help='Also emit full effective tensor state as .fstate')
    p.add_argument('--shard-state-manifest',type=Path,help='Also export effective weights as shards, no .fstate intermediate')
    p.add_argument('--shard-store',type=Path,help='Storage for --shard-state-manifest')
    p.add_argument('--shard-mb',type=int,default=4)


def add_identity(p):
    p.add_argument('--signing-key',help='Ed25519 private key PEM for outgoing file signatures')
    p.add_argument('--peer-public-key',help='Pinned Ed25519 public key PEM; reject unsigned incoming')

