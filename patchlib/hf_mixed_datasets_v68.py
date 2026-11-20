"""Immutable-revision, bounded-memory Hugging Face iterable sources for SFT/PPO.

All HF loading happens within the local trainer; URLs, dataset scripts and
untrusted Python factories are not accepted as source IDs.  Dataset source
settings are local and intentionally absent from the room-wide objective policy.
"""
from __future__ import annotations
import re
from collections.abc import Mapping

_REPO = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$')
_SHA = re.compile(r'^[0-9a-fA-F]{40}$')
_COMPONENT = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$')

class HFMixedDatasetError(ValueError):
    pass


def source_spec(args, role):
    """Validate and return pinned source for role, or None for local JSONL."""
    if role not in ('sft','ppo'):
        raise HFMixedDatasetError('unknown dataset role')
    prefix=role+'_hf_'
    repo=str(getattr(args,prefix+'repo','') or '').strip()
    local=getattr(args,role+'_jsonl',None)
    if not repo:
        if any(str(getattr(args,prefix+field,'') or '').strip() for field in ('revision','config')):
            raise HFMixedDatasetError(f'{role}: Hub revision/config provided without repository')
        return None
    if local:
        raise HFMixedDatasetError(f'{role}: choose local JSONL OR a Hub dataset, not both')
    sha=str(getattr(args,prefix+'revision','') or '').strip()
    config=str(getattr(args,prefix+'config','') or '').strip()
    split=str(getattr(args,prefix+'split','train') or '').strip()
    prompt=str(getattr(args,prefix+'prompt_column','prompt') or '').strip()
    other_name='response' if role=='sft' else 'reference'
    other=str(getattr(args,prefix+other_name+'_column',other_name) or '').strip()
    if not _REPO.fullmatch(repo):raise HFMixedDatasetError(f'{role}: Hub repo must be organization/name')
    if not _SHA.fullmatch(sha):raise HFMixedDatasetError(f'{role}: pin a 40-character immutable Hub commit SHA')
    if config and not _COMPONENT.fullmatch(config):raise HFMixedDatasetError(f'{role}: invalid dataset config')
    if not _COMPONENT.fullmatch(split):raise HFMixedDatasetError(f'{role}: split must be a single named split')
    for value in (prompt,other):
        if not _COMPONENT.fullmatch(value):raise HFMixedDatasetError(f'{role}: columns must be simple names or dotted fields')
    if role=='sft' and prompt==other:
        raise HFMixedDatasetError('SFT prompt and response columns must differ')
    result={'repo':repo,'revision':sha.lower(),'config':config or None,'split':split,
            'prompt_column':prompt,other_name+'_column':other}
    if role=='sft':
        rejected=str(getattr(args,'sft_hf_rejected_column','rejected') or '').strip()
        if not _COMPONENT.fullmatch(rejected) or rejected in (prompt,other):
            raise HFMixedDatasetError('SFT rejected column must be a distinct simple or dotted field')
        result['rejected_column']=rejected
        # Ordinary SFT must ignore unrelated/unparseable 'rejected' metadata.
        # Only explicit preference training interprets the extra column.
        result['require_rejected']=float(getattr(args,'sft_pref_coef',0) or 0)>0
    return result


def source_selected(args, role):
    return bool(getattr(args,role+'_jsonl',None) or getattr(args,role+'_hf_repo',''))


def _field(row,name):
    item=row
    for key in name.split('.'):
        if not isinstance(item,Mapping) or key not in item:
            raise HFMixedDatasetError(f'Missing Hub column {name!r} (check config, split and field mapping)')
        item=item[key]
    return item


def _text(value, field):
    # Chat turns in a single column can be projected to a plain text prompt/answer.
    # No hidden Jinja or arbitrary Python eval is performed.
    if isinstance(value,list) and value and all(isinstance(item,Mapping) for item in value):
        result=[]
        for item in value:
            content=item.get('content',item.get('text'))
            if not isinstance(content,str):
                raise HFMixedDatasetError(f'{field}: chat message needs string content')
            role=item.get('role','user')
            if role not in ('system','user','assistant','tool'):
                raise HFMixedDatasetError(f'{field}: unsupported chat role')
            result.append(f'{role}: {content}')
        value='\n'.join(result)
    if not isinstance(value,str) or not value or len(value)>16384:
        raise HFMixedDatasetError(f'{field}: expected nonempty string (or chat turns) <=16Ki chars')
    return value



# Only recognize exact, widely used prompt/target pairs.  Arbitrary column
# guessing would silently train on the wrong supervision signal.
_COMMON_PAIRS = {
    'sft': (
        ('instruction', 'output'), ('question', 'answer'),
        ('prompt', 'completion'), ('query', 'response'),
        ('question', 'response'), ('instruction', 'response'),
        ('query', 'answer'), ('input', 'output'),
    ),
    'ppo': (
        ('question', 'answer'), ('instruction', 'output'),
        ('prompt', 'completion'), ('query', 'reference'),
        ('question', 'reference'), ('prompt', 'answer'),
        ('query', 'answer'), ('input', 'output'),
    ),
}
_DEFAULT_PAIR = {'sft': ('prompt', 'response'), 'ppo': ('prompt', 'reference')}
_EMPTY = object()


def _available(row):
    """Column names only; never put training data in exceptions/logs."""
    return ', '.join(sorted(str(k)[:80] for k in row.keys())[:40]) or '(none)'


def _exists(row, path):
    try:
        _field(row, path)
        return True
    except HFMixedDatasetError:
        return False


def _mapping(row, spec, role):
    other = 'response' if role == 'sft' else 'reference'
    requested = (spec['prompt_column'], spec[other+'_column'])
    prompt_ok, other_ok = (_exists(row, part) for part in requested)
    if prompt_ok and (other_ok or role == 'ppo'):
        return ('columns', requested)

    # Only defaults can be safely adapted automatically. A caller who supplied
    # explicit field names gets a precise error rather than an unexpected map.
    inferred = []
    if requested == _DEFAULT_PAIR[role]:
        if role == 'sft' and not prompt_ok and not other_ok and 'messages' in row:
            # Standard OpenAI-style chat dataset. Last assistant message is
            # training target, all prior turns are prompt, never its answer.
            return ('messages', None)
        for pair in _COMMON_PAIRS[role]:
            if _exists(row, pair[0]) and _exists(row, pair[1]):
                inferred.append(pair)
        # With external HF Reward Model only a prompt is needed; a reference
        # is neither required nor synthesized. Never choose among candidates.
        if role == 'ppo' and not prompt_ok and not inferred:
            prompts = [name for name in ('question','instruction','query','input','text')
                       if _exists(row, name)]
            if len(prompts) == 1:
                return ('columns', (prompts[0], requested[1]))
    if len(inferred) == 1:
        return ('columns', inferred[0])

    suggestion = ('; plausible mappings: ' + ', '.join(f'{a}/{b}' for a,b in inferred)
                  if inferred else '')
    explanation = ('ambiguous dataset fields' if len(inferred)>1 else
                   'Missing Hub column ' + repr(requested[0] if not prompt_ok else requested[1]))
    raise HFMixedDatasetError(
        f'{role.upper()} dataset mapping: {explanation}. '
        f'Available columns: [{_available(row)}]. '
        f'Configured prompt={requested[0]!r}, {other}={requested[1]!r}'
        f'{suggestion}. Select the matching {role.upper()} prompt/{other} column '
        f'in TRAINING GUI or use --{role}-hf-prompt-column and '
        f'--{role}-hf-{other}-column. '
        'Nested dict paths such as instruction.text are supported; '
        'datasets with unsupported conversation formats require an explicit conversion.'
    )


def _chat_sft(row):
    messages=row.get('messages')
    if not isinstance(messages,list) or len(messages)<2:
        raise HFMixedDatasetError('SFT messages requires at least two chat turns')
    answer=messages[-1]
    if not isinstance(answer,Mapping) or answer.get('role')!='assistant':
        raise HFMixedDatasetError('SFT messages must finish with an assistant answer')
    # _text validates role vocabulary and string content for previous messages.
    prompt=_text(messages[:-1], 'prompt')
    response=_text(answer.get('content',answer.get('text')), 'response')
    return {'prompt':prompt, 'response':response}


class HFRowCycle:
    """Stream Parquet/Arrow-backed Hub rows without staging an entire corpus."""
    def __init__(self, spec, role, *, dataset_factory=None):
        if role not in ('sft','ppo'):raise HFMixedDatasetError('unknown role')
        self.spec=dict(spec)
        self.role=role
        if dataset_factory is None:
            try:from datasets import load_dataset
            except ImportError as exc:
                raise HFMixedDatasetError('Install huggingface datasets for SFT/PPO Hub streaming') from exc
            dataset_factory=load_dataset
        self._factory=dataset_factory
        self._resolved_mapping=None
        self._peeked=_EMPTY
        self._open()

    def _open(self):
        s=self.spec
        # HF Datasets no longer runs remote dataset loading scripts. We never
        # set trust_remote_code=True and do not execute Hub-provided Python.
        try:
            self._dataset=self._factory(s['repo'],name=s['config'],split=s['split'],
                                        revision=s['revision'],streaming=True)
            self._rows=iter(self._dataset)
        except Exception as exc:
            raise HFMixedDatasetError(f"Cannot stream {self.role} Hub dataset {s['repo']} at pinned SHA: {exc}") from exc

    def peek(self):
        """Validate and retain the first row, before HF reward/critic downloads."""
        if self._peeked is _EMPTY:
            self._peeked=self.next()
        return self._peeked

    def _project(self,row):
        if self._resolved_mapping is None:
            self._last_columns=tuple(sorted(str(key) for key in row.keys()))[:40]
            self._resolved_mapping=_mapping(row,self.spec,self.role)
            mode,fields=self._resolved_mapping
            if mode=='messages':
                display='messages -> prompt + final assistant response'
            else:
                other='response' if self.role=='sft' else 'reference'
                display=f'{fields[0]} -> prompt; {fields[1]} -> {other}'
            # Only field names are logged, never dataset contents.
            print(f'KCIA HF DATASET MAP role={self.role} repo={self.spec["repo"]} {display}',flush=True)
        mode,fields=self._resolved_mapping
        if mode=='messages':
            if self.role=='sft' and self.spec.get('require_rejected',False):
                raise HFMixedDatasetError('Preference SFT needs explicit prompt/response/rejected columns, not messages-only rows')
            return _chat_sft(row)
        prompt=_text(_field(row,fields[0]),'prompt')
        result={'prompt':prompt}
        other='response' if self.role=='sft' else 'reference'
        try:
            value=_field(row,fields[1])
        except HFMixedDatasetError:
            if self.role=='ppo':return result  # Valid only for an external RM.
            raise
        result[other]=_text(value,other)
        if self.role=='sft' and self.spec.get('require_rejected',False):
            # Fail at Hub preflight before loading expensive actor/critic weights.
            rejected=self.spec.get('rejected_column','rejected')
            result['rejected']=_text(_field(row,rejected),'rejected')
        return result

    def next(self):
        if self._peeked is not _EMPTY:
            result=self._peeked
            self._peeked=_EMPTY
            return result
        for attempt in range(2):
            try:
                row=next(self._rows)
            except StopIteration:
                if attempt:break
                restarted=iter(self._dataset)
                if restarted is self._rows:
                    self._open()
                else:
                    self._rows=restarted
                continue
            except Exception as exc:
                raise HFMixedDatasetError(f'{self.role}: Hub streaming failed, check login/network/dataset: {exc}') from exc
            if not isinstance(row,Mapping):raise HFMixedDatasetError('Hub dataset must yield dict rows')
            return self._project(row)
        raise HFMixedDatasetError(f'{self.role}: selected Hub split is empty')

    def close(self):
        self._rows=iter(())
        self._dataset=None
