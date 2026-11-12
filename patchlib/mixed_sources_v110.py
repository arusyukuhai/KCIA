"""Bounded, streaming local JSONL reader used by mixed training."""
from __future__ import annotations
import json
from pathlib import Path
from .mixed_errors_v110 import MixedLearningError


class JsonlCycle:
    """Streaming fixed-size lines; never keeps a multi-GB corpus in RAM."""
    def __init__(self, path):
        self.path=Path(path)
        self.file=self.path.open('r',encoding='utf8')
        self.line_number=0
        self.file.seek(0,2)
        if self.file.tell()==0:
            raise MixedLearningError(f'{path}: empty dataset')
        self.file.seek(0)

    def next(self):
        attempts=0
        while True:
            line=self.file.readline(65537)
            if not line:
                if attempts:
                    raise MixedLearningError(f'{self.path}: no nonempty JSONL rows')
                self.file.seek(0);self.line_number=0;attempts+=1;continue
            self.line_number+=1
            if len(line)>65536 or (line and not line.endswith('\n') and not self.file.tell()==self.path.stat().st_size):
                raise MixedLearningError(f'{self.path}:{self.line_number} JSONL line too large')
            if not line.strip():
                continue
            try:
                row=json.loads(line)
            except ValueError as exc:
                raise MixedLearningError(f'{self.path}:{self.line_number} invalid JSON') from exc
            if not isinstance(row,dict):
                raise MixedLearningError('training row must be a JSON object')
            return row

    def close(self):
        self.file.close()

