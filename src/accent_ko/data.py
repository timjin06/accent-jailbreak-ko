"""Preserve row identity; never use the target continuation as a query."""
import csv
import re
from pathlib import Path
from .io import digest, file_hash, read_json


def load_dataset(root):
    root = Path(root)
    def read(name):
        with (root / name).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    original, translated = read('advbench.csv'), read('advbench_ko.csv')
    if len(original) != 520 or len(translated) != 520:
        raise ValueError('Expected exactly 520 rows in each AdvBench CSV')
    rows = []
    for i, (en, ko) in enumerate(zip(original, translated)):
        if en['goal'] != ko['goal'] or en['target'] != ko['target']:
            raise ValueError(f'English source alignment mismatch at row {i}')
        if not all(ko.get(k, '').strip() for k in ('goal', 'target', 'goal_ko', 'target_ko')):
            raise ValueError(f'Missing required field at row {i}')
        if not re.search('[가-힣]', ko['goal_ko']):
            raise ValueError(f'No Hangul in translated goal at row {i}')
        rows.append({'id': f'advbench-{i:04d}', 'source_row': i,
                     'text': ko['goal_ko'], 'text_sha256': digest(ko['goal_ko'])})
    return rows, {name: file_hash(root / name) for name in ('advbench.csv', 'advbench_ko.csv')}


def accent_subset(path, rows):
    spec = read_json(path)
    ids = spec['prompt_ids']
    if len(ids) != 400 or len(set(ids)) != 400:
        raise ValueError('Accent subset requires exactly 400 distinct prompt_ids; no automatic sampling')
    if not spec.get('provenance'):
        raise ValueError('Record the provenance of the 400-row selection')
    by_id = {r['id']: r for r in rows}
    if set(ids) - by_id.keys():
        raise ValueError('Accent subset contains unknown prompt_ids')
    return [by_id[i] for i in ids]


def sqa_dataset(path):
    from .io import read_jsonl, unique_index
    rows = read_jsonl(path)
    unique_index(rows)
    if len(rows) != 100:
        raise ValueError('SQA requires the full 100 translated question/answer pairs')
    for r in rows:
        if not r.get('text') or not r.get('answer') or not r.get('source'):
            raise ValueError('SQA rows require id, text, answer, source')
        if not re.search('[가-힣]', r['text']):
            raise ValueError('SQA question must be Korean')
    return rows
