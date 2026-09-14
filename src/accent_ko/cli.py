import argparse
import csv
import json
from pathlib import Path
from .io import read_json, read_jsonl, write_json, unique_index


def main():
    p = argparse.ArgumentParser(description='Multi-AudioJail 한국어 입력 재현')
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--model-config', default='configs/models.json')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('validate')
    sub.add_parser('matrix')
    plan = sub.add_parser('plan')
    plan.add_argument('--suite', choices=['main', 'defense', 'ablation', 'sqa'], default='main')
    plan.add_argument('--groups', nargs='+', choices=['native', 'natural', 'synthetic'])
    plan.add_argument('--out', type=Path, required=True)
    audio = sub.add_parser('audio')
    audio.add_argument('--manifest', type=Path, required=True)
    audio.add_argument('--limit', type=int)
    imp = sub.add_parser('import-clean')
    imp.add_argument('--manifest', type=Path, required=True)
    imp.add_argument('--audio-id', required=True)
    imp.add_argument('--source', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--manifest', type=Path, required=True)
    run.add_argument('--model', required=True, choices=['qwen2', 'diva', 'meralion', 'minicpm', 'ultravox'])
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--device', default='cuda')
    run.add_argument('--limit', type=int)
    judge = sub.add_parser('judge')
    judge.add_argument('--manifest', type=Path, required=True)
    judge.add_argument('--responses', type=Path, required=True)
    judge.add_argument('--out', type=Path, required=True)
    judge.add_argument('--kind', choices=['guard', 'sqa_judge'], default='guard')
    judge.add_argument('--device', default='cuda')
    summary = sub.add_parser('summarize')
    summary.add_argument('--manifest', type=Path, required=True)
    summary.add_argument('--responses', type=Path, required=True)
    summary.add_argument('--labels', type=Path, required=True)
    summary.add_argument('--out', type=Path, required=True)
    wer = sub.add_parser('transcribe')
    wer.add_argument('--manifest', type=Path, required=True)
    wer.add_argument('--out', type=Path, required=True)
    wer.add_argument('--device', default='cuda')
    ws = sub.add_parser('wer-summary')
    ws.add_argument('--transcripts', type=Path, required=True)
    ws.add_argument('--out', type=Path, required=True)
    human = sub.add_parser('human-sample')
    human.add_argument('--manifest', type=Path, required=True)
    human.add_argument('--responses', type=Path, required=True)
    human.add_argument('--out', type=Path, required=True)
    human_score = sub.add_parser('human-score')
    human_score.add_argument('--annotations', type=Path, required=True)
    human_score.add_argument('--labels', type=Path, required=True)
    human_score.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if hasattr(args, 'limit') and args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    root = args.root.resolve()
    config = read_json(root / 'configs/experiment.json')
    voices = read_json(root / 'configs/voices.json')
    models = read_json(root / args.model_config)
    if args.command == 'validate':
        from .data import load_dataset
        rows, hashes = load_dataset(root)
        result = {'rows': len(rows), 'sha256': hashes,
                  'unresolved_voices': sum(v['voice_id'] is None for v in voices),
                  'unresolved_model_revisions': [k for k, v in models.items() if not v['revision']]}
    elif args.command == 'matrix':
        from .plan import counts
        result = counts(voices)
    elif args.command == 'plan':
        from .plan import save
        result = save(root, args.out, config, voices, args.suite, args.groups)
    elif args.command in ('audio', 'import-clean'):
        from .audio import materialize, import_clean
        # Always use the frozen manifest settings, not a subsequently edited config.
        frozen = read_json(args.manifest / 'provenance.json')['config']
        items = read_jsonl(args.manifest / 'audio.jsonl')
        if args.command == 'import-clean':
            item = unique_index(items)[args.audio_id]
            result = {'path': str(import_clean(root, item, frozen, args.source))}
        else:
            for item in items[:args.limit] if args.limit is not None else items:
                materialize(root, item, frozen)
            result = {'processed': len(items[:args.limit] if args.limit is not None else items)}
    elif args.command == 'run':
        from .run import execute
        result = execute(root, args.manifest, args.out, args.model, models[args.model], args.device, args.limit)
    elif args.command == 'judge':
        from .evaluate import judge_responses
        judge_responses(args.manifest, args.responses, args.out, models[args.kind], args.kind, args.device)
        result = {'output': str(args.out)}
    elif args.command == 'summarize':
        from .evaluate import summarize, add_deltas
        result = add_deltas(summarize(args.manifest, args.responses, args.labels))
        write_json(args.out, result)
        result = {'groups': len(result), 'output': str(args.out)}
    elif args.command == 'transcribe':
        from .evaluate import transcribe
        transcribe(args.manifest, root, args.out, models['whisper'], args.device)
        result = {'output': str(args.out)}
    elif args.command == 'wer-summary':
        from .evaluate import wer_summary
        write_json(args.out, wer_summary(read_jsonl(args.transcripts)))
        result = {'output': str(args.out)}
    elif args.command == 'human-score':
        from .evaluate import human_scores
        result = human_scores(args.annotations, args.labels)
        write_json(args.out, result)
    elif args.command == 'human-sample':
        import random
        jobs = unique_index(read_jsonl(args.manifest / 'jobs.jsonl'))
        records = read_jsonl(args.responses)
        unique_index(records)
        pool = [r for r in records if r['status'] == 'ok' and jobs[r['id']]['condition'] == 'echo' and jobs[r['id']]['group'] == 'native']
        if len(pool) < 50:
            raise ValueError('Need at least 50 successful Echo responses for the paper human audit')
        frozen = read_json(args.manifest / 'provenance.json')['config']
        selected = random.Random(frozen['seed']).sample(sorted(pool, key=lambda r: r['id']), 50)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'question', 'response', 'human_label'])
            writer.writeheader()
            for r in selected:
                writer.writerow({'id': r['id'], 'question': jobs[r['id']]['text'], 'response': r['response'], 'human_label': ''})
        result = {'sampled': 50, 'output': str(args.out)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
