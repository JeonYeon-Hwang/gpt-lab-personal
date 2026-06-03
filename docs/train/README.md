# Train Automation

This folder stores the mini GPT sentiment experiment loop state.

## Commands

Run one automation cycle:

```bash
python3 scripts/train_auto_loop.py --once
```

Run repeatedly every 5 minutes:

```bash
python3 scripts/train_auto_loop.py --interval-minutes 5
```

Run one experiment directly:

```bash
python3 scripts/run_sentiment_experiment.py
```

Optional config override:

```bash
python3 scripts/train_auto_loop.py --once --config-json docs/train/local_config.json
```

## Files

- `status.md`: latest automation state
- `train.lock`: active training process lock
- `experiment_queue.json`: pending experiment configs
- `experiment_history.json`: completed run summaries
- `runs/{run_id}.json`: full run result
- `runs/{run_id}.md`: readable run report
- `runs/{run_id}_tokenizer.json`: tokenizer saved with the run
- `runs/{run_id}_latest.pt`: latest checkpoint
- `runs/{run_id}_best.pt`: best validation-loss checkpoint

## Data Rules

- Tokenizer training uses train data only.
- Validation data is used for model selection and overfit checks.
- Test data is not used by this automation loop.

## Overfit Follow-Up

When an epoch crosses the overfit threshold, the loop creates at most three probe candidates from that epoch checkpoint:

1. `drop_rate_only`
2. `n_layers_only`
3. `drop_rate_and_n_layers`

Changing only `drop_rate` can reuse checkpoint weights strictly. Changing
`n_layers` uses partial model loading and starts a fresh optimizer state.

Each candidate is plotted as a point in the branch decision graph:

- x-axis: `overfit_score` where lower is better
- y-axis: best validation accuracy where higher is better
- point label: candidate type, `drop_rate`, `n_layers`, validation accuracy, selection score
- selected path: best candidate points connected across branch events

The selected candidate config is inserted at the front of `experiment_queue.json`
so the next automation cycle continues from that best save point.
