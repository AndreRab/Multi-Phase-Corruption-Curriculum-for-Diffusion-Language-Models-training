# Experiment configs

Create one JSON file per experiment and pass it to `main.py`.

Example:

```bash
python3 main.py --config experiments/example.json --dry-run
python3 main.py --config experiments/example.json
```

Use all visible GPUs with PyTorch DistributedDataParallel:

```bash
torchrun --nproc_per_node=4 main.py --config experiments/example.json
```

Set `--nproc_per_node` to the number of GPUs you want to use. Each process
uses one GPU, and the dataset is sharded across processes.
`batch_size` is per GPU, so the global batch size is
`batch_size * nproc_per_node`.

You can also override a single value from the command line:

```bash
python3 main.py --config experiments/example.json --set model_id=v2
python3 main.py --config experiments/example.json --set learning_rate=0.000003
python3 main.py --config experiments/example.json --set iterations_intervals='[1, 1, 1]'
```

`run_experiment.py` still works as a short alias:

```bash
python3 run_experiment.py experiments/example.json --dry-run
```

Supported JSON keys:

- `mode` (`train` or `eval`)
- `model_name`
- `seed` (optional; omit it to leave random number generators unseeded)
- `dataset_path`
- `dataset_name`
- `iterations_intervals`
- `learning_rate`
- `result_folder`
- `model_save_path`
- `model_id`
- `training_output_save_path`
- `train_output_id`
- `num_diffusion_steps`
- `train_diffusion_steps`
- `rollout_loss_decay`
- `corruption_methods`
- `corruption_minimum_probability`
- `corruption_maximum_probability`
- `similar_number_of_neighbors`
- `denoise_iterations`
- `max_length`
- `batch_size`

If `seed` is omitted or set to `null`, the run is intentionally unseeded. New
training outputs and evaluation rows store `"seed": null`, making such runs
visibly different from seeded runs. Historical result files created before
this field was added remain unchanged and should be treated according to the
configuration used to create them. When a non-negative integer is provided,
Python, NumPy, and PyTorch random generators are initialized with that value.

Evaluation always creates its initial corrupted sequences at the maximum
timestep (`num_diffusion_steps - 1`). This makes one-step and iterative
evaluation start from the same maximally noisy state; training keeps its
per-example random starting timesteps.

Training and evaluation share the same corruption contract:

- method order is `similar`, `mask`, `random`;
- `iterations_intervals` follows that order during training;
- `corruption_minimum_probability`, `corruption_maximum_probability`, and
  `similar_number_of_neighbors` are explicit training parameters;
- evaluation `corruption_rates` controls the per-condition maximum probability
  while using the same method implementations and minimum probability.

## Intentional train/evaluation difference

Training and final evaluation do not use exactly the same position-update
policy by design. During training, each rollout step feeds the model's own
predictions back into the next step for all corrupted positions. This exposes
the model to its self-generated errors and provides a broad learning signal
for iterative correction.

During evaluation, denoising is more conservative: only the currently most
confident positions are updated, while the remaining positions are left for
later iterations. This confidence-based selective refinement models the
intended inference behaviour, where reliable corrections improve the context
before less certain tokens are changed. The difference is therefore an
intentional training strategy versus final inference policy, not an accidental
implementation mismatch.

For evaluation, use `mode: "eval"` and the evaluation keys below:

- `models_path` (list of checkpoint paths)
- `dataset_path` and `dataset_name` (or only `dataset_name` as a full dataset id)
- `dataset_split`
- `corruption_methods` (np. `similar`, `mask`, `random`)
- `corruption_rates` (każdy rate zostanie przetestowany z każdą metodą)
- `corruption_minimum_probability`
- `similar_number_of_neighbors`
- `result_folder`
- `output_file`
- `num_diffusion_steps`
- `denoise_iterations`
- `max_length`
- `batch_size`

Example evaluation command:

```bash
python3 main.py --config experiments/eval.json --dry-run
torchrun --nproc_per_node=2 main.py --config experiments/eval.json
```

## Reconstruction evaluation metrics

Evaluation now adds `bleu4`, `rouge_l_f1`, `top5_accuracy_denoising`, and
`mauve` to each model/corruption/rate result. Install project dependencies
with `pip install -e .` before evaluation. All four scores use a 0–1 scale.

BLEU is corpus BLEU-4 (SacreBLEU 13a tokenizer, exponential smoothing,
no effective-order fallback); its signature is saved with results.
ROUGE-L is mean per-text F1 (English rouge-score tokenizer, no stemming).
Both compare final full reconstructions against the same truncated clean
sequences. Padding is excluded via attention masks; generated special tokens
remain visible. At low corruption rates, unchanged context contributes to
these scores.

Top-5 counts only originally corrupted, non-padding positions, exactly once
when committed. It uses the logits from that commitment step and two running
counters, without storing logits or a trajectory. Existing argmax decoding and
accuracy metrics are preserved.

MAUVE is computed on the complete gathered corpus on rank 0, never averaged
across batches or workers. Test shards do not duplicate examples. Options:

- `metrics`: list of metrics to compute; defaults to all supported metrics.
  Use `["bleu4", "rouge_l_f1"]` to compute only these missing scores.
  Enum names: `loss`, `accuracy`, `denoising_accuracy`, `bleu_4`,
  `rouge_l`, `top_5_accuracy`, `mauve`, `masked_perplexity`. Earlier result-key names are accepted
  as config aliases. JSON result keys stay unchanged for compatibility.
  Disabled metric dependencies are not imported; disabled metric fields are omitted.
- `mauve_model`: defaults to `gpt2-large` (downloaded on first use).
- `mauve_device_id`: defaults to `-1` (CPU); use `0` for GPU 0.
- `mauve_max_text_length`: defaults to `256` evaluator tokens.

MAUVE uses the evaluation seed, or 25 when none is supplied. Keep evaluator,
length, seed, and corpus size fixed between comparisons. A few thousand texts
per corpus are recommended by MAUVE's authors. Corpora with fewer than two texts or any empty text
produce a null score with status metadata when computation is undefined;
missing dependencies or evaluator download failures raise an error.

Example lightweight run:

```bash
python main.py --config experiments_config/eval/eval_2.json --set 'metrics=["bleu4","rouge_l_f1"]' --set output_file=missing_text_metrics.json
```

Use a separate `output_file` for incremental metric runs to preserve previous results.
Evaluation regenerates reconstructions; existing aggregate JSON files do not contain
the texts or commitment logits needed to calculate these new metrics retroactively.
Set the same seed and configuration to reproduce the corruption benchmark.

Metric implementations live in separate classes under `metrics/` and extend
`BaseMetric`. `metrics/factory.py` maps each member of your `METRICS` enum
(in `metrics/metirc_enum.py`) to its implementation. Config strings are parsed
into enum members; only selected metric classes are instantiated. Adding a
metric requires an enum member, a class, and its factory mapping. Token metrics
accumulate sufficient statistics; corpus metrics score the gathered final texts.

### Masked-token perplexity and new seeded checkpoints

`masked_perplexity` is `exp(sum(masked token NLL) / num_masked_positions)`
from the initial, one-step forward pass at the configured corruption level.
Only positions selected for corruption whose input ID equals the tokenizer's
mask token ID are scored; padding is excluded. NLL sums and token counts are
summed across batches and distributed workers before exponentiation.
`num_masked_positions` reports the denominator. If there are no masked tokens,
the score is `null`. This is conditional denoising perplexity, not autoregressive
sequence perplexity; compare scores at the same corruption rate and dataset.
It does not depend on the number of iterative denoising steps.

48 configs under `experiments_config/eval/new/seed_{42,123,2026}/` cover all
39 new training checkpoints at `/net/tscratch/people/plgarebowski/results/new/`.
Each seed has `denoising_<K>/` for K = 1, 2, 4, 8, 16, 32, 64, 100.
Each K contains `one_step_training.json` (7 models trained with one step) and
`four_step_training.json` (6 models trained with four steps).
`denoise_iterations` is K in both configs; corruption rates remain unchanged.
They retain loss, accuracy,
denoising accuracy, BLEU-4, ROUGE-L and top-5 accuracy, and add masked perplexity.
Evaluation results go to
`results/eval/new/seed_<seed>/denoising_<K>/<training phase>/evaluation_results.json`.
The same seed and checkpoint order are used across K to reproduce the same
corrupted inputs. Initial-pass masked perplexity should therefore stay the same
across K; iterative reconstruction metrics can change.
Missing checkpoints raise an error before model loading instead of silently
evaluating a newly initialized model.

Example on an allocated GPU:

```bash
python main.py --config experiments_config/eval/new/seed_42/denoising_8/one_step_training.json
```

To submit all 48 configs as a Slurm array, using the same four-GPU torchrun
setup as training:

```bash
bash scripts/submit_new_eval.sh
```

The submitter snapshots a sorted config list into `logs/eval_configs.*` for each
submission. `scripts/eval_new.slurm` selects one config per array task and logs
its seed, denoising iteration count, and training variant. Logs are written to
`logs/%x_%A_%a.out` and `.err`. Slurm determines concurrency.

The default time limit is four hours. You can override it when submitting, e.g.:

```bash
bash scripts/submit_new_eval.sh --time=12:00:00
```

For perplexity alone, pass `--set 'metrics=["masked_perplexity"]'`; this skips
iterative denoising and text metrics. Keep the same seed to regenerate the same
corruptions. Checkpoints must be copied back to these scratch paths if they were
removed after uploading to Hugging Face.
