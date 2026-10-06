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
