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
- `max_length`
- `batch_size`

For evaluation, use `mode: "eval"` and the evaluation keys below:

- `models_path` (list of checkpoint paths)
- `dataset_path` and `dataset_name` (or only `dataset_name` as a full dataset id)
- `dataset_split`
- `corruption_methods` (np. `similar`, `mask`, `random`)
- `corruption_rates` (każdy rate zostanie przetestowany z każdą metodą)
- `result_folder`
- `output_file`
- `num_diffusion_steps`
- `max_length`
- `batch_size`

Example evaluation command:

```bash
python3 main.py --config experiments/eval.json --dry-run
torchrun --nproc_per_node=2 main.py --config experiments/eval.json
```
