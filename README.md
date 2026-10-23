# EVOP–V2G Benchmark

Implementation and experimental results for qubit-efficient hybrid quantum–classical optimisation of EV routing and vehicle-to-grid scheduling.

## Files

`run_main.py` : Main experiment runner for calibration, configuration selection and held-out transfer.

`evop_model.py` : EVOP–V2G formulation, exact route validation and reference solver.

`pce_model.py` : Pauli Correlation Encoding, variational circuit and relaxed objective.

`optimisers.py` : Differential Evolution (DE) and SPSA controllers.

`instances.py` : Benchmark instance loading and development/held-out splits.

`experiment_config.json` : Experimental settings, calibration grid, seeds and evaluation budgets.

`final_instances.json` : Frozen benchmark instances.

`requirements.txt` : Python dependencies.

`fleet_results.zip` : Archived experimental results.

## Running

```powershell
python run_main.py --stage all --n 6 7 8 9 10 11 --workers 4
```

Arguments:

- `--stage` : `calibrate`, `select`, `transfer`, or `all`. The `all` option runs calibration, selects configurations and evaluates them on held-out instances.
- `--n` : One or more problem sizes, where N is the total number of customers and charging stations. The reported study uses N = 6–11.
- `--workers` : Number of parallel worker processes.
- `--dry-run` : Displays the experiment plan without executing optimisation.

The final experiments were distributed across virtual machines on the **Nectar Research Cloud**, using sharded calibration and transfer runs. The command above runs the complete pipeline on a single machine.

## Results

New runs are written to `results/main_track/`.

The completed experimental results are stored in `fleet_results.zip`, including:

- Calibration run records
- Selected configurations
- Held-out transfer results
- Exact reference solutions

Extract the archive using:

```powershell
Expand-Archive .\fleet_results.zip -DestinationPath .
```

The reported study consists of 19,440 calibration runs and 5,340 held-out transfer runs across 89 held-out instances.
