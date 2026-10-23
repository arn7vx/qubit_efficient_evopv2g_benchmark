# Qubit-Efficient Hybrid Quantum-Classical EVOP–V2G Optimisation

Research code and frozen experimental evidence accompanying the Monash University honours thesis by Arnav Patial.

## Benchmark implementation

- `run_main.py` — calibration, configuration selection and held-out transfer (including exact reference computation).
- `evop_model.py` — simplified EVOP–V2G model, exact route validation and classical reference solver.
- `pce_model.py` — Pauli Correlation Encoding, variational circuit and relaxed objective.
- `optimisers.py` — differential evolution (DE), SPSA and run execution.
- `instances.py` — benchmark instance loading and development/held-out splits.
- `experiment_config.json` — experiment design, calibration grid, controllers, seeds and budgets.
- `final_instances.json` — frozen benchmark instance catalogue.
- `requirements.txt` — runner dependencies (unpinned; not an exact environment lockfile).

## Recorded experiments

`fleet_results.zip` is a curated copy of the original experiment archive. It preserves:

- `fleet_results/progress.csv`, identifying the six final calibration/transfer batches;
- the calibration run CSV shards and exact reference baselines;
- the six calibration selection directories, including selected settings and metadata;
- the six held-out transfer directories, including run records and metadata.

The archive excludes VM coordinator logs, deployment files, duplicate master analysis material and nested ZIPs. The full original archive is retained privately by the researcher. Checksums of the original and curated archives are in `ARCHIVE_PROVENANCE.txt`.

The completed study covers **N = 6–11**, with **19,440 calibration runs** and **5,340 held-out transfer runs** across **89 held-out instances**. The experiment configuration also defines larger N values that were not evaluated in the reported study.

To inspect the recorded results in PowerShell:

```powershell
Expand-Archive -LiteralPath .\fleet_results.zip -DestinationPath .
Import-Csv .\fleet_results\progress.csv | Format-Table
```

## Running the benchmark

```powershell
python -m pip install -r requirements.txt
python run_main.py --stage all --n 6 7 8 9 10 11 --workers 4 --dry-run
```

Remove `--dry-run` only to run the computationally intensive experiment. New runs write to the runner's output directories and do not change the preserved historical ZIP. Results and runtimes can vary with the environment; the requirements file does not pin dependency versions.

## Attribution

Research and development: Arnav Patial, under the supervision and guidance of Charith Jayasekara, Muhammad Aamir Cheema, Muhammad Usman and Harish Vallury. The thesis cites the underlying methods and prior work.
