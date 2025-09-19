# QBF — Phase 2 snapshot

## Summary
Phase 2 produced a hybrid quantum-classical protein design engine and hardware validation runs.

Key files:
- reports/hw_vs_sim_results.csv
- reports/calibration_M_q2_shots256.npz
- reports/mitigated_hw_runs/metrics_per_run.csv
- reports/mitigated_hw_runs/02_hw_validation_analysis.ipynb
- reports/mitigated_hw_runs/02_hw_validation_analysis.html (export)

## Reproduce (local)
1. Activate conda environment:
   conda env create -f environment.yml
   conda activate qbf_clean

2. Generate analysis (already done in this snapshot):
   python src/compare_mitigated_vs_raw.py --dir reports/mitigated_hw_runs
   python src/analyze_hw_vs_sim_stats.py --csv reports/hw_vs_sim_results.csv --out reports/hw_vs_sim_analysis

Notes:
