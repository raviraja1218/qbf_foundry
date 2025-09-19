# src/generate_hw_analysis_notebook.py
"""
Generate a starter Jupyter notebook for hardware validation analysis.

Usage:
  python src/generate_hw_analysis_notebook.py --dir reports/mitigated_hw_runs
"""
import os, json, argparse
import nbformat as nbf

NOTEBOOK_FILENAME = "02_hw_validation_analysis.ipynb"

def make_notebook(out_dir):
    nb = nbf.v4.new_notebook()
    cells = []

    intro = """# Hardware Validation — Analysis
This notebook loads the per-run mitigated hardware results and runs statistical comparisons and plots.
Generated automatically; run the cells to perform the analysis and customize figures for publication.
"""
    cells.append(nbf.v4.new_markdown_cell(intro))

    load_cell = f"""import os, pandas as pd, json
base_dir = "{out_dir}"
summary_csv = os.path.join(base_dir, "summary.csv")
metrics_csv = os.path.join(base_dir, "metrics_per_run.csv")
metrics_json = os.path.join(base_dir, "metrics_summary.json")

print("Files found:", [p for p in [summary_csv, metrics_csv, metrics_json] if os.path.exists(p)])
if os.path.exists(metrics_csv):
    metrics = pd.read_csv(metrics_csv)
    display(metrics)
else:
    print("metrics_per_run.csv not found; run src/compare_mitigated_vs_raw.py first.")"""
    cells.append(nbf.v4.new_code_cell(load_cell))

    stats_cell = """# Basic statistics & plots
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
if 'metrics' in globals():
    figs = []
    for col in ['js_div','kl','l1','l2','classical_fidelity']:
        if col in metrics.columns:
            vals = metrics[col].dropna().values
            if len(vals) > 0:
                plt.figure(figsize=(5,2))
                plt.title(col)
                plt.boxplot(vals, vert=False)
                plt.savefig(os.path.join(base_dir, f"plots/{col}_boxplot.png"))
                plt.show()
    # paired test example (if you have simulator comparisons, replace arrays accordingly)
else:
    print("metrics dataframe missing.")"""
    cells.append(nbf.v4.new_code_cell(stats_cell))

    advanced_cell = """# Bootstrap CI example for classical_fidelity
import numpy as np
def bootstrap_ci(data, n_boot=10000, alpha=0.05):
    n = len(data)
    boots = []
    for _ in range(n_boot):
        s = np.random.choice(data, size=n, replace=True)
        boots.append(np.mean(s))
    lo = np.percentile(boots, 100*alpha/2)
    hi = np.percentile(boots, 100*(1-alpha/2))
    return lo, hi

if 'metrics' in globals() and 'classical_fidelity' in metrics.columns:
    arr = metrics['classical_fidelity'].dropna().values
    print("n=", len(arr), "mean=", np.mean(arr))
    lo, hi = bootstrap_ci(arr, n_boot=2000)
    print("Bootstrap 95% CI:", (lo, hi))"""
    cells.append(nbf.v4.new_code_cell(advanced_cell))

    nb['cells'] = cells
    path = os.path.join(out_dir, NOTEBOOK_FILENAME)
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="reports/mitigated_hw_runs")
    args = p.parse_args()
    out_dir = args.dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    make_notebook(out_dir)

if __name__ == "__main__":
    main()
