import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

def plot_results(results):
    # Make the figure a little wider to accommodate a denser table
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    workers = results['worker_counts']

    # Statistical values (use defaults if missing)
    mp_median = results.get('mp_times', [0]*len(workers))
    mp_ci = results.get('mp_ci95_half', [0]*len(workers))
    mp_std = results.get('mp_std', [0]*len(workers))

    cf_median = results.get('cf_times', [0]*len(workers))
    cf_ci = results.get('cf_ci95_half', [0]*len(workers))
    cf_std = results.get('cf_std', [0]*len(workers))

    # Plot 1: Execution Times with 95% CI error bars
    axes[0, 0].errorbar(workers, mp_median, yerr=mp_ci, fmt='o-', label='Multiprocessing (median ±95% CI)', capsize=5, color='b')
    axes[0, 0].errorbar(workers, cf_median, yerr=cf_ci, fmt='o-', label='Concurrent.futures (median ±95% CI)', capsize=5, color='r')
    axes[0, 0].set_xlabel('Number of Workers')
    axes[0, 0].set_ylabel('Execution Time (seconds)')
    axes[0, 0].set_title('Execution Time vs Workers (with 95% CI)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Speedup
    axes[0, 1].plot(workers, results['mp_speedup'], 'bo-', label='Multiprocessing')
    axes[0, 1].plot(workers, results['cf_speedup'], 'ro-', label='Concurrent.futures')
    axes[0, 1].plot(workers, workers, 'k--', label='Ideal')
    axes[0, 1].set_xlabel('Number of Workers')
    axes[0, 1].set_ylabel('Speedup')
    axes[0, 1].set_title('Speedup vs Workers')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Efficiency
    axes[1, 0].plot(workers, results['mp_efficiency'], 'bo-', label='Multiprocessing')
    axes[1, 0].plot(workers, results['cf_efficiency'], 'ro-', label='Concurrent.futures')
    axes[1, 0].set_xlabel('Number of Workers')
    axes[1, 0].set_ylabel('Efficiency')
    axes[1, 0].set_title('Efficiency vs Workers')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Comparison Table with statistical columns
    axes[1, 1].axis('off')
    table_data = []
    for i, w in enumerate(workers):
        table_data.append([
            w,
            f"{mp_median[i]:.2f}±{mp_ci[i]:.2f}",
            f"{mp_std[i]:.2f}",
            f"{cf_median[i]:.2f}±{cf_ci[i]:.2f}",
            f"{cf_std[i]:.2f}",
            f"{results['mp_speedup'][i]:.2f}",
            f"{results['cf_speedup'][i]:.2f}",
            f"{results['mp_efficiency'][i]:.2f}",
            f"{results['cf_efficiency'][i]:.2f}"
        ])

    # Use shorter / wrapped column labels to help fitting
    col_labels = ['Workers', 'MP\nMedian±CI(s)', 'MP\nStd(s)', 'CF\nMedian±CI(s)', 'CF\nStd(s)', 'MP\nSpeedup', 'CF\nSpeedup', 'MP\nEff', 'CF\nEff']

    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )

    # Auto-adjust column widths and font size for readability
    try:
        table.auto_set_column_width(col=list(range(len(col_labels))))
    except Exception:
        # older matplotlib versions may not have auto_set_column_width
        pass

    table.auto_set_font_size(False)
    ncols = len(col_labels)
    # pick a font size that scales with number of columns
    fontsize = 8 if ncols <= 8 else max(6, 10 - ncols)
    table.set_fontsize(fontsize)
    table.scale(1, 1.2)

    # Give the table more horizontal room
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.05)

    plt.tight_layout()
    plt.savefig('performance_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Performance graphs saved to 'performance_results.png'")

def save_results_to_csv(results, filename='performance_metrics.csv'):
    # Flatten results into per-method rows including statistics and raw runs
    rows = []
    repeats = results.get('repeats', '')
    for i, w in enumerate(results['worker_counts']):
        mp_runs = results.get('mp_runs', [[]])[i]
        cf_runs = results.get('cf_runs', [[]])[i]
        rows.append({
            'Method': 'multiprocessing',
            'Workers': w,
            'Repeats': repeats,
            'Mean': results.get('mp_mean', [None])[i],
            'Median': results.get('mp_times', [None])[i],
            'Std': results.get('mp_std', [None])[i],
            'CI95Half': results.get('mp_ci95_half', [None])[i],
            'Runs': ";".join(f"{x:.4f}" for x in mp_runs)
        })
        rows.append({
            'Method': 'concurrent',
            'Workers': w,
            'Repeats': repeats,
            'Mean': results.get('cf_mean', [None])[i],
            'Median': results.get('cf_times', [None])[i],
            'Std': results.get('cf_std', [None])[i],
            'CI95Half': results.get('cf_ci95_half', [None])[i],
            'Runs': ";".join(f"{x:.4f}" for x in cf_runs)
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Results saved to '{filename}'")


def plot_distributions(results, out_path='performance_distributions.png'):
    """Create boxplots of per-run timings and a bar chart of mean ± 95% CI for each method and worker count."""
    import numpy as np
    from matplotlib.patches import Patch

    workers = results['worker_counts']
    mp_runs = results.get('mp_runs', [[] for _ in workers])
    cf_runs = results.get('cf_runs', [[] for _ in workers])

    mp_means = results.get('mp_mean', [None]*len(workers))
    mp_ci = results.get('mp_ci95_half', [0]*len(workers))
    cf_means = results.get('cf_mean', [None]*len(workers))
    cf_ci = results.get('cf_ci95_half', [0]*len(workers))

    # Create figure with two subplots: boxplots and mean±CI bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Boxplots grouped by worker count: MP and CF side-by-side
    pos_mp = [i*2 + 1 for i in range(len(workers))]
    pos_cf = [i*2 + 2 for i in range(len(workers))]

    # Safe: if a run list is empty, replace with [np.nan] so boxplot can handle it
    mp_box_data = [r if r else [np.nan] for r in mp_runs]
    cf_box_data = [r if r else [np.nan] for r in cf_runs]

    b1 = axes[0].boxplot(mp_box_data, positions=pos_mp, widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='blue'))
    b2 = axes[0].boxplot(cf_box_data, positions=pos_cf, widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor='lightpink', color='red'), medianprops=dict(color='red'))

    # X ticks between each MP/CF pair
    xticks = [(a + b) / 2.0 for a, b in zip(pos_mp, pos_cf)]
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels([str(w) for w in workers])
    axes[0].set_xlabel('Number of Workers')
    axes[0].set_ylabel('Execution Time (s)')
    axes[0].set_title('Per-run Timing Distributions (boxplots)')
    # Legend
    axes[0].legend([Patch(facecolor='lightblue'), Patch(facecolor='lightpink')], ['Multiprocessing', 'Concurrent'], loc='upper right')
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    # Bar chart: mean ± CI
    x = np.arange(len(workers))
    width = 0.35
    axes[1].bar(x - width/2, mp_means, width, yerr=mp_ci, capsize=5, label='Multiprocessing', color='steelblue')
    axes[1].bar(x + width/2, cf_means, width, yerr=cf_ci, capsize=5, label='Concurrent', color='indianred')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(w) for w in workers])
    axes[1].set_xlabel('Number of Workers')
    axes[1].set_ylabel('Mean Execution Time (s)')
    axes[1].set_title('Mean ± 95% CI')
    axes[1].legend()
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution plots saved to '{out_path}'")