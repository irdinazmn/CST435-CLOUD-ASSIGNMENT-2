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

    # Plot 2: Speedup (mean ± 95% CI when available)
    mp_s_mean = results.get('mp_speedup_mean', results.get('mp_speedup'))
    mp_s_err = results.get('mp_speedup_ci_half', [0]*len(workers))
    cf_s_mean = results.get('cf_speedup_mean', results.get('cf_speedup'))
    cf_s_err = results.get('cf_speedup_ci_half', [0]*len(workers))

    axes[0, 1].errorbar(workers, mp_s_mean, yerr=mp_s_err, fmt='o-', label='Multiprocessing (mean ±95% CI)', capsize=5, color='b')
    axes[0, 1].errorbar(workers, cf_s_mean, yerr=cf_s_err, fmt='o-', label='Concurrent.futures (mean ±95% CI)', capsize=5, color='r')
    axes[0, 1].plot(workers, workers, 'k--', label='Ideal')
    axes[0, 1].set_xlabel('Number of Workers')
    axes[0, 1].set_ylabel('Speedup')
    axes[0, 1].set_title('Speedup vs Workers (with 95% CI)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Efficiency (mean ± 95% CI when available)
    mp_eff_mean = results.get('mp_eff_mean', results.get('mp_efficiency'))
    mp_eff_err = results.get('mp_eff_ci_half', [0]*len(workers))
    cf_eff_mean = results.get('cf_eff_mean', results.get('cf_efficiency'))
    cf_eff_err = results.get('cf_eff_ci_half', [0]*len(workers))

    axes[1, 0].errorbar(workers, mp_eff_mean, yerr=mp_eff_err, fmt='o-', label='Multiprocessing (mean ±95% CI)', capsize=5, color='b')
    axes[1, 0].errorbar(workers, cf_eff_mean, yerr=cf_eff_err, fmt='o-', label='Concurrent.futures (mean ±95% CI)', capsize=5, color='r')
    axes[1, 0].set_xlabel('Number of Workers')
    axes[1, 0].set_ylabel('Efficiency')
    axes[1, 0].set_title('Efficiency vs Workers (with 95% CI)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Throughput vs Workers with CPU correlation (replaces the table in the main figure)
    axes[1, 1].clear()
    mp_thr_means = results.get('mp_throughput_means', [0]*len(workers))
    mp_thr_ci = results.get('mp_throughput_ci_half', [0]*len(workers))
    cf_thr_means = results.get('cf_throughput_means', [0]*len(workers))
    cf_thr_ci = results.get('cf_throughput_ci_half', [0]*len(workers))

    ax_thr = axes[1, 1]
    ax_thr.errorbar(workers, mp_thr_means, yerr=mp_thr_ci, fmt='o-', capsize=5, label='Multiprocessing (imgs/s, ±95% CI)', color='b')
    ax_thr.errorbar(workers, cf_thr_means, yerr=cf_thr_ci, fmt='o-', capsize=5, label='Concurrent (imgs/s, ±95% CI)', color='r')
    ax_thr.set_xlabel('Number of Workers')
    ax_thr.set_ylabel('Throughput (images/sec)')
    ax_thr.set_title('Throughput vs Workers (with CPU correlation)')
    ax_thr.grid(True)

    # twin axis for average CPU %
    ax_cpu = ax_thr.twinx()
    mp_cpu = results.get('mp_cpu_means', [None]*len(workers))
    cf_cpu = results.get('cf_cpu_means', [None]*len(workers))
    ax_cpu.plot(workers, mp_cpu, 'b--x', label='MP avg CPU%')
    ax_cpu.plot(workers, cf_cpu, 'r--x', label='CF avg CPU%')
    ax_cpu.set_ylabel('Avg CPU%')

    # combined legend for throughput and CPU
    handles1, labels1 = ax_thr.get_legend_handles_labels()
    handles2, labels2 = ax_cpu.get_legend_handles_labels()
    ax_thr.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    # keep the same layout adjustments but ensure enough room for plots
    plt.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.03)

    plt.tight_layout()
    plt.savefig('performance_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Performance graphs saved to 'performance_results.png'")

    # save the table as a separate image for convenience
    try:
        _save_table_image(results, out_path='performance_table.png')
    except Exception:
        pass

    # generate distribution plots (kept as separate file)
    try:
        plot_distributions(results)
    except Exception as e:
        print(f"Failed to generate distribution plots: {e}")

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
            'ThroughputMean': results.get('mp_throughput_means', [None])[i],
            'ThroughputStd': results.get('mp_throughput_std', [None])[i],
            'ThroughputCI95Half': results.get('mp_throughput_ci_half', [None])[i],
            'CPU_Mean': results.get('mp_cpu_means', [None])[i],
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
            'ThroughputMean': results.get('cf_throughput_means', [None])[i],
            'ThroughputStd': results.get('cf_throughput_std', [None])[i],
            'ThroughputCI95Half': results.get('cf_throughput_ci_half', [None])[i],
            'CPU_Mean': results.get('cf_cpu_means', [None])[i],
            'Runs': ";".join(f"{x:.4f}" for x in cf_runs)
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Results saved to '{filename}'")


# def plot_distributions(results, out_path='performance_distributions.png'):
#     """Create boxplots of per-run timings and a bar chart of mean ± 95% CI for each method and worker count."""
#     import numpy as np
#     from matplotlib.patches import Patch

#     workers = results['worker_counts']
#     mp_runs = results.get('mp_runs', [[] for _ in workers])
#     cf_runs = results.get('cf_runs', [[] for _ in workers])

#     mp_means = results.get('mp_mean', [None]*len(workers))
#     mp_ci = results.get('mp_ci95_half', [0]*len(workers))
#     cf_means = results.get('cf_mean', [None]*len(workers))
#     cf_ci = results.get('cf_ci95_half', [0]*len(workers))

#     # Create figure with two subplots: boxplots and mean±CI bar chart
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#     # Boxplots grouped by worker count: MP and CF side-by-side
#     pos_mp = [i*2 + 1 for i in range(len(workers))]
#     pos_cf = [i*2 + 2 for i in range(len(workers))]

#     # Safe: if a run list is empty, replace with [np.nan] so boxplot can handle it
#     mp_box_data = [r if r else [np.nan] for r in mp_runs]
#     cf_box_data = [r if r else [np.nan] for r in cf_runs]

#     b1 = axes[0].boxplot(mp_box_data, positions=pos_mp, widths=0.6, patch_artist=True,
#                          boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='blue'))
#     b2 = axes[0].boxplot(cf_box_data, positions=pos_cf, widths=0.6, patch_artist=True,
#                          boxprops=dict(facecolor='lightpink', color='red'), medianprops=dict(color='red'))

#     # X ticks between each MP/CF pair
#     xticks = [(a + b) / 2.0 for a, b in zip(pos_mp, pos_cf)]
#     axes[0].set_xticks(xticks)
#     axes[0].set_xticklabels([str(w) for w in workers])
#     axes[0].set_xlabel('Number of Workers')
#     axes[0].set_ylabel('Execution Time (s)')
#     axes[0].set_title('Per-run Timing Distributions (boxplots)')
#     # Legend
#     axes[0].legend([Patch(facecolor='lightblue'), Patch(facecolor='lightpink')], ['Multiprocessing', 'Concurrent'], loc='upper right')
#     axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

#     # Bar chart: mean ± CI
#     x = np.arange(len(workers))
#     width = 0.35
#     axes[1].bar(x - width/2, mp_means, width, yerr=mp_ci, capsize=5, label='Multiprocessing', color='steelblue')
#     axes[1].bar(x + width/2, cf_means, width, yerr=cf_ci, capsize=5, label='Concurrent', color='indianred')
#     axes[1].set_xticks(x)
#     axes[1].set_xticklabels([str(w) for w in workers])
#     axes[1].set_xlabel('Number of Workers')
#     axes[1].set_ylabel('Mean Execution Time (s)')
#     axes[1].set_title('Mean ± 95% CI')
#     axes[1].legend()
#     axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Distribution plots saved to '{out_path}'")


def plot_throughput(results, out_path='performance_throughput.png'):
    """Plot images/sec (throughput) vs workers with 95% CI error bars for both methods."""
    workers = results['worker_counts']

    mp_means = results.get('mp_throughput_means', [0]*len(workers))
    mp_ci = results.get('mp_throughput_ci_half', [0]*len(workers))
    cf_means = results.get('cf_throughput_means', [0]*len(workers))
    cf_ci = results.get('cf_throughput_ci_half', [0]*len(workers))

    plt.figure(figsize=(8,5))
    plt.errorbar(workers, mp_means, yerr=mp_ci, fmt='o-', capsize=5, label='Multiprocessing (mean ±95% CI)', color='b')
    plt.errorbar(workers, cf_means, yerr=cf_ci, fmt='o-', capsize=5, label='Concurrent (mean ±95% CI)', color='r')
    plt.xlabel('Number of Workers')
    plt.ylabel('Throughput (images/sec)')
    plt.title('Throughput vs Workers')
    plt.grid(True)

    # add avg CPU% on secondary y-axis to help correlate CPU vs throughput
    mp_cpu = results.get('mp_cpu_means', [None]*len(workers))
    cf_cpu = results.get('cf_cpu_means', [None]*len(workers))
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(workers, mp_cpu, 'b--x', label='MP avg CPU%')
    ax2.plot(workers, cf_cpu, 'r--x', label='CF avg CPU%')
    ax2.set_ylabel('Avg CPU%')

    # combined legend
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc='upper left')

    plt.xticks(workers)
    # annotate analysis summary if available
    summary = results.get('analysis_summary', '')
    if summary:
        plt.gcf().text(0.98, 0.02, summary.split('\n')[0], fontsize=8, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Throughput plot saved to '{out_path}'")


def _save_table_image(results, out_path='performance_table.png'):
    """Render the summary table as a standalone PNG for easy inspection in reports."""
    workers = results['worker_counts']

    mp_median = results.get('mp_times', [0]*len(workers))
    mp_ci = results.get('mp_ci95_half', [0]*len(workers))
    mp_std = results.get('mp_std', [0]*len(workers))

    cf_median = results.get('cf_times', [0]*len(workers))
    cf_ci = results.get('cf_ci95_half', [0]*len(workers))
    cf_std = results.get('cf_std', [0]*len(workers))

    table_data = []
    for i, w in enumerate(workers):
        table_data.append([
            w,
            f"{mp_median[i]:.2f}±{mp_ci[i]:.2f}",
            f"{mp_std[i]:.2f}",
            f"{cf_median[i]:.2f}±{cf_ci[i]:.2f}",
            f"{cf_std[i]:.2f}",
            f"{results.get('mp_speedup', [None]*len(workers))[i]:.2f}",
            f"{results.get('cf_speedup', [None]*len(workers))[i]:.2f}",
            f"{results.get('mp_efficiency', [None]*len(workers))[i]:.2f}",
            f"{results.get('cf_efficiency', [None]*len(workers))[i]:.2f}"
        ])

    col_labels = ['Workers', 'MP Median±CI(s)', 'MP Std(s)', 'CF Median±CI(s)', 'CF Std(s)', 'MP Speedup', 'CF Speedup', 'MP Eff', 'CF Eff']

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.2), max(4, len(workers) * 0.35)))
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')

    try:
        table.auto_set_column_width(col=list(range(len(col_labels))))
    except Exception:
        pass

    table.auto_set_font_size(False)
    ncols = len(col_labels)
    fontsize = 8 if ncols <= 8 else max(6, 10 - ncols)
    table.set_fontsize(fontsize)
    table.scale(1, 1.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Table saved to '{out_path}'")