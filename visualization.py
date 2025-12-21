# visualization.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Execution Times
    axes[0, 0].plot(results['worker_counts'], results['mp_times'], 'bo-', label='Multiprocessing')
    axes[0, 0].plot(results['worker_counts'], results['cf_times'], 'ro-', label='Concurrent.futures')
    axes[0, 0].set_xlabel('Number of Workers')
    axes[0, 0].set_ylabel('Execution Time (seconds)')
    axes[0, 0].set_title('Execution Time vs Workers')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Speedup
    axes[0, 1].plot(results['worker_counts'], results['mp_speedup'], 'bo-', label='Multiprocessing')
    axes[0, 1].plot(results['worker_counts'], results['cf_speedup'], 'ro-', label='Concurrent.futures')
    axes[0, 1].plot(results['worker_counts'], results['worker_counts'], 'k--', label='Ideal')
    axes[0, 1].set_xlabel('Number of Workers')
    axes[0, 1].set_ylabel('Speedup')
    axes[0, 1].set_title('Speedup vs Workers')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Efficiency
    axes[1, 0].plot(results['worker_counts'], results['mp_efficiency'], 'bo-', label='Multiprocessing')
    axes[1, 0].plot(results['worker_counts'], results['cf_efficiency'], 'ro-', label='Concurrent.futures')
    axes[1, 0].set_xlabel('Number of Workers')
    axes[1, 0].set_ylabel('Efficiency')
    axes[1, 0].set_title('Efficiency vs Workers')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Comparison Table
    axes[1, 1].axis('off')
    table_data = []
    for i, w in enumerate(results['worker_counts']):
        table_data.append([
            w,
            f"{results['mp_times'][i]:.2f}",
            f"{results['cf_times'][i]:.2f}",
            f"{results['mp_speedup'][i]:.2f}",
            f"{results['cf_speedup'][i]:.2f}",
            f"{results['mp_efficiency'][i]:.2f}",
            f"{results['cf_efficiency'][i]:.2f}"
        ])
    
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Workers', 'MP Time', 'CF Time', 'MP Speedup', 'CF Speedup', 'MP Eff', 'CF Eff'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.savefig('performance_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Performance graphs saved to 'performance_results.png'")

def save_results_to_csv(results, filename='performance_metrics.csv'):
    df = pd.DataFrame({
        'Workers': results['worker_counts'],
        'Multiprocessing_Time': results['mp_times'],
        'Concurrent_Time': results['cf_times'],
        'MP_Speedup': results['mp_speedup'],
        'CF_Speedup': results['cf_speedup'],
        'MP_Efficiency': results['mp_efficiency'],
        'CF_Efficiency': results['cf_efficiency']
    })
    df.to_csv(filename, index=False)
    print(f"Results saved to '{filename}'")