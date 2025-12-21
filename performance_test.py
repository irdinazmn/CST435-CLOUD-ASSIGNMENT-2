# performance_test.py
import time
import matplotlib.pyplot as plt
from parallel_multiprocessing import parallel_process_multiprocessing
from parallel_concurrent import parallel_process_concurrent

def run_performance_tests(input_dir, output_base_dir):
    worker_counts = [1, 2, 4, 8]  # Test with different number of workers
    multiprocessing_times = []
    concurrent_times = []
    
    print("=" * 50)
    print("PERFORMANCE TESTING")
    print("=" * 50)
    
    for workers in worker_counts:
        print(f"\nTesting with {workers} worker(s)...")
        
        # Test multiprocessing
        output_dir = f"{output_base_dir}/multiprocessing_{workers}"
        print("  Multiprocessing...", end=" ")
        mp_time = parallel_process_multiprocessing(input_dir, output_dir, workers)
        multiprocessing_times.append(mp_time)
        print(f"Time: {mp_time:.2f}s")
        
        # Test concurrent.futures
        output_dir = f"{output_base_dir}/concurrent_{workers}"
        print("  Concurrent.futures...", end=" ")
        cf_time = parallel_process_concurrent(input_dir, output_dir, workers)
        concurrent_times.append(cf_time)
        print(f"Time: {cf_time:.2f}s")
    
    # Calculate speedup
    baseline_mp = multiprocessing_times[0]  # 1 worker time
    baseline_cf = concurrent_times[0]       # 1 worker time
    
    mp_speedup = [baseline_mp / t for t in multiprocessing_times]
    cf_speedup = [baseline_cf / t for t in concurrent_times]
    
    # Calculate efficiency
    mp_efficiency = [s / w for s, w in zip(mp_speedup, worker_counts)]
    cf_efficiency = [s / w for s, w in zip(cf_speedup, worker_counts)]
    
    return {
        'worker_counts': worker_counts,
        'mp_times': multiprocessing_times,
        'cf_times': concurrent_times,
        'mp_speedup': mp_speedup,
        'cf_speedup': cf_speedup,
        'mp_efficiency': mp_efficiency,
        'cf_efficiency': cf_efficiency
    }

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
    plt.show()

if __name__ == "__main__":
    # Create subset first
    from create_subset import create_subset
    subset_dir = create_subset(classes=10, images_per_class=10)  # 100 images
    
    # Run tests
    results = run_performance_tests(subset_dir, "output_results")
    
    # Plot results
    plot_results(results)
    
    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame({
        'Workers': results['worker_counts'],
        'Multiprocessing_Time': results['mp_times'],
        'Concurrent_Time': results['cf_times'],
        'MP_Speedup': results['mp_speedup'],
        'CF_Speedup': results['cf_speedup'],
        'MP_Efficiency': results['mp_efficiency'],
        'CF_Efficiency': results['cf_efficiency']
    })
    df.to_csv('performance_metrics.csv', index=False)
    print("\nResults saved to 'performance_metrics.csv' and 'performance_results.png'")