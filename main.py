# main.py
from performance_test import run_performance_tests
from visualization import plot_results, save_results_to_csv

def main():
    # Configuration
    INPUT_DIR = "food_subset"
    OUTPUT_BASE_DIR = "output_results"
    
    print("=" * 50)
    print("PARALLEL IMAGE PROCESSING PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Run performance tests
    results = run_performance_tests(INPUT_DIR, OUTPUT_BASE_DIR)
    
    # Generate visualizations
    plot_results(results)
    
    # Save data to CSV
    save_results_to_csv(results)
    
    # Print summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)

    # Detailed per-worker results
    for i, w in enumerate(results['worker_counts']):
        print(f"\nWorkers: {w}")

        mp_runs = results.get('mp_runs', [])
        if mp_runs and i < len(mp_runs):
            for r_idx, t in enumerate(mp_runs[i], 1):
                print(f"  MP run {r_idx}: {t:.2f}s")
            print(f"  MP median time: {results['mp_times'][i]:.2f}s")
        else:
            print(f"  MP time: {results['mp_times'][i]:.2f}s")

        cf_runs = results.get('cf_runs', [])
        if cf_runs and i < len(cf_runs):
            for r_idx, t in enumerate(cf_runs[i], 1):
                print(f"  CF run {r_idx}: {t:.2f}s")
            print(f"  CF median time: {results['cf_times'][i]:.2f}s")
        else:
            print(f"  CF time: {results['cf_times'][i]:.2f}s")

    print("\nBest Performance:")
    print(f"- Multiprocessing: {min(results['mp_times']):.2f}s ({results['worker_counts'][results['mp_times'].index(min(results['mp_times']))]} workers)")
    print(f"- Concurrent.futures: {min(results['cf_times']):.2f}s ({results['worker_counts'][results['cf_times'].index(min(results['cf_times']))]} workers)")
    
    print("\nMaximum Speedup:")
    print(f"- Multiprocessing: {max(results['mp_speedup']):.2f}x")
    print(f"- Concurrent.futures: {max(results['cf_speedup']):.2f}x")
    
    print("\nFiles Generated:")
    print("1. performance_results.png - Performance graphs")
    print("2. performance_metrics.csv - Raw performance data")
    print("3. output_results/ - Processed images by worker count")

if __name__ == "__main__":
    main()