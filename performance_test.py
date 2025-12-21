# performance_test.py
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
