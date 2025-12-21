# performance_test.py
import statistics
from parallel_multiprocessing import parallel_process_multiprocessing
from parallel_concurrent import parallel_process_concurrent

def run_performance_tests(input_dir, output_base_dir, repeats=3):
    worker_counts = [1, 2, 4, 8]
    multiprocessing_times = []
    concurrent_times = []

    mp_runs = []
    cf_runs = []

    print("=" * 50)
    print("PERFORMANCE TESTING")
    print("=" * 50)

    for workers in worker_counts:
        print(f"\nTesting with {workers} worker(s)...")

        # Multiprocessing (multiple repeats, take median)
        mp_times_this = []
        for i in range(repeats):
            output_dir = f"{output_base_dir}/multiprocessing_{workers}_run{i}"
            try:
                t = parallel_process_multiprocessing(input_dir, output_dir, workers)
                mp_times_this.append(t)
                print(f"  MP run {i+1}: {t:.3f}s")
            except Exception as e:
                print(f"  MP run {i+1} failed: {e}")
        mp_median = statistics.median(mp_times_this) if mp_times_this else float('inf')
        multiprocessing_times.append(mp_median)
        mp_runs.append(mp_times_this)
        print(f"  MP median Time: {mp_median:.3f}s")

        # Concurrent.futures (multiple repeats, take median)
        cf_times_this = []
        for i in range(repeats):
            output_dir = f"{output_base_dir}/concurrent_{workers}_run{i}"
            try:
                t = parallel_process_concurrent(input_dir, output_dir, workers)
                cf_times_this.append(t)
                print(f"  CF run {i+1}: {t:.3f}s")
            except Exception as e:
                print(f"  CF run {i+1} failed: {e}")
        cf_median = statistics.median(cf_times_this) if cf_times_this else float('inf')
        concurrent_times.append(cf_median)
        cf_runs.append(cf_times_this)
        print(f"  CF median Time: {cf_median:.3f}s")

    # Safe baselines
    baseline_mp = multiprocessing_times[0] or 1.0
    baseline_cf = concurrent_times[0] or 1.0

    mp_speedup = [baseline_mp / t if t and t != float('inf') else 0.0 for t in multiprocessing_times]
    cf_speedup = [baseline_cf / t if t and t != float('inf') else 0.0 for t in concurrent_times]

    mp_efficiency = [s / w for s, w in zip(mp_speedup, worker_counts)]
    cf_efficiency = [s / w for s, w in zip(cf_speedup, worker_counts)]

    return {
        'worker_counts': worker_counts,
        'mp_times': multiprocessing_times,
        'cf_times': concurrent_times,
        'mp_runs': mp_runs,
        'cf_runs': cf_runs,
        'mp_speedup': mp_speedup,
        'cf_speedup': cf_speedup,
        'mp_efficiency': mp_efficiency,
        'cf_efficiency': cf_efficiency
    }

