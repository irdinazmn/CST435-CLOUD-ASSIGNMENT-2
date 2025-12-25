import statistics
from parallel_multiprocessing import parallel_process_multiprocessing
from parallel_concurrent import parallel_process_concurrent


def compute_metrics(worker_counts, mp_times, cf_times):
    """
    Compute speedup and efficiency given baseline (first) timings and per-worker times.

    Parameters
    - worker_counts: list of ints (e.g. [1,2,4,8])
    - mp_times: list of median times for multiprocessing corresponding to worker_counts
    - cf_times: list of median times for concurrent.futures corresponding to worker_counts

    Returns a dict with keys:
    - 'mp_speedup', 'cf_speedup', 'mp_efficiency', 'cf_efficiency'

    Notes:
    - Baseline is the first (index 0) time for each approach. If the baseline is missing or
      infinite, a safe default of 1.0 is used to avoid divide-by-zero.
    - Times that are `float('inf')` (e.g., all runs failed) are treated as unavailable and
      mapped to speedup 0.0 and efficiency 0.0.
    """
    # Defensive baseline selection
    baseline_mp = mp_times[0] if mp_times and mp_times[0] not in (0.0, float('inf')) else 1.0
    baseline_cf = cf_times[0] if cf_times and cf_times[0] not in (0.0, float('inf')) else 1.0

    mp_speedup = [baseline_mp / t if t and t != float('inf') else 0.0 for t in mp_times]
    cf_speedup = [baseline_cf / t if t and t != float('inf') else 0.0 for t in cf_times]

    mp_efficiency = [s / w for s, w in zip(mp_speedup, worker_counts)]
    cf_efficiency = [s / w for s, w in zip(cf_speedup, worker_counts)]

    return {
        'mp_speedup': mp_speedup,
        'cf_speedup': cf_speedup,
        'mp_efficiency': mp_efficiency,
        'cf_efficiency': cf_efficiency
    }


def run_performance_tests(input_dir, output_base_dir, repeats=3):
    # Run repeated runs for each worker count and compute median times, speedup, and efficiency.
    #
    #   Returns:
    #       dict with keys: 'worker_counts', 'mp_times', 'cf_times', 'mp_runs', 'cf_runs',
    #       'mp_speedup', 'cf_speedup', 'mp_efficiency', 'cf_efficiency'
    worker_counts = [1, 2, 4, 8]
    multiprocessing_times = []
    concurrent_times = []

    mp_runs = []
    cf_runs = []

    # containers for statistics per worker count
    mp_means = []
    mp_stds = []
    mp_ci95_half = []

    cf_means = []
    cf_stds = []
    cf_ci95_half = []

    print("=" * 50)
    print("PERFORMANCE TESTING")
    print("=" * 50)

    for workers in worker_counts:
        print(f"\nTesting with {workers} worker(s)...")

        # Multiprocessing (multiple repeats, collect stats)
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
        mp_mean = statistics.mean(mp_times_this) if mp_times_this else float('inf')
        mp_std = statistics.stdev(mp_times_this) if len(mp_times_this) > 1 else 0.0
        # t-critical values for 95% CI for small samples; default to 1.96 for larger samples
        _t_table = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,29:2.045,30:2.042}
        n_mp = len(mp_times_this)
        tcrit = _t_table.get(n_mp-1, 1.96) if n_mp > 1 else 1.96
        mp_ci = tcrit * (mp_std / (n_mp ** 0.5)) if n_mp > 1 else 0.0

        multiprocessing_times.append(mp_median)
        mp_runs.append(mp_times_this)
        mp_means.append(mp_mean)
        mp_stds.append(mp_std)
        mp_ci95_half.append(mp_ci)

        print(f"  MP median Time: {mp_median:.3f}s (mean={mp_mean:.3f}s, std={mp_std:.3f}s, 95%CI±{mp_ci:.3f}s)")

        # Concurrent.futures (multiple repeats, collect stats)
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
        cf_mean = statistics.mean(cf_times_this) if cf_times_this else float('inf')
        cf_std = statistics.stdev(cf_times_this) if len(cf_times_this) > 1 else 0.0
        n_cf = len(cf_times_this)
        tcrit_cf = _t_table.get(n_cf-1, 1.96) if n_cf > 1 else 1.96
        cf_ci = tcrit_cf * (cf_std / (n_cf ** 0.5)) if n_cf > 1 else 0.0

        concurrent_times.append(cf_median)
        cf_runs.append(cf_times_this)
        cf_means.append(cf_mean)
        cf_stds.append(cf_std)
        cf_ci95_half.append(cf_ci)

        print(f"  CF median Time: {cf_median:.3f}s (mean={cf_mean:.3f}s, std={cf_std:.3f}s, 95%CI±{cf_ci:.3f}s)")

    # Compute metrics using helper (based on medians)
    metrics = compute_metrics(worker_counts, multiprocessing_times, concurrent_times)

    return {
        'worker_counts': worker_counts,
        'mp_times': multiprocessing_times,           # medians
        'cf_times': concurrent_times,                # medians
        'mp_runs': mp_runs,                          # raw runs per worker
        'cf_runs': cf_runs,
        'mp_mean': mp_means,
        'mp_std': mp_stds,
        'mp_ci95_half': mp_ci95_half,
        'cf_mean': cf_means,
        'cf_std': cf_stds,
        'cf_ci95_half': cf_ci95_half,
        'repeats': repeats,
        **metrics
    }

