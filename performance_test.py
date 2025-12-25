import statistics
import os
import threading
import time
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    psutil = None
    HAS_PSUTIL = False

from parallel_multiprocessing import parallel_process_multiprocessing
from parallel_concurrent import parallel_process_concurrent


def _count_images_in_dir(input_dir):
    cnt = 0
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                cnt += 1
    return cnt


def _metric_sampler(out_list, stop_event, interval=0.1):
    # samples overall cpu% and mem% periodically into out_list
    # Each sample: {'t': time.time(), 'cpu_avg': float, 'mem_percent': float}
    # If psutil is not available, do nothing
    if not HAS_PSUTIL:
        return
    # prime
    psutil.cpu_percent(percpu=True)
    while not stop_event.is_set():
        cpu_per_core = psutil.cpu_percent(percpu=True)
        mem = psutil.virtual_memory().percent
        out_list.append({'t': time.time(), 'cpu_avg': sum(cpu_per_core) / len(cpu_per_core) if cpu_per_core else 0.0, 'mem_percent': mem})
        time.sleep(interval)


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

    # containers for throughput and system metrics
    mp_throughput_runs = []
    cf_throughput_runs = []
    mp_throughput_means = []
    mp_throughput_stds = []
    mp_throughput_ci_half = []
    cf_throughput_means = []
    cf_throughput_stds = []
    cf_throughput_ci_half = []
    mp_cpu_runs = []
    cf_cpu_runs = []
    mp_disk_bps_runs = []
    cf_disk_bps_runs = []

    # t-critical table for small sample CIs
    _t_table = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,29:2.045,30:2.042}
    for workers in worker_counts:
        print(f"\nTesting with {workers} worker(s)...")

        # Multiprocessing (multiple repeats, collect stats)
        mp_times_this = []
        mp_throughput_this = []
        mp_cpu_this = []
        mp_disk_this = []

        for i in range(repeats):
            output_dir = f"{output_base_dir}/multiprocessing_{workers}_run{i}"
            # prepare sampling
            samples = []
            stop_event = threading.Event()
            sampler = threading.Thread(target=_metric_sampler, args=(samples, stop_event, 0.1), daemon=True)
            start_io = psutil.disk_io_counters() if HAS_PSUTIL else None
            sampler.start()
            try:
                ret = parallel_process_multiprocessing(input_dir, output_dir, workers)
            except Exception as e:
                print(f"  MP run {i+1} failed: {e}")
                stop_event.set()
                sampler.join(timeout=1.0)
                continue
            # stop sampling
            stop_event.set()
            sampler.join(timeout=1.0)
            end_io = psutil.disk_io_counters() if HAS_PSUTIL else None

            # parse return value (backwards compatible)
            if isinstance(ret, tuple) and len(ret) == 2:
                t, processed = ret
            else:
                t = float(ret)
                processed = _count_images_in_dir(input_dir)

            throughput = (processed / t) if t > 0 else 0.0
            avg_cpu = statistics.mean([s['cpu_avg'] for s in samples]) if samples else None
            max_cpu = max([s['cpu_avg'] for s in samples]) if samples else None
            disk_bytes = None
            disk_bps = None
            if start_io and end_io and t > 0:
                disk_bytes = (end_io.read_bytes + end_io.write_bytes) - (start_io.read_bytes + start_io.write_bytes)
                disk_bps = disk_bytes / t

            mp_times_this.append(t)
            mp_throughput_this.append(throughput)
            mp_cpu_this.append(avg_cpu if avg_cpu is not None else 0.0)
            mp_disk_this.append(disk_bps if disk_bps is not None else 0.0)

            print(f"  MP run {i+1}: {t:.3f}s, {processed} images, {throughput:.2f} imgs/s, avg_cpu={avg_cpu if avg_cpu is not None else 'N/A'}")

        mp_median = statistics.median(mp_times_this) if mp_times_this else float('inf')
        mp_mean = statistics.mean(mp_times_this) if mp_times_this else float('inf')
        mp_std = statistics.stdev(mp_times_this) if len(mp_times_this) > 1 else 0.0
        n_mp = len(mp_times_this)
        tcrit = _t_table.get(n_mp-1, 1.96) if n_mp > 1 else 1.96
        mp_ci = tcrit * (mp_std / (n_mp ** 0.5)) if n_mp > 1 else 0.0

        multiprocessing_times.append(mp_median)
        mp_runs.append(mp_times_this)
        mp_means.append(mp_mean)
        mp_stds.append(mp_std)
        mp_ci95_half.append(mp_ci)

        # throughput/cpu stats
        mp_throughput_runs.append(mp_throughput_this)
        mp_throughput_mean = statistics.mean(mp_throughput_this) if mp_throughput_this else 0.0
        mp_throughput_std = statistics.stdev(mp_throughput_this) if len(mp_throughput_this) > 1 else 0.0
        mp_throughput_ci = tcrit * (mp_throughput_std / (len(mp_throughput_this) ** 0.5)) if len(mp_throughput_this) > 1 else 0.0
        mp_throughput_means.append(mp_throughput_mean)
        mp_throughput_stds.append(mp_throughput_std)
        # store CI half-width for throughput
        mp_throughput_ci_half.append(mp_throughput_ci)

        mp_cpu_runs.append(mp_cpu_this)
        mp_cpu_mean = statistics.mean(mp_cpu_this) if mp_cpu_this else 0.0
        mp_cpu_std = statistics.stdev(mp_cpu_this) if len(mp_cpu_this) > 1 else 0.0
        mp_disk_bps_runs.append(mp_disk_this)

        print(f"  MP median Time: {mp_median:.3f}s (mean={mp_mean:.3f}s, std={mp_std:.3f}s, 95%CI±{mp_ci:.3f}s)")
        print(f"    Throughput mean={mp_throughput_mean:.2f} imgs/s, std={mp_throughput_std:.2f}, 95%CI±{mp_throughput_ci:.2f}")

        # Concurrent.futures (multiple repeats, collect stats)
        cf_times_this = []
        cf_throughput_this = []
        cf_cpu_this = []
        cf_disk_this = []

        for i in range(repeats):
            output_dir = f"{output_base_dir}/concurrent_{workers}_run{i}"
            # prepare sampling
            samples = []
            stop_event = threading.Event()
            sampler = threading.Thread(target=_metric_sampler, args=(samples, stop_event, 0.1), daemon=True)
            start_io = psutil.disk_io_counters() if HAS_PSUTIL else None
            sampler.start()
            try:
                ret = parallel_process_concurrent(input_dir, output_dir, workers)
            except Exception as e:
                print(f"  CF run {i+1} failed: {e}")
                stop_event.set()
                sampler.join(timeout=1.0)
                continue
            # stop sampling
            stop_event.set()
            sampler.join(timeout=1.0)
            end_io = psutil.disk_io_counters() if HAS_PSUTIL else None

            # parse return value (backwards compatible)
            if isinstance(ret, tuple) and len(ret) == 2:
                t, processed = ret
            else:
                t = float(ret)
                processed = _count_images_in_dir(input_dir)

            throughput = (processed / t) if t > 0 else 0.0
            avg_cpu = statistics.mean([s['cpu_avg'] for s in samples]) if samples else None
            max_cpu = max([s['cpu_avg'] for s in samples]) if samples else None
            disk_bytes = None
            disk_bps = None
            if start_io and end_io and t > 0:
                disk_bytes = (end_io.read_bytes + end_io.write_bytes) - (start_io.read_bytes + start_io.write_bytes)
                disk_bps = disk_bytes / t

            cf_times_this.append(t)
            cf_throughput_this.append(throughput)
            cf_cpu_this.append(avg_cpu if avg_cpu is not None else 0.0)
            cf_disk_this.append(disk_bps if disk_bps is not None else 0.0)

            print(f"  CF run {i+1}: {t:.3f}s, {processed} images, {throughput:.2f} imgs/s, avg_cpu={avg_cpu if avg_cpu is not None else 'N/A'}")

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

        # throughput/cpu stats
        cf_throughput_runs.append(cf_throughput_this)
        cf_throughput_mean = statistics.mean(cf_throughput_this) if cf_throughput_this else 0.0
        cf_throughput_std = statistics.stdev(cf_throughput_this) if len(cf_throughput_this) > 1 else 0.0
        cf_throughput_ci = tcrit_cf * (cf_throughput_std / (len(cf_throughput_this) ** 0.5)) if len(cf_throughput_this) > 1 else 0.0
        cf_throughput_means.append(cf_throughput_mean)
        cf_throughput_stds.append(cf_throughput_std)
        # store CI half-width for throughput
        cf_throughput_ci_half.append(cf_throughput_ci)

        cf_cpu_runs.append(cf_cpu_this)
        cf_cpu_mean = statistics.mean(cf_cpu_this) if cf_cpu_this else 0.0
        cf_cpu_std = statistics.stdev(cf_cpu_this) if len(cf_cpu_this) > 1 else 0.0
        cf_disk_bps_runs.append(cf_disk_this)

        print(f"  CF median Time: {cf_median:.3f}s (mean={cf_mean:.3f}s, std={cf_std:.3f}s, 95%CI±{cf_ci:.3f}s)")
        print(f"    Throughput mean={cf_throughput_mean:.2f} imgs/s, std={cf_throughput_std:.2f}, 95%CI±{cf_throughput_ci:.2f}")

    # Compute metrics using helper (based on medians)
    metrics = compute_metrics(worker_counts, multiprocessing_times, concurrent_times)

    # Compute speedup means and approximate 95% CI using delta method on ratio
    mp_speedup_means = []
    mp_speedup_ci = []
    mp_eff_means = []
    mp_eff_ci = []

    cf_speedup_means = []
    cf_speedup_ci = []
    cf_eff_means = []
    cf_eff_ci = []

    # Baselines (means and stds)
    base_mp_mean = mp_means[0] if mp_means else 1.0
    base_mp_std = mp_stds[0] if mp_stds else 0.0
    n_base_mp = len(mp_runs[0]) if mp_runs and mp_runs[0] else 1

    base_cf_mean = cf_means[0] if cf_means else 1.0
    base_cf_std = cf_stds[0] if cf_stds else 0.0
    n_base_cf = len(cf_runs[0]) if cf_runs and cf_runs[0] else 1

    for i, w in enumerate(worker_counts):
        # MP
        mean_i = mp_means[i]
        std_i = mp_stds[i]
        n_i = len(mp_runs[i]) if i < len(mp_runs) else 1
        if mean_i and mean_i != float('inf'):
            s = base_mp_mean / mean_i
            # standard error via propagation: se = s * sqrt((var_base/(n_base*base^2)) + (var_i/(n_i*mean_i^2)))
            se = 0.0
            if n_base_mp > 1 or n_i > 1:
                se = s * (( (base_mp_std**2) / (n_base_mp * (base_mp_mean**2 if base_mp_mean else 1.0)) ) + ((std_i**2) / (n_i * (mean_i**2 if mean_i else 1.0))))**0.5
            df = min(max(n_base_mp-1,1), max(n_i-1,1))
            tcrit = _t_table.get(df, 1.96)
            ci_half = tcrit * se
        else:
            s = 0.0
            ci_half = 0.0
        mp_speedup_means.append(s)
        mp_speedup_ci.append(ci_half)
        eff = s / w if w else 0.0
        mp_eff_means.append(eff)
        mp_eff_ci.append(ci_half / w if w else 0.0)

        # CF
        mean_i = cf_means[i]
        std_i = cf_stds[i]
        n_i = len(cf_runs[i]) if i < len(cf_runs) else 1
        if mean_i and mean_i != float('inf'):
            s = base_cf_mean / mean_i
            se = 0.0
            if n_base_cf > 1 or n_i > 1:
                se = s * (( (base_cf_std**2) / (n_base_cf * (base_cf_mean**2 if base_cf_mean else 1.0)) ) + ((std_i**2) / (n_i * (mean_i**2 if mean_i else 1.0))))**0.5
            df = min(max(n_base_cf-1,1), max(n_i-1,1))
            tcrit = _t_table.get(df, 1.96)
            ci_half = tcrit * se
        else:
            s = 0.0
            ci_half = 0.0
        cf_speedup_means.append(s)
        cf_speedup_ci.append(ci_half)
        eff = s / w if w else 0.0
        cf_eff_means.append(eff)
        cf_eff_ci.append(ci_half / w if w else 0.0)

    # Throughput stats already computed per worker in mp_throughput_means / cf_throughput_means and their ci lists
    # CPU and disk summary
    all_cpu_vals = [val for sub in (mp_cpu_runs + cf_cpu_runs) for val in sub if val is not None]
    avg_cpu_overall = statistics.mean(all_cpu_vals) if all_cpu_vals else 0.0
    all_disk_vals = [val for sub in (mp_disk_bps_runs + cf_disk_bps_runs) for val in sub if val is not None]
    avg_disk_bps = statistics.mean(all_disk_vals) if all_disk_vals else 0.0

    # Simple heuristic analysis
    if avg_cpu_overall > 70.0:
        bound = 'CPU-bound'
    elif avg_cpu_overall < 50.0 and avg_disk_bps > 1e6:
        bound = 'I/O-bound'
    else:
        bound = 'mixed'

    # Per-worker winner summary (statistical overlap test using CIs)
    per_worker_winners = []
    for i, w in enumerate(worker_counts):
        mp_mu = mp_means[i]
        mp_ci = mp_ci95_half[i]
        cf_mu = cf_means[i]
        cf_ci = cf_ci95_half[i]
        if mp_mu + mp_ci < cf_mu - cf_ci:
            per_worker_winners.append((w, 'multiprocessing'))
        elif cf_mu + cf_ci < mp_mu - mp_ci:
            per_worker_winners.append((w, 'concurrent'))
        else:
            per_worker_winners.append((w, 'no_significant_diff'))

    # Best overall
    best_mp_idx = 0 if not multiprocessing_times else multiprocessing_times.index(min(multiprocessing_times))
    best_cf_idx = 0 if not concurrent_times else concurrent_times.index(min(concurrent_times))
    best_mp = multiprocessing_times[best_mp_idx]
    best_cf = concurrent_times[best_cf_idx]

    overall_winner = 'tie'
    if best_mp < best_cf - (mp_ci95_half[best_mp_idx] + cf_ci95_half[best_cf_idx]):
        overall_winner = 'multiprocessing'
    elif best_cf < best_mp - (mp_ci95_half[best_mp_idx] + cf_ci95_half[best_cf_idx]):
        overall_winner = 'concurrent'

    summary_lines = []
    summary_lines.append(f"Workload classification: {bound} (avg_cpu={avg_cpu_overall:.1f}%, avg_disk_bps={avg_disk_bps:.1f})")
    summary_lines.append(f"Overall winner by median time: {overall_winner} (MP best {best_mp:.2f}s @ {worker_counts[best_mp_idx]} workers, CF best {best_cf:.2f}s @ {worker_counts[best_cf_idx]} workers)")
    summary_lines.append("Per-worker winners:")
    for w, s in per_worker_winners:
        if s == 'no_significant_diff':
            summary_lines.append(f" - {w} workers: no significant difference")
        else:
            summary_lines.append(f" - {w} workers: {s} performs significantly better")

    summary_text = "\n".join(summary_lines)

    # write summary to file for convenience
    try:
        with open(os.path.join(output_base_dir, 'analysis_summary.txt'), 'w') as f:
            f.write(summary_text)
    except Exception:
        pass

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
        'mp_throughput_means': mp_throughput_means,
        'mp_throughput_std': mp_throughput_stds,
        'mp_throughput_ci_half': mp_throughput_ci_half,
        'cf_throughput_means': cf_throughput_means,
        'cf_throughput_std': cf_throughput_stds,
        'cf_throughput_ci_half': cf_throughput_ci_half,
        'mp_cpu_means': [statistics.mean(x) if x else 0.0 for x in mp_cpu_runs],
        'cf_cpu_means': [statistics.mean(x) if x else 0.0 for x in cf_cpu_runs],
        'repeats': repeats,
        'analysis_summary': summary_text,
        'mp_speedup_mean': mp_speedup_means,
        'mp_speedup_ci_half': mp_speedup_ci,
        'mp_eff_mean': mp_eff_means,
        'mp_eff_ci_half': mp_eff_ci,
        'cf_speedup_mean': cf_speedup_means,
        'cf_speedup_ci_half': cf_speedup_ci,
        'cf_eff_mean': cf_eff_means,
        'cf_eff_ci_half': cf_eff_ci,
        **metrics
    }

