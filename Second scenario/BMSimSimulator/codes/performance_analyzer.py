# performance_analyzer.py
# Destination PDR (paper style): delivered/generated
# delivered + latency are parsed from EACH SINK LOG's "(main)" lines (e.g., 141.log)
# generated is parsed from node logs' "(generate)" lines

import os
import glob
import re
import ast
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import csv

# -----------------------
# Paths / settings
# -----------------------
LOG_DIR = "log"
NODE_LOG_GLOB = os.path.join(LOG_DIR, "[0-9]*.log")  # e.g., log/38.log
OUT_DIR = "result_chart"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Regex patterns
# -----------------------
# (generate)    src    gen_time    [dst1, dst2]    seq
GEN_RE = re.compile(
    r'^\s*\(generate\)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+(\[[^\]]*\]|\d+)\s+(\d+)\b'
)

# (main)    src    seq    gen_time    dst    recv_time
# Example from 141.log:
# (main)    136    6    625.25    141    628.4
MAIN_RE = re.compile(
    r'^\s*\(main\)\s+(\d+)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\b'
)

# -----------------------
# Helpers
# -----------------------
def parse_generated_from_node_logs(node_log_paths):
    """
    Returns:
      generated: dict dst -> set((src, seq))
      origin_set: set(src)
      debug dict
    """
    generated = defaultdict(set)
    origin_set = set()

    files_scanned = 0
    files_with_generate = 0
    lines_with_generate = 0
    lines_matched = 0
    parse_errors = 0

    for path in node_log_paths:
        files_scanned += 1
        saw_generate = False

        with open(path, "r", errors="ignore") as f:
            for line in f:
                if "(generate)" in line:
                    saw_generate = True
                    lines_with_generate += 1

                m = GEN_RE.match(line)
                if not m:
                    continue

                lines_matched += 1
                try:
                    src = int(m.group(1))
                    # gen_time = float(m.group(2))  # not needed here
                    dst_raw = m.group(3)
                    seq = int(m.group(4))

                    dst_list = ast.literal_eval(dst_raw)
                    if not isinstance(dst_list, list):
                        dst_list = [dst_list]

                    origin_set.add(src)
                    for dst in dst_list:
                        generated[int(dst)].add((src, seq))

                except Exception:
                    parse_errors += 1
                    continue

        if saw_generate:
            files_with_generate += 1

    debug = {
        "files_scanned": files_scanned,
        "files_with_generate": files_with_generate,
        "lines_with_generate": lines_with_generate,
        "lines_matched": lines_matched,
        "parse_errors": parse_errors,
    }
    return generated, origin_set, debug


def parse_delivered_and_latency_from_sink_logs(sink_ids):
    """
    Parse delivered packets and latency from each sink's "<sink>.log" file using "(main)" lines.

    Returns:
      delivered: dst -> set((src, seq))
      best_latency: (src,dst,seq) -> min latency
      latencies_all: list(float)
      debug: counters
    """
    delivered = defaultdict(set)
    best_latency = {}  # (src,dst,seq) -> min latency

    missing_sink_logs = []
    main_lines = 0
    main_matched = 0
    parse_errors = 0

    for dst in sorted(sink_ids):
        sink_path = os.path.join(LOG_DIR, f"{dst}.log")
        if not os.path.exists(sink_path):
            missing_sink_logs.append(dst)
            continue

        with open(sink_path, "r", errors="ignore") as f:
            for line in f:
                if "(main)" in line:
                    main_lines += 1
                m = MAIN_RE.match(line)
                if not m:
                    continue

                main_matched += 1
                try:
                    src = int(m.group(1))
                    seq = int(m.group(2))
                    gen_time = float(m.group(3))
                    dst_in_line = int(m.group(4))
                    recv_time = float(m.group(5))

                    # Sanity: only count if matches this sink set
                    if dst_in_line not in sink_ids:
                        continue

                    delivered[dst_in_line].add((src, seq))

                    lat = recv_time - gen_time
                    if lat < 0:
                        continue
                    k = (src, dst_in_line, seq)
                    if (k not in best_latency) or (lat < best_latency[k]):
                        best_latency[k] = lat

                except Exception:
                    parse_errors += 1
                    continue

    latencies_all = list(best_latency.values())
    debug = {
        "missing_sink_logs": missing_sink_logs,
        "main_lines_seen": main_lines,
        "main_lines_matched": main_matched,
        "parse_errors": parse_errors,
    }
    return delivered, best_latency, latencies_all, debug


def percentile_summary(values, name="values"):
    if not values:
        return f"{name}: empty"
    arr = np.array(values, dtype=float)
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    mean = float(np.mean(arr))
    mx = float(np.max(arr))
    return (f"{name}: count={len(arr)} mean={mean:.3f} "
            f"p50={p50:.3f} p90={p90:.3f} p95={p95:.3f} p99={p99:.3f} max={mx:.3f}")


def plot_dst_pdr_bar(dst_pdr, out_path):
    """
    Plot destination-level PDR.
    x-axis: destination node_id
    y-axis: PDR (%)
    """
    if not dst_pdr:
        print("[PLOT] No dst_pdr to plot.")
        return

    # sort by destination node id
    dsts = sorted(dst_pdr.keys())
    vals = [dst_pdr[d] for d in dsts]

    plt.figure(figsize=(max(10, 0.35 * len(dsts)), 5))

    x_pos = range(len(dsts))
    plt.bar(x_pos, vals, color="steelblue", edgecolor="black")

    # x-axis: destination node id
    plt.xticks(
        x_pos,
        [str(d) for d in dsts],
        rotation=45,
        ha="right"
    )

    plt.xlabel("Destination node ID")
    plt.ylabel("PDR (%)")
    plt.title("Destination-level PDR (Random topology)")
    plt.ylim(0, 100)

    # horizontal grid for readability
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # annotate bars (optional but useful)
    for i, v in enumerate(vals):
        plt.text(
            i, 
            min(100, v + 1.0),
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[PLOT] Saved: {out_path}")



def plot_latency_cdf(latencies, out_path):
    if not latencies:
        print("[PLOT] No latencies to plot.")
        return
    arr = np.sort(np.array(latencies, dtype=float))
    y = np.arange(1, len(arr) + 1) / len(arr)

    plt.figure()
    plt.plot(arr, y)
    plt.xlabel("End-to-end latency (time units)")
    plt.ylabel("CDF")
    plt.title("Latency CDF (from sink (main) lines)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


# -----------------------
# Main
# -----------------------
def main():
    node_logs = sorted(glob.glob(NODE_LOG_GLOB))
    if not node_logs:
        print(f"[ERROR] No node logs found: {NODE_LOG_GLOB}")
        return

    print(f"[INFO] Found {len(node_logs)} node log files.")
    generated, origin_set, gen_debug = parse_generated_from_node_logs(node_logs)

    print("[INFO] Generated parse stats:", gen_debug)
    print(f"[INFO] Unique destinations in generated: {len(generated)} -> {sorted(generated.keys())[:20]}")
    print(f"[INFO] Unique originators observed in generate logs: {len(origin_set)}")

    if not generated:
        print("[ERROR] generated is empty. Check TOTAL_LOG==1 and (generate) lines.")
        return

    sink_ids = set()
    if os.path.exists("log/sinks.txt"):
        with open("log/sinks.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    sink_ids.add(int(line))
    else:
        # fallback: use generated destinations
        sink_ids = set(generated.keys())


    # delivered + latency from sink logs (main lines)
    delivered, best_latency, latencies, sink_debug = parse_delivered_and_latency_from_sink_logs(sink_ids)
    print("[INFO] Sink (main) parse stats:", sink_debug)
    if sink_debug["missing_sink_logs"]:
        print("[WARN] Missing sink log files for dst:", sink_debug["missing_sink_logs"][:20])

    # Destination PDR
    dst_pdr = {}
    print("\n=== Destination PDR (from sink (main)) ===")
    for dst in sorted(sink_ids):
        gen_set = generated.get(dst, set())
        del_set = delivered.get(dst, set())
        denom = len(gen_set)
        numer = len(del_set)
        pdr = (100.0 * numer / denom) if denom > 0 else 0.0
        dst_pdr[dst] = pdr
        print(f"dst={dst}: delivered={numer} generated={denom} PDR={pdr:.2f}%")

    # Latency summary (all sinks)
    print("\n=== Latency summary (min per (src,dst,seq)) ===")
    print(percentile_summary(latencies, name="latency"))

    # Latency per dst (optional but useful)
    lat_by_dst = defaultdict(list)
    for (src, dst, seq), lat in best_latency.items():
        lat_by_dst[dst].append(lat)

    for dst in sorted(lat_by_dst.keys()):
        print(percentile_summary(lat_by_dst[dst], name=f"latency(dst={dst})"))

    # Plots
    plot_dst_pdr_bar(dst_pdr, os.path.join(OUT_DIR, "dst_pdr_bar.png"))
    plot_latency_cdf(latencies, os.path.join(OUT_DIR, "latency_cdf.png"))

    # Compact output
    valid_PDR = list(dst_pdr.values())
    print("\n=== PDR values (destinations) ===")
    print(valid_PDR)
    p_list = []
    with open("log/collision_stats.csv", "r") as f:
        r = csv.DictReader(f)
        for row in r:
            p_list.append(float(row["p_collision"]))
    print("avg collision prob:", sum(p_list)/len(p_list))


if __name__ == "__main__":
    main()
