# performance_analyzer.py
# Destination PDR (paper style): delivered/generated
# delivered + latency are parsed from EACH SINK LOG's "(main)" lines (e.g., 141.log)
# generated is parsed from node logs' "(generate)" lines

import os
import glob
import re
import ast
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import csv

# -----------------------
# Regex patterns
# -----------------------
# (generate)    src    gen_time    [dst1, dst2]    seq
GEN_RE = re.compile(
    r'^\s*\(generate\)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+(\[[^\]]*\]|\d+)\s+(\d+)\b'
)

# (main)    src    seq    gen_time    dst    recv_time
MAIN_RE = re.compile(
    r'^\s*\(main\)\s+(\d+)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\b'
)

# -----------------------
# Helpers
# -----------------------
def parse_generated_from_node_logs(node_log_paths):
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


def parse_delivered_and_latency_from_sink_logs(log_dir, sink_ids):
    delivered = defaultdict(set)
    best_latency = {}  # (src,dst,seq) -> min latency

    missing_sink_logs = []
    main_lines = 0
    main_matched = 0
    parse_errors = 0

    for dst in sorted(sink_ids):
        sink_path = os.path.join(log_dir, f"{dst}.log")
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


def plot_dst_pdr_bar(dst_pdr, out_path, title):
    if not dst_pdr:
        print("[PLOT] No dst_pdr to plot.")
        return

    dsts = sorted(dst_pdr.keys())
    vals = [dst_pdr[d] for d in dsts]

    plt.figure(figsize=(max(10, 0.35 * len(dsts)), 5))
    x_pos = range(len(dsts))
    plt.bar(x_pos, vals, edgecolor="black")

    plt.xticks(x_pos, [str(d) for d in dsts], rotation=45, ha="right")
    plt.xlabel("Destination node ID")
    plt.ylabel("PDR (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate(vals):
        plt.text(i, min(100, v + 1.0), f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


def plot_latency_boxplot_by_dst(lat_by_dst, out_path, title):
    """
    Paper-like boxplot: each dst is one box (latency distribution of delivered packets).
    """
    # only keep dst that has data
    dsts = sorted([d for d, v in lat_by_dst.items() if v])
    if not dsts:
        print("[PLOT] No latency data to plot.")
        return

    data = [lat_by_dst[d] for d in dsts]

    plt.figure(figsize=(max(10, 0.35 * len(dsts)), 5))
    plt.boxplot(
        data,
        labels=[str(d) for d in dsts],
        showfliers=True,   # outliers
        whis=1.5
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Destination node ID")
    plt.ylabel("Latency (time units)")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


def load_sink_ids(log_dir, generated_keys):
    sink_path = os.path.join(log_dir, "sinks.txt")
    sink_ids = set()
    if os.path.exists(sink_path):
        with open(sink_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    sink_ids.add(int(line))
    else:
        sink_ids = set(generated_keys)
    return sink_ids


def load_collision_avg(log_dir):
    csv_path = os.path.join(log_dir, "collision_stats.csv")
    if not os.path.exists(csv_path):
        return None

    p_list = []
    with open(csv_path, "r", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                p_list.append(float(row["p_collision"]))
            except Exception:
                continue
    if not p_list:
        return None
    return sum(p_list) / len(p_list)



# -----------------------
# Energy (from energy.log)
# -----------------------
def parse_energy_log(energy_path):
    """
    Parse energy.log lines like:
        node_id   Scan_Time   Switch_Time   Transmit_Time   Sleep_Time
    All time units are whatever your simulator accumulates (typically ms).

    Returns:
      times_by_node: dict[node_id] -> dict(scan=..., switch=..., tx=..., sleep=...)
    """
    if not os.path.exists(energy_path):
        return {}

    times_by_node = {}
    with open(energy_path, "r", errors="ignore") as f:
        for line in f:
            parts = [p for p in line.strip().split() if p]
            if len(parts) < 5:
                continue
            try:
                nid = int(float(parts[0]))
                scan_t = float(parts[1])
                sw_t = float(parts[2])
                tx_t = float(parts[3])
                sl_t = float(parts[4])
            except Exception:
                continue
            times_by_node[nid] = {"scan": scan_t, "switch": sw_t, "tx": tx_t, "sleep": sl_t}
    return times_by_node


def compute_energy_mj(times_by_node, voltage_v=3.0, c_tx_ma=8.45, c_rx_ma=13.9, c_sw_ma=3.66, c_sleep_ma=0.015):
    """
    Convert time spent in each radio state to energy (mJ):
        E(mJ) = time(ms) * current(mA) * voltage(V) / 1000
    """
    scan_e = {}
    sw_e = {}
    tx_e = {}
    sl_e = {}
    total_e = {}

    for nid, t in times_by_node.items():
        scan = t.get("scan", 0.0)
        sw = t.get("switch", 0.0)
        tx = t.get("tx", 0.0)
        sl = t.get("sleep", 0.0)

        scan_e[nid] = (scan * c_rx_ma * voltage_v) / 1000.0
        sw_e[nid] = (sw * c_sw_ma * voltage_v) / 1000.0
        tx_e[nid] = (tx * c_tx_ma * voltage_v) / 1000.0
        sl_e[nid] = (sl * c_sleep_ma * voltage_v) / 1000.0
        total_e[nid] = scan_e[nid] + sw_e[nid] + tx_e[nid] + sl_e[nid]

    return scan_e, sw_e, tx_e, sl_e, total_e


def plot_energy_bar(values_by_node, out_path, title, ylabel):
    if not values_by_node:
        print(f"[PLOT] No energy data for: {out_path}")
        return
    nodes_sorted = sorted(values_by_node.keys())
    vals = [values_by_node[n] for n in nodes_sorted]

    plt.figure(figsize=(max(10, 0.25 * len(nodes_sorted)), 5))
    x_pos = range(len(nodes_sorted))
    plt.bar(x_pos, vals, edgecolor="black")
    plt.xticks(x_pos, [str(n) for n in nodes_sorted], rotation=45, ha="right", fontsize=7)
    plt.xlabel("Node ID")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


def parse_roles_from_cluster_nodes(cluster_nodes_path):
    """Return node_id -> role using the last snapshot in cluster_nodes.csv.

    Expected columns: time,node_id,x,y,role[,my_ch_id]
    """
    if not os.path.exists(cluster_nodes_path):
        return {}

    rows = []
    try:
        with open(cluster_nodes_path, "r", errors="ignore") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r:
                    continue
                try:
                    t = float(r.get("time", "nan"))
                    nid = int(float(r.get("node_id", "nan")))
                    role = (r.get("role") or "Legacy").strip()
                except Exception:
                    continue
                rows.append((t, nid, role))
    except Exception:
        return {}

    if not rows:
        return {}

    t_last = max(t for t, _, _ in rows)
    role_by_node = {}
    for t, nid, role in rows:
        if t == t_last:
            role_by_node[nid] = role
    return role_by_node


def plot_role_avg_energy(role_avg_mj, out_path, title):
    if not role_avg_mj:
        print(f"[PLOT] No role energy data for: {out_path}")
        return

    roles_pref = ["CH", "GW", "OD", "UC", "Legacy"]
    roles = [r for r in roles_pref if r in role_avg_mj]
    if not roles:
        roles = sorted(role_avg_mj.keys())

    vals = [role_avg_mj[r] for r in roles]
    plt.figure(figsize=(8, 4.8))
    x = range(len(roles))
    plt.bar(x, vals, edgecolor="black")
    plt.xticks(x, roles)
    plt.ylabel("Average energy per node (mJ)")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["DF", "MF"], default=None,
                        help="Which run mode logs to parse (DF or MF). "
                             "If not set, read from env RUN_MODE; default DF.")
    parser.add_argument("--voltage", type=float, default=3.0, help="Voltage (V) used for energy calc. Default: 3.0")
    parser.add_argument("--c_tx", type=float, default=8.45, help="TX current (mA). Default: 8.45")
    parser.add_argument("--c_rx", type=float, default=13.9, help="RX/scan current (mA). Default: 13.9")
    parser.add_argument("--c_sw", type=float, default=3.66, help="Channel switch current (mA). Default: 3.66")
    parser.add_argument("--c_sleep", type=float, default=0.015, help="Sleep current (mA). Default: 0.015")
    args = parser.parse_args()

    run_mode = (args.mode or os.environ.get("RUN_MODE") or "DF").upper()
    if run_mode not in ("DF", "MF"):
        run_mode = "DF"

    LOG_DIR = os.path.join("log", run_mode)
    NODE_LOG_GLOB = os.path.join(LOG_DIR, "[0-9]*.log")

    OUT_DIR = "result_chart"
    os.makedirs(OUT_DIR, exist_ok=True)

    pdr_png = os.path.join(OUT_DIR, f"{run_mode}_dst_pdr_bar.png")
    lat_box_png = os.path.join(OUT_DIR, f"{run_mode}_latency_boxplot.png")

    node_logs = sorted(glob.glob(NODE_LOG_GLOB))
    if not node_logs:
        print(f"[ERROR] No node logs found: {NODE_LOG_GLOB}")
        return

    print(f"[INFO] RUN_MODE={run_mode}")
    print(f"[INFO] Found {len(node_logs)} node log files in {LOG_DIR}")

    generated, origin_set, gen_debug = parse_generated_from_node_logs(node_logs)
    print("[INFO] Generated parse stats:", gen_debug)
    print(f"[INFO] Unique destinations in generated: {len(generated)} -> {sorted(generated.keys())[:20]}")
    print(f"[INFO] Unique originators observed in generate logs: {len(origin_set)}")

    if not generated:
        print("[ERROR] generated is empty. Check TOTAL_LOG==1 and (generate) lines.")
        return

    sink_ids = load_sink_ids(LOG_DIR, generated.keys())
    print(f"[INFO] sink_ids({len(sink_ids)}): {sorted(list(sink_ids))[:20]}")

    delivered, best_latency, latencies, sink_debug = parse_delivered_and_latency_from_sink_logs(LOG_DIR, sink_ids)
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

    # Latency summary
    print("\n=== Latency summary (min per (src,dst,seq)) ===")
    print(percentile_summary(latencies, name="latency"))

    # Latency per dst (for boxplot)
    lat_by_dst = defaultdict(list)
    for (src, dst, seq), lat in best_latency.items():
        lat_by_dst[dst].append(lat)

    for dst in sorted(lat_by_dst.keys()):
        print(percentile_summary(lat_by_dst[dst], name=f"latency(dst={dst})"))

    # Plots
    plot_dst_pdr_bar(dst_pdr, pdr_png, title=f"{run_mode}: Destination-level PDR")
    plot_latency_boxplot_by_dst(lat_by_dst, lat_box_png, title=f"{run_mode}: Latency boxplot per destination")

    # Optional collision stats
    avg_col = load_collision_avg(LOG_DIR)
    if avg_col is not None:
        print(f"\n[INFO] avg collision prob ({run_mode}): {avg_col}")
    else:
        print(f"\n[INFO] collision_stats.csv not found in {LOG_DIR} (or empty). Skipping.")

    
    # -----------------------
    # Energy (optional)
    # -----------------------
    energy_path = os.path.join(LOG_DIR, "energy.log")
    times_by_node = parse_energy_log(energy_path)
    if times_by_node:
        scan_e, sw_e, tx_e, sl_e, total_e = compute_energy_mj(
            times_by_node,
            voltage_v=args.voltage,
            c_tx_ma=args.c_tx,
            c_rx_ma=args.c_rx,
            c_sw_ma=args.c_sw,
            c_sleep_ma=args.c_sleep,
        )
        network_energy = sum(total_e.values())
        network_avg_energy = network_energy / max(1, len(total_e))
        print("\n=== Energy summary ===")
        print(f"[INFO] energy.log nodes={len(times_by_node)}  network_energy={network_energy:.3f} mJ")
        print(f"[INFO] network_avg_energy_per_node={network_avg_energy:.3f} mJ")

        # Save plots
        plot_energy_bar(scan_e, os.path.join(OUT_DIR, f"{run_mode}_scan_energy.png"),
                        title=f"{run_mode}: Scanning energy per node", ylabel="Energy (mJ)")
        plot_energy_bar(sw_e, os.path.join(OUT_DIR, f"{run_mode}_switch_energy.png"),
                        title=f"{run_mode}: Switching energy per node", ylabel="Energy (mJ)")
        plot_energy_bar(tx_e, os.path.join(OUT_DIR, f"{run_mode}_tx_energy.png"),
                        title=f"{run_mode}: TX energy per node", ylabel="Energy (mJ)")
        plot_energy_bar(sl_e, os.path.join(OUT_DIR, f"{run_mode}_sleep_energy.png"),
                        title=f"{run_mode}: Sleep energy per node", ylabel="Energy (mJ)")
        plot_energy_bar(total_e, os.path.join(OUT_DIR, f"{run_mode}_total_energy.png"),
                        title=f"{run_mode}: Total energy per node", ylabel="Energy (mJ)")

        # -----------------------
        # Energy by role (avg)
        # -----------------------
        cluster_nodes_path = os.path.join("cluster_nodes.csv")
        role_by_node = parse_roles_from_cluster_nodes(cluster_nodes_path)
        if role_by_node:
            role_to_vals = defaultdict(list)
            for nid, e in total_e.items():
                role = role_by_node.get(nid, "Legacy")
                role_to_vals[role].append(e)

            role_avg = {r: float(np.mean(v)) for r, v in role_to_vals.items() if v}
            print("\n=== Average energy per role (using last cluster_nodes snapshot) ===")
            for r in ["CH", "GW", "OD", "UC", "Legacy"]:
                if r in role_avg:
                    print(f"role={r}: avg_energy={role_avg[r]:.3f} mJ   n={len(role_to_vals[r])}")

            # Save CSV
            out_csv = os.path.join(OUT_DIR, f"{run_mode}_avg_energy_by_role.csv")
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["role", "n_nodes", "avg_energy_mJ"])
                for r in sorted(role_to_vals.keys()):
                    w.writerow([r, len(role_to_vals[r]), role_avg.get(r, "")])
            print(f"[INFO] Saved: {out_csv}")

            # Plot role averages
            plot_role_avg_energy(
                role_avg,
                os.path.join(OUT_DIR, f"{run_mode}_avg_energy_by_role.png"),
                title=f"{run_mode}: Average energy per role"
            )
        else:
            print(f"[INFO] cluster_nodes.csv not found (or empty): {cluster_nodes_path}. Skipping role energy stats.")
    else:
        print(f"\n[INFO] energy.log not found (or empty): {energy_path}. Skipping energy metrics.")

    print("\n=== PDR values (destinations) ===")
    print(list(dst_pdr.values()))


if __name__ == "__main__":
    main()
