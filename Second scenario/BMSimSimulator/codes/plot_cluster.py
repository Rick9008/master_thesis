import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

# =========================
# Helpers
# =========================
def _stable_color_map(keys, cmap_name="tab20"):
    cmap = get_cmap(cmap_name)
    keys = list(keys)
    out = {}
    for idx, k in enumerate(keys):
        out[k] = cmap(idx % cmap.N)
    return out

def _as_int_or_none(v):
    if pd.isna(v):
        return None
    try:
        return int(v)
    except Exception:
        return str(v)

# =========================
# (1) Role counts over time
# =========================
try:
    counts = pd.read_csv("cluster_counts.csv")

    plt.figure(figsize=(10.5, 4.8))
    for col in ["CH", "GW", "OD", "UC", "Legacy"]:
        if col in counts.columns:
            plt.plot(counts["time"], counts[col], label=col, linewidth=2.4)

    plt.title("Cluster role counts over time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Number of nodes")
    plt.grid(True, linewidth=0.7, alpha=0.35)
    plt.legend(
        ncol=5, frameon=True, fontsize=10,
        loc="upper center", bbox_to_anchor=(0.5, 1.18)
    )
    plt.tight_layout()
    plt.savefig("cluster_counts.png", dpi=220)
    plt.show()
except FileNotFoundError:
    print("[WARN] cluster_counts.csv not found. Skip counts plot.")

# =========================
# (2) Topology snapshot
#   CH/OD: color = my_ch_id (OD same color as its CH)
#   GW:    fixed gray
#   UC:    hollow
#   Labels: CH + GW only
# =========================
nodes = pd.read_csv("cluster_nodes.csv")
t_last = nodes["time"].max()
snap = nodes[nodes["time"] == t_last].copy()

required_cols = {"time", "node_id", "x", "y", "role", "my_ch_id"}
missing = required_cols - set(snap.columns)
if missing:
    raise ValueError(f"cluster_nodes.csv missing columns: {sorted(missing)}")

snap["my_ch_id"] = snap["my_ch_id"].apply(_as_int_or_none)

# color mapping for clusters (by my_ch_id)
ch_ids = sorted([c for c in snap["my_ch_id"].unique() if c is not None])
color_map = _stable_color_map(ch_ids, cmap_name="tab20")

NONE_COLOR = (0.75, 0.75, 0.75, 0.9)   # my_ch_id=None
GW_COLOR = (0.90, 0.10, 0.85, 0.98)  # vivid magenta (avoid tab20)

def _face_color(row):
    if row["role"] == "GW":
        return GW_COLOR
    if row["role"] in ("UC", "Legacy"):
        return NONE_COLOR
    return color_map.get(row["my_ch_id"], NONE_COLOR)



# role style
# === role style (match pasive_clustering.py shapes/sizes) ===
ROLE_MARKER = {
    "CH": "^",
    "GW": "s",
    "OD": "o",
    "UC": "x",      # ✅ UC 改成 x
    "Legacy": "o",
}

# ✅ OD / UC 放大（你可以再調）
ROLE_SIZE = {
    "CH": 150,
    "GW": 150,
    "OD": 150,       # ✅ 原本 20 太小
    "UC": 150,       # ✅ x 要大一點才看得見
    "Legacy": 150,
}

ROLE_EDGE = {
    "CH": ("black", 1.2),
    "GW": ("black", 1.3),
    "OD": ("gray", 1.0),
    "UC": ("black", 1.5),     # ✅ x 用粗一點
    "Legacy": ("lightgrey", 0.8),
}

ROLE_ALPHA = {
    "CH": 1.0,
    "GW": 1.0,
    "OD": 0.95,
    "UC": 0.80,     # ✅ UC 別太淡，不然 x 看不到
    "Legacy": 0.45,
}


ROLE_Z      = {"Legacy": 1, "UC": 2, "OD": 6, "GW": 7, "CH": 8}

plt.figure(figsize=(8.6, 8.6))
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")

# draw order: background -> foreground
draw_order = ["Legacy", "UC", "OD", "GW", "CH"]

for role in draw_order:
    sub = snap[snap["role"] == role]
    if sub.empty:
        continue

    marker = ROLE_MARKER.get(role, "o")
    size = ROLE_SIZE.get(role, 60)
    edge_c, lw = ROLE_EDGE.get(role, ("black", 1.2))
    alpha = ROLE_ALPHA.get(role, 0.8)
    zorder = ROLE_Z.get(role, 3)

    colors = [_face_color(r) for _, r in sub.iterrows()]

    # UC: hollow
    if role == "UC":
        plt.scatter(
            sub["x"], sub["y"],
            color="black",          # ✅ 保證 x 是黑色
            marker="x",
            s=ROLE_SIZE["UC"],
            alpha=ROLE_ALPHA["UC"],
            linewidths=ROLE_EDGE["UC"][1],
            zorder=zorder
        )
        continue


    plt.scatter(
        sub["x"], sub["y"],
        c=colors,
        edgecolors=edge_c,
        linewidths=lw,
        marker=marker,
        s=size,
        alpha=alpha,
        zorder=zorder
    )


# labels: CH + GW only
chs = snap[snap["role"] == "CH"]
for _, row in chs.iterrows():
    plt.text(row["x"], row["y"], f"{int(row['node_id'])}", fontsize=10,
             ha="center", va="bottom", zorder=30)


plt.title(f"Topology @ t={t_last} ms  (CH/OD color=my_ch_id, GW=gray; labels: CH+GW)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, linewidth=0.7, alpha=0.28)

# legend
legend_handles = [
    Line2D([0],[0], marker="^", color="w", markerfacecolor="white",
           markeredgecolor="black", markeredgewidth=1.2, markersize=10, label="CH"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=GW_COLOR,
           markeredgecolor="black", markeredgewidth=1.1, markersize=10, label="GW"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="gray", markeredgewidth=0.9, markersize=10, label="OD"),
    Line2D([0],[0], marker="x", color="black",
           markersize=10, linewidth=0, label="UC"),

]
ods = snap[snap["role"] == "OD"]
for _, row in ods.iterrows():
    v = row["my_ch_id"]
    if pd.isna(v):
        continue
    try:
        v_int = int(v)
    except Exception:
        continue

    plt.text(
        row["x"], row["y"],
        f"{v_int}",          # ✅ 用 int 顯示
        fontsize=8,
        ha="center",
        va="top",
        alpha=0.9,
        zorder=25
    )


plt.legend(handles=legend_handles, frameon=True, fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig("cluster_topology_by_my_ch_id.png", dpi=360)
plt.show()

print("\n=== node_id -> (role, my_ch_id) @ t_last ===")
for _, row in snap.sort_values("node_id").iterrows():
    print(f"node {int(row['node_id']):>3}  role={row['role']:<6}  my_ch_id={row['my_ch_id']}")
