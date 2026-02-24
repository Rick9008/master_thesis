""" Settings of the BM network is specified in this file.
It can be used for plugging in static models and using static algorithms
 at the network design time.
 """
import random
from turtle import pos
import networkx as nx
import numpy as np
from node import node
import pymobility
from pymobility.models.mobility import random_waypoint
from logger import setup_logger
import logging
from detect_neighbor import detect_neighbor
from choice_feature import choice_feature
import matplotlib.pyplot as plt
import math
from pylab import plot, show, savefig, xlim, figure, \
                 ylim, legend, boxplot, setp, axes
from destinations import get_fixed_df_targets, F_destination
import os
import json
import time
#############global variables##################
global update_mobility_interval, lowpower_Poll_interval
global BUFFER_SIZE, reception_ratio
global NUMBER_NODES, ENVIRONMENT, NUMBER_RELAY_NODE, NUMBER_RELAY_G_NODE
global heartbeat_log, logger, energy_log, mobility_flag, R_h, x, y, next_update, Relay_Node, Relay_G_Node
global NETWORK_TTL, PACKET_LENGTH, EXECUTION_TIME, TOTAL_LOG, BYTE, DATA_RATE, SCAN_STEP, SWITCH_TIME
global Advertise_time, Receive_Delay, sleep_time, Receive_window, destination, max_seq
global GENERATION_EVENT_Adv37, HEARTBEAT_EVENT_Adv37, RELAY_EVENT_Adv37, AD38_EVENT, AD39_EVENT, AD39_EVENT_End
global SCAN37_EVENT, SCAN38_EVENT, SCAN39_EVENT, SCAN37_C_EVENT, SCAN38_C_EVENT, SCAN39_C_EVENT, SWITCH_37TO38
global SWITCH_38TO39, SEND_POLL, FRIEND_RELAY, NODE_TIME, NODE_EVENT, Time, i_node
global SINK_NODE, JUST_RELAY, JUST_GENERATION, RELAY_AND_GENERATION, LOW_POWER, FRIEND_RELAY_NODE, FRIEND_NODE
global nodes, Show_progress_interval, Show_progress
#########################################
#######  initial settings ##############
# RUN_MODE = "DF"   # DF-only run
RUN_MODE = "MF" # MF-only run
# Algo Flag: 100% True (不支援 DF 的節點也會分群)
algo_enabled = True
SEED_DIR = os.path.join("log", RUN_MODE)
SEED_FILE = os.path.join(SEED_DIR, "topology_seed.txt")

os.makedirs(SEED_DIR, exist_ok=True)
def load_or_create_seed(path, name):
    if os.path.exists(path):
        with open(path, "r") as f:
            seed = int(f.read().strip())
        print(f"[SEED] Loaded {name} seed = {seed}")
    else:
        seed = int(time.time() * 2342) % (2**32 - 1)
        with open(path, "w") as f:
            f.write(str(seed))
        print(f"[SEED] Generated {name} seed = {seed}")
        print(f"[SEED] Saved to {path}")
    return seed


# --- define seed files ---
SEED_TOPO_FILE    = os.path.join(SEED_DIR, "seed_topology.txt")
SEED_ROLE_FILE    = os.path.join(SEED_DIR, "seed_role.txt")
SEED_TRAFFIC_FILE = os.path.join(SEED_DIR, "seed_traffic.txt")

# --- load / create ---
SEED_TOPO    = load_or_create_seed(SEED_TOPO_FILE, "TOPOLOGY")
SEED_ROLE    = load_or_create_seed(SEED_ROLE_FILE, "ROLE")
SEED_TRAFFIC = load_or_create_seed(SEED_TRAFFIC_FILE, "TRAFFIC")
LOG_ROOT = os.path.join("log", RUN_MODE)
os.makedirs(LOG_ROOT, exist_ok=True)
TRAFFIC_PLAN_PATH = "log/traffic_plan.json"
###### deployment setting ###########
NUMBER_NODES = 150  # the number of nodes in the network
x = []  # the positions of nodes in the network
y = []



DF_RATIO = 1
ENVIRONMENT = 100  # the dimension of the environment (square) i meters that the network nodes are spread in it
random.seed(SEED_TOPO)
for i in range(NUMBER_NODES):
    x.append(random.uniform(0, ENVIRONMENT))
    y.append(random.uniform(0, ENVIRONMENT))


NUMBER_RELAY_NODE = 145  # number of nodes with relay feature in the network
NUMBER_RELAY_G_NODE = 5 # number of nodes with relay and generator features in the network
NODE_RANGE = 10  # the communication range of the nodes, assumin a unit disk model
EXECUTION_TIME = 100000  # the execution time of the simulator in milliseconds
#####loging##
"""when TOTAL_LOG is one, each node has a separate log file
 with detailed data about its operations during network simulation. 
when TOTAL_LOG is zero, there is one log file for all network nodes.
 In this file, the data needed for calculating performance metrics are logged
"""
TOTAL_LOG = 1  # 1: detailed log files for each node, 0: one log file for all nodes
heartbeat_log = 0  # when set to one, some information in detailed log files in each node about
# the heartbeat messages is logged
##############network setting#############
R_h = 2  # used in determining TTL by heartbeat message, this variable is added to min hop for determining the TTL  
mobility_flag = 0  # mobility flag determines the number of mobility updates during the simulation
update_flag = 0  # update flag determines the presence of run-time adjustment
# for parameters and models during the simulation
BUFFER_SIZE = 6  # the size of the nodes' buffer
Show_progress_interval = EXECUTION_TIME/100  # the resolution (%) of the simulation progress
update_mobility_interval = 6000  # The interval for calling the Network_updator module (in milisecond)
NETWORK_TTL = NUMBER_NODES  # the initial value for the network's TTL if the user
# does not want to use the heartbeat message
#######lowpower and friend #####
Receive_Delay = 10  # the Receive Delay parameter in friendship mechanismb (ms)
sleep_time = 5      # the sleep time in friendship mechanism (ms)
Receive_window = 98  # the Receive window parameter in friendship mechanism( ms)
lowpower_Poll_interval = 4000  # the request[friend poll] interval parameter in friendship mechanism (ms)
#####################
logger = []
energy_log = []
destination = []
max_seq = []
Gar = nx.Graph()
Relay_Node = []
Relay_G_Node = []
Show_progress = 0
Show_progress = 0 + Show_progress_interval
next_update = 0
next_update = 0 + update_mobility_interval
# === 新增 ===
num_DF_nodes = int(NUMBER_NODES * DF_RATIO)
num_non_DF_nodes = NUMBER_NODES - num_DF_nodes
df_flags = [True] * num_DF_nodes + [False] * num_non_DF_nodes
random.seed(SEED_ROLE)
random.shuffle(df_flags)

# =================================================


#########initialize nodes#######################
nodes = []
for i1 in range(NUMBER_NODES):  
    is_df_capable = df_flags[i1]
    # 建立節點
    nodes.append(node(i1, x[i1], y[i1], 
                      enable_df=is_df_capable, 
                      enable_proposed_algo=algo_enabled))
    # === [Timer Optimization] ===
    nodes[i1].DISCOVERY_TIMEOUT = 8000.0
    nodes[i1].COLLECTION_WINDOW = 800.0
    
    Gar.add_node(nodes[i1].ID, pos=(nodes[i1].Xposition, nodes[i1].Yposition))
for i_r in range(NUMBER_NODES):
    ########## node setting ###########
    nodes[i_r].SCAN_INTERVAL = 30
    nodes[i_r].SCAN_WINDOW = 30
    nodes[i_r].Relay_Retransmit_Count = 0
    nodes[i_r].Network_Transmit_Count = 0
    nodes[i_r].Rris = 1
    nodes[i_r].Ntis = 1
    nodes[i_r].Advertise_Interval = 20
    nodes[i_r].GENERATION_INTERVAL = 100
    # Relay_Retransmission_Interval
    nodes[i_r].Relay_Retransmission_Interval = (nodes[i_r].Rris + 1) * 10 + random.randint(1, 10)
    nodes[i_r].Transmission_Interval = (nodes[i_r].Ntis + 1) * 10 + random.randint(1, 10)  # Transmission_Interval
    ##########Initial value ##############
    nodes[i_r].cache = []
    nodes[i_r].buffer = []
    nodes[i_r].keynet = []
    nodes[i_r].channel37 = 0
    nodes[i_r].channel38 = 0
    nodes[i_r].channel39 = 0
    nodes[i_r].advertisetag37 = 0  # these flags are used for collisions detection
    nodes[i_r].advertisetag38 = 0
    nodes[i_r].advertisetag39 = 0
    nodes[i_r].message = 0
    nodes[i_r].L_scan = 8  # the last scanning channel for determining the next scanning channel 
    nodes[i_r].first_time_scan = 0  # the beginning of the scan window is saved in each node
    nodes[i_r].seq_number = 0
    nodes[i_r].h_seq_number = 0
    nodes[i_r].Gen_cache = []
    nodes[i_r].heart_cache = []
    nodes[i_r].Sleep_Time = 0  # the node sleeping time
    nodes[i_r].Scan_Time = 0  # the node scanning time
    nodes[i_r].low_power_ack = 0
    nodes[i_r].Switch_Time = 0  # the node switching time
    nodes[i_r].Transmit_Time = 0  # the node transmission time 
    nodes[i_r].node_TTL = 127  # the initial value for the nodes' TTL
    nodes[i_r].n_count = 0  # this variable is used for counting the retransmissions in the generator nodes
    nodes[i_r].r_count = 0  # this variable is used for counting the retransmissions in the relay nodes 
    nodes[i_r].minhop = 127
    nodes[i_r].maxhop = 0
    nodes[i_r].Sleep_Time = 0
    nodes[i_r].init_time = 0
    ########## Initial random value########
    # it is necessary to save the last time of executing each event, for calculating the next time of executing this event
    # the last time that the retransmission event is executed  in the relay nodes
    nodes[i_r].last_T_relay = random.randint(0, nodes[i_r].Relay_Retransmission_Interval)
    # the last time that the retransmission event is executed  in the generator nodes
    nodes[i_r].last_T_generation = random.randint(0, nodes[i_r].Transmission_Interval)
    # the last time that advertising event is executed  in the nodes
    nodes[i_r].last_relay_time = random.randint(0, nodes[i_r].Advertise_Interval)
    # the last time that scanning event is executed  in the nodes
    nodes[i_r].last_t_scan = random.randint(0, nodes[i_r].SCAN_INTERVAL)
    # the last time that sending request[friend poll] event is executed in the low-power nodes
    nodes[i_r].last_poll_time = random.randint(0, lowpower_Poll_interval)
    # the last time that the heartbeat event is executed in the nodes
    nodes[i_r].time_heartbeat = random.randint(0, 1000)
    nodes[i_r].last_seq_number = np.full(NUMBER_NODES, 0)
    nodes[i_r].h_last_seq_number = np.full(NUMBER_NODES, 0)

for i1 in range(NUMBER_NODES):
    # the last time that generation event is executed in the generator nodes
    nodes[i1].last_generation_time = random.randint(0, nodes[i1].GENERATION_INTERVAL)

##########statice algorithms ##############
############detect neighbors ##########
# by calling the detect_neighbor function the neighbors of each node is determined
for node_source in range(NUMBER_NODES):  
    neighbor = detect_neighbor(node_source, NODE_RANGE, NUMBER_NODES, nodes, Gar)
    nodes[node_source].neighbors = neighbor

# plt.show()
####### static algorithms for determining center node and relay nodes########################
# we use the closeness centrality characteristic in the network topology for choosing the sink node
# closeness = nx.closeness_centrality(Gar)  
# Sink = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:1]
# Center_node = Sink[0][0]
# Center_node_static = Center_node
# # we use the betweenness centrality characteristic in the network topology for choosing the relay nodes
# Betweenness = nx.betweenness_centrality(Gar) 
# best_Relay = sorted(Betweenness.items(), key=lambda x: x[1], reverse=True)[:NUMBER_RELAY_NODE]
# best_G_Relay = sorted(Betweenness.items(), key=lambda x: x[1], reverse=True)[
#                NUMBER_RELAY_NODE:NUMBER_RELAY_G_NODE + NUMBER_RELAY_NODE]
# for j1 in range(NUMBER_RELAY_NODE):
#     Relay_Node.append(best_Relay[j1][0])
# for j1 in range(NUMBER_RELAY_G_NODE):
#     Relay_G_Node.append(best_G_Relay[j1][0])
####################choice feature############
# by calling the choice_feature function the feature of each node is determined
# 先清空
random.seed(SEED_ROLE)
Relay_G_Node = []  # 確保只用我們指定的 3 個 generator

# 從非 sink 節點中選 3 個當 generator（可改成固定名單）
candidates = [i for i in range(NUMBER_NODES)]
Relay_G_Node = random.sample(candidates, NUMBER_RELAY_G_NODE)
#_probe_src = 0 if Center_node != 0 else 1

random.seed(SEED_TRAFFIC)

_ = F_destination(
    NUMBER_NODES=NUMBER_NODES,
    NETWORK_TTL=NETWORK_TTL,
    nodes=nodes,
    num_dst=5
)
df_targets = get_fixed_df_targets()   # 可能還沒選過 -> []
traffic_plan = None
if RUN_MODE == "MF" and os.path.exists(TRAFFIC_PLAN_PATH):
    # MF run：讀 DF run 已經寫好的 plan
    with open(TRAFFIC_PLAN_PATH, "r", encoding="utf-8") as f:
        traffic_plan = json.load(f)
    print("[PLAN] Loaded traffic plan:", TRAFFIC_PLAN_PATH)
else:
    # DF run（或 MF run 但找不到 plan）：建立新 plan
    traffic_plan = {
        "seed_topo": SEED_TOPO,
        "seed_role": SEED_ROLE,
        "seed_traffic": SEED_TRAFFIC,
        "df_targets": df_targets,
        "sources": Relay_G_Node,
        "src_to_dsts": {}
    }
    # --- 一對一配對：dst 不重複 ---
    sources = list(Relay_G_Node)
    dst_pool = list(df_targets)

    # 先 shuffle，讓配對是隨機但可重現（你前面已 random.seed(SEED_TRAFFIC)）
    random.shuffle(dst_pool)

    # 如果 pool 不夠大，直接保護（不然無法一對一）
    if len(dst_pool) < len(sources):
        raise ValueError(f"[PLAN] df_targets({len(dst_pool)}) < sources({len(sources)}); 無法一對一配對")

    # 先做初始配對
    pairs = {s: dst_pool[idx] for idx, s in enumerate(sources)}

    # 避免 src == dst（derangement 修正：遇到相同就跟下一個交換；最後一個就跟第一個交換）
    for idx, s in enumerate(sources):
        if pairs[s] == s:
            j = (idx + 1) % len(sources)
            s2 = sources[j]
            pairs[s], pairs[s2] = pairs[s2], pairs[s]

    # 寫進 plan：每個 src 只給一個 dst（list 長度 = 1）
    for s in sources:
        traffic_plan["src_to_dsts"][str(s)] = [int(pairs[s])]

    
    with open(TRAFFIC_PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(traffic_plan, f, ensure_ascii=False, indent=2)
    print("[PLAN] Wrote traffic plan:", TRAFFIC_PLAN_PATH)

# 用 plan 覆蓋 df_targets / sources（確保 MF run 用完全相同的）
df_targets = list(traffic_plan.get("df_targets", df_targets))
Relay_G_Node = [int(x) for x in traffic_plan.get("sources", Relay_G_Node)]

# (D) sinks：DF targets 一定是 sink（你要求的）
sinks = set(df_targets)

with open("log/sinks.txt", "w") as f:
    for s in sorted(sinks):
        f.write(f"{s}\n")

for i in range(NUMBER_NODES):
    nodes[i].is_sink = (i in sinks)

# (E) 設定 features / data sources + 把 “固定 dst pair” 掛到 node 上
for i_f in range(NUMBER_NODES):
    if i_f in Relay_G_Node:
        nodes[i_f].feature = 3  # RELAY_AND_GENERATION
        nodes[i_f].is_data_source = True

        # 關鍵：固定 dst pair（destinations.py 會優先回傳這個）
        fixed_dsts = traffic_plan["src_to_dsts"].get(str(i_f), [])
        nodes[i_f].fixed_dsts = list(fixed_dsts)

        # 方便 debug
        # print(f"[PLAN] src={i_f} fixed_dsts={nodes[i_f].fixed_dsts}")
    else:
        nodes[i_f].feature = 1  # JUST_RELAY
        nodes[i_f].is_data_source = False
        nodes[i_f].fixed_dsts = []
print("[SRC] Relay_G_Node =", Relay_G_Node)
print("[SRC] node65 is_data_source =", nodes[65].is_data_source, "feature=", nodes[65].feature)

#################### Heartbeat_period #######################################
for i_heart in range(NUMBER_NODES):
    nodes[i_heart].Heartbeat_period = 0
# nodes[Center_node].Heartbeat_period = 4000
############# choose a log-normal shadowing path loss model for reception_ratio ##############
import radio_model

# === Radio model params (your given) ===
PT_DBM = 0.0
ETA = 3.5
SIGMA_DB = 4.0
D0 = 1.0

# PL0 你沒給，這裡用常見 2.4GHz @1m 約 40 dB（可自行調）
PL0_DB = 40.0

# Noise / WLAN interference (你沒給，先做可調)
NF_DBM = -90.0
IWLAN_DBM = -95.0

# PER params
ALPHA = 0.68
B_BITS = 312

# Build Pr(dBm) matrix (store in reception_ratio variable to minimize code changes)
reception_ratio = radio_model.build_pr_dbm_matrix(
    nodes,
    pt_dbm=PT_DBM,
    d0=D0,
    pl0_db=PL0_DB,
    eta=ETA,
    sigma_db=SIGMA_DB,
    seed=1
)

# 把這些參數也一起丟出去（scan 會用）
radio_params = {
    "NF_DBM": NF_DBM,
    "IWLAN_DBM": IWLAN_DBM,
    "ALPHA": ALPHA,
    "B_BITS": B_BITS,
}

#############constants##############
all_event = []  # the array for saving all network events and their time
DATA_RATE = 1000
PACKET_LENGTH = 39
BYTE = 8
SCAN_STEP = 0.2  # each scan function increases the simulator's current time as much as  SCAN_STEP = 0.2 ms
SWITCH_TIME = 0.15  # when the node switches between channels, it takes time as much as SWITCH_TIME (ms)
Advertise_time = (PACKET_LENGTH * BYTE) / DATA_RATE  # when the node advertises a packet,
# it takes time as much as Advertise_time
### define events#####
GENERATION_EVENT_Adv37 = 1
HEARTBEAT_EVENT_Adv37 = 11
RELAY_EVENT_Adv37 = 2
AD38_EVENT = 3
AD39_EVENT = 4
AD39_EVENT_End = 5
SCAN37_EVENT = 6
SCAN38_EVENT = 7
SCAN39_EVENT = 8
SCAN37_C_EVENT = 60
SCAN38_C_EVENT = 70
SCAN39_C_EVENT = 80
SWITCH_37TO38 = 9
SWITCH_38TO39 = 10
SEND_POLL = 12
FRIEND_RELAY = 13
NODE_TIME = 0
NODE_EVENT = 1
### define features####
SINK_NODE = 0
JUST_RELAY = 1
JUST_GENERATION = 2
RELAY_AND_GENERATION = 3
LOW_POWER = 4
FRIEND_RELAY_NODE = 5  # FRIEND_RELAY_NODE
FRIEND_NODE = 6
#################################################
#######Initialize logging########################
import os  # [新增] 引入 os 套件
# [新增] 檢查 log 資料夾是否存在，不存在則建立
if not os.path.exists('log'):
    os.makedirs('log')
# some settings for logging during the simulation
formatter = logging.Formatter('%(message)s')
# [修改] 加上 'log/' 前綴
energy_log = setup_logger('energy_log', os.path.join(LOG_ROOT, 'energy.log'))
if TOTAL_LOG == 1:
    # [修改] 加上 'log/' 前綴
    detail_log = setup_logger('detail_log', os.path.join(LOG_ROOT, 'detail_log.log'))
if TOTAL_LOG == 1:
    for init1 in range(NUMBER_NODES):
        # [修改] 加上 'log/' 前綴
        logger.append(setup_logger(str(init1), os.path.join(LOG_ROOT, f"{init1}.log")))
else:
    # [修改] 加上 'log/' 前綴
    logger = setup_logger('logger', os.path.join(LOG_ROOT, 'network_detail.log'))
####### Initializing  events and their time ########################
for init in range(NUMBER_NODES):
    max_seq.append(0)
# the initial event times are compared with each other, and the event with the minimum time is chosen as the initial event
    if nodes[init].last_generation_time <= nodes[init].last_t_scan and \
            (nodes[init].feature == FRIEND_RELAY_NODE or nodes[init].feature == FRIEND_NODE or nodes[init].feature ==
             JUST_GENERATION or nodes[init].feature == RELAY_AND_GENERATION):
        First_time = nodes[init].last_generation_time  # determining the time of the first event in each node
        First_event = GENERATION_EVENT_Adv37  # determining the first event in each node
    elif nodes[init].feature == LOW_POWER:
        First_time = nodes[init].last_generation_time
        First_event = GENERATION_EVENT_Adv37
    else:
        First_time = nodes[init].last_t_scan
        First_event = SCAN37_EVENT
    list_node = [First_time, First_event]
    all_event.append(list_node)  # events and event's time are stored in this array
i_node = all_event.index(min((x for x in all_event), key=lambda k: k[0]))  # The minimum time of events is specified
# then the event of this time is going to do.
Time = all_event[i_node][NODE_TIME]  # minimum time of events is selected as the simulator current time
# used in our mobility model (pymobility)
rw = random_waypoint(NUMBER_NODES, dimensions=(ENVIRONMENT, ENVIRONMENT), velocity=(0.25, 1.0), wt_max=10.0)
print("initial", all_event)

for init_t in range(NUMBER_NODES):
    nodes[init_t].init_time = all_event[init_t][0]

def _ensure_result_dir():
    os.makedirs("result_chart", exist_ok=True)

def _draw_directed_arrows(ax, pos, edges, color, lw=3, alpha=1.0, zorder=5):
    """
    用 matplotlib annotate 畫箭頭，確保是「有方向」的路徑。
    """
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=lw, color=color, alpha=alpha),
            zorder=zorder
        )

def plot_and_print_df_paths(nodes, Gar, dpi=200):
    """
    Print all DF paths and save ONE image per (origin, target) under result_chart/.
    - topology 非路徑邊：虛線背景
    - DF 路徑：箭頭（不同 lane 不同顏色）
    - origin/target 標示：最上層（最後畫 + 高 zorder）
    """
    _ensure_result_dir()
    #########plot network topology##########
    print(Gar)

    # 高對比、不跟 node 藍色混淆
    DF_LANE_COLORS = [
        "#d62728",  # red
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
    ]

    pos = nx.get_node_attributes(Gar, "pos")
    if not pos:
        raise ValueError("Graph has no 'pos' attribute.")

    fig = figure()
    nx.draw(Gar, pos=pos, with_labels=True)
    plt.savefig('result_chart/topology.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    data_sources = [idx for idx, nd in enumerate(nodes) if getattr(nd, "is_data_source", False)]
    print("[DF] data_sources =", data_sources)
    for s in data_sources:
        has = bool(getattr(nodes[s], "df_lane_nodes", {}))
        print(f"[DF] source {s} has_df_paths={has}")

    # 收集所有 DF paths
    df_paths = []
    for origin_id, n in enumerate(nodes):
        lane_nodes_map = getattr(n, "df_lane_nodes", {})
        for target_id, lanes in lane_nodes_map.items():
            if lanes:
                df_paths.append((origin_id, target_id, lanes))

    if not df_paths:
        print("[DF] No DF paths established.")
        return

    print("\n========== DF PATHS ==========")
    for origin_id, target_id, lanes in df_paths:
        print(f"DF path: origin={origin_id}, target={target_id}")
        for lane_idx, path in sorted(lanes.items()):
            print(f"  lane {lane_idx}: {path}")
    print("================================\n")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    neighbor_edges = []
    for u in range(len(nodes)):
        for v in nodes[u].neighbors:
            if u < v:  # 避免重複
                neighbor_edges.append((u, v))
    for origin_id, target_id, lanes in df_paths:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ----------------- (A) 背景拓樸：虛線邊（非路徑） -----------------
        # 先畫虛線 edges 當背景，不要太搶眼
        nx.draw_networkx_edges(
            Gar,
            pos=pos,
            edgelist=neighbor_edges,
            style="dashed",
            width=1.2,
            alpha=0.5,
            ax=ax
        )
        # 節點與 label 先畫一層（之後 origin/target 會再疊最上層）
        nx.draw_networkx_nodes(Gar, pos=pos, ax=ax, node_size=300)
        nx.draw_networkx_labels(Gar, pos=pos, ax=ax, font_size=9)
        # -------------------------------------------------------------------

        # ----------------- (B) DF lanes：用箭頭畫路徑（不同 lane 不同顏色） ---
        for k, (lane_idx, path) in enumerate(sorted(lanes.items())):
            if not path or len(path) < 2:
                continue

            # 確保端點
            if path[0] != origin_id:
                path = [origin_id] + list(path)
            if path[-1] != target_id:
                path = list(path) + [target_id]

            edges = list(zip(path[:-1], path[1:]))

            # 箭頭疊上去（高 zorder）
            _draw_directed_arrows(
                ax, pos, edges,
                color=DF_LANE_COLORS[k % len(DF_LANE_COLORS)],
                lw=3,
                alpha=0.95,
                zorder=6
            )
        # -------------------------------------------------------------------

        # ----------------- (C) Origin/Target：最上層標示 --------------------
        # 這裡「最後畫」+ zorder 拉最高，確保蓋在箭頭和其他元素上
        origin_nodes = nx.draw_networkx_nodes(
            Gar, pos=pos, nodelist=[origin_id],
            ax=ax,
            node_size=520,
            linewidths=3,
            edgecolors="black"
        )
        origin_nodes.set_zorder(10)
        target_nodes = nx.draw_networkx_nodes(
            Gar, pos=pos, nodelist=[target_id],
            ax=ax,
            node_size=520,
            linewidths=3,
            edgecolors="black"
        )
        target_nodes.set_zorder(10)
        ox, oy = pos[origin_id]
        tx, ty = pos[target_id]
        ax.text(ox, oy, f"  O({origin_id})", fontsize=11, fontweight="bold", zorder=11)
        ax.text(tx, ty, f"  T({target_id})", fontsize=11, fontweight="bold", zorder=11)
        # -------------------------------------------------------------------

        ax.set_title(f"DF lanes (arrows) on dashed topology (origin={origin_id}, target={target_id})")
        ax.axis("equal")

        out = f"result_chart/df_o{origin_id}_t{target_id}_arrows.png"
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"[DF] Saved: {out}")

