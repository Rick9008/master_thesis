
import numpy as np
import random
class node:
    def __init__(self, ID, Xposition=0, Yposition=9, enable_df=False, enable_proposed_algo=True):
        self.ID = ID
        self.Xposition = Xposition
        self.Yposition = Yposition
        self.GENERATION_INTERVAL = 1000
        self.Advertise_Interval = 20
        self.Relay_Retransmission_Interval = 20
        self.Transmission_Interval = 20
        self.SCAN_WINDOW = 30
        self.SCAN_INTERVAL = 30
        self.Relay_Retransmit_Count = 0
        self.Network_Transmit_Count = 0
        self.Rris = 1
        self.Ntis = 1
        self.NODE_RANGE = 10
        self.unicastaddress = 1
        self.groupaddress = 0
        self.last_seq_number = np.full(50, 0)
        self.friend_queue = []
        self.last_poll_time = 0
        self.response_friend_time = 50000
        self.LOW_POWER_ID = -1
        self.friend_Id = -1
        self.previous_ack = 1
        self.not_receive = 0
        self.Low_Scan_Time = 0
        self.low_power_ack = 0
        self.feature = 6
        self.cache = []
        self.five_hop = []
        self.buffer = []
        self.keynet = []
        self.channel37 = 0
        self.channel38 = 0
        self.channel39 = 0
        self.advertisetag37 = 0  # these flags are used for collision detection
        self.advertisetag38 = 0
        self.advertisetag39 = 0
        self.neighbors = []
        self.message = 0
        self.L_scan = 8
        self.init_time = 0
        self.first_time_scan = 0
        self.last_T_relay = random.randint(0, 30)  # Initial random value
        self.last_T_generation = random.randint(0, 30)
        self.last_t_scan = random.randint(0, 50)
        self.h_last_seq_number = np.full(50, 0)
        self.node_TTL = 0
        self.time_heartbeat = random.randint(0, 1000)
        self.seq_number = 0
        self.h_seq_number = 0
        self.Gen_cache = []
        self.heart_cache = []
        self.Sleep_Time = 0
        self.Scan_Time = 0
        self.Switch_Time = 0
        self.Transmit_Time = 0
        self.minhop = 127
        self.maxhop = 0
        self.Heartbeat_period = 0
        self.n_count = 0
        self.r_count = 0
        self.last_relay_time = random.randint(0, 120)  # Initial random value
        self.last_generation_time = random.randint(0, 1000)
        self.is_data_source = False
        self.DF_AVOID_PENALTY = 10
        # === Year 1 Plan 新增屬性 ===
        self.busy_steps = 0  # 節點資料接收的次數
        self.collision_steps = 0  # 節點發生碰撞的次數
        # DF 能力開關 (控制是否發送 Path Request / 查表)
        self.enable_df = enable_df
        # 即使不支援 DF，保持空表以避免其他邏輯報錯
        self.forwarding_table = {}
        # --- DF path_id generator (only needed for originator) ---
        self._path_id_counter = 0
        # 控制是否執行被動分群 & 自適應 Duty Cycle
        self.enable_proposed_algo = enable_proposed_algo
        
        # --- DF Path Discovery 狀態 (scan37.py 會用到) ---
        # 目的端用來「收集」多條 Path Request 的時間窗 (ms)
        self.COLLECTION_WINDOW = 20

        # 目的端暫存候選路徑：
        # 每筆通常包含 {'packet': <pkt>, 'ttl':..., 'rssi':...}
        self.path_candidates = []

        # 目的端收集視窗截止時間；None 代表目前沒有在收集
        self.discovery_end_time = None

        # 要選幾條不重複的 path (lanes) 回覆 Path Reply
        self.LANES = 3

        if self.enable_proposed_algo:
            # 初始化角色與演算法參數
            self.role = "UC"
            self.duty_cycle = 1.0
            self.neighbor_counts = {"CH": 0, "GW": 0}
            self.ch_timer = None
            self.alpha = 1
            self.beta = 0
            self.NG = 2
        else:
            # 純 Legacy 模式 (完全不分群，通常用於 Baseline 對照組)
            self.role = "Legacy"
            self.duty_cycle = 1.0
            self.neighbor_counts = {}
            self.ch_timer = None
    def scanchannel37(self, ch1):
        self.cache.append(ch1)

    def scanchannel38(self, ch2):
        self.cache.append(ch2)

    def scanchannel39(self, ch3):
        self.cache.append(ch3)

    def advertising_37(self, message):
        self.channel37 = message

    def advertising_38(self, message):
        self.channel38 = message

    def advertising_39(self, message):
        self.channel39 = message
    
    # === 第一年計畫 (Year 1 Plan) 新增邏輯方法 ===
    def update_neighbor_counts(self, src_role):
        if self.enable_proposed_algo:
            if src_role == "CH":
                self.neighbor_counts["CH"] += 1
            elif src_role == "GW":
                self.neighbor_counts["GW"] += 1

    def new_path_id(self):
        self._path_id_counter += 1
        return (self.ID, self._path_id_counter)
    
    def run_passive_clustering(self, current_time):
        """
        演算法 1: 被動式分群 (Passive Clustering)
        """
        if not self.enable_proposed_algo:
            return

        if self.role == "UC":
            if self.neighbor_counts["CH"] > 0:
                self.ch_timer = None 
                score = (self.alpha * self.neighbor_counts["CH"]) + self.beta
                
                if score >= self.NG:
                    self.role = "GW"
                    # GW 需要幫忙轉發，也要產生資料
                    self.feature = 3  # RELAY_AND_GENERATION
                else:
                    self.role = "OD"
                    # [修正] OD 只產生資料，不幫忙轉發 (Leaf Node)
                    self.feature = 2  # JUST_GENERATION (原本是 1)
            else:
                if self.ch_timer is None:
                    delay = random.uniform(0, 10) 
                    self.ch_timer = current_time + delay
    
    def check_ch_timer(self, current_time):
        if not self.enable_proposed_algo:
            return
        if self.role == "UC" and self.ch_timer is not None:
            if current_time >= self.ch_timer:
                if self.neighbor_counts["CH"] == 0:
                    self.role = "CH"
                self.ch_timer = None

    def calculate_adaptive_duty_cycle(self, neighbors_list):
        # 只有開啟演算法的節點才調整 Duty Cycle
        if not self.enable_proposed_algo:
            return 1.0

        dc_target = 1.0
        if self.role == "CH":
            dc_target = 1.0
        elif self.role == "GW":
            redundant_count = len(neighbors_list)
            if redundant_count > 2:
                dc_target = 0.50 
            else:
                dc_target = 0.75 
        elif self.role == "OD":
            dc_target = 0.25

        rho = 0.5
        self.duty_cycle = (1 - rho) * self.duty_cycle + (rho * dc_target)
        self.SCAN_WINDOW = self.duty_cycle * self.SCAN_INTERVAL
        return self.duty_cycle