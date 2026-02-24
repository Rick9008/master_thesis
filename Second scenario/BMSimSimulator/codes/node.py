
import numpy as np
import random

from trio import current_time
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
        self.COLLECTION_WINDOW = 800
        # 要選幾條不重複的 path (lanes) 回覆 Path Reply
        self.LANES = 2
        # 目的端收集視窗截止時間；None 代表目前沒有在收集
        self.discovery_end_time = 8000.0

        # 目的端暫存候選路徑：
        # 每筆通常包含 {'packet': <pkt>, 'ttl':..., 'rssi':...}
        self.path_candidates = []

        # --------------------------------------------------------
        self.cluster_count_window = 100.0  # ms，你可先用 100
        self._last_cluster_reset = 0.0
        if self.enable_proposed_algo:
            # 初始化角色與演算法參數
            self.role = "UC"
            self.duty_cycle = 1.0
            self.neighbor_counts = {"CH": 0, "GW": 0}
            self.ch_timer = None
            self.alpha = 1
            self.beta = 0.5
            self.NG = 2
            self._role_prev = self.role
            self.seen_ch = {}                  # dict: ch_id -> last_seen_time
            self.candidate_ch_id = None         # 最早聽到的 CH（第一次聽到就固定）
            self.candidate_ch_time = -1.0

            self.cluster_hold = 200.0           # ms，視為「還在範圍內」的時間窗
            self.my_ch_id = None
            self.my_ch_expire = 0.0

            self.ch_timer = None
            self.ch_delay_offset = random.randint(1, 10)
            self.GW_HYST_K = 5   # 連續 3 個 window 才升/降（建議 2~5）
            self.gw_good_streak = 0
            self.gw_bad_streak = 0
            # --- CH demotion (soft-state) ---
            self.ch_fwd_in_window = 0          # 本 window CH 轉發次數
            self.ch_child_od_in_window = 0     # 本 window 聽到屬於我群的 OD 次數（近似下游 OD 數量）
            self.ch_demote_bad_streak = 0

            self.CH_DEMOTE_K = 3               # 連續 K 個 window 才退化（避免抖動）
            self.CH_FWD_MIN = 1                # window 內轉發 < 1 視為「沒被需要」（可調）
            self.CH_CHILD_OD_MIN = 1           # window 內下游 OD < 1 視為「沒被需要」（可調）

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
    
    # === 新增分群邏輯 ===
    def update_neighbor_counts(self, src_role):
        if self.enable_proposed_algo:
            if src_role == "CH":
                self.neighbor_counts["CH"] += 1
            elif src_role == "GW":
                self.neighbor_counts["GW"] += 1

    def new_path_id(self):
        self._path_id_counter += 1
        return (self.ID, self._path_id_counter)
    

    def has_any_reachable_ch(self, t):
        # 清掉過期紀錄
        expired = [cid for cid, ts in self.seen_ch.items() if (t - ts) > self.cluster_hold]
        for cid in expired:
            del self.seen_ch[cid]
            if self.candidate_ch_id == cid:
                # 如果最早那個 CH 過期了，讓它重新選（或你也可以改成選下一個最早）
                self.candidate_ch_id = None
                self.candidate_ch_time = -1.0
        return len(self.seen_ch) > 0


    def on_hear_sender(self, sender_id, sender_role, t):
        if not self.enable_proposed_algo:
            return
        # (1) hearing CH
        if sender_role == "CH":
            self.seen_ch[sender_id] = t
            # 固定最早聽到的 CH
            if self.candidate_ch_id is None:
                self.candidate_ch_id = sender_id
                self.candidate_ch_time = t
            # 聽到 CH 就取消競爭倒數
            self.ch_timer = None
        # (2) hearing GW
        elif sender_role == "GW":
            # 若沒看到 CH，才啟動倒數
            if self.role == "UC" and not self.has_any_reachable_ch(t):
                if self.ch_timer is None:
                    self.ch_timer = t + float(self.ch_delay_offset)
        


    def run_passive_clustering(self, current_time):
        """
        Passive Clustering with GW hysteresis
        """
        if not self.enable_proposed_algo:
            return
        # 放在 run_passive_clustering() 一開始，enable_proposed_algo 檢查後
        if getattr(self, "is_data_source", False):
            # data source 一律維持 CH
            if self.role != "CH":
                self.role = "CH"
            self.my_ch_id = self.ID
            self.my_ch_expire = current_time + self.cluster_hold
            self.ch_timer = None
            self.gw_good_streak = 0
            self.gw_bad_streak = 0
            self.ch_demote_bad_streak = 0
            return
        # 在 run_passive_clustering() 開頭 (enable_proposed_algo 檢查後) 加：
        self.has_any_reachable_ch(current_time)
        # reset counts per window
        if current_time - self._last_cluster_reset >= self.cluster_count_window:
            self.neighbor_counts = {"CH": 0, "GW": 0}
            self._last_cluster_reset = current_time
            self.ch_fwd_in_window = 0
            self.ch_child_od_in_window = 0
        K = getattr(self, "GW_HYST_K", 3)
        ch_cnt = self.neighbor_counts.get("CH", 0)
        reachable = self.has_any_reachable_ch(current_time)  # 也會 purge 過期 CH
        effective_ch = ch_cnt if ch_cnt > 0 else (1 if reachable else 0)
        # ---------- UC decision: promote to GW or become OD (with hysteresis) ----------
        if self.role == "UC":
            if effective_ch > 0:
                self.ch_timer = None
                score = self.alpha * effective_ch + self.beta

                if score >= self.NG:
                    # hysteresis: need K consecutive "good" windows to become GW
                    self.gw_good_streak += 1
                    self.gw_bad_streak = 0

                    if self.gw_good_streak >= K:
                        self.role = "GW"
                        self.my_ch_id = None
                        self.my_ch_expire = 0.0
                        # once promoted, reset streaks to avoid immediate flip
                        self.gw_good_streak = 0
                        self.gw_bad_streak = 0
                else:
                    # score not enough => become OD immediately
                    self.gw_good_streak = 0
                    self.gw_bad_streak = 0

                    self.role = "OD"
                    if self.candidate_ch_id is not None and self.has_any_reachable_ch(current_time):
                        self.my_ch_id = self.candidate_ch_id
                        self.my_ch_expire = current_time + self.cluster_hold
                    else:
                        self.my_ch_id = None
                        self.my_ch_expire = 0.0
            else:
                # UC 沒聽到 CH：這裡不開 timer（由 on_hear_sender(GW) 觸發）
                self.gw_good_streak = 0
                self.gw_bad_streak = 0
                pass
        elif self.role == "OD":
            # 若我的 CH 過期（時間到）或我最近根本沒再聽到我的 CH，就回 UC 重新加入
            if self.my_ch_id is None:
                self.role = "UC"
            else:
                if current_time >= self.my_ch_expire:
                    self.my_ch_id = None
                    self.my_ch_expire = 0.0
                    self.role = "UC"
                else:
                    # 若你有 seen_ch：檢查我的 CH 是否仍可達（更準）
                    if hasattr(self, "seen_ch"):
                        last = self.seen_ch.get(self.my_ch_id)
                        if last is None or (current_time - last) > self.cluster_hold:
                            self.my_ch_id = None
                            self.my_ch_expire = 0.0
                            self.role = "UC"
        # ---------- GW decision: re-evaluate score and possibly demote (with hysteresis) ----------
        elif self.role == "GW":
        # 用 score 判斷是否需要維持 GW（跟 UC 同規則）
            score = self.alpha * effective_ch + self.beta
            if score >= self.NG:
                # keep GW
                self.gw_good_streak += 1
                self.gw_bad_streak = 0
            else:
                # bad window -> accumulate demotion streak
                self.gw_bad_streak += 1
                self.gw_good_streak = 0
                if self.gw_bad_streak >= K:
                    self.role = "OD"
                    self.gw_good_streak = 0
                    self.gw_bad_streak = 0
                    # 用 candidate_ch_id 直接加入最早聽到的 CH
                    if self.candidate_ch_id is not None and self.has_any_reachable_ch(current_time):
                        self.my_ch_id = self.candidate_ch_id
                        self.my_ch_expire = current_time + self.cluster_hold
                    else:
                        # 保底：如果真的完全沒聽過任何 CH，才回 UC（或你也可以選擇維持 GW）
                        self.role = "UC"
                        self.my_ch_id = None
                        self.my_ch_expire = 0.0
                # OD/CH/Legacy 暫時不處理（你說 CH 退化還在想）
                # elif self.role == "OD": pass
                # elif self.role == "CH": pass
        elif self.role == "CH":
            # 1) 在 hold 期間內聽到「其他 CH」？
            heard_other_ch = False
            if self.candidate_ch_id is not None and self.candidate_ch_id != self.ID:
                # candidate 是最早聽到的 CH，若不是自己，代表附近存在其他 CH
                heard_other_ch = True
            # 2) 自己是否「被需要」？（轉發次數或下游 OD 足夠就算被需要）
            needed = (self.ch_fwd_in_window >= self.CH_FWD_MIN) or (self.ch_child_od_in_window >= self.CH_CHILD_OD_MIN)
            if heard_other_ch and (not needed) and (not getattr(self, "is_data_source", False)):
                self.ch_demote_bad_streak += 1
                if self.ch_demote_bad_streak >= self.CH_DEMOTE_K:
                    # demote CH -> OD, join earliest heard CH
                    self.role = "OD"
                    self.ch_demote_bad_streak = 0

                    if self.candidate_ch_id is not None and self.has_any_reachable_ch(current_time):
                        self.my_ch_id = self.candidate_ch_id
                        self.my_ch_expire = current_time + self.cluster_hold
                    else:
                        # 保底：沒有可加入的 CH，就先保持 CH（避免 OD 沒群）
                        self.role = "CH"
                        self.my_ch_id = self.ID
                        self.my_ch_expire = current_time + self.cluster_hold
            else:
                self.ch_demote_bad_streak = 0
        if self.role != self._role_prev:
            print(f"[CLUSTER] t={current_time:.2f} node={self.ID} {self._role_prev}->{self.role} counts={self.neighbor_counts} good={getattr(self,'gw_good_streak',0)} bad={getattr(self,'gw_bad_streak',0)}")
            self._role_prev = self.role

    def check_ch_timer(self, current_time):
        if not self.enable_proposed_algo:
            return
        if self.role != "UC":
            self.ch_timer = None
            return
        if self.ch_timer is None:
            return

        # 倒數到之前如果已經看到 CH，就不升
        if self.has_any_reachable_ch(current_time):
            self.ch_timer = None
            return

        if current_time >= self.ch_timer:
            self.role = "CH"
            self.ch_timer = None
            self.my_ch_id = self.ID
            self.my_ch_expire = current_time + self.cluster_hold
    
    def calculate_adaptive_duty_cycle(self):
        if not self.enable_proposed_algo:
            return 1.0

        dc_target = 1.0
        if self.role == "CH":
            dc_target = 1.0
        elif self.role == "GW":
            redundant_count = getattr(self, "neighbor_count", None)
            if redundant_count is None:
                # fallback: 若沒有總鄰居數，就用近期聽到的 GW 數當冗餘代理
                redundant_count = self.neighbor_counts.get("GW", 0)
            dc_target = 0.50 if redundant_count > 2 else 0.75
        elif self.role == "OD":
            dc_target = 0.5
        else:
            dc_target = 1.0  # UC/Legacy 不調整或維持 1.0 也行
        rho = 0.5
        self.duty_cycle = (1 - rho) * self.duty_cycle + rho * dc_target
        self.SCAN_WINDOW = self.duty_cycle * self.SCAN_INTERVAL
        return self.duty_cycle
