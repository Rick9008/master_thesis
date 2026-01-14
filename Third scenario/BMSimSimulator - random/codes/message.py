
import random
class message:
    def __init__(self, TTL, source, destination, data):
        self.heartbeat = 0
        self.friend_update = 0
        self.friend_poll = 0
        self.data = data
        self.TTL = TTL
        self.source = source
        self.destination = destination
        self.latency = 0
        self.seq_number = 0
        self.pathlose = random.uniform(0, 1)
        self.generation_time = 0
        # === [新增] 用於 Passive Clustering 的 Piggyback 資訊 ===
        self.src_role = "Legacy"
        # 封包類型: 
        # 0: Standard Data / Non-DF
        # 1: DF_DATA (走路徑的資料)
        # 2: PATH_REQUEST (路徑探索)
        # 3: PATH_REPLY (路徑建立)
        self.type = 0