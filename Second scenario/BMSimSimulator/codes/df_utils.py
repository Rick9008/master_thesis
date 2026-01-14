import copy

def df_init_and_cleanup(nodes, i, Time):
    """Ensure DF state exists and cleanup expired discovery_table entries."""
    n = nodes[i]
    if not getattr(n, "enable_df", False):
        return

    # defaults
    if not hasattr(n, "COLLECTION_WINDOW"):
        n.COLLECTION_WINDOW = 20.0
    if not hasattr(n, "DISCOVERY_TIMEOUT"):
        n.DISCOVERY_TIMEOUT = 200.0

    # state containers
    if not hasattr(n, "discovery_table"):
        # key(pid,origin,target) -> {"prev_hop","metric","expires_at", ...}
        n.discovery_table = {}
    if not hasattr(n, "df_lane_count"):
        n.df_lane_count = {}    # (origin,target) -> lanes count (penalty)
    if not hasattr(n, "df_lanes_established"):
        n.df_lanes_established = {}  # target -> established lanes
    if not hasattr(n, "df_lane_nodes"):
        n.df_lane_nodes = {}    # df_lane_nodes[target][lane_idx] = [...]

    if not hasattr(n, "df_collect"):
        # Target-side Path Reply Timer state:
        #   key(pid,origin,target) -> {"end_time": <float>}
        # (We keep the name df_collect to avoid touching many call sites.)
        n.df_collect = {}

    # Target-side replied markers (to ignore late duplicate PATH_REQUEST for the same transaction)
    if not hasattr(n, "df_replied"):
        # key(pid,origin,target) -> expires_at
        n.df_replied = {}

    # cleanup discovery_table
    for k in list(n.discovery_table.keys()):
        if Time > n.discovery_table[k].get("expires_at", 0):
            del n.discovery_table[k]

    # cleanup replied markers
    for k in list(n.df_replied.keys()):
        if Time > n.df_replied.get(k, 0):
            del n.df_replied[k]


def df_gate_df_data(nodes, i, received_packet, Time, SCAN_STEP, scan_c_event):
    """
    DF_DATA(type==1) selective relaying gating.
    Return (handled:bool, ret_time, ret_event) where handled=True means caller should return.
    """
    n = nodes[i]
    if not getattr(n, "enable_df", False):
        return False, None, None
    pkt_type = getattr(received_packet, "type", 0)
    if pkt_type != 1:
        return False, None, None

    # unicast only
    if not (isinstance(received_packet.destination, list) and len(received_packet.destination) == 1):
        return False, None, None

    final_dest = received_packet.destination[0]
    if (n.ID != final_dest) and (final_dest not in n.forwarding_table):
        # drop (do not relay)
        if received_packet in n.cache:
            n.cache.remove(received_packet)
        return True, Time + SCAN_STEP, scan_c_event
    return False, None, None


def df_handle_path_request(nodes, i, received_packet, Advertise_node, link_quality, Time):
    """
    Handle PATH_REQUEST(type==2).
    - If I'm target: start/maintain a Path Reply Timer (COLLECTION_WINDOW) and keep ONLY
      the current-best sequence in discovery_table (spec-style). No candidate list.
    - Else: update discovery_table with best metric + prev_hop; update packet metric and trace for forwarding.
    Returns:
      action: "COLLECTED_RETURN" | "FORWARD_UPDATED" | "NO_DF"
      updated_packet: packet to keep in cache[-1] if FORWARD_UPDATED else None
    """
    n = nodes[i]
    if not getattr(n, "enable_df", False):
        return "NO_DF", None
    if getattr(received_packet, "type", 0) != 2:
        return "NO_DF", None

    origin = getattr(received_packet, "pd_origin", received_packet.source)
    target = getattr(received_packet, "pd_target", None)
    if target is None and isinstance(received_packet.destination, list) and len(received_packet.destination) >= 1:
        target = received_packet.destination[0]

    pid = getattr(received_packet, "path_id", None)
    if pid is None:
        # fallback (should not happen if you set it at generator)
        pid = (origin, getattr(received_packet, "seq_number", 0))

    # (A) I'm target -> start/maintain Path Reply Timer + keep ONLY best sequence in discovery_table
    if target is not None and n.ID == target:
        key = (pid, origin, target)

        # If we already replied for this transaction (within a guard window), ignore late duplicates.
        if key in getattr(n, "df_replied", {}) and Time <= n.df_replied.get(key, 0):
            return "COLLECTED_RETURN", None

        # Start Path Reply Timer on first receive (do not extend on later receives).
        if key not in n.df_collect:
            n.df_collect[key] = {"end_time": Time + n.COLLECTION_WINDOW}

        metric = getattr(received_packet, "path_metric", 0)
        ttl = getattr(received_packet, "TTL", 0)
        rssi = link_quality

        # Build a stable trace snapshot for lane_nodes.
        trace = copy.deepcopy(getattr(received_packet, "path_trace", []) or [])
        # Ensure last hop is present in trace (needed for overlap/disjoint heuristics).
        if len(trace) == 0 or trace[-1] != Advertise_node:
            trace.append(Advertise_node)

        avoid = set(getattr(received_packet, "avoid_nodes", []) or [])
        overlap = len(set(trace) & avoid)

        # score: smaller is better
        score = (metric, overlap, -ttl, -rssi)

        # Store/update the current-best sequence in discovery_table.
        ent = n.discovery_table.get(key)
        if (ent is None) or (ent.get("score", (10**9, 10**9, 0, 0)) > score):
            n.discovery_table[key] = {
                "prev_hop": Advertise_node,
                "metric": metric,
                "ttl": ttl,
                "rssi": rssi,
                "overlap": overlap,
                "score": score,
                "trace": trace,
                # Keep a copy of request as a template for PATH_REPLY.
                "req_pkt": copy.deepcopy(received_packet),
                # Ensure it survives at least through the reply timer window
                "expires_at": Time + max(getattr(n, "DISCOVERY_TIMEOUT", 0),
                                        getattr(n, "COLLECTION_WINDOW", 0) + 1.0),
            }
        return "COLLECTED_RETURN", None

    # (B) intermediate -> update discovery_table best + update outgoing request metric/trace
    if target is None:
        return "NO_DF", None

    # node-count metric approximation:
    #   +1 hop
    #   +avoid penalty if this node is on avoid_nodes (best-effort disjoint)
    #   +lane_counter penalty if I'm already on an established lane toward this target
    cur_metric = getattr(received_packet, "path_metric", 0)
    lane_counter = getattr(received_packet, "lane_counter", 0)
    avoid_nodes = set(getattr(received_packet, "avoid_nodes", []) or [])

    new_metric = cur_metric + 1
    if n.ID in avoid_nodes:
        new_metric += getattr(n, "DF_AVOID_PENALTY", 10)
    if target in n.forwarding_table:
        new_metric += lane_counter

    dkey = (pid, origin, target)
    ent = n.discovery_table.get(dkey)
    if (ent is None) or ("metric" not in ent) or (new_metric < ent["metric"]):
        n.discovery_table[dkey] = {
            "prev_hop": Advertise_node,
            "metric": new_metric,
            # Ensure it survives at least through target reply timer
            "expires_at": Time + max(getattr(n, "DISCOVERY_TIMEOUT", 0),
                                    getattr(n, "COLLECTION_WINDOW", 0) + 1.0)
        }

    fwd_pkt = copy.deepcopy(received_packet)
    fwd_pkt.path_id = pid
    fwd_pkt.pd_origin = origin
    fwd_pkt.pd_target = target
    fwd_pkt.path_metric = new_metric

    # trace (for diversity)
    if not hasattr(fwd_pkt, "path_trace") or fwd_pkt.path_trace is None:
        fwd_pkt.path_trace = [origin]
    if len(fwd_pkt.path_trace) == 0 or fwd_pkt.path_trace[-1] != n.ID:
        fwd_pkt.path_trace.append(n.ID)

    return "FORWARD_UPDATED", fwd_pkt


def df_try_send_path_reply_on_timeout(nodes, i, Time, BUFFER_SIZE):
    """
    Target-side Path Reply Timer.

    When COLLECTION_WINDOW expires for a (path_id, origin, target) transaction, enqueue a single
    PATH_REPLY based on the *current-best* discovery_table entry (spec-style: no candidates list).

    Returns True if a reply was enqueued.
    """
    n = nodes[i]
    if not getattr(n, "enable_df", False):
        return False

    enqueued = False
    expired_keys = [k for k, st in n.df_collect.items() if Time >= st.get("end_time", 0)]

    for key in expired_keys:
        # stop the timer (do not restart for late duplicates; df_replied will gate those)
        n.df_collect.pop(key, None)

        pid, origin, target = key
        ent = n.discovery_table.get(key)
        if not ent:
            continue

        prev_hop = ent.get("prev_hop")
        req_pkt = ent.get("req_pkt")
        trace = ent.get("trace", []) or []
        if prev_hop is None or req_pkt is None:
            continue

        reply = copy.deepcopy(req_pkt)
        reply.type = 3
        reply.path_id = pid
        reply.pd_origin = origin
        reply.pd_target = target
        reply.source = n.ID
        reply.destination = [prev_hop]
        reply.lane_counter = getattr(req_pkt, "lane_counter", 0)
        reply.lane_nodes = list(dict.fromkeys(trace + [n.ID]))

        if len(n.buffer) < BUFFER_SIZE:
            n.buffer.append(reply)
            enqueued = True

        # mark replied to ignore late duplicate PATH_REQUEST for the same transaction
        n.df_replied[key] = Time + max(getattr(n, "DISCOVERY_TIMEOUT", 0),
                                       getattr(n, "COLLECTION_WINDOW", 0) + 1.0)

    return enqueued


def df_handle_path_reply(nodes, i, received_packet, Advertise_node, Time, BUFFER_SIZE):
    """
    Handle PATH_REPLY(type==3) hop-by-hop unicast back to origin.

    Returns:
      handled(bool), need_return(bool), enqueued(bool)

    - handled=True：代表 DF 已經處理（包含 drop / 建表 / 反向轉發），scan 應該直接 return
    - need_return=True：代表 scan 應該結束本輪（避免後面 generic buffer 邏輯重複處理）
    - enqueued=True：代表我有把 reply 往回塞進 buffer（不是必要，但方便 debug）
    """
    n = nodes[i]
    if not getattr(n, "enable_df", False):
        return False, False, False

    if getattr(received_packet, "type", 0) != 3:
        return False, False, False

    # hop-by-hop unicast：只有 destination 裡包含我，才處理
    if not (isinstance(received_packet.destination, list) and (n.ID in received_packet.destination)):
        if received_packet in n.cache:
            n.cache.remove(received_packet)
        return True, True, False

    origin = getattr(received_packet, "pd_origin", None)
    target = getattr(received_packet, "pd_target", getattr(received_packet, "pd_target", received_packet.source))
    pid = getattr(received_packet, "path_id", None)

    # (1) forward direction：我學到「往 target 的 next hop = 送我 reply 的鄰居」
    if target is not None:
        n.forwarding_table[target] = Advertise_node

    enqueued = False

    # (2) backward direction：用 discovery_table 找我當初記的 prev_hop，把 reply 往回送
    if pid is not None and origin is not None and target is not None:
        dkey = (pid, origin, target)
        ent = getattr(n, "discovery_table", {}).get(dkey)

        if ent and (Time <= ent.get("expires_at", -1)):
            prev_hop = ent.get("prev_hop")

            # 不是 origin：把 reply 往回 unicast 給 prev_hop
            if n.ID != origin and prev_hop is not None:
                reply_back = copy.deepcopy(received_packet)
                reply_back.destination = [prev_hop]
                if len(n.buffer) < BUFFER_SIZE:
                    n.buffer.append(reply_back)
                    enqueued = True

            # 是 origin：lane 建立成功，紀錄 lane_node
            elif n.ID == origin:
                if not hasattr(n, "df_lanes_established"):
                    n.df_lanes_established = {}
                if not hasattr(n, "df_lane_nodes"):
                    n.df_lane_nodes = {}

                lane_idx = n.df_lanes_established.get(target, 0) + 1
                n.df_lanes_established[target] = lane_idx

                lane_nodes = getattr(received_packet, "lane_nodes", []) or []
                n.df_lane_nodes.setdefault(target, {})[lane_idx] = list(lane_nodes)
            # 用完就刪，避免同一 transaction 重複處理
            try:
                del n.discovery_table[dkey]
            except Exception:
                pass

    # consume this packet (不讓後面 generic buffer 邏輯再處理一次)
    if received_packet in n.cache:
        n.cache.remove(received_packet)

    return True, True, enqueued


