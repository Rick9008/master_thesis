
""" 
 this function is almost similar to the scan37 function please for comments reference the scan37 function
 for giving more freedom to the developer in future development we preferred to write them in the separate functions 
"""
from initializer import *
import random 
import copy
import df_utils
import radio_model
def Scan_function38(nodes, i, all_event, NODE_TIME, NODE_EVENT, reception_ratio, logger, Time):
    if Time < nodes[i].first_time_scan+nodes[i].SCAN_WINDOW:
        if nodes[i].enable_proposed_algo:
            nodes[i].check_ch_timer(Time)
        # ====================== [新增]DF Init / Cleanup / Collection timeout ======================
        df_utils.df_init_and_cleanup(nodes, i, Time)
        df_utils.df_try_send_path_reply_on_timeout(nodes, i, Time, BUFFER_SIZE)
        # =================================================================================================
        Num_AD_nodes = 0
        if TOTAL_LOG == 1:
            detail_log.info("scan38 node %s at Time %s ", i, all_event[i][NODE_TIME])
        for j in range(len(nodes[i].neighbors)):
            if nodes[nodes[i].neighbors[j]].advertisetag38 == 1:
                Num_AD_nodes += 1
                Advertise_node = nodes[i].neighbors[j]
        if Num_AD_nodes >= 1:
            nodes[i].busy_steps += 1
        if Num_AD_nodes >= 2:
            nodes[i].collision_steps += 1
        if Num_AD_nodes == 1:
            nodes[i].scanchannel38(nodes[Advertise_node].channel38)
             # if random.randint(0, 100) > reception_ratio[nodes[i].ID][Advertise_node]:
            pr_dbm = reception_ratio[nodes[i].ID][Advertise_node]  # now it's Pr(dBm)
            if not radio_model.can_receive_once(
                    pr_dbm_signal=pr_dbm,
                    nf_dbm=radio_params["NF_DBM"],
                    iwlan_dbm=radio_params["IWLAN_DBM"],
                    ib_mw=0.0,  # 這裡先不算 BLE mesh 干擾（因為 Num_AD_nodes==1）
                    alpha=radio_params["ALPHA"],
                    b_bits=radio_params["B_BITS"],
                    rnd=random
            ):  
                nodes[i].cache.remove(nodes[i].cache[len(nodes[i].cache)-1])
                all_event[i][NODE_TIME] += SCAN_STEP
                all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                nodes[i].Scan_Time += SCAN_STEP
                return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
            # =================================================================
            received_packet = nodes[i].cache[len(nodes[i].cache) - 1]
            link_quality = reception_ratio[nodes[i].ID][Advertise_node]
            pkt_type = getattr(received_packet, "type", None)
            # --- 分群邏輯 (Clustering) ---
            if nodes[i].enable_proposed_algo:
                sender_id = received_packet.source
                sender_role = getattr(received_packet, 'src_role', 'Legacy')
                if nodes[i].role == "CH":
                    pkt_ch = getattr(received_packet, "src_ch_id", getattr(received_packet, "my_ch_id", None))
                    if sender_role == "OD" and pkt_ch == nodes[i].ID:
                        nodes[i].ch_child_od_in_window += 1
                nodes[i].on_hear_sender(sender_id, sender_role, Time)
                nodes[i].update_neighbor_counts(sender_role)
                nodes[i].run_passive_clustering(Time)
                nodes[i].calculate_adaptive_duty_cycle()
            if pkt_type == 1:
                is_sink = isinstance(received_packet.destination, list) and (nodes[i].ID in received_packet.destination)
                if is_sink:
                    print(f"[RX-SINK] t={round(Time,3)} sink={nodes[i].ID} got DATA from={received_packet.source} "
                          f"seq={received_packet.seq_number} gen_t={round(getattr(received_packet,'generation_time',-1),3)}, 38")
           # ====================== DF common handling (shared by scan37/38/39) ======================
            # (A) DF_DATA selective relaying gate
            handled, ret_time, ret_event = df_utils.df_gate_df_data(
                nodes, i, received_packet, Time, SCAN_STEP, SCAN38_C_EVENT
            )
            if handled:
                all_event[i][NODE_TIME] = ret_time
                all_event[i][NODE_EVENT] = ret_event
                nodes[i].Scan_Time += SCAN_STEP
                return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]

            # (B) PATH_REQUEST handling
            action, updated_pkt = df_utils.df_handle_path_request(
                nodes, i, received_packet, Advertise_node, link_quality, Time
            )
            if action == "COLLECTED_RETURN":
                # dst 收集後：消耗 packet，不走 generic buffer
                if received_packet in nodes[i].cache:
                    nodes[i].cache.remove(received_packet)
                all_event[i][NODE_TIME] += SCAN_STEP
                all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                nodes[i].Scan_Time += SCAN_STEP
                return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
            elif action == "FORWARD_UPDATED" and updated_pkt is not None:
                # intermediate：用更新過 metric/trace 的 copy 取代 cache[-1]，後面 generic buffer.append 會轉發它
                nodes[i].cache[len(nodes[i].cache) - 1] = updated_pkt
                received_packet = updated_pkt  # 後面若還要用，保持一致

            # (C) PATH_REPLY handling (hop-by-hop)
            handled, need_return, _ = df_utils.df_handle_path_reply(
                nodes, i, received_packet, Advertise_node, Time, BUFFER_SIZE
            )
            if handled and need_return:
                all_event[i][NODE_TIME] += SCAN_STEP
                all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                nodes[i].Scan_Time += SCAN_STEP
                return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
            # =========================================================================================
            f = nodes[i].ID in nodes[i].cache[len(nodes[i].cache)-1].destination
            if f == False and nodes[i].cache[len(nodes[i].cache)-1].friend_poll == 1:
                nodes[i].cache.remove(nodes[i].cache[len(nodes[i].cache)-1])
                all_event[i][NODE_TIME] += SCAN_STEP
                all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                nodes[i].Scan_Time += SCAN_STEP
                return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
            if (nodes[i].feature == LOW_POWER or nodes[i].feature == SINK_NODE) and f == False:
                nodes[i].cache.remove(nodes[i].cache[len(nodes[i].cache)-1])
                all_event[i][NODE_TIME] += SCAN_STEP
                all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                nodes[i].Scan_Time += SCAN_STEP
                return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
            if (nodes[i].feature == LOW_POWER and nodes[i].friend_Id != Advertise_node) and f == True:
                nodes[i].cache.remove(nodes[i].cache[len(nodes[i].cache)-1])
                all_event[i][NODE_TIME] += SCAN_STEP
                all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                nodes[i].Scan_Time += SCAN_STEP
                return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
            L = nodes[i].LOW_POWER_ID in nodes[i].cache[len(nodes[i].cache)-1].destination
            if(nodes[i].feature == JUST_GENERATION or (nodes[i].feature == FRIEND_NODE and L == False)) and f == False\
                    and nodes[i].cache[len(nodes[i].cache)-1].heartbeat == 0 and \
                    nodes[i].cache[len(nodes[i].cache)-1].friend_poll == 0:
                nodes[i].cache.remove(nodes[i].cache[len(nodes[i].cache)-1])
                all_event[i][NODE_TIME] += SCAN_STEP
                all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                nodes[i].Scan_Time += SCAN_STEP
                return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
            if nodes[i].cache[len(nodes[i].cache)-1].friend_poll == 0:
                if nodes[i].cache[len(nodes[i].cache)-1].heartbeat == 0:
                    comp_temp = nodes[i].last_seq_number[nodes[i].cache[len(nodes[i].cache)-1].source]
                else:
                    comp_temp = nodes[i].h_last_seq_number[nodes[i].cache[len(nodes[i].cache)-1].source]
                if nodes[i].cache[len(nodes[i].cache)-1].seq_number > comp_temp:
                    if nodes[i].cache[len(nodes[i].cache)-1].heartbeat == 0:
                        nodes[i].last_seq_number[nodes[i].cache[len(nodes[i].cache)-1].source] = \
                            nodes[i].cache[len(nodes[i].cache)-1].seq_number
                    else:
                        nodes[i].h_last_seq_number[nodes[i].cache[len(nodes[i].cache)-1].source] = \
                            nodes[i].cache[len(nodes[i].cache)-1].seq_number
                else:
                    nodes[i].cache.remove(nodes[i].cache[len(nodes[i].cache)-1])
                    all_event[i][NODE_TIME] += SCAN_STEP
                    all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                    nodes[i].Scan_Time += SCAN_STEP
                    return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
            # ====================== MF role/group forwarding gate (after seq de-dup, before buffer.append) ======================
            # only gate "main data" (avoid breaking friend/heartbeat control traffic)
            if nodes[i].cache[-1].friend_poll == 0 and nodes[i].cache[-1].heartbeat == 0:
                is_dst = nodes[i].ID in nodes[i].cache[-1].destination
                if not is_dst:
                    pkt_ch = getattr(nodes[i].cache[-1], "src_ch_id", None)
                    if pkt_ch is None:
                        pass
                    else:                    
                        # expire OD's membership if needed
                        if getattr(nodes[i], "my_ch_id", None) is not None and Time >= getattr(nodes[i], "my_ch_expire", 0):
                            nodes[i].my_ch_id = None
                            nodes[i].my_ch_expire = 0.0
                        if nodes[i].role == "OD":
                            # OD forwards only its own cluster
                            if nodes[i].my_ch_id is None or pkt_ch is None or pkt_ch != nodes[i].my_ch_id:
                                nodes[i].cache.remove(nodes[i].cache[-1])
                                all_event[i][NODE_TIME] += SCAN_STEP
                                all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                                nodes[i].Scan_Time += SCAN_STEP
                                return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
                        elif nodes[i].role == "CH":
                            # CH forwards only packets tagged to itself
                            if pkt_ch is None or pkt_ch != nodes[i].ID:
                                nodes[i].cache.remove(nodes[i].cache[-1])
                                all_event[i][NODE_TIME] += SCAN_STEP
                                all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                                nodes[i].Scan_Time += SCAN_STEP
                                return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
                        elif nodes[i].role == "GW":
                            # GW forwards everything
                            pass
                        else:
                            # UC / Legacy: follow managed flooding
                            pass
            # ===================================================================================================================
            
            if nodes[i].cache[len(nodes[i].cache)-1].friend_poll == 0 or \
                    nodes[i].cache[len(nodes[i].cache)-1].friend_poll == 1:
                if len(nodes[i].buffer) < BUFFER_SIZE:
                    if nodes[i].enable_proposed_algo and nodes[i].role == "CH":
                        nodes[i].ch_fwd_in_window += 1
                    nodes[i].buffer.append(nodes[i].cache[len(nodes[i].cache)-1])
                    f = nodes[i].ID in nodes[i].buffer[len(nodes[i].buffer)-1].destination
                    if TOTAL_LOG == 1 and ((nodes[i].feature == JUST_RELAY) or
                                           (nodes[i].feature == RELAY_AND_GENERATION) or
                                           (nodes[i].feature == FRIEND_RELAY_NODE)) and f == False and\
                            nodes[i].cache[len(nodes[i].cache)-1].friend_poll == 0:
                        logger[i].info('(relay)    %s    %s    %s    %s    %s    %s    %s', Advertise_node,
                                       nodes[i].buffer[len(nodes[i].buffer)-1].source, round(Time, 2),
                                       nodes[i].buffer[len(nodes[i].buffer)-1].seq_number,
                                       round(nodes[i].buffer[len(nodes[i].buffer)-1].generation_time, 2),
                                       nodes[i].buffer[len(nodes[i].buffer)-1].TTL, len(nodes[i].buffer))
                    if nodes[i].feature == FRIEND_NODE or nodes[i].feature == FRIEND_RELAY_NODE:
                        f = nodes[i].LOW_POWER_ID in nodes[i].buffer[len(nodes[i].buffer)-1].destination
                        if f == True:
                                nodes[i].friend_queue.append(nodes[i].buffer[len(nodes[i].buffer)-1])
                                nodes[i].buffer.remove(nodes[i].buffer[len(nodes[i].buffer)-1])
                                all_event[i][NODE_TIME] += SCAN_STEP
                                all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                                nodes[i].Scan_Time += SCAN_STEP
                                return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
                    f = nodes[i].ID in nodes[i].buffer[len(nodes[i].buffer)-1].destination
                    if f == True:  # the packet is received by destination
                          if TOTAL_LOG == 1 and nodes[i].buffer[len(nodes[i].buffer)-1].heartbeat == 0 and\
                                  nodes[i].cache[len(nodes[i].cache)-1].friend_poll == 0:
                              logger[i].info('(main)    %s    %s    %s    %s    %s',
                                             nodes[i].buffer[len(nodes[i].buffer)-1].source,
                                             nodes[i].buffer[len(nodes[i].buffer)-1].seq_number,
                                             round(nodes[i].buffer[len(nodes[i].buffer)-1].generation_time, 2),
                                             nodes[i].ID, round(Time, 2),
                                             )
                          if TOTAL_LOG == 1 and nodes[i].buffer[len(nodes[i].buffer)-1].heartbeat == 1 and \
                                  nodes[i].cache[len(nodes[i].cache)-1].friend_poll == 0:
                              logger[i].info('(heartbeat)    %s    %s    %s    %s    %s    %s    %s',
                                             nodes[i].buffer[len(nodes[i].buffer)-1].source,
                                             nodes[i].buffer[len(nodes[i].buffer)-1].seq_number,
                                             round(nodes[i].buffer[len(nodes[i].buffer)-1].generation_time, 2),
                                             nodes[i].ID, round(Time, 2),
                                             nodes[i].minhop, nodes[i].maxhop)
                          if TOTAL_LOG == 0 and nodes[i].cache[len(nodes[i].cache)-1].friend_poll == 0 and \
                                  nodes[i].buffer[len(nodes[i].buffer)-1].heartbeat == 0:
                              logger.info('%s    %s    %s    %s    %s', nodes[i].buffer[len(nodes[i].buffer)-1].source,
                                                 nodes[i].ID, round(Time, 2),
                                          nodes[i].buffer[len(nodes[i].buffer)-1].seq_number,
                                                 round(nodes[i].buffer[len(nodes[i].buffer)-1].generation_time, 2))
                          if (nodes[i].feature == FRIEND_NODE or nodes[i].feature == FRIEND_RELAY_NODE) and\
                                  nodes[i].buffer[len(nodes[i].buffer)-1].friend_poll == 1:
                              nodes[i].response_friend_time = all_event[i][NODE_TIME]+Receive_Delay +\
                                                              random.randint(0, Receive_window)
                              all_event[i][NODE_TIME] += SCAN_STEP
                              all_event[i][NODE_EVENT] = SCAN38_C_EVENT
                              nodes[i].Scan_Time += SCAN_STEP
                              if nodes[i].previous_ack < nodes[i].buffer[len(nodes[i].buffer)-1].acknowledge:
                                  if len(nodes[i].friend_queue) >= 1:
                                      nodes[i].friend_queue.remove(nodes[i].friend_queue[0])
                                  nodes[i].previous_ack = nodes[i].buffer[len(nodes[i].buffer)-1].acknowledge
                              nodes[i].buffer.remove(nodes[i].buffer[len(nodes[i].buffer)-1])
                              return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
                          if nodes[i].feature == LOW_POWER:
                              if nodes[i].buffer[len(nodes[i].buffer)-1].friend_update == 1:
                                  nodes[i].buffer[len(nodes[i].buffer)-1].friend_update = 0
                                  nodes[i].low_power_ack = 0
                                  nodes[i].not_receive = 0
                                  Generat_Time = nodes[i].last_generation_time+nodes[i].GENERATION_INTERVAL
                                  scan_time_low = all_event[i][NODE_TIME] - nodes[i].Low_Scan_Time
                                  remain_receive_window = Receive_window-scan_time_low
                                  nodes[i].last_poll_time = all_event[i][NODE_TIME] + sleep_time + \
                                                            remain_receive_window
                                  Poll_time = nodes[i].last_poll_time+lowpower_Poll_interval
                                  if Generat_Time < Poll_time:
                                    all_event[i][NODE_TIME] = Generat_Time
                                    all_event[i][NODE_EVENT] = GENERATION_EVENT_Adv37
                                    nodes[i].Scan_Time += SCAN_STEP
                                    nodes[i].buffer.remove(nodes[i].buffer[len(nodes[i].buffer)-1])
                                    return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
                                  else:
                                    all_event[i][NODE_TIME] = Poll_time
                                    all_event[i][NODE_EVENT] = SEND_POLL
                                    nodes[i].Scan_Time += SCAN_STEP
                                    nodes[i].buffer.remove(nodes[i].buffer[len(nodes[i].buffer)-1])
                                    return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
                              else:
                                  nodes[i].not_receive = 0
                                  scan_time_low = all_event[i][NODE_TIME]-nodes[i].Low_Scan_Time
                                  remain_receive_window = Receive_window-scan_time_low
                                  all_event[i][NODE_TIME] += sleep_time+remain_receive_window
                                  all_event[i][NODE_EVENT] = SEND_POLL
                                  nodes[i].Scan_Time += SCAN_STEP
                                  if nodes[i].buffer[len(nodes[i].buffer)-1].heartbeat == 0:
                                      nodes[i].buffer.remove(nodes[i].buffer[len(nodes[i].buffer)-1])
                                      return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
                          if nodes[i].buffer[len(nodes[i].buffer)-1].heartbeat == 1:
                              hop = nodes[i].buffer[len(nodes[i].buffer)-1].initTTL-\
                                    nodes[i].buffer[len(nodes[i].buffer)-1].TTL + 1
                              nodes[i].five_hop.append(hop)
                              if len(nodes[i].five_hop) > 5:
                                  nodes[i].five_hop.pop(0)
                              nodes[i].minhop = min(nodes[i].five_hop)
                              nodes[i].maxhop = max(nodes[i].five_hop)
                              nodes[i].node_TTL = nodes[i].minhop+R_h
                              if nodes[i].feature == LOW_POWER:
                                  all_event[i][NODE_TIME] += sleep_time
                                  all_event[i][NODE_EVENT] = SEND_POLL
                                  nodes[i].Scan_Time += SCAN_STEP
                                  nodes[i].buffer.remove(nodes[i].buffer[len(nodes[i].buffer)-1])
                                  return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
                              if nodes[i].feature == JUST_GENERATION:
                                  nodes[i].buffer.remove(nodes[i].buffer[len(nodes[i].buffer)-1])
                          if len(nodes[i].buffer) > 0:
                              if len(nodes[i].buffer[len(nodes[i].buffer)-1].destination) == 1:
                                  nodes[i].buffer.remove(nodes[i].buffer[len(nodes[i].buffer) - 1])
                              elif nodes[i].feature != RELAY_AND_GENERATION and nodes[i].feature != JUST_RELAY\
                                      and nodes[i].feature != FRIEND_RELAY_NODE:
                                  nodes[i].buffer.remove(nodes[i].buffer[len(nodes[i].buffer) - 1])
                    else:
                        if nodes[i].feature == JUST_GENERATION:
                            nodes[i].buffer.remove(nodes[i].buffer[len(nodes[i].buffer) - 1])
                            nodes[i].cache.remove(nodes[i].cache[len(nodes[i].cache) - 1])

        all_event[i][NODE_TIME] += SCAN_STEP
        nodes[i].Scan_Time += SCAN_STEP
        all_event[i][NODE_EVENT] = SCAN38_C_EVENT
    return all_event[i][NODE_TIME], all_event[i][NODE_EVENT]
############################################

