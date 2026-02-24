

""" in this function, the node's destinations are determined
 you can use this function, if you have algorithms for determining network TTL also nodes' data. 
in this function, you can choose one destination or a group of destinations or choose them based on the algorithms """
import random
_fixed_df_targets = None
def F_destination(NUMBER_NODES, NETWORK_TTL,
                  nodes=None, src_id=None, num_dst=2):
    """
    Choose `num_dst` DF-capable destinations.
    """
    if nodes is not None and src_id is not None:
        nd = nodes[src_id]
        fixed = getattr(nd, "fixed_dsts", None)
        if fixed:
            destination_c = list(fixed)[:num_dst]
            data = random.randint(1, 100)
            TTL = NETWORK_TTL
            return destination_c, data, TTL
    candidates = list(range(NUMBER_NODES))
    global _fixed_df_targets
  
    # 不選自己
    if src_id is not None and src_id in candidates:
        candidates.remove(src_id)

    # 保護：如果 DF-capable 不夠，就退回一般節點
    if len(candidates) < num_dst:
        candidates = list(range(NUMBER_NODES))
        if src_id is not None and src_id in candidates:
            candidates.remove(src_id)

    if _fixed_df_targets is None:
        _fixed_df_targets = random.sample(candidates, num_dst)
    destination_c = list(_fixed_df_targets)

    data = random.randint(1, 100)
    TTL = NETWORK_TTL
    return destination_c, data, TTL

def get_fixed_df_targets():
    global _fixed_df_targets
    return list(_fixed_df_targets) if _fixed_df_targets is not None else []
