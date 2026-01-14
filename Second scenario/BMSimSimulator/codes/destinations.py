

""" in this function, the node's destinations are determined
 you can use this function, if you have algorithms for determining network TTL also nodes' data. 
in this function, you can choose one destination or a group of destinations or choose them based on the algorithms """
import random
_fixed_df_targets = None
def F_destination(NUMBER_NODES, Center_node, NETWORK_TTL,
                  nodes=None, src_id=None, require_df=True, num_dst=2):
    """
    Choose `num_dst` DF-capable destinations.
    """
    candidates = list(range(NUMBER_NODES))
    global _fixed_df_targets
    # 不選 sink
    if Center_node in candidates:
        candidates.remove(Center_node)

    # 不選自己
    if src_id is not None and src_id in candidates:
        candidates.remove(src_id)

    # 只選 DF-capable
    if require_df and nodes is not None:
        candidates = [i for i in candidates if getattr(nodes[i], "enable_df", False)]

    # 保護：如果 DF-capable 不夠，就退回一般節點
    if len(candidates) < num_dst:
        candidates = list(range(NUMBER_NODES))
        if Center_node in candidates:
            candidates.remove(Center_node)
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
