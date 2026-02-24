# radio_model.py
# Log-distance path loss + log-normal shadowing + SINR -> PER model

import math
import random

def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)

def mw_to_dbm(mw: float) -> float:
    if mw <= 0:
        return -1e9
    return 10.0 * math.log10(mw)

def build_pr_dbm_matrix(nodes,
                        pt_dbm=0.0,
                        d0=1.0,
                        pl0_db=40.0,
                        eta=3.5,
                        sigma_db=4.0,
                        seed=1):
    """
    Build Pr_dBm[i][j] for all links based on distance and log-normal shadowing.

    PL(d) = PL0 + 10*eta*log10(d/d0) + X_sigma, X_sigma ~ N(0, sigma_db)
    Pr = Pt - PL

    Notes:
      - pl0_db is NOT given by you; common 2.4GHz FSPL@1m is ~40 dB, so default=40.
      - shadowing is fixed per (i,j) for fairness and stability across time.
    """
    rng = random.Random(seed)

    n = len(nodes)
    pr = [[-200.0 for _ in range(n)] for __ in range(n)]

    # pre-sample shadowing per directed link
    shadow = [[0.0 for _ in range(n)] for __ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Gaussian(0, sigma)
            shadow[i][j] = rng.gauss(0.0, sigma_db)

    for rx in range(n):
        for tx in range(n):
            if rx == tx:
                continue
            dx = float(nodes[rx].Xposition) - float(nodes[tx].Xposition)
            dy = float(nodes[rx].Yposition) - float(nodes[tx].Yposition)
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1e-9:
                d = d0

            # log-distance PL
            pl = pl0_db + 10.0 * eta * math.log10(d / d0) + shadow[tx][rx]
            pr_dbm = pt_dbm - pl
            pr[rx][tx] = pr_dbm

    return pr

def per_from_sinr(sinr_lin: float, alpha=0.68, b_bits=312) -> float:
    """
    PER = (0.5 * (1 - sqrt(alpha * SINR)))^b   (your provided formula)
    We clamp to [0,1]. If inside term <=0 then PER ~ 0.
    """
    if sinr_lin <= 0:
        return 1.0

    x = alpha * sinr_lin
    if x <= 0:
        return 1.0

    term = 0.5 * (1.0 - math.sqrt(x))
    if term <= 0:
        return 0.0

    # term in (0,0.5] usually -> term^b becomes very small quickly
    per = term ** b_bits
    if per < 0:
        per = 0.0
    if per > 1:
        per = 1.0
    return per

def can_receive_once(pr_dbm_signal: float,
                     nf_dbm=-90.0,
                     iwlan_dbm=-95.0,
                     ib_mw=0.0,
                     alpha=0.68,
                     b_bits=312,
                     rnd=None) -> bool:
    """
    Decide reception success for a single packet given signal Pr(dBm).
    SINR = Pr / (NF + IWLAN + IB)
    PER -> random draw
    """
    if rnd is None:
        rnd = random

    sig_mw = dbm_to_mw(pr_dbm_signal)
    nf_mw = dbm_to_mw(nf_dbm)
    iwlan_mw = dbm_to_mw(iwlan_dbm)
    den = nf_mw + iwlan_mw + float(ib_mw)
    if den <= 0:
        den = 1e-12

    sinr = sig_mw / den
    per = per_from_sinr(sinr, alpha=alpha, b_bits=b_bits)

    return (rnd.random() > per)
