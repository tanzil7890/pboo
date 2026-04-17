"""
Reproduction of the PBOO Monte Carlo simulation from:
"Provenance-Backed Optimistic Oracle Resolution for Prediction Markets"
Built solely from the parameters disclosed in Table 1 of the paper.
N = 200,000 markets, seed = 42.
"""

import numpy as np
from math import comb

SEED = 42
N = 200_000

# ---------------------------------------------------------------
# Shared parameters (from Table 1)
# ---------------------------------------------------------------
ALPHA = 0.22  # baseline semantic ambiguity rate

# ---------------------------------------------------------------
# System 1: Manual
# ---------------------------------------------------------------
def simulate_manual(rng):
    # Effective ambiguity = 0.220 (no template reduction)
    ambig = rng.random(N) < 0.220
    # P(wrong | ambiguous) = 0.110; P(wrong | clear) = 0.004
    wrong = np.where(
        ambig,
        rng.random(N) < 0.110,
        rng.random(N) < 0.004,
    )
    # Dispute: 0.065 if ambiguous else 0.003 (manual moderators review)
    disputed = np.where(
        ambig,
        rng.random(N) < 0.065,
        rng.random(N) < 0.003,
    )
    # Latency: log-normal(mu=3.22, sigma=0.50) hours, normal flow
    # Disputed cases add a log-normal(mu=3.80, sigma=0.45) resolution time
    base_lat = rng.lognormal(mean=3.22, sigma=0.50, size=N)
    disp_add = rng.lognormal(mean=3.80, sigma=0.45, size=N)
    latency = base_lat + disputed * disp_add
    # Note: manual moderators don't reliably "correct" disputed wrong cases
    # (the paper doesn't give qf for manual); assume disputes surface but
    # final wrong rate is just the wrong-finalization rate observed above.
    return wrong, disputed, latency

# ---------------------------------------------------------------
# System 2: Single-source optimistic oracle
# ---------------------------------------------------------------
def simulate_single(rng):
    # Template cuts ambiguity by 50%: effective = 0.110
    ambig = rng.random(N) < 0.110
    # Semantic wrong given effective ambiguous: 0.040
    semantic_wrong = ambig & (rng.random(N) < 0.040)
    # Source error: k = 1, e = 0.012
    source_wrong = rng.random(N) < 0.012
    proposal_wrong = semantic_wrong | source_wrong

    # Dispute dynamics
    # On wrong proposals: dispute with prob qd = 0.50
    # On correct proposals: false dispute prob 0.008
    disputed = np.where(
        proposal_wrong,
        rng.random(N) < 0.50,
        rng.random(N) < 0.008,
    )
    # qf = 0.82: adjudication flips wrong -> correct
    fixed = disputed & proposal_wrong & (rng.random(N) < 0.82)
    final_wrong = proposal_wrong & ~fixed

    # Latency: challenge window 2.0h + overhead 0.8h = 2.8h normal
    base_lat = np.full(N, 2.8)
    disp_add = rng.lognormal(mean=3.85, sigma=0.42, size=N)
    latency = base_lat + disputed * disp_add
    return final_wrong, disputed, latency

# ---------------------------------------------------------------
# System 3: PBOO (typed templates + 3-source + provenance + optimistic)
# ---------------------------------------------------------------
def simulate_pboo(rng):
    # Template cuts ambiguity by 85%: effective = 0.033
    ambig = rng.random(N) < 0.033
    semantic_wrong = ambig & (rng.random(N) < 0.020)

    # 3-source majority with common-mode failure pc = 0.003
    common_fail = rng.random(N) < 0.003
    # For non-common-fail markets, draw k=3 independent source errors
    s1 = rng.random(N) < 0.012
    s2 = rng.random(N) < 0.012
    s3 = rng.random(N) < 0.012
    majority_wrong = (s1.astype(int) + s2.astype(int) + s3.astype(int)) >= 2
    source_wrong = common_fail | ((~common_fail) & majority_wrong)

    proposal_wrong = semantic_wrong | source_wrong

    # Dispute: qd = 0.82 on wrong, false-dispute 0.002 on correct
    disputed = np.where(
        proposal_wrong,
        rng.random(N) < 0.82,
        rng.random(N) < 0.002,
    )
    # qf = 0.95
    fixed = disputed & proposal_wrong & (rng.random(N) < 0.95)
    final_wrong = proposal_wrong & ~fixed

    # Latency: challenge window 3.0h + overhead 0.3h = 3.3h normal
    base_lat = np.full(N, 3.3)
    disp_add = rng.lognormal(mean=1.80, sigma=0.25, size=N)
    latency = base_lat + disputed * disp_add
    return final_wrong, disputed, latency

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
def pct(a):
    return 100.0 * float(np.mean(a))

def summarize(name, wrong, disputed, latency):
    return (
        f"{name:<26s}  "
        f"Wrong {pct(wrong):6.3f}%   "
        f"Dispute {pct(disputed):6.3f}%   "
        f"med {np.median(latency):6.2f}h   "
        f"p90 {np.percentile(latency,90):6.2f}h   "
        f"p99 {np.percentile(latency,99):6.2f}h"
    )

def main():
    rng = np.random.default_rng(SEED)
    m = simulate_manual(rng)
    s = simulate_single(rng)
    p = simulate_pboo(rng)

    print("=" * 92)
    print(f"PBOO simulation reproduction   N = {N:,}   seed = {SEED}")
    print("=" * 92)
    print(summarize("Manual",                   *m))
    print(summarize("Single-source optimistic", *s))
    print(summarize("PBOO (proposed)",          *p))
    print()

    # Equation (6) analytical check for PBOO proposal error
    k, ei, pc = 3, 0.012, 0.003
    maj = sum(comb(k, j) * ei**j * (1 - ei)**(k - j) for j in range(2, k + 1))
    eq6 = pc + (1 - pc) * maj
    print(f"Eq.(6) analytical PBOO proposal error: {eq6:.5f}  (paper: 0.00343)")

    # Eq. (7) final error prediction for PBOO
    eq7 = eq6 * (1 - 0.82 * 0.95)
    print(f"Eq.(7) predicted PBOO final error:    {eq7:.5f}  (paper: 0.00095)")

    # -----------------------------------------------------------
    # Sensitivity analysis (Table 3): vary ei and qd for PBOO
    # -----------------------------------------------------------
    print()
    print("Sensitivity (PBOO final error %):  k=3, qf=0.95, pc=0.005")
    print(f"{'ei':>8s}  {'qd=0.40':>10s}  {'qd=0.60':>10s}  {'qd=0.82':>10s}")
    pc_sens = 0.005
    for ei_s in [0.005, 0.012, 0.025, 0.050, 0.100]:
        row = []
        maj_s = sum(comb(k, j) * ei_s**j * (1 - ei_s)**(k - j) for j in range(2, k + 1))
        e_prop = pc_sens + (1 - pc_sens) * maj_s
        for qd in [0.40, 0.60, 0.82]:
            e_fin = e_prop * (1 - qd * 0.95) * 100
            row.append(f"{e_fin:10.3f}")
        print(f"{ei_s:8.3f}  " + "  ".join(row))

    # -----------------------------------------------------------
    # Provenance Value Theorem numerical check
    # Theta ~ Exp(mean = 100), gamma = 0.15
    # -----------------------------------------------------------
    print()
    print("Provenance Value Theorem numerical check (theta ~ Exp(mean=100), gamma=0.15):")
    Cv, gamma, qf, Bc = 100.0, 0.15, 0.95, 1.0
    for Bp in [5, 50]:
        R = qf * Bp - (1 - qf) * Bc
        if R <= 0:
            print(f"  Bp={Bp}: R<=0, skip")
            continue
        q_noprov = 1 - np.exp(-R / Cv)         # F(R)   for Exp(mean=Cv)
        q_prov   = 1 - np.exp(-(R / gamma) / Cv)  # F(R/gamma)
        print(f"  Bp={Bp:>3d}  q_noprov={q_noprov:.3f}  q_prov={q_prov:.3f}  "
              f"ratio={q_prov/max(q_noprov,1e-9):.2f}x")

if __name__ == "__main__":
    main()
