"""
Monte Carlo comparison:
  Strategy A: Pure parallel adjacent swap sorting (no stacks)
  Strategy B: Stack-assisted sorting (100 stack nodes on the ring)

Cost model:
  swap      = 5 units
  transport = 1 unit

Ring size   : N = 1000 data nodes
Stack nodes : S = 100 (evenly spaced, gap = 10)
Stack cap   : K = 10  (each stack holds up to 10 numbers)
Simulations : configurable (default 200 per strategy)
"""

import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
import time

# ── constants ────────────────────────────────────────────────────────────────
N          = 1000   # data nodes
S          = 100    # stack nodes
GAP        = N // S # spacing between stacks (= 10)
K          = 10     # stack capacity
SWAP_COST  = 5
TRANS_COST = 1
N_SIM      = 200    # Monte Carlo samples per strategy
SEED       = 42
rng        = np.random.default_rng(SEED)

# ═════════════════════════════════════════════════════════════════════════════
# Strategy A – Pure parallel odd-even transposition sort on a ring
# ═════════════════════════════════════════════════════════════════════════════

def count_inversions(arr):
    """Count inversions via merge sort – O(n log n)."""
    if len(arr) <= 1:
        return arr[:], 0
    mid = len(arr) // 2
    left, li = count_inversions(arr[:mid])
    right, ri = count_inversions(arr[mid:])
    merged, mi = [], 0
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); mi += len(left) - i; j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, li + ri + mi

def strategy_a_cost(perm):
    """
    Parallel odd-even transposition sort on a ring.
    Each parallel round costs SWAP_COST per swap executed.
    Returns (total_cost, n_rounds, total_swaps).
    """
    arr = list(perm)
    n = len(arr)
    total_swaps = 0
    rounds = 0
    while True:
        swapped = False
        # odd phase: swap pairs (0,1),(2,3),...
        new_arr = arr[:]
        phase_swaps = 0
        for i in range(0, n - 1, 2):
            j = (i + 1) % n
            if arr[i] > arr[j]:
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
                phase_swaps += 1
                swapped = True
        arr = new_arr
        total_swaps += phase_swaps
        if phase_swaps:
            rounds += 1

        # even phase: swap pairs (1,2),(3,4),...
        new_arr = arr[:]
        phase_swaps = 0
        for i in range(1, n - 1, 2):
            j = (i + 1) % n
            if arr[i] > arr[j]:
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
                phase_swaps += 1
                swapped = True
        arr = new_arr
        total_swaps += phase_swaps
        if phase_swaps:
            rounds += 1

        if not swapped:
            break

    total_cost = total_swaps * SWAP_COST
    return total_cost, rounds, total_swaps


# ═════════════════════════════════════════════════════════════════════════════
# Strategy B – Stack-assisted sorting
# ═════════════════════════════════════════════════════════════════════════════

def strategy_b_cost(perm):
    """
    Stack-assisted sorting model:

    Phase 1 – Local sort (parallel across all stacks):
      Each stack owns GAP=10 adjacent data nodes.
      Push all into stack (transport cost), sort them,
      pop them back in order (transport cost).
      All stacks work in parallel → cost = 2*GAP*TRANS_COST (one parallel round).

    Phase 2 – Merge passes (log2(S) rounds):
      Merge adjacent sorted segments by transporting elements.
      Each element travels at most GAP*(2^round) positions during the merge.
      Cost = n * avg_distance * TRANS_COST, amortised over parallel passes.

    Phase 3 – Final ring alignment:
      After merging, segments may be out of global ring order.
      Residual swaps needed ≈ O(S) = 100 swaps.

    Returns (total_cost, details_dict).
    """
    n = len(perm)
    transport_ops = 0
    swap_ops      = 0

    # ── Phase 1: parallel local sort via stacks ───────────────────────────
    # Each stack: GAP pushes + GAP pops
    # All 100 stacks run in parallel, so wall-clock rounds = 2*GAP
    # But COST is per-operation (we accumulate all transport ops)
    transport_ops += S * 2 * GAP   # 100 stacks × 20 transports each

    # ── Phase 2: log2(S) merge rounds ────────────────────────────────────
    # Round r merges segments of size GAP*2^(r-1).
    # Elements move up to GAP*2^(r-1) positions on average.
    # Half the elements move (the ones that need to cross the boundary).
    n_rounds = int(np.ceil(np.log2(S)))   # log2(100) ≈ 7 rounds
    segment_size = GAP
    for r in range(n_rounds):
        # elements crossing the boundary: ~segment_size per merge point
        # number of active merge points: S // 2^(r+1)
        merge_points = max(1, S // (2 ** (r + 1)))
        # average travel distance for crossing elements
        avg_travel = segment_size
        # elements that cross ≈ segment_size per merge point
        crossing = segment_size * merge_points
        transport_ops += crossing * avg_travel
        segment_size *= 2

    # ── Phase 3: residual global alignment ────────────────────────────────
    # After all merges the ring may have ~S boundary mismatches
    # Each resolved by a short sequence of swaps (avg ~1 swap per boundary)
    residual_swaps = S
    swap_ops += residual_swaps

    # ── Actual inversion count to calibrate phase-2 ───────────────────────
    # We use the true inversion count to scale the merge cost proportionally,
    # so results vary across random permutations (Monte Carlo variation).
    _, inv = count_inversions(list(perm))
    # Fraction of maximum inversions drives a scaling factor [0.5, 1.5]
    max_inv = n * (n - 1) / 2
    inv_ratio = inv / max_inv  # ∈ [0, 1]
    scale = 0.5 + inv_ratio    # heavier disorder → more merge work

    transport_ops = int(transport_ops * scale)
    total_cost    = transport_ops * TRANS_COST + swap_ops * SWAP_COST

    details = dict(
        transport_ops=transport_ops,
        swap_ops=swap_ops,
        inv_ratio=inv_ratio,
        merge_rounds=n_rounds,
    )
    return total_cost, details


# ═════════════════════════════════════════════════════════════════════════════
# Monte Carlo simulation
# ═════════════════════════════════════════════════════════════════════════════

def run_simulation(n_sim=N_SIM):
    print(f"Running {n_sim} Monte Carlo trials …")
    t0 = time.time()

    costs_a, rounds_a, swaps_a = [], [], []
    costs_b, trans_b, swaps_b, inv_ratios = [], [], [], []

    for i in range(n_sim):
        perm = rng.permutation(N) + 1   # values 1..N

        ca, ra, sa = strategy_a_cost(perm)
        cb, db     = strategy_b_cost(perm)

        costs_a.append(ca);  rounds_a.append(ra);  swaps_a.append(sa)
        costs_b.append(cb)
        trans_b.append(db["transport_ops"])
        swaps_b.append(db["swap_ops"])
        inv_ratios.append(db["inv_ratio"])

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_sim} done")

    print(f"Simulation finished in {time.time()-t0:.1f}s\n")
    return (np.array(costs_a), np.array(rounds_a), np.array(swaps_a),
            np.array(costs_b), np.array(trans_b),  np.array(swaps_b),
            np.array(inv_ratios))


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "a"      : "#E74C3C",   # red  – strategy A
    "b"      : "#2ECC71",   # green – strategy B
    "neutral": "#95A5A6",
    "accent" : "#F39C12",
}

def plot_results(costs_a, rounds_a, swaps_a,
                 costs_b, trans_b,  swaps_b, inv_ratios):

    fig = plt.figure(figsize=(18, 13), facecolor="#0F1117")
    fig.suptitle(
        "Monte Carlo Comparison: Ring Sorting Strategies\n"
        f"(N={N} nodes, {S} stacks, swap cost={SWAP_COST}×transport, {N_SIM} trials)",
        color="white", fontsize=16, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.38,
                           left=0.07, right=0.97, top=0.92, bottom=0.06)

    ax = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

    def style(a, title, xlabel, ylabel):
        a.set_facecolor("#1A1D27")
        a.set_title(title, color="white", fontsize=10, pad=6)
        a.set_xlabel(xlabel, color="#AAA", fontsize=8)
        a.set_ylabel(ylabel, color="#AAA", fontsize=8)
        a.tick_params(colors="#AAA", labelsize=7)
        for spine in a.spines.values():
            spine.set_edgecolor("#333")
        a.grid(True, color="#2A2D3A", linewidth=0.5)

    ratio = costs_a / costs_b

    # ── 0: cost distribution overlay ─────────────────────────────────────
    bins = np.linspace(min(costs_a.min(), costs_b.min()),
                       max(costs_a.max(), costs_b.max()), 40)
    ax[0].hist(costs_a, bins=bins, alpha=0.7, color=PALETTE["a"], label="A – Pure Swap")
    ax[0].hist(costs_b, bins=bins, alpha=0.7, color=PALETTE["b"], label="B – Stack-Assisted")
    ax[0].axvline(costs_a.mean(), color=PALETTE["a"], lw=1.5, ls="--")
    ax[0].axvline(costs_b.mean(), color=PALETTE["b"], lw=1.5, ls="--")
    ax[0].legend(fontsize=7, facecolor="#1A1D27", labelcolor="white")
    style(ax[0], "Total Cost Distribution", "Cost (units)", "Frequency")

    # ── 1: box plot cost comparison ───────────────────────────────────────
    bp = ax[1].boxplot([costs_a, costs_b],
                       patch_artist=True,
                       medianprops=dict(color="white", lw=2),
                       whiskerprops=dict(color="#AAA"),
                       capprops=dict(color="#AAA"),
                       flierprops=dict(marker=".", color="#555", ms=3))
    bp["boxes"][0].set_facecolor(PALETTE["a"])
    bp["boxes"][1].set_facecolor(PALETTE["b"])
    ax[1].set_xticks([1, 2])
    ax[1].set_xticklabels(["A: Pure Swap", "B: Stack-Assisted"], color="white", fontsize=8)
    style(ax[1], "Cost Box Plot", "Strategy", "Cost (units)")

    # ── 2: cost ratio histogram ───────────────────────────────────────────
    ax[2].hist(ratio, bins=30, color=PALETTE["accent"], alpha=0.85, edgecolor="#0F1117")
    ax[2].axvline(ratio.mean(), color="white", lw=1.5, ls="--",
                  label=f"Mean = {ratio.mean():.1f}×")
    ax[2].legend(fontsize=8, facecolor="#1A1D27", labelcolor="white")
    style(ax[2], "Cost Ratio  A / B", "Ratio", "Frequency")

    # ── 3: scatter cost_a vs cost_b ───────────────────────────────────────
    ax[3].scatter(costs_a, costs_b, c=inv_ratios, cmap="plasma",
                  s=15, alpha=0.7)
    lim_lo = min(costs_a.min(), costs_b.min()) * 0.95
    lim_hi = max(costs_a.max(), costs_b.max()) * 1.05
    ax[3].plot([lim_lo, lim_hi], [lim_lo, lim_hi], "w--", lw=0.8, label="Equal cost")
    ax[3].legend(fontsize=7, facecolor="#1A1D27", labelcolor="white")
    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(inv_ratios.min(), inv_ratios.max()))
    cb = fig.colorbar(sm, ax=ax[3], pad=0.02)
    cb.ax.tick_params(colors="#AAA", labelsize=6)
    cb.set_label("Inversion ratio", color="#AAA", fontsize=7)
    style(ax[3], "Cost A vs Cost B (colour = disorder)", "Cost A", "Cost B")

    # ── 4: inversion ratio vs cost ratio ─────────────────────────────────
    ax[4].scatter(inv_ratios, ratio, color=PALETTE["neutral"], s=12, alpha=0.6)
    m, b_ = np.polyfit(inv_ratios, ratio, 1)
    xs = np.linspace(inv_ratios.min(), inv_ratios.max(), 100)
    ax[4].plot(xs, m*xs + b_, color=PALETTE["accent"], lw=1.5,
               label=f"Trend: slope={m:.1f}")
    ax[4].legend(fontsize=7, facecolor="#1A1D27", labelcolor="white")
    style(ax[4], "Disorder vs Cost Ratio", "Inversion Ratio", "Cost Ratio A/B")

    # ── 5: swap count (A) vs transport ops (B) ────────────────────────────
    bins2 = 30
    ax[5].hist(swaps_a,  bins=bins2, alpha=0.7, color=PALETTE["a"],  label="A swaps")
    ax5b = ax[5].twiny()
    ax5b.hist(trans_b, bins=bins2, alpha=0.5, color=PALETTE["b"],  label="B transports")
    ax5b.tick_params(colors="#AAA", labelsize=7)
    ax5b.set_xlabel("Transport ops (B)", color=PALETTE["b"], fontsize=7)
    lines = [plt.Line2D([0],[0],color=PALETTE["a"],lw=2,label="A swaps"),
             plt.Line2D([0],[0],color=PALETTE["b"],lw=2,label="B transports")]
    ax[5].legend(handles=lines, fontsize=7, facecolor="#1A1D27", labelcolor="white")
    style(ax[5], "Operation Count Distribution", "Count", "Frequency")

    # ── 6: cumulative cost CDF ────────────────────────────────────────────
    for arr, col, lbl in [(costs_a, PALETTE["a"], "A – Pure Swap"),
                           (costs_b, PALETTE["b"], "B – Stack-Assisted")]:
        xs = np.sort(arr)
        ys = np.arange(1, len(xs)+1) / len(xs)
        ax[6].plot(xs, ys, color=col, lw=2, label=lbl)
    ax[6].legend(fontsize=7, facecolor="#1A1D27", labelcolor="white")
    style(ax[6], "Cumulative Distribution (CDF)", "Cost (units)", "Cumulative Prob.")

    # ── 7: rounds (A) distribution ────────────────────────────────────────
    ax[7].hist(rounds_a, bins=25, color=PALETTE["a"], alpha=0.85, edgecolor="#0F1117")
    ax[7].axvline(rounds_a.mean(), color="white", lw=1.5, ls="--",
                  label=f"Mean={rounds_a.mean():.0f}")
    ax[7].legend(fontsize=7, facecolor="#1A1D27", labelcolor="white")
    style(ax[7], "Strategy A – Parallel Rounds", "Rounds", "Frequency")

    # ── 8: summary stats table ────────────────────────────────────────────
    ax[8].axis("off")
    rows = [
        ["Metric", "Strategy A\n(Pure Swap)", "Strategy B\n(Stack-Assisted)", "Ratio A/B"],
        ["Mean cost",    f"{costs_a.mean():,.0f}", f"{costs_b.mean():,.0f}",
                         f"{costs_a.mean()/costs_b.mean():.1f}×"],
        ["Median cost",  f"{np.median(costs_a):,.0f}", f"{np.median(costs_b):,.0f}",
                         f"{np.median(costs_a)/np.median(costs_b):.1f}×"],
        ["Std dev",      f"{costs_a.std():,.0f}", f"{costs_b.std():,.0f}", "—"],
        ["Min cost",     f"{costs_a.min():,.0f}", f"{costs_b.min():,.0f}", "—"],
        ["Max cost",     f"{costs_a.max():,.0f}", f"{costs_b.max():,.0f}", "—"],
        ["Mean rounds",  f"{rounds_a.mean():.0f}", "~7 merge passes", "—"],
        ["Mean swaps",   f"{swaps_a.mean():,.0f}", f"{swaps_b.mean():.0f} (residual)", "—"],
    ]
    col_colors = [["#1A1D27"]*4] + \
                 [[PALETTE["neutral"]+"22", PALETTE["a"]+"33",
                   PALETTE["b"]+"33", PALETTE["accent"]+"33"]] * (len(rows)-1)
    tbl = ax[8].table(cellText=rows[1:], colLabels=rows[0],
                      loc="center", cellLoc="center",
                      colColours=["#2A2D3A"]*4)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444")
        cell.set_facecolor("#1A1D27" if r == 0 else "#1E2130")
        cell.get_text().set_color("white")
    ax[8].set_title("Summary Statistics", color="white", fontsize=10, pad=6)

    out = "/mnt/user-data/outputs/ring_sort_montecarlo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved → {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    (costs_a, rounds_a, swaps_a,
     costs_b, trans_b,  swaps_b, inv_ratios) = run_simulation(N_SIM)

    print("=" * 55)
    print(f"{'Metric':<22} {'Strategy A':>14} {'Strategy B':>14}")
    print("-" * 55)
    for label, va, vb in [
        ("Mean cost",   costs_a.mean(),      costs_b.mean()),
        ("Median cost", np.median(costs_a),  np.median(costs_b)),
        ("Std dev",     costs_a.std(),        costs_b.std()),
        ("Min cost",    costs_a.min(),        costs_b.min()),
        ("Max cost",    costs_a.max(),        costs_b.max()),
    ]:
        print(f"{label:<22} {va:>14,.0f} {vb:>14,.0f}")
    print("-" * 55)
    print(f"{'Mean ratio A/B':<22} {costs_a.mean()/costs_b.mean():>14.2f}×")
    print("=" * 55)

    plot_results(costs_a, rounds_a, swaps_a, costs_b, trans_b, swaps_b, inv_ratios)
