"""
Monte Carlo comparison: Ring Sorting Strategies
================================================
Strategy A : Pure parallel odd-even transposition sort (no stacks)
Strategy B : Stack-assisted sorting with TWO-STACK insertion sort for
             local phase (correctly models LIFO behaviour).

Cost model (parallel rounds):
  One parallel swap      round = SWAP_COST  = 5
  One parallel transport round = TRANS_COST = 1

Stack configurations compared:
  S =  50  (gap = 20, k = 20 elements per stack)
  S = 100  (gap = 10, k = 10 elements per stack)   ← baseline
  S = 500  (gap =  2, k =  2 elements per stack)

N_SIM random permutations are shared across all configurations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# ── global constants ──────────────────────────────────────────────────────────
N          = 1000
SWAP_COST  = 5
TRANS_COST = 1
N_SIM      = 300
SEED       = 42
rng        = np.random.default_rng(SEED)

STACK_CONFIGS = [50, 100, 500]          # stack counts to compare
PALETTE = {
    50  : "#F39C12",   # orange
    100 : "#2ECC71",   # green
    500 : "#3498DB",   # blue
    "a" : "#E74C3C",   # red – strategy A
}

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def count_inversions(arr):
    """Merge-sort inversion count — O(n log n)."""
    if len(arr) <= 1:
        return arr[:], 0
    mid = len(arr) // 2
    left,  li = count_inversions(arr[:mid])
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


def two_stack_sort_rounds(k):
    """
    Count parallel transport rounds needed to sort k elements using
    two-stack insertion sort (correctly models LIFO stacks).

    Algorithm:
      Main stack S1, auxiliary stack S2.
      For each new element x from the input segment:
        - While S1 top > x: pop S1 → push S2  (1 transport round each)
        - Push x onto S1                        (1 transport round)
        - While S2 non-empty: pop S2 → push S1 (1 transport round each)
      Result: S1 is sorted ascending from bottom to top.
      Pop all from S1 back to data nodes        (k transport rounds)

    Each push/pop between stacks or data↔stack = 1 sequential transport round
    (within a stack's local neighbourhood these are serial, not parallel).

    Worst case (reverse-sorted input): O(k²) rounds.
    Average case (random input):       O(k²/2) rounds  ≈ k²/2.
    We use average-case for the Monte Carlo model.
    """
    # Average transport rounds for insertion sort via two stacks ≈ k²/2 + k
    # (k²/2 for the comparisons/moves, k for final pop-back to data nodes)
    return k * k // 2 + k


# ═════════════════════════════════════════════════════════════════════════════
# Strategy A — pure parallel odd-even transposition sort
# ═════════════════════════════════════════════════════════════════════════════

def strategy_a_cost(perm):
    """
    Simulate odd-even transposition sort on a ring.
    Cost = number of parallel rounds × SWAP_COST.
    Each round (odd or even phase) where at least one swap occurs = 1 round.
    """
    arr = list(perm)
    n   = len(arr)
    rounds = 0

    while True:
        moved = False

        # odd phase — pairs (0,1),(2,3),… all swap simultaneously
        new = arr[:]
        did = False
        for i in range(0, n - 1, 2):
            if arr[i] > arr[i + 1]:
                new[i], new[i+1] = new[i+1], new[i]
                did = True
        arr = new
        if did:
            rounds += 1
            moved = True

        # even phase — pairs (1,2),(3,4),…
        new = arr[:]
        did = False
        for i in range(1, n - 1, 2):
            if arr[i] > arr[i + 1]:
                new[i], new[i+1] = new[i+1], new[i]
                did = True
        arr = new
        if did:
            rounds += 1
            moved = True

        if not moved:
            break

    return rounds * SWAP_COST, rounds


# ═════════════════════════════════════════════════════════════════════════════
# Strategy B — stack-assisted sort (parameterised by stack count S)
# ═════════════════════════════════════════════════════════════════════════════

def strategy_b_cost(perm, s):
    """
    Stack-assisted sorting with S stacks evenly spaced on the ring.

    gap = N // S   elements per stack segment
    k   = gap      elements each stack is responsible for

    Phase 1 — Local sort via two-stack insertion sort (parallel across all stacks):
      Rounds = two_stack_sort_rounds(k)
      All S stacks run in parallel → wall-clock = two_stack_sort_rounds(k) rounds.
      Cost = two_stack_sort_rounds(k) × TRANS_COST

    Phase 2 — Iterative merge (ceil(log2(S)) passes):
      Pass r: merge pairs of sorted segments of size gap*2^r.
      Max element travel distance in pass r = gap*2^r (the segment size).
      All merge points are parallel within a pass.
      Rounds per pass = segment_size × scale  (scaled by inversion ratio).
      Cost per pass = rounds × TRANS_COST

    Phase 3 — Residual boundary fix:
      1 parallel swap round to fix ~S remaining boundary mismatches.
      Cost = 1 × SWAP_COST
    """
    n   = len(perm)
    gap = max(1, n // s)   # elements per stack
    k   = gap

    _, inv    = count_inversions(list(perm))
    max_inv   = n * (n - 1) / 2
    inv_ratio = inv / max_inv
    scale     = 0.5 + inv_ratio        # [0.5, 1.5] — more disorder → more merge work

    # ── Phase 1: two-stack local sort ────────────────────────────────────────
    phase1_rounds = two_stack_sort_rounds(k)

    # ── Phase 2: merge passes ────────────────────────────────────────────────
    n_merge_passes = int(np.ceil(np.log2(max(s, 2))))
    phase2_rounds  = 0
    seg = gap
    for _ in range(n_merge_passes):
        pass_rounds    = int(seg * scale)
        phase2_rounds += pass_rounds
        seg           *= 2

    # ── Phase 3: residual swap ────────────────────────────────────────────────
    phase3_rounds = 1   # 1 parallel swap round

    transport_rounds = phase1_rounds + phase2_rounds
    swap_rounds      = phase3_rounds

    total_cost = transport_rounds * TRANS_COST + swap_rounds * SWAP_COST

    return total_cost, dict(
        phase1=phase1_rounds,
        phase2=phase2_rounds,
        phase3_swap=swap_rounds,
        transport_rounds=transport_rounds,
        inv_ratio=inv_ratio,
        gap=gap,
        n_merge_passes=n_merge_passes,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Monte Carlo simulation
# ═════════════════════════════════════════════════════════════════════════════

def run_simulation():
    print(f"Running {N_SIM} Monte Carlo trials …")
    t0 = time.time()

    # Pre-generate all permutations (shared across strategies)
    perms = [rng.permutation(N) + 1 for _ in range(N_SIM)]

    results_a = dict(cost=[], rounds=[])
    results_b = {s: dict(cost=[], tr=[], p1=[], p2=[], inv=[]) for s in STACK_CONFIGS}

    for i, perm in enumerate(perms):
        ca, ra = strategy_a_cost(perm)
        results_a["cost"].append(ca)
        results_a["rounds"].append(ra)

        for s in STACK_CONFIGS:
            cb, db = strategy_b_cost(perm, s)
            results_b[s]["cost"].append(cb)
            results_b[s]["tr"].append(db["transport_rounds"])
            results_b[s]["p1"].append(db["phase1"])
            results_b[s]["p2"].append(db["phase2"])
            results_b[s]["inv"].append(db["inv_ratio"])

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N_SIM} done")

    # Convert to numpy
    for k in results_a:
        results_a[k] = np.array(results_a[k])
    for s in STACK_CONFIGS:
        for k in results_b[s]:
            results_b[s][k] = np.array(results_b[s][k])

    print(f"Simulation finished in {time.time()-t0:.1f}s\n")
    return results_a, results_b


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(results_a, results_b):
    fig = plt.figure(figsize=(20, 15), facecolor="#0F1117")
    fig.suptitle(
        "Monte Carlo Ring Sort: Strategy A (Pure Swap) vs Strategy B (Stack-Assisted)\n"
        f"N={N} nodes · swap={SWAP_COST}× transport · {N_SIM} trials · "
        f"Stack counts: {STACK_CONFIGS}",
        color="white", fontsize=14, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.48, wspace=0.35,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)
    ax = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

    def style(a, title, xlabel, ylabel, legend=True):
        a.set_facecolor("#1A1D27")
        a.set_title(title, color="white", fontsize=9.5, pad=5)
        a.set_xlabel(xlabel, color="#AAA", fontsize=8)
        a.set_ylabel(ylabel, color="#AAA", fontsize=8)
        a.tick_params(colors="#AAA", labelsize=7)
        for sp in a.spines.values():
            sp.set_edgecolor("#333")
        a.grid(True, color="#2A2D3A", linewidth=0.5)
        if legend:
            a.legend(fontsize=7, facecolor="#1A1D27", labelcolor="white",
                     framealpha=0.8)

    costs_a = results_a["cost"]
    rounds_a = results_a["rounds"]
    inv_ratios = results_b[100]["inv"]   # same across configs (shared perms)

    # ── 0: cost distribution — all strategies ────────────────────────────────
    all_vals = np.concatenate([costs_a] + [results_b[s]["cost"] for s in STACK_CONFIGS])
    bins = np.linspace(all_vals.min(), all_vals.max(), 50)
    ax[0].hist(costs_a, bins=bins, alpha=0.6, color=PALETTE["a"], label="A – Pure Swap")
    for s in STACK_CONFIGS:
        ax[0].hist(results_b[s]["cost"], bins=bins, alpha=0.6,
                   color=PALETTE[s], label=f"B – S={s}")
        ax[0].axvline(results_b[s]["cost"].mean(), color=PALETTE[s], lw=1.2, ls="--")
    ax[0].axvline(costs_a.mean(), color=PALETTE["a"], lw=1.2, ls="--")
    style(ax[0], "Cost Distribution (all strategies)", "Cost (rounds)", "Frequency")

    # ── 1: mean cost bar chart with std error bars ────────────────────────────
    labels  = ["A\n(no stack)"] + [f"B\nS={s}" for s in STACK_CONFIGS]
    means   = [costs_a.mean()] + [results_b[s]["cost"].mean() for s in STACK_CONFIGS]
    stds    = [costs_a.std()]  + [results_b[s]["cost"].std()  for s in STACK_CONFIGS]
    colors  = [PALETTE["a"]]   + [PALETTE[s] for s in STACK_CONFIGS]
    xpos    = np.arange(len(labels))
    bars = ax[1].bar(xpos, means, color=colors, alpha=0.85, width=0.6,
                     yerr=stds, capsize=4,
                     error_kw=dict(ecolor="#AAA", lw=1.2))
    for bar, m in zip(bars, means):
        ax[1].text(bar.get_x() + bar.get_width()/2, m + max(stds)*0.05,
                   f"{m:.0f}", ha="center", va="bottom", color="white", fontsize=7.5)
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels(labels, color="white", fontsize=8)
    style(ax[1], "Mean Cost ± Std Dev", "Strategy", "Cost (rounds)", legend=False)

    # ── 2: cost ratio A/B for each stack config ───────────────────────────────
    for s in STACK_CONFIGS:
        ratio = costs_a / results_b[s]["cost"]
        ax[2].hist(ratio, bins=25, alpha=0.65, color=PALETTE[s],
                   label=f"S={s}  μ={ratio.mean():.2f}×")
        ax[2].axvline(ratio.mean(), color=PALETTE[s], lw=1.5, ls="--")
    ax[2].axvline(1.0, color="white", lw=0.8, ls=":", label="Equal cost")
    style(ax[2], "Cost Ratio  A / B  (per stack config)", "Ratio A/B", "Frequency")

    # ── 3: phase breakdown — stacked bar per stack config ─────────────────────
    configs_lbl = [f"S={s}" for s in STACK_CONFIGS]
    p1_means = [results_b[s]["p1"].mean() for s in STACK_CONFIGS]
    p2_means = [results_b[s]["p2"].mean() for s in STACK_CONFIGS]
    p3_cost  = [SWAP_COST] * len(STACK_CONFIGS)   # always 1 swap round × 5
    xpos2 = np.arange(len(STACK_CONFIGS))
    b1 = ax[3].bar(xpos2, p1_means, color="#9B59B6", alpha=0.85, label="Phase 1 (local sort)")
    b2 = ax[3].bar(xpos2, p2_means, bottom=p1_means, color="#1ABC9C", alpha=0.85, label="Phase 2 (merge)")
    bottom2 = [a+b for a,b in zip(p1_means, p2_means)]
    b3 = ax[3].bar(xpos2, p3_cost, bottom=bottom2, color="#E74C3C", alpha=0.85, label="Phase 3 (residual swap×5)")
    ax[3].set_xticks(xpos2)
    ax[3].set_xticklabels(configs_lbl, color="white", fontsize=9)
    style(ax[3], "Phase Cost Breakdown (Strategy B)", "Stack Config", "Rounds")

    # ── 4: cost vs inversion ratio (scatter, one config each) ─────────────────
    for s in STACK_CONFIGS:
        ax[4].scatter(inv_ratios, results_b[s]["cost"],
                      color=PALETTE[s], s=10, alpha=0.5, label=f"B S={s}")
    ax[4].scatter(inv_ratios, costs_a, color=PALETTE["a"], s=10, alpha=0.3, label="A")
    style(ax[4], "Cost vs Disorder (inversion ratio)", "Inversion Ratio", "Cost")

    # ── 5: phase 1 cost vs gap size ───────────────────────────────────────────
    gaps   = [N // s for s in STACK_CONFIGS]
    p1_th  = [two_stack_sort_rounds(g) for g in gaps]
    ax[5].plot(gaps, p1_th, "o-", color="#9B59B6", lw=2, ms=8, label="Phase 1 (two-stack sort)")
    ax[5].plot(gaps, [2*g for g in gaps], "s--", color="#AAA", lw=1.5, ms=6,
               label="Naïve 2k (incorrect)")
    for g, v in zip(gaps, p1_th):
        ax[5].annotate(f"gap={g}\n{v}r", (g, v), textcoords="offset points",
                       xytext=(6, 4), color="white", fontsize=7.5)
    ax[5].set_xlabel("Gap size (k elements per stack)", color="#AAA", fontsize=8)
    style(ax[5], "Phase 1 Rounds vs Gap Size\n(two-stack insertion sort)", "Gap (k)", "Rounds")

    # ── 6: CDF ────────────────────────────────────────────────────────────────
    for arr, col, lbl in (
        [(costs_a, PALETTE["a"], "A – Pure Swap")] +
        [(results_b[s]["cost"], PALETTE[s], f"B – S={s}") for s in STACK_CONFIGS]
    ):
        xs = np.sort(arr)
        ys = np.arange(1, len(xs)+1) / len(xs)
        ax[6].plot(xs, ys, color=col, lw=2, label=lbl)
    style(ax[6], "Cumulative Cost Distribution (CDF)", "Cost", "Cumulative Prob.")

    # ── 7: merge passes contribution per config ────────────────────────────────
    for s in STACK_CONFIGS:
        ax[7].scatter(inv_ratios, results_b[s]["p2"],
                      color=PALETTE[s], s=10, alpha=0.55, label=f"S={s}")
    style(ax[7], "Phase 2 (Merge) Rounds vs Disorder", "Inversion Ratio", "Phase 2 Rounds")

    # ── 8: summary table ──────────────────────────────────────────────────────
    ax[8].axis("off")
    header = ["Metric", "A (no stack)", "B S=50", "B S=100", "B S=500"]
    rows = []

    def fmt(v): return f"{v:,.0f}"

    for label, getter in [
        ("Mean cost",   lambda d: d["cost"].mean()),
        ("Median cost", lambda d: np.median(d["cost"])),
        ("Std dev",     lambda d: d["cost"].std()),
        ("Min cost",    lambda d: d["cost"].min()),
        ("Max cost",    lambda d: d["cost"].max()),
    ]:
        row = [label, fmt(getter(results_a))]
        for s in STACK_CONFIGS:
            row.append(fmt(getter(results_b[s])))
        rows.append(row)

    # ratio rows
    for s in STACK_CONFIGS:
        ratio = costs_a / results_b[s]["cost"]
        rows.append([f"Mean ratio A/B (S={s})",
                     "—",
                     fmt(ratio.mean()) if s == 50 else "—",
                     fmt(ratio.mean()) if s == 100 else "—",
                     fmt(ratio.mean()) if s == 500 else "—"])

    # Phase 1 cost
    p1_row = ["Mean Phase1 rounds", "N/A"]
    for s in STACK_CONFIGS:
        p1_row.append(fmt(results_b[s]["p1"].mean()))
    rows.append(p1_row)

    tbl = ax[8].table(cellText=rows, colLabels=header,
                      loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444")
        cell.set_facecolor("#2A2D3A" if r == 0 else "#1E2130")
        cell.get_text().set_color("white")
    ax[8].set_title("Summary Statistics", color="white", fontsize=10, pad=6)

    out = "/mnt/user-data/outputs/ring_sort_v2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved → {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results_a, results_b = run_simulation()

    print("=" * 72)
    print(f"{'Metric':<28} {'A':>10} {'B S=50':>10} {'B S=100':>10} {'B S=500':>10}")
    print("-" * 72)
    for label, getter in [
        ("Mean cost",   lambda d: d["cost"].mean()),
        ("Median cost", lambda d: np.median(d["cost"])),
        ("Std dev",     lambda d: d["cost"].std()),
        ("Min cost",    lambda d: d["cost"].min()),
        ("Max cost",    lambda d: d["cost"].max()),
    ]:
        print(f"{label:<28} {getter(results_a):>10,.0f}", end="")
        for s in STACK_CONFIGS:
            print(f" {getter(results_b[s]):>10,.0f}", end="")
        print()
    print("-" * 72)
    for s in STACK_CONFIGS:
        ratio = results_a["cost"] / results_b[s]["cost"]
        print(f"Mean ratio A / B (S={s:<3})     {ratio.mean():>10.2f}×")
    print("-" * 72)
    print(f"\nPhase 1 rounds (two-stack sort):")
    for s in STACK_CONFIGS:
        gap = N // s
        print(f"  S={s:<4} gap={gap:<4}  k²/2+k = {two_stack_sort_rounds(gap):>6} rounds")
    print("=" * 72)

    plot_results(results_a, results_b)
