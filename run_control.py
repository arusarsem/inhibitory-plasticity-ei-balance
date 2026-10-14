import sys
import json
from pathlib import Path
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

TARGETS = [2.0, 5.0, 10.0, 15.0]
SEEDS = [0, 1, 2]
T_LEARN = 80.0
ETA = 0.02


def job(args):
    condition, rho0, seed = args
    from brian2 import ms, second, Hz
    from model import build, make_signals, channel_means, excitatory_tuning
    dt = 1 * ms
    rng = np.random.default_rng(seed)
    rates = make_signals((T_LEARN + 1) * second, dt, rng)
    inh_rates = None
    if condition == "decorrelated":
        inh_rates = make_signals((T_LEARN + 1) * second, dt, np.random.default_rng(1000 + seed))
    m = build(rates, dt, eta=ETA, rho0=rho0 * Hz, rng_seed=seed + 1, inh_rates=inh_rates)
    m["net"].run(T_LEARN * second)
    spikes = np.array(m["spikes"].t / second)
    final = np.sum(spikes > T_LEARN - 30) / 30.0
    inh = channel_means(np.array(m["inh_syn"].w), m["inh_channel"])
    return dict(
        condition=condition, rho0=rho0, seed=seed, final_rate=float(final),
        inh_profile=inh.tolist(),
        tuning_corr=float(np.corrcoef(excitatory_tuning(), inh)[0, 1]),
    )


def simulate():
    jobs = [(c, r, s) for c in ("shared", "decorrelated") for r in TARGETS for s in SEEDS]
    with Pool(2) as pool:
        results = pool.map(job, jobs)
    (OUT / "summary_control.json").write_text(json.dumps(results, indent=2))
    return results


def plot(results):
    from model import excitatory_tuning, N_CHANNELS
    from plotting import plt, save, NET, MUTED, EXC, INH, SERIES

    styles = {
        "shared": dict(color=SERIES[0], label="I inputs share the E channel signals"),
        "decorrelated": dict(color=SERIES[2], label="I inputs driven by independent signals"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    lim = 17.5
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1.2, ls="--", label="output = target")
    summary = {}
    for cond, st in styles.items():
        means, sds = [], []
        for r in TARGETS:
            v = np.array([d["final_rate"] for d in results if d["condition"] == cond and d["rho0"] == r])
            means.append(v.mean())
            sds.append(v.std(ddof=1))
        means, sds = np.array(means), np.array(sds)
        x = np.array(TARGETS)
        slope = float(np.sum(x * means) / np.sum(x * x))
        summary[cond] = dict(means=means.round(2).tolist(), sds=sds.round(2).tolist(), slope=round(slope, 3))
        ax.errorbar(x, means, yerr=sds, color=st["color"], marker="o", ms=6, lw=2, capsize=3,
                    label=f"{st['label'].split(' driven')[0].split(' share')[0]} {'share E signals' if cond == 'shared' else 'independent'} — slope {slope:.2f}")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("target rate ρ₀ (Hz)")
    ax.set_ylabel("output rate after learning (Hz)")
    ax.set_title("5a · Independent I inputs: bias disappears")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    k = np.arange(1, N_CHANNELS + 1)
    exc = excitatory_tuning()
    ax.plot(k, exc / exc.max(), color=MUTED, lw=1.5, ls="--", label="E tuning (scaled)")
    for cond, st in styles.items():
        prof = np.array([d["inh_profile"] for d in results if d["condition"] == cond and d["rho0"] == 5.0])
        mean = prof.mean(axis=0)
        corr = np.mean([d["tuning_corr"] for d in results if d["condition"] == cond and d["rho0"] == 5.0])
        ax.plot(k, mean / mean.max(), color=st["color"], marker="o", ms=5,
                label=f"I weights, {'shared' if cond == 'shared' else 'independent'} (r = {corr:.2f})")
    ax.set_xticks(k)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("input channel")
    ax.set_ylabel("weight (normalised to max)")
    ax.set_title("5b · …and so does co-tuning (ρ₀ = 5 Hz)")
    ax.legend(loc="upper left", fontsize=8, ncol=1, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    save(fig, OUT / "fig5_control.png")
    print(json.dumps(summary, indent=2))
    for cond in styles:
        print(cond, "tuning r @5Hz:", [round(d["tuning_corr"], 3) for d in results if d["condition"] == cond and d["rho0"] == 5.0])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("sim", "all"):
        res = simulate()
    else:
        res = json.loads((OUT / "summary_control.json").read_text())
    plot(res)
