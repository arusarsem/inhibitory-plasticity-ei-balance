import sys
import json
from pathlib import Path

import numpy as np
from brian2 import ms, second, Hz

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import build, make_signals, channel_means, binned_rate, excitatory_tuning
from plotting import plt, save, NET, MUTED, SERIES

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
SIGNAL_DT = 1 * ms


def run(eta, rho0, t_learn, seed=0):
    rng = np.random.default_rng(seed)
    rates = make_signals((t_learn + 1) * second, SIGNAL_DT, rng)
    m = build(rates, SIGNAL_DT, eta=eta, rho0=rho0 * Hz, rng_seed=seed + 1)
    m["net"].run(t_learn * second)
    spikes = np.array(m["spikes"].t / second)
    tc, r = binned_rate(spikes, t_learn, 1.0)
    inh = channel_means(np.array(m["inh_syn"].w), m["inh_channel"])
    return tc, r, inh


def time_to_settle(tc, r, target, tol=0.5, hold=10):
    sm = np.convolve(r, np.ones(5) / 5, mode="same")
    inside = np.abs(sm - target) <= tol * target
    for i in range(len(inside) - hold):
        if inside[i:i + hold].all():
            return float(tc[i])
    return float("nan")


def rho_sweep():
    targets = [2.0, 5.0, 10.0, 15.0]
    t_learn = 80.0
    results = []
    for rho0 in targets:
        tc, r, inh = run(0.02, rho0, t_learn)
        final = r[tc > t_learn - 30].mean()
        results.append(dict(rho0=rho0, final_rate=float(final), inh_total=float(inh.sum()), inh_profile=inh.tolist()))
        print("rho0", rho0, "final", round(final, 2), flush=True)
    return results


def eta_sweep():
    etas = [0.005, 0.02, 0.08]
    t_learn = 120.0
    results = []
    for eta in etas:
        tc, r, inh = run(eta, 5.0, t_learn)
        late = r[tc > t_learn - 40]
        results.append(dict(
            eta=eta, tc=tc.tolist(), rate=r.tolist(),
            settle_s=time_to_settle(tc, r, 5.0),
            late_mean=float(late.mean()), late_std=float(late.std()),
            tuning_corr=float(np.corrcoef(excitatory_tuning(), inh)[0, 1]),
        ))
        print("eta", eta, "settle", results[-1]["settle_s"], "late", round(late.mean(), 2), round(late.std(), 2), flush=True)
    return results


def plot(rho_res, eta_res):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axes[0]
    x = np.array([d["rho0"] for d in rho_res])
    y = np.array([d["final_rate"] for d in rho_res])
    lim = max(x.max(), y.max()) * 1.15
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1.2, ls="--", label="output = target")
    ax.plot(x, y, color=SERIES[0], marker="o", ms=7, label="simulation")
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(6, -12), fontsize=8.5, color=NET)
    slope = float(np.sum(x * y) / np.sum(x * x))
    ax.text(0.97, 0.05, f"output ≈ {slope:.2f} × target", transform=ax.transAxes, ha="right", fontsize=9, color=NET)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("target rate ρ₀ (Hz)")
    ax.set_ylabel("output rate after learning (Hz)")
    ax.set_title("4a · Output follows the target, but sits below it")
    ax.legend(loc="upper left")

    ax = axes[1]
    for color, d in zip(SERIES, eta_res):
        tc = np.array(d["tc"])
        sm = np.convolve(np.array(d["rate"]), np.ones(5) / 5, mode="same")
        ax.plot(tc, sm, color=color, lw=1.8, label=f"η = {d['eta']:g}")
    ax.axhline(5.0, color=NET, lw=1.2, ls="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("output rate, 5 s average (Hz)")
    ax.set_title("4b · Larger η converges faster, to the same level")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, OUT / "fig4_own_experiment.png")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("rho", "all"):
        (OUT / "summary_rho.json").write_text(json.dumps(rho_sweep(), indent=2))
    if mode in ("eta", "all"):
        (OUT / "summary_eta.json").write_text(json.dumps(eta_sweep(), indent=2))
    if mode in ("plot", "all"):
        plot(json.loads((OUT / "summary_rho.json").read_text()), json.loads((OUT / "summary_eta.json").read_text()))
