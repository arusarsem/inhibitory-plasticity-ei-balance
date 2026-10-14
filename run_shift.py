import sys
import json
from pathlib import Path

import numpy as np
from brian2 import StateMonitor, ms, second, Hz

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import build, make_signals, channel_means, binned_rate, excitatory_tuning, N_CHANNELS
from plotting import plt, save, NET, MUTED, SERIES

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

T_BEFORE = 100.0
T_AFTER = 150.0
ETA = 0.02
RHO0 = 5.0
OLD_PREF = 4
NEW_PREF = 1


def main():
    total = T_BEFORE + T_AFTER
    rng = np.random.default_rng(7)
    rates = make_signals((total + 1) * second, 1 * ms, rng)
    m = build(rates, 1 * ms, eta=ETA, rho0=RHO0 * Hz, rng_seed=8, preferred=OLD_PREF)
    wmon = StateMonitor(m["inh_syn"], "w", record=True, dt=2 * second)
    m["net"].add(wmon)

    m["net"].run(T_BEFORE * second)
    new_tuning = excitatory_tuning(preferred=NEW_PREF)
    noise = m["w_exc"] / excitatory_tuning(preferred=OLD_PREF)[m["exc_channel"]]
    m["exc_syn"].w = new_tuning[m["exc_channel"]] * noise
    m["net"].run(T_AFTER * second)

    old_tuning = excitatory_tuning(preferred=OLD_PREF)
    t_w = np.array(wmon.t / second)
    w = np.array(wmon.w)
    profiles = np.array([channel_means(w[:, i], m["inh_channel"]) for i in range(w.shape[1])])
    r_old = np.array([np.corrcoef(p, old_tuning)[0, 1] for p in profiles])
    r_new = np.array([np.corrcoef(p, new_tuning)[0, 1] for p in profiles])

    spikes = np.array(m["spikes"].t / second)
    tc, r = binned_rate(spikes, total, 1.0)
    sm = np.convolve(r, np.ones(5) / 5, mode="same")

    after = t_w > T_BEFORE
    cross = t_w[after][np.argmax(r_new[after] > r_old[after])] - T_BEFORE
    reach = t_w[after][np.argmax(r_new[after] > 0.9)] - T_BEFORE

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.2]))
    ax = axes[0]
    ax.plot(tc, sm, color=SERIES[0])
    ax.axhline(RHO0, color=NET, lw=1.2, ls="--")
    ax.axvline(T_BEFORE, color=MUTED, lw=1.2)
    ax.set_ylabel("output rate (Hz)")
    ax.set_title("6a · Excitatory tuning switches from channel 5 to channel 2 at t = 100 s")
    ax.text(T_BEFORE + 2, ax.get_ylim()[1] * 0.85, "E tuning switched", fontsize=8.5, color=NET)

    ax = axes[1]
    ax.plot(t_w, r_old, color=SERIES[1], label="corr(I weights, old E tuning)")
    ax.plot(t_w, r_new, color=SERIES[2], label="corr(I weights, new E tuning)")
    ax.axvline(T_BEFORE, color=MUTED, lw=1.2)
    ax.set_ylim(-0.6, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Pearson r across channels")
    ax.set_title(f"6b · Inhibition re-learns the new tuning (r > 0.9 after ~{reach:.0f} s)")
    ax.legend(loc="center left")
    fig.tight_layout()
    save(fig, OUT / "fig6_retuning.png")

    summary = dict(
        crossover_s=float(cross), reach_0_9_s=float(reach),
        rate_peak_after_switch=float(sm[(tc > T_BEFORE) & (tc < T_BEFORE + 20)].max()),
        rate_last_30s=float(r[tc > total - 30].mean()),
        r_new_final=float(r_new[-1]), r_old_final=float(r_old[-1]),
        r_old_before_switch=float(r_old[t_w <= T_BEFORE][-1]),
    )
    (OUT / "summary_shift.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
