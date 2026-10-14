import sys
import json
from pathlib import Path

import numpy as np
from brian2 import StateMonitor, ms, second, Hz, nA

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import build, make_signals, channel_means, binned_rate, excitatory_tuning, N_CHANNELS, PREFERRED_CHANNEL
from plotting import plt, save, EXC, INH, NET, MUTED

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

T_PRE = 5.0
T_LEARN = 120.0
T_POST = 5.0
ETA = 0.02
RHO0 = 5.0
SIGNAL_DT = 1 * ms


def lowpass(x, dt_s, tau_s=0.05):
    a = dt_s / tau_s
    y = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc += a * (v - acc)
        y[i] = acc
    return y


def main():
    rng = np.random.default_rng(0)
    total = T_PRE + T_LEARN + T_POST
    rates = make_signals((total + 1) * second, SIGNAL_DT, rng)
    m = build(rates, SIGNAL_DT, eta=0.0, rho0=RHO0 * Hz)

    trace = StateMonitor(m["neuron"], ["I_e", "I_i"], record=0, dt=1 * ms)
    weights = StateMonitor(m["inh_syn"], "w", record=True, dt=2 * second)
    m["net"].add(trace, weights)

    w_initial = np.array(m["inh_syn"].w)
    m["net"].run(T_PRE * second)
    m["inh_syn"].eta = ETA
    m["net"].run(T_LEARN * second)
    m["inh_syn"].eta = 0.0
    m["net"].run(T_POST * second)

    t = np.array(trace.t / second)
    i_e = np.array(trace.I_e[0] / nA)
    i_i = np.array(trace.I_i[0] / nA)
    spikes = np.array(m["spikes"].t / second)

    pre_mask = t < T_PRE
    post_mask = t >= T_PRE + T_LEARN
    i_e_f, i_i_f = lowpass(i_e, 1e-3), lowpass(i_i, 1e-3)
    corr_pre = np.corrcoef(i_e_f[pre_mask][200:], -i_i_f[pre_mask][200:])[0, 1]
    corr_post = np.corrcoef(i_e_f[post_mask][200:], -i_i_f[post_mask][200:])[0, 1]
    rate_pre = np.sum(spikes < T_PRE) / T_PRE
    rate_post = np.sum(spikes >= T_PRE + T_LEARN) / T_POST

    w_final = np.array(m["inh_syn"].w)
    inh_ch = m["inh_channel"]
    exc_tuning = channel_means(m["w_exc"], m["exc_channel"])
    inh_init = channel_means(w_initial, inh_ch)
    inh_final = channel_means(w_final, inh_ch)
    tuning_corr = np.corrcoef(exc_tuning, inh_final)[0, 1]

    tc, r = binned_rate(spikes, total, 1.0)
    r_smooth = np.convolve(r, np.ones(5) / 5, mode="same")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axvspan(0, T_PRE, color="#efeeea", lw=0)
    ax.axvspan(T_PRE + T_LEARN, total, color="#efeeea", lw=0)
    ax.plot(tc, r, color=EXC, lw=0.8, alpha=0.35)
    ax.plot(tc, r_smooth, color=EXC, label="output rate (5 s average)")
    ax.axhline(RHO0, color=NET, lw=1.2, ls="--", label=f"target rate ρ₀ = {RHO0:g} Hz")
    from matplotlib.patches import Patch
    ax.set_xlabel("time (s)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("1 · Inhibitory plasticity pulls the output rate down towards the target")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color="#efeeea"))
    labels.append("plasticity off (test windows)")
    ax.legend(handles, labels, loc="upper right")
    save(fig, OUT / "fig1_rate.png")

    k = np.arange(1, N_CHANNELS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    axes[0].bar(k, exc_tuning, color=EXC, width=0.62)
    axes[0].set_title("2a · Excitatory weights (fixed)")
    axes[0].set_xlabel("input channel")
    axes[0].set_ylabel("mean weight (× g̅ₑ)")
    axes[0].set_xticks(k)
    axes[1].plot(k, inh_init, color=MUTED, marker="o", ms=5, lw=1.5, label="before learning")
    axes[1].plot(k, inh_final, color=INH, marker="o", ms=6, label="after learning")
    axes[1].set_title("2b · Inhibitory weights (learned)")
    axes[1].set_xlabel("input channel")
    axes[1].set_ylabel("mean weight (× g̅ᵢ)")
    axes[1].set_xticks(k)
    axes[1].legend(loc="upper left")
    axes[1].text(0.98, 0.95, f"r(E, I) = {tuning_corr:.2f}", transform=axes[1].transAxes, ha="right", va="top", fontsize=9, color=NET)
    fig.tight_layout()
    save(fig, OUT / "fig2_tuning.png")

    win = 1.0
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, sharey=True)
    for ax, start, label, c, rr in [
        (axes[0], 1.0, "before learning", corr_pre, rate_pre),
        (axes[1], T_PRE + T_LEARN + 1.0, "after learning", corr_post, rate_post),
    ]:
        sel = (t >= start) & (t < start + win)
        tt = (t[sel] - start) * 1000
        ax.plot(tt, i_e_f[sel], color=EXC, lw=1.6, label="excitatory current")
        ax.plot(tt, -i_i_f[sel], color=INH, lw=1.6, label="inhibitory current (sign flipped)")
        ax.set_title(f"3 · {label}: corr = {c:.2f}, rate = {rr:.1f} Hz", fontsize=10)
        ax.set_ylabel("current (nA)")
    axes[1].set_xlabel("time (ms)")
    axes[0].legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, OUT / "fig3_currents.png")

    summary = dict(
        eta=ETA, rho0=RHO0, t_learn_s=T_LEARN,
        rate_before_hz=round(float(rate_pre), 2),
        rate_after_hz=round(float(rate_post), 2),
        ei_current_corr_before=round(float(corr_pre), 3),
        ei_current_corr_after=round(float(corr_post), 3),
        tuning_corr_after=round(float(tuning_corr), 3),
        exc_tuning=np.round(exc_tuning, 3).tolist(),
        inh_initial=np.round(inh_init, 3).tolist(),
        inh_final=np.round(inh_final, 3).tolist(),
    )
    (OUT / "summary_main.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
