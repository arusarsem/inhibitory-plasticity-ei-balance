import numpy as np
from brian2 import (
    NeuronGroup, PoissonGroup, Synapses, SpikeMonitor, StateMonitor,
    Network, TimedArray, defaultclock, prefs, seed,
    ms, second, mV, pF, nS, psiemens, Hz,
)

prefs.codegen.target = "cython"

N_CHANNELS = 8
N_EXC_PER_CHANNEL = 100
N_INH_PER_CHANNEL = 25
PREFERRED_CHANNEL = 4

PARAMS = dict(
    C=200 * pF,
    g_leak=10 * nS,
    E_leak=-60 * mV,
    V_th=-50 * mV,
    V_reset=-60 * mV,
    t_ref=5 * ms,
    E_exc=0 * mV,
    E_inh=-80 * mV,
    tau_exc=5 * ms,
    tau_inh=10 * ms,
    g_exc_bar=140 * psiemens,
    g_inh_bar=350 * psiemens,
    tau_stdp=20 * ms,
    w_inh_init=0.1,
    w_inh_max=100.0,
    signal_tau=50 * ms,
    signal_base=5.0,
    signal_gain=10.0,
)


def excitatory_tuning(n_channels=N_CHANNELS, preferred=PREFERRED_CHANNEL):
    k = np.arange(n_channels)
    return 0.3 + 1.1 / (1.0 + (k - preferred) ** 4)


def make_signals(duration, dt, rng, n_channels=N_CHANNELS, p=PARAMS):
    n_steps = int(round(duration / dt))
    tau = float(p["signal_tau"] / dt)
    decay = np.exp(-1.0 / tau)
    noise_scale = np.sqrt(1.0 - decay ** 2)
    x = np.zeros((n_steps, n_channels))
    state = rng.standard_normal(n_channels)
    for i in range(n_steps):
        state = decay * state + noise_scale * rng.standard_normal(n_channels)
        x[i] = state
    rates = np.clip(p["signal_base"] + p["signal_gain"] * x, 0.0, None)
    return rates


def build(rates, signal_dt, eta, rho0, rng_seed=1, p=PARAMS, inh_rates=None, preferred=PREFERRED_CHANNEL):
    seed(rng_seed)
    rng = np.random.default_rng(rng_seed)
    rate_array = TimedArray(rates * Hz, dt=signal_dt)
    inh_rate_array = rate_array if inh_rates is None else TimedArray(inh_rates * Hz, dt=signal_dt)

    n_exc = N_CHANNELS * N_EXC_PER_CHANNEL
    n_inh = N_CHANNELS * N_INH_PER_CHANNEL

    neuron = NeuronGroup(
        1,
        """
        dv/dt = (g_leak*(E_leak - v) + g_e*(E_exc - v) + g_i*(E_inh - v)) / C : volt (unless refractory)
        dg_e/dt = -g_e / tau_exc : siemens
        dg_i/dt = -g_i / tau_inh : siemens
        I_e = g_e*(E_exc - v) : amp
        I_i = g_i*(E_inh - v) : amp
        """,
        threshold="v > V_th",
        reset="v = V_reset",
        refractory=p["t_ref"],
        method="euler",
        namespace=p,
    )
    neuron.v = p["E_leak"]

    exc_channel = np.repeat(np.arange(N_CHANNELS), N_EXC_PER_CHANNEL)
    inh_channel = np.repeat(np.arange(N_CHANNELS), N_INH_PER_CHANNEL)

    exc_inputs = NeuronGroup(
        n_exc, "channel : integer (constant)",
        threshold="rand() < rate_array(t, channel) * dt",
        namespace={"rate_array": rate_array},
    )
    exc_inputs.channel = exc_channel
    inh_inputs = NeuronGroup(
        n_inh, "channel : integer (constant)",
        threshold="rand() < rate_array(t, channel) * dt",
        namespace={"rate_array": inh_rate_array},
    )
    inh_inputs.channel = inh_channel

    tuning = excitatory_tuning(preferred=preferred)
    w_exc = tuning[exc_channel] * (1.0 + 0.1 * rng.standard_normal(n_exc))
    w_exc = np.clip(w_exc, 0.0, None)

    exc_syn = Synapses(
        exc_inputs, neuron, "w : 1",
        on_pre="g_e_post += w * g_exc_bar",
        namespace=p,
    )
    exc_syn.connect()
    exc_syn.w = w_exc

    alpha = 2.0 * float(rho0 / Hz) * float(p["tau_stdp"] / second)
    inh_syn = Synapses(
        inh_inputs, neuron,
        """
        w : 1
        eta : 1 (shared)
        dApre/dt = -Apre / tau_stdp : 1 (event-driven)
        dApost/dt = -Apost / tau_stdp : 1 (event-driven)
        """,
        on_pre="""
        Apre += 1.
        w = clip(w + eta * (Apost - alpha), 0, w_inh_max)
        g_i_post += w * g_inh_bar
        """,
        on_post="""
        Apost += 1.
        w = clip(w + eta * Apre, 0, w_inh_max)
        """,
        namespace={**p, "alpha": alpha},
    )
    inh_syn.connect()
    inh_syn.w = p["w_inh_init"]
    inh_syn.eta = eta

    spikes = SpikeMonitor(neuron)
    net = Network(neuron, exc_inputs, inh_inputs, exc_syn, inh_syn, spikes)
    return dict(
        net=net, neuron=neuron, exc_syn=exc_syn, inh_syn=inh_syn,
        spikes=spikes, exc_channel=exc_channel, inh_channel=inh_channel,
        w_exc=w_exc,
    )


def channel_means(values, channels):
    return np.array([values[channels == k].mean() for k in range(N_CHANNELS)])


def binned_rate(spike_times_s, t_end_s, bin_s=1.0):
    edges = np.arange(0.0, t_end_s + bin_s, bin_s)
    counts, _ = np.histogram(spike_times_s, bins=edges)
    return edges[:-1] + bin_s / 2, counts / bin_s
