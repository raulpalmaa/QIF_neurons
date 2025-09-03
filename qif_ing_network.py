"""
QIF network with second-order (rise/decay) current-based synapses.
Check 'Biological Cybernetics: https://doi.org/10.1007/s00422-022-00952-7'
for reference.
"""
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class QIFParams:
    N: int = 200                  # number of neurons
    T: float = 100.0              # total sim time [ms]
    dt: float = 0.001             # time step [ms]
    v_peak: float = 100.0         # spike threshold surrogate
    v_reset: float = -100.0       # reset value after spike
    I_ext_mean: float = 0.0       # mean external drive
    I_ext_std: float = 0.0        # std of external drive
    J: float = -0.2               # synaptic coupling (negative = inhibitory)
    p_connect: float = 0.1        # connection probability
    w_scale: float = 1.0          # scale of random weights
    tau_m: float = 7.5            # membrane time constant [ms]
    tau_s: float = 2.0            # synaptic time constant [ms]
    eta: float = 20.0             # RNG seed for reproducibility
    refrac_ms: float = 0.0        # absolute refractory (ms); 0 to disable
    noise_std: float = 0.0        # optional Gaussian noise in v dynamics
    seed: int = 7                 # RNG seed for reproducibility


def build_connectivity(N, p_connect, w_scale, rng):
    #Erdős–Rényi inhibitory connectivity.
    #We'll be using a fully connected network, p = 1.0.
    W = (rng.random((N, N)) < p_connect).astype(np.float32)
    np.fill_diagonal(W, 0.0)
    if w_scale != 1.0:
        W *= w_scale
    # Normalize by in-degree to keep input scale reasonable (optional)
    indeg = W.sum(axis=1, keepdims=True)
    indeg[indeg == 0] = 1.0
    W = W / indeg    
    return W.astype(np.float32)


def simulate_qif_network(params: QIFParams):
    rng = np.random.default_rng(params.seed)
    N = params.N
    dt = params.dt
    steps = int(np.round(params.T / dt))
    times = np.arange(steps) * dt
    
    # Connectivity (all neurons treated as inhibitory via sign of J)
    W = build_connectivity(N, params.p_connect, params.w_scale, rng)

    #rE, vE, sE, zE
    # State variables -   1.360e-02, -1.173, 1.362e-02, 1.228e-06  
    v = np.ones(N, dtype=np.float32) * -1.173   # initial voltages
    s = np.ones(N, dtype=np.float32) * 1.362e-02# synaptic rise state
    z = np.ones(N, dtype=np.float32) * 1.228e-06# synaptic decay state
    r = np.ones(N, dtype=np.float32) * 1.360e-02

    # Optional refractory handling
    # We'll be using refrerefrac_ms = 0.0 - so no refractory time.
    refrac_steps = int(np.round(params.refrac_ms / dt))
    refrac_counter = np.zeros(N, dtype=np.int32)
    conductance = np.zeros(steps, dtype=np.float32) 
    # Spike recording
    spike_times = []
    spike_neurons = []
    # Load constants
    tau_m =  params.tau_m
    tau_s =  params.tau_s
    eta = params.eta
    J = params.J
    noise_std = float(params.noise_std)

    for k in range(steps):
        # We can model external inputs
        I_ext = 0.0

        if refrac_steps > 0:
            in_refrac = refrac_counter > 0
            v[in_refrac] = params.v_reset
            refrac_counter[in_refrac] -= 1

        # Synaptic current from previous s
        I_syn = J * (W @ s) * tau_m # shape (N,)

        # QIF membrane update
        dv = (v * v + eta + I_ext + I_syn) / tau_m
        v += dt * dv
        if noise_std > 0:
            v += dt * (noise_std * np.random.standard_cauchy(size=N) / tau_m)
            #v += noise_std * np.sqrt(dt) * rng.normal(0.0, noise_std, size=N)

        # Spike detection
        spiking = v >= params.v_peak
        if np.any(spiking):
            # Record spikes
            n_ids = np.nonzero(spiking)[0]
            spike_times.append(np.full(n_ids.shape, times[k], dtype=np.float32))
            spike_neurons.append(n_ids.astype(np.int32))

            # Reset voltages
            v[spiking] = params.v_reset

            # Refractory
            if refrac_steps > 0:
                refrac_counter[spiking] = refrac_steps
       
        r_pop = np.sum(spiking) / N / dt
        s += dt * z / tau_s
        z += dt * (r_pop - 2*z - s) / tau_s

        conductance[k] = np.sum(s)
    

    if spike_times:
        spike_times = np.concatenate(spike_times)
        spike_neurons = np.concatenate(spike_neurons)
    else:
        spike_times = np.empty(0, dtype=np.float32)
        spike_neurons = np.empty(0, dtype=np.int32)

    results = {
        "times": times,                # shape (steps,)
        "spike_times": spike_times,    # shape (n_spikes,)
        "spike_neurons": spike_neurons,# shape (n_spikes,)
        "v": v,                        # final voltages (for curiosity)
        "s": s,                        # final syn state (decay component)
        "z": z,                        # final syn rise component
        "W": W,                        # connectivity matrix
        "I_ext": I_ext,                # per-neuron external currents
        "params": params,
        "conductance": conductance,
    }
    return results


def quick_raster(results, max_neurons_to_show=100):
    """Plot a simple spike raster."""
    st = results["spike_times"]
    sn = results["spike_neurons"]
    if st.size == 0:
        print("No spikes recorded.")
        return
    # Optionally downselect for readability
    keep = sn < max_neurons_to_show
    st = st[keep]
    sn = sn[keep]

    plt.figure(figsize=(8, 4))
    plt.scatter(st, sn, s=2, color='black',marker='.', linewidths=0)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.title("QIF inhibitory network: spike raster")
    plt.tight_layout()
    plt.xlim(1400,1500)
    plt.show()


def compute_isi(results, bins=50, show=True):
    spike_times = results["spike_times"]
    spike_neurons = results["spike_neurons"]
    N = results["params"].N
    
    isi_dict = {}
    all_isis = []

    for n in range(N):
        st = spike_times[spike_neurons == n]
        if len(st) > 1:
            isi_vals = np.diff(st)
            isi_dict[n] = isi_vals
            all_isis.append(isi_vals)
        else:
            isi_dict[n] = np.array([])

    if all_isis:
        all_isis = np.concatenate(all_isis)
    else:
        all_isis = np.array([])
    # Plot ISI histogram
    if show and len(all_isis) > 0:
        plt.figure(figsize=(6,4))
        plt.hist(all_isis, bins=bins, edgecolor="k", alpha=0.7)
        plt.xlabel("ISI (ms)")
        plt.ylabel("Count")
        plt.title("Inter-Spike Interval Histogram (all neurons)")
        plt.tight_layout()
        plt.show()
    elif show:
        print("No spikes -> no ISI histogram")

    return isi_dict, all_isis



def plot_conduc(results):
    """Plot a simple spike raster."""
    st = results["times"]
    sc = results["conductance"]

    plt.figure(figsize=(8, 4))
    plt.plot(st, sc,'b',lw = 2)
    plt.xlabel("Time (ms)")
    plt.ylabel("Conductance")
    plt.title("QIF inhibitory network: conductance")
    plt.tight_layout()
    plt.show()


def population_rate(results):
    dt = results["params"].dt
    tau_r = 10 * dt
    spike_times = np.asarray(results["spike_times"])
    N = results["params"].N
    t_max = spike_times.max()
    times = np.arange(0, t_max, dt)
    # histogram of spikes into bins of dt (ms)
    counts, _ = np.histogram(spike_times, bins=np.append(times, t_max + dt))
    # rectangular window length in bins
    win_len = int(np.round(tau_r / dt))
    kernel = np.ones(win_len)
    # convolve counts -> total spikes in last tau_r ms
    counts_tau = np.convolve(counts, kernel, mode="same")
    # convert to rate (Hz)
    rates = counts_tau / (N * (tau_r / 1000.0))  # spikes / (neuron * second)
    return times, rates



if __name__ == "__main__":
    cc = []
    for kk in [-20]:
        params = QIFParams(
            N=1024,
            T=1500.0,
            dt=0.01,
            I_ext_mean=0.0,
            I_ext_std=0.0,
            J=kk,          # inhibitory coupling
            p_connect=1.0,
            w_scale=1.0,
            tau_m=7.5,
            tau_s=2.0,
            eta = 20,                 # RNG seed for reproducibility
            refrac_ms=0.0,
            noise_std=1.0,
            seed=42,
        )
        res = simulate_qif_network(params)
        print(res["spike_times"])
        print(res["spike_neurons"])
        quick_raster(res, max_neurons_to_show=1024)
        compute_isi(res)
        plot_conduc(res)

times, rates = population_rate(res)

plt.figure(figsize=(6,4))
plt.plot(times, rates, lw=2)
plt.xlabel("t (ms)")
plt.title("r(t)")
plt.show()
