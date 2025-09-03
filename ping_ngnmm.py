import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rc


# --- Plotting defaults ---
plt.rcParams.update({'font.size': 14, 'figure.autolayout': True})


# --- Model equations ---
def NG(r, v, tau_m, delta, eta, input_current):
    Delta = delta / (np.pi * tau_m)
    dr = (Delta + 2 * r * v) / tau_m
    dv = (eta - (np.pi * r * tau_m) ** 2 + v ** 2 + input_current) / tau_m
    return np.array([dr, dv])


def SOS(s, z, r, tau_s):
    ds = z / tau_s
    dz = (r - 2 * z - s) / tau_s
    return np.array([ds, dz])


def PING(t, u, taus, deltas, eta, J):
    #State vector u = [rE, vE, sE, zE, rI, vI, sI, zI].
    rE, vE, sE, zE, rI, vI, sI, zI = u

    # External inputs (can be made time-dependent)
    Ie, Ii = 0.0, 0.0

    # Synaptic inputs
    inptE = taus[0] * (J[0] * sE + J[1] * sI) + Ie
    inptI = taus[2] * (J[2] * sE + J[3] * sI) + Ii

    # Excitatory population
    rrE = NG(rE, vE, taus[0], deltas[0], eta[0], inptE)
    ssE = SOS(sE, zE, rE, taus[1])

    # Inhibitory population
    rrI = NG(rI, vI, taus[2], deltas[1], eta[1], inptI)
    ssI = SOS(sI, zI, rI, taus[3])

    return np.concatenate([rrE, ssE, rrI, ssI])


# --- Parameters ---
delta_e, delta_i = 1.0, 1.0
tau_e, tau_i = 15.0, 7.5
tau_a, tau_g = 10.0, 2.0

taus = [tau_e, tau_a, tau_i, tau_g]
deltas = [delta_e, delta_i]

eta = [10.0,20.0]
J = np.array([10.0, 0.0, 0.0, -20.0])  # connectivity

# Initial conditions (exc, inh)
u0 = [
    6.692e-03, -2.379, 6.697e-03, 2.227e-07,  # excitatory
    1.360e-02, -1.173, 1.362e-02, 1.228e-06   # inhibitory
]
#rE, vE, sE, zE
# --- Transient run ---
t_span = (0.0, 1000.0)
sol = solve_ivp(lambda t, y: PING(t, y, taus, deltas, eta, J),
                t_span, u0, method='RK23', max_step=0.1)
u0 = sol.y[:, -1]

# --- Main run ---
t_span = (0.0, 1500.0)
sol = solve_ivp(lambda t, y: PING(t, y, taus, deltas, eta, J),
                t_span, u0, method='RK23', max_step=0.1)

# --- Plot last 100 ms ---
fig, ax = plt.subplots(figsize=(6, 4))
labels = ['Excitatory', 'Inhibitory']

# Select indices where time is within the last 100 ms
mask = (sol.t >= sol.t - 100) & (sol.t <= sol.t[-1])

for i, label in zip([0, 4], labels):
    ax.plot(sol.t[mask], sol.y[i, mask], lw=2, alpha=0.9, label=label)

ax.set_ylabel(r"$r$")
ax.set_xlabel(r"$t$")
ax.set_ylim(0, np.max(sol.y[[0, 4], :]) + 0.2)
ax.set_xlim(1400,1500)
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis="y", direction='in', length=5)

ax.legend(bbox_to_anchor=(0.75, 1.105), loc="upper left", frameon=False)
plt.tight_layout(pad=0.0)
plt.show()

