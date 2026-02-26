import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
dt = 0.01          # ms
t_total = 100      # ms
t = np.arange(0, t_total, dt)
V_rest = -65.0     # mV

# Membrane properties
Cm = 1.0           # uF/cm²
gNa = 120.0        # mS/cm²
gK  = 36.0
gL  = 0.3
ENa = 50.0         # mV
EK  = -77.0
EL  = -54.4

# Stimulation
stim_amp = 100.0   # uA/cm²
stim_dur = 50.0     # ms

# Functions
def alpha_m(V):   return 0.1 * (V + 40) / (1 - np.exp(-(V + 40)/10))
def beta_m(V):    return 4.0 * np.exp(-(V + 65)/18)
def alpha_h(V):   return 0.07 * np.exp(-(V + 65)/20)
def beta_h(V):    return 1.0 / (1 + np.exp(-(V + 35)/10))
def alpha_n(V):   return 0.01 * (V + 55) / (1 - np.exp(-(V + 55)/10))
def beta_n(V):    return 0.125 * np.exp(-(V + 65)/80)

# Inactivation time constant multiplier (main difference between scenarios)
tau_h_multiplier = 1.0    # 1.0 = fast inactivation, 10.0 = slow inactivation

# Steady-state values
def m_inf(V):  return alpha_m(V) / (alpha_m(V) + beta_m(V))
def h_inf(V):  return alpha_h(V) / (alpha_h(V) + beta_h(V))
def n_inf(V):  return alpha_n(V) / (alpha_n(V) + beta_n(V))

# Time constants (we slow down h only)
def tau_m(V):  return 1.0 / (alpha_m(V) + beta_m(V))
def tau_h(V):  return tau_h_multiplier / (alpha_h(V) + beta_h(V))   # ← key change
def tau_n(V):  return 1.0 / (alpha_n(V) + beta_n(V))

# ----------------------------------------------------------------------
# Simulation function
# ----------------------------------------------------------------------
def simulate_AP(tau_h_mult=1.0, stim_amp=stim_amp, stim_start=10.0, stim_dur=stim_dur):
    global tau_h_multiplier
    tau_h_multiplier = tau_h_mult

    V = np.full(len(t), V_rest)
    m = np.full(len(t), m_inf(V_rest))
    h = np.full(len(t), h_inf(V_rest))
    n = np.full(len(t), n_inf(V_rest))

    I_stim = np.zeros(len(t))
    stim_idx = (t >= stim_start) & (t < stim_start + stim_dur)
    I_stim[stim_idx] = stim_amp

    spikes = []

    for i in range(1, len(t)):
        # Currents
        INa = gNa * m[i-1]**3 * h[i-1] * (V[i-1] - ENa)
        IK  = gK  * n[i-1]**4           * (V[i-1] - EK)
        IL  = gL                        * (V[i-1] - EL)

        dV = (-INa - IK - IL + I_stim[i-1]) / Cm

        V[i] = V[i-1] + dV * dt

        # Gating variables (exponential Euler)
        m[i] = m[i-1] + (m_inf(V[i-1]) - m[i-1]) * (1 - np.exp(-dt / tau_m(V[i-1])))
        h[i] = h[i-1] + (h_inf(V[i-1]) - h[i-1]) * (1 - np.exp(-dt / tau_h(V[i-1])))
        n[i] = n[i-1] + (n_inf(V[i-1]) - n[i-1]) * (1 - np.exp(-dt / tau_n(V[i-1])))

        # Simple spike detection
        if i > 1 and V[i-1] > 0 and V[i] < V[i-1] and V[i-1] > 20:
            spikes.append(t[i])

    return V, spikes, t

# ----------------------------------------------------------------------
# Run both scenarios
# ----------------------------------------------------------------------
print("Simulating fast inactivation (normal τ_h)...")
V_fast, spikes_fast, t_fast = simulate_AP(tau_h_mult=1.0, stim_amp=stim_amp)

print("Simulating slow inactivation (10× τ_h)...")
V_slow, spikes_slow, t_slow = simulate_AP(tau_h_mult=10.0, stim_amp=stim_amp)

# ----------------------------------------------------------------------
# Plot the action potentials
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_fast, V_fast, label='Fast Inactivation', color='blue')
plt.plot(t_slow, V_slow, label='Slow Inactivation', color='red')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Action Potentials: Fast vs. Slow Na⁺ Inactivation')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------
# Basic analysis (optional, as before)
# ----------------------------------------------------------------------
def analyze_AP(V, t):
    above_thresh = V > -20
    if not np.any(above_thresh):
        return {"height_mV": np.nan, "width_FWHM_ms": np.nan, "V_threshold_mV": np.nan}

    peak_idx = np.argmax(V)
    peak_V = V[peak_idx]
    height = peak_V - V_rest

    # FWHM
    half_height = V_rest + height/2
    above_half = V > half_height
    crossings = np.where(np.diff(above_half.astype(int)))[0]
    if len(crossings) >= 2:
        width_idx = crossings[1] - crossings[0]
        width_ms = t[width_idx] - t[crossings[0]]
    else:
        width_ms = np.nan

    # Rough threshold estimate (where dV/dt max before peak)
    dV = np.diff(V) / dt
    dv_max_idx = np.argmax(dV[:peak_idx]) if peak_idx > 10 else 10
    V_thresh = V[dv_max_idx]

    return {
        "height_mV": height,
        "width_FWHM_ms": width_ms,
        "V_threshold_mV": V_thresh
    }


stats_fast = analyze_AP(V_fast, t_fast)
stats_slow = analyze_AP(V_slow, t_slow)
print(stats_fast)
print("\nFast inactivation:")
print(f"  V threshold ≈ {stats_fast['V_threshold_mV']:.1f} mV")
print(f"  AP height   ≈ {stats_fast['height_mV']:.0f} mV")
print(f"  AP width    ≈ {stats_fast['width_FWHM_ms']:.2f} ms")

print("\nSlow inactivation:")
print(f"  V threshold ≈ {stats_slow['V_threshold_mV']:.1f} mV")
print(f"  AP height   ≈ {stats_slow['height_mV']:.0f} mV")
print(f"  AP width    ≈ {stats_slow['width_FWHM_ms']:.2f} ms")
