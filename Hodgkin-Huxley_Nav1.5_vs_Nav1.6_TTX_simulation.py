import numpy as np
import matplotlib.pyplot as plt


# *Neurons in Culture A express the TTX-sensitive Nav1.6 as the only Nav channel.
# An analysis of steady-state inactivation reveals that 50% of Nav1.6 channels are inactive at -70mV.
# *Neurons in Culture B express Nav1.5 as the only Nav channel. This channel is
# TTX-resistant with 50% inactivation at -40mV.
# Simulate:
# 1) the whole-cell Nav current generated in response to a single, 10ms, voltage pulse of
# -20mV from a resting voltage of -90mV for these two types of neurons.
# 2) a single action potential following membrane depolarization from -70mV to -45mV for 2ms. (Artificially bringing membrane potential to V-thresh to induce AP).

# ----------------------------------------------------------------------
# Global parameters (standard Hodgkin-Huxley style)
# ----------------------------------------------------------------------
dt = 0.01          # ms
Cm = 1.0           # uF/cm²
gNa = 120.0        # mS/cm²
gK  = 36.0
gL  = 0.3
ENa = 50.0         # mV
EK  = -77.0
EL  = -67.0        # adjusted so resting potential = -70 mV

# Activation (m) and delayed rectifier (n) - identical for both cultures
def alpha_m(V):   return 0.1 * (V + 40) / (1 - np.exp(-(V + 40)/10))
def beta_m(V):    return 4.0 * np.exp(-(V + 65)/18)
def alpha_n(V):   return 0.01 * (V + 55) / (1 - np.exp(-(V + 55)/10))
def beta_n(V):    return 0.125 * np.exp(-(V + 65)/80)

def m_inf(V):  return alpha_m(V) / (alpha_m(V) + beta_m(V))
def tau_m(V):  return 1.0 / (alpha_m(V) + beta_m(V))
def n_inf(V):  return alpha_n(V) / (alpha_n(V) + beta_n(V))
def tau_n(V):  return 1.0 / (alpha_n(V) + beta_n(V))

# ----------------------------------------------------------------------
# Inactivation (h) - DIFFERENT for each culture via voltage shift
# Standard HH V½_inact ≈ -62.3 mV
# ----------------------------------------------------------------------
Vhalf_old = -62.3
Vhalf_A = -70.0   # Nav1.6 - 50% inactivated at -70 mV
Vhalf_B = -40.0   # Nav1.5 - 50% inactivated at -40 mV

delta_A = Vhalf_old - Vhalf_A   # +7.7 mV
delta_B = Vhalf_old - Vhalf_B   # -22.3 mV

def get_h_gates(delta):
    def alpha_h(V):
        return 0.07 * np.exp(-(V + delta + 65) / 20)
    def beta_h(V):
        return 1.0 / (1 + np.exp(-(V + delta + 35) / 10))
    def h_inf(V):
        a = alpha_h(V)
        b = beta_h(V)
        return a / (a + b) if (a + b) > 1e-9 else 0.0
    def tau_h(V):
        a = alpha_h(V)
        b = beta_h(V)
        return 1.0 / (a + b) if (a + b) > 1e-9 else 1000.0
    return alpha_h, beta_h, h_inf, tau_h

# Create gating functions for each culture
_, _, h_inf_A, tau_h_A = get_h_gates(delta_A)
_, _, h_inf_B, tau_h_B = get_h_gates(delta_B)

# ----------------------------------------------------------------------
# 1) Voltage-clamp simulation - Nav current only
# ----------------------------------------------------------------------
def simulate_vc(h_inf_func, tau_h_func):
    t_total = 25.0
    t = np.arange(0, t_total, dt)
    
    V_hold = -90.0
    V_step = -20.0
    step_start = 5.0
    step_dur = 10.0
    
    V = np.full(len(t), V_hold)
    mask = (t >= step_start) & (t < step_start + step_dur)
    V[mask] = V_step
    
    m = np.full(len(t), m_inf(V_hold))
    h = np.full(len(t), h_inf_func(V_hold))
    INa = np.zeros(len(t))
    
    for i in range(1, len(t)):
        Vm = V[i-1]
        m[i] = m[i-1] + (m_inf(Vm) - m[i-1]) * (1 - np.exp(-dt / tau_m(Vm)))
        h[i] = h[i-1] + (h_inf_func(Vm) - h[i-1]) * (1 - np.exp(-dt / tau_h_func(Vm)))
        INa[i] = gNa * m[i]**3 * h[i] * (V[i] - ENa)   # negative = inward
    
    return t, INa

# Run VC for both cultures
t_vc, INa_A = simulate_vc(h_inf_A, tau_h_A)
t_vc, INa_B = simulate_vc(h_inf_B, tau_h_B)

# ----------------------------------------------------------------------
# 2) Hybrid current-clamp: artificial 2 ms depolarization to -45 mV
#    (force V = -45 mV for exactly 2 ms, then release to free evolution)
# ----------------------------------------------------------------------
def simulate_hybrid_AP(h_inf_func, tau_h_func):
    t_total = 50.0
    t = np.arange(0, t_total, dt)
    
    V_rest = -70.0
    V_forced = -45.0
    forced_start = 5.0
    forced_dur = 2.0
    forced_end = forced_start + forced_dur
    
    V = np.full(len(t), V_rest)
    m = np.full(len(t), m_inf(V_rest))
    h = np.full(len(t), h_inf_func(V_rest))
    n = np.full(len(t), n_inf(V_rest))
    
    for i in range(1, len(t)):
        is_forced = (t[i-1] >= forced_start) and (t[i-1] < forced_end)
        
        if is_forced:
            # Artificially clamp voltage
            V[i] = V_forced
            Vm = V_forced
        else:
            # Free current-clamp (I_stim = 0)
            Vm = V[i-1]
            INa = gNa * m[i-1]**3 * h[i-1] * (Vm - ENa)
            IK  = gK  * n[i-1]**4           * (Vm - EK)
            IL  = gL                        * (Vm - EL)
            dV = (-INa - IK - IL) / Cm
            V[i] = Vm + dV * dt
            Vm = V[i-1]   # use previous Vm for gate update (standard Euler)
        
        # Update gates (always use appropriate Vm)
        m[i] = m[i-1] + (m_inf(Vm) - m[i-1]) * (1 - np.exp(-dt / tau_m(Vm)))
        h[i] = h[i-1] + (h_inf_func(Vm) - h[i-1]) * (1 - np.exp(-dt / tau_h_func(Vm)))
        n[i] = n[i-1] + (n_inf(Vm) - n[i-1]) * (1 - np.exp(-dt / tau_n(Vm)))
    
    return t, V

# Run AP for both cultures
t_ap, V_A = simulate_hybrid_AP(h_inf_A, tau_h_A)
t_ap, V_B = simulate_hybrid_AP(h_inf_B, tau_h_B)

# ----------------------------------------------------------------------
# Plot 1: Whole-cell Nav currents (voltage clamp)
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(t_vc, INa_A, 'b-', linewidth=2, label='Culture A – Nav1.6 (TTX-sensitive, 50% inactivated at -70 mV)')
plt.plot(t_vc, INa_B, 'r-', linewidth=2, label='Culture B – Nav1.5 (TTX-resistant, 50% inactivated at -40 mV)')
plt.axvspan(5, 15, alpha=0.15, color='gray', label='10 ms voltage step to -20 mV')
plt.xlabel('Time (ms)')
plt.ylabel('I_Na (µA/cm²)  (negative = inward)')
plt.title('Whole-cell Nav Current (hold -90 mV → step to -20 mV for 10 ms)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Plot 2: Action potentials (hybrid clamp)
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(t_ap, V_A, 'b-', linewidth=2, label='Culture A – Nav1.6')
plt.plot(t_ap, V_B, 'r-', linewidth=2, label='Culture B – Nav1.5')
plt.axvspan(5, 7, alpha=0.2, color='gray', label='Artificial 2 ms depol to -45 mV')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Single Action Potential (rest -70 mV → artificial 2 ms to -45 mV → free evolution)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()