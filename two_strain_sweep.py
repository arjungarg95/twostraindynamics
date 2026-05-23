import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
import os

# ODE system with flags for scenarios
def SIR2str(pops, dispars, flowpars, simpars, same_reinfect=0, stairway=0):
    s0, i10, i20, r10, r20, w0 = pops
    beta1, beta2, eta1, eta2, delta, eta_same = dispars  # Added eta_same for reinfection
    gamma1, gamma2 = flowpars
    t0, tf = simpars
    pop = sum(pops)

    def eqs(t, y):
        s, i1, i2, r1, r2, w = y
        ds = -beta1 * s * i1 / pop - beta2 * s * i2 / pop + delta * (r1 + r2)
        di1 = beta1 * s * i1 / pop + eta1 * beta1 * r2 * i1 / pop - gamma1 * i1
        if same_reinfect:
            di1 += eta_same * beta1 * r1 * i1 / pop  # Reinfection with same strain 1
        di2 = beta2 * s * i2 / pop + eta2 * beta2 * r1 * i2 / pop - gamma2 * i2
        if same_reinfect:
            di2 += eta_same * beta2 * r2 * i2 / pop  # Reinfection with same strain 2
        if stairway:
            di2 = eta2 * beta2 * r1 * i2 / pop - gamma2 * i2  # Strain 2 only from R1
        dr1 = gamma1 * i1 - eta2 * beta2 * r1 * i2 / pop - delta * r1
        if same_reinfect:
            dr1 -= eta_same * beta1 * r1 * i1 / pop  # Outflow for reinfection
        dr2 = gamma2 * i2 - eta1 * beta1 * r2 * i1 / pop - delta * r2
        if same_reinfect:
            dr2 -= eta_same * beta2 * r2 * i2 / pop  # Outflow for reinfection
        dw = delta * (r1 + r2)
        return [ds, di1, di2, dr1, dr2, dw]

    sol = solve_ivp(eqs, [t0, tf], pops, method='Radau', max_step=1, rtol=1e-9, atol=1e-9)
    return sol if sol.success else None

# Global parameters
N = 1000000
inpops = [N - 1, 1, 0, 0, 0, 0]  # S, I1, I2, R1, R2, W

beta_base = 0.26666666666666666
beta1_grid = [beta_base]  # Fixed; expand if needed
beta2_mult_grid = [1.0, 1.2, 1.4, 1.6, 1.8]  # beta2 = beta1 * mult
eta1_grid = np.arange(0.0, 1.05, 0.2)  # Small for test; expand to 0.1 step for full
eta2_grid = np.arange(0.0, 1.05, 0.2)
eta_same_grid = [0.0, 0.5, 1.0]  # For reinfection scenario
tem_grid = [50, 100, 150, 200]
gamma1 = gamma2 = 1 / 9.0
delta = 1 / 180.0
ft = 1825

max_t_transient = 365
start_check = 30
eps = 0.05
win = 20

# Core analysis function
def analyse(tem, beta1, beta2_mult, eta1, eta2, eta_same, scenario='base'):
    beta2 = beta1 * beta2_mult
    dispars = [beta1, beta2, eta1, eta2, delta, eta_same]
    same_reinfect = 1 if scenario == 'reinfection' else 0
    stairway = 1 if scenario == 'stairway' else 0

    sol1 = SIR2str(inpops, dispars, [gamma1, gamma2], [0, tem], same_reinfect, stairway)
    if sol1 is None:
        return None

    y0 = [sol1.y[k][-1] for k in range(6)]
    y0[2] = 1.0
    if not np.all(np.isfinite(y0)):
        return None

    sol2 = SIR2str(y0, dispars, [gamma1, gamma2], [tem, ft], same_reinfect, stairway)
    if sol2 is None:
        return None

    t = np.concatenate([sol1.t, sol2.t[1:]])
    i1 = np.concatenate([sol1.y[1], sol2.y[1][1:]])
    i2 = np.concatenate([sol1.y[2], sol2.y[2][1:]])

    daily_t = np.arange(0, max_t_transient + 1)
    i1d = np.interp(daily_t, t, i1)
    i2d = np.interp(daily_t, t, i2)

    p1 = find_peaks(i1d, height=100)[0]
    p2 = find_peaks(i2d, height=100)[0]
    num_peaks1 = len(p1)
    num_peaks2 = len(p2)
    peak_sizes1 = i1d[p1].tolist() if num_peaks1 > 0 else []
    peak_times1 = daily_t[p1].tolist() if num_peaks1 > 0 else []
    peak_sizes2 = i2d[p2].tolist() if num_peaks2 > 0 else []
    peak_times2 = daily_t[p2].tolist() if num_peaks2 > 0 else []

    avg_time_between1 = np.mean(np.diff(peak_times1)) if num_peaks1 > 1 else 0
    avg_time_between2 = np.mean(np.diff(peak_times2)) if num_peaks2 > 1 else 0

    fixation = max_t_transient
    idx_start = np.searchsorted(daily_t, tem + start_check)
    for k in range(idx_start, len(daily_t) - win):
        ch1 = np.max(np.abs(i1d[k+1:k+win+1] - i1d[k:k+win]))
        ch2 = np.max(np.abs(i2d[k+1:k+win+1] - i2d[k:k+win]))
        if ch1 < eps and ch2 < eps:
            fixation = daily_t[k]
            break

    if fixation < max_t_transient:
        stable_start = np.where(daily_t >= fixation)[0][0]
        i1f = np.mean(i1d[stable_start:stable_start+win])
        i2f = np.mean(i2d[stable_start:stable_start+win])
    else:
        i1f = i1d[-1]
        i2f = i2d[-1]

    return {
        'scenario': scenario,
        'beta1': beta1,
        'beta2_mult': beta2_mult,
        'eta1': eta1,
        'eta2': eta2,
        'eta_same': eta_same,
        'tem': tem,
        'num_peaks1': num_peaks1,
        'peak_sizes1': peak_sizes1,
        'peak_times1': peak_times1,
        'avg_time_between1': avg_time_between1,
        'num_peaks2': num_peaks2,
        'peak_sizes2': peak_sizes2,
        'peak_times2': peak_times2,
        'avg_time_between2': avg_time_between2,
        'fixation_time': fixation,
        'i1f': i1f,
        'i2f': i2f
    }

# Grid sweep for all scenarios
scenarios = ['base', 'reinfection', 'stairway']

results = []
for scenario in scenarios:
    for beta1 in beta1_grid:
        for beta2_mult in beta2_mult_grid:
            for eta1 in eta1_grid:
                for eta2 in eta2_grid:
                    eta_same = 0.0  # Default; vary only for reinfection
                    if scenario == 'reinfection':
                        for eta_same in eta_same_grid:
                            result = analyse(tem, beta1, beta2_mult, eta1, eta2, eta_same, scenario)
                            if result:
                                results.append(result)
                    else:
                        for tem in tem_grid:
                            result = analyse(tem, beta1, beta2_mult, eta1, eta2, eta_same, scenario)
                            if result:
                                results.append(result)

# Save to CSV (already good)
df = pd.DataFrame(results)
df.to_csv('grid_sweep_all_scenarios.csv', index=False)
print("Saved grid_sweep_all_scenarios.csv with", len(results), "simulations.")

# Example plot: Fixation time vs eta1 x eta2 for base scenario, tem=50
df_base = df[(df['scenario'] == 'base') & (df['tem'] == 50)]

# Drop duplicates (safety) and pivot
df_pivot = df_base.drop_duplicates(subset=['eta1', 'eta2'])
pivot = df_pivot.pivot(index='eta1', columns='eta2', values='fixation_time')

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(pivot, origin='lower', cmap='viridis',
               extent=[pivot.columns.min(), pivot.columns.max(),
                       pivot.index.min(), pivot.index.max()])
ax.set_title('Fixation Time - Base Scenario, tem=50')
ax.set_xlabel('η₂')
ax.set_ylabel('η₁')
plt.colorbar(im, ax=ax, label='Days to Fixation')
plt.tight_layout()
plt.savefig('fixation_time_base_tem50.png')
plt.close()

print("Example plot saved: fixation_time_base_tem50.png")
print("Tip: For full analysis, load 'grid_sweep_all_scenarios.csv' in Jupyter and explore with pivot_table() or seaborn heatmaps.")