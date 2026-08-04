import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime

# =============================================================================
# 1. System Setup and Directory Management
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"model_outputs_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

FIG_EXT = "pdf"

mpl.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'legend.fontsize': 10.5,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
})

print("Initializing Comprehensive Multi-Scenario Epidemic Pipeline...")
print(f"All outputs will be safely routed to: ./{output_dir}/\n")

# =============================================================================
# 2. Core ODE System (The Cyclic Reinfection "Inner Loop")
# =============================================================================
def two_strain_inner_loop(t, y, beta1, beta2, gamma1, gamma2, eta1, eta2, delta, N):
    S, I1, I2, R1, R2 = y
    lambda1 = beta1 * I1 / N
    lambda2 = beta2 * I2 / N
    dSdt  = -lambda1 * S - lambda2 * S + delta * R1 + delta * R2
    dI1dt = lambda1 * S + eta1 * lambda1 * R2 - gamma1 * I1
    dI2dt = lambda2 * S + eta2 * lambda2 * R1 - gamma2 * I2
    dR1dt = gamma1 * I1 - eta2 * lambda2 * R1 - delta * R1
    dR2dt = gamma2 * I2 - eta1 * lambda1 * R2 - delta * R2
    return [dSdt, dI1dt, dI2dt, dR1dt, dR2dt]

# =============================================================================
# 3. Two-Phase Integration Engine & Mass Conservation
# =============================================================================
def simulate_epidemic(params, t_em, t_max):
    N = params['N']
    y0_phase1 = [N - 10, 10, 0, 0, 0]
    ode_args = (params['beta1'], params['beta2'], params['gamma1'],
                params['gamma2'], params['eta1'], params['eta2'],
                params['delta'], N)

    sol1 = solve_ivp(two_strain_inner_loop, [0, t_em], y0_phase1, args=ode_args,
                     method='Radau', dense_output=True, max_step=1.0)

    y0_phase2 = sol1.y[:, -1].copy()
    seed_size = 10
    if y0_phase2[0] >= seed_size:
        y0_phase2[0] -= seed_size
    elif y0_phase2[3] >= seed_size:
        y0_phase2[3] -= seed_size
    else:
        y0_phase2[0] = max(0, y0_phase2[0] - seed_size)
    y0_phase2[2] += seed_size

    sol2 = solve_ivp(two_strain_inner_loop, [t_em, t_max], y0_phase2, args=ode_args,
                     method='Radau', dense_output=True, max_step=1.0)

    t_combined = np.concatenate((sol1.t, sol2.t[1:]))
    y_combined = np.concatenate((sol1.y, sol2.y[:, 1:]), axis=1)

    total_population = np.sum(y_combined, axis=0)
    if not np.allclose(total_population, N, atol=1e-5):
        print(f"CRITICAL WARNING: Population conservation violated at eta1={params['eta1']}, eta2={params['eta2']}")

    return t_combined, y_combined

# =============================================================================
# 4. High-Fidelity Metric Extraction
# =============================================================================
def extract_metrics(t, y, N, epsilon=1e-5, window=30):
    I1, I2 = y[1], y[2]

    peaks1, _ = find_peaks(I1, prominence=N * 0.0005, distance=14)
    peaks2, _ = find_peaks(I2, prominence=N * 0.0005, distance=14)

    avg_dist_1 = np.mean(np.diff(t[peaks1])) if len(peaks1) > 1 else 0
    avg_dist_2 = np.mean(np.diff(t[peaks2])) if len(peaks2) > 1 else 0

    p1_sizes = [float(round(val / N, 4)) for val in I1[peaks1]]
    p2_sizes = [float(round(val / N, 4)) for val in I2[peaks2]]

    dI1 = np.abs(np.diff(I1))
    dI2 = np.abs(np.diff(I2))
    threshold = epsilon * N
    violating = (dI1 >= threshold) | (dI2 >= threshold)

    violation_idx = np.nonzero(violating)[0]
    if len(violation_idx) == 0:
        fixation_time = t[0]
    else:
        last_violation = violation_idx[-1]
        candidate_i = last_violation + 1
        if candidate_i >= len(t):
            fixation_time = t[-1]
        else:
            remaining_days = t[-1] - t[candidate_i]
            if remaining_days < window:
                fixation_time = t[-1]
            else:
                fixation_time = t[candidate_i]

    endemic_I1_prop = float(np.mean(I1[-50:]) / N)
    endemic_I2_prop = float(np.mean(I2[-50:]) / N)
    endemic_total = endemic_I1_prop + endemic_I2_prop
    endemic_I2_share = float(endemic_I2_prop / endemic_total) if endemic_total > 0 else 0.5

    return {
        'P1_count': len(peaks1), 'P1_sizes': p1_sizes, 'P1_dist': avg_dist_1,
        'P2_count': len(peaks2), 'P2_sizes': p2_sizes, 'P2_dist': avg_dist_2,
        'Fixation_Time': float(fixation_time),
        'Endemic_I1_prop': endemic_I1_prop, 'Endemic_I2_prop': endemic_I2_prop,
        'Endemic_I2_share': endemic_I2_share,
        'peaks1_idx': peaks1, 'peaks2_idx': peaks2
    }

# =============================================================================
# 5. Helper: determine a wide x-axis window to prevent truncation
# =============================================================================
def compute_plot_xlim(t, I1, I2, fixation_time, N, floor_prop=1e-4, hard_max=2500):
    """
    Extends the x-axis to capture the full range of meaningful dynamics.
    Ensures plots aren't truncated mid-oscillation and zooms out appropriately.
    """
    active = np.where((I1 / N > floor_prop) | (I2 / N > floor_prop))[0]
    t_start = max(0.0, t[active[0]] - 10) if len(active) else 0.0

    peaks1, _ = find_peaks(I1, prominence=N * 0.0005, distance=14)
    peaks2, _ = find_peaks(I2, prominence=N * 0.0005, distance=14)
    last_peak_t = 0.0
    if len(peaks1):
        last_peak_t = max(last_peak_t, t[peaks1[-1]])
    if len(peaks2):
        last_peak_t = max(last_peak_t, t[peaks2[-1]])

    # Pad by 35% of the active duration or at least 150 days to show the long tail
    duration = max(fixation_time, last_peak_t) - t_start
    t_end = max(fixation_time, last_peak_t) + max(150.0, duration * 0.35)
    t_end = min(hard_max, t_end)
    return t_start, t_end

# =============================================================================
# 6. Composite 9-Panel Figure: Shape of Competition (3 scenarios x 3 views)
# =============================================================================
def generate_nine_panel_composite(scenario_runs, base_params, filename="Figure2_ShapeOfCompetition_9Panel"):
    n_rows = len(scenario_runs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 6.0 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    panel_letters = [chr(ord('A') + i) for i in range(3 * n_rows)]
    letter_idx = 0

    # Specific legend placements requested per panel layout
    legend_locs = [
        ['upper right', 'upper right', 'upper right'], # Row 0 (Adversarial): A, B, C
        ['upper left', 'upper left', 'upper left'],    # Row 1 (Cooperative): D, E, F
        ['upper left', 'upper right', 'upper left']    # Row 2 (Asymmetric): G, H, I
    ]

    for row, (label, t, y, metrics, params, t_em) in enumerate(scenario_runs):
        N = params['N']
        S, I1, I2, R1, R2 = y / N
        t_start, t_end = compute_plot_xlim(t, I1 * N, I2 * N, metrics['Fixation_Time'], N)

        # ---------------- Column A: Active Infections Time Series ----------------
        axA = axes[row, 0]
        axA.plot(t, I1, label=r'Strain 1 ($I_1/N$)', color='#1f77b4', lw=2.2)
        axA.plot(t, I2, label=r'Strain 2 ($I_2/N$)', color='#d62728', lw=2.2)
        axA.scatter(t[metrics['peaks1_idx']], I1[metrics['peaks1_idx']], color='black', zorder=5, marker='x', s=45)
        axA.scatter(t[metrics['peaks2_idx']], I2[metrics['peaks2_idx']], color='black', zorder=5, marker='x', s=45)
        if metrics['Fixation_Time'] <= t_end:
            axA.axvline(metrics['Fixation_Time'], color='green', ls='--', lw=1.6,
                        label=f"Fixation (Day {int(metrics['Fixation_Time'])})")
        axA.set_xlim(t_start, t_end)
        axA.set_ylim(bottom=0)
        axA.set_ylabel(r"Proportion ($I_i/N$)")
        if row == n_rows - 1:
            axA.set_xlabel("Days")
        axA.set_title(f"({panel_letters[letter_idx]}) {label}: Active Infections", fontsize=13, pad=8)
        axA.legend(loc=legend_locs[row][0], framealpha=0.92, fontsize=9.5)
        axA.grid(alpha=0.3)
        letter_idx += 1

        # ---------------- Column B: Immune Landscape (single stacked panel) ------
        axB = axes[row, 1]
        axB.stackplot(
            t, I1, I2, S, R1, R2,
            labels=[r'Infected, Strain 1 ($I_1/N$)', r'Infected, Strain 2 ($I_2/N$)',
                    r'Naive Susceptible ($S/N$)', r'Recovered from Strain 1 ($R_1/N$)',
                    r'Recovered from Strain 2 ($R_2/N$)'],
            colors=['#1f77b4', '#d62728', '#d3d3d3', '#aec7e8', '#ff9896'], alpha=0.85
        )
        axB.set_xlim(t_start, t_end)
        axB.set_ylim(0, 1.0)
        axB.set_ylabel("Population Proportion")
        if row == n_rows - 1:
            axB.set_xlabel("Days")
        axB.set_title(f"({panel_letters[letter_idx]}) {label}: Immune Landscape", fontsize=13, pad=8)
        axB.legend(loc=legend_locs[row][1], framealpha=0.95, fontsize=9)
        letter_idx += 1

        # ---------------- Column C: Dominance (relative competition, invasion-up) -
        axC = axes[row, 2]
        I_total = I1 + I2
        prop_I1 = np.zeros_like(I1)
        prop_I2 = np.zeros_like(I2)
        mask = I_total > 0
        prop_I1[mask] = I1[mask] / I_total[mask]
        prop_I2[mask] = I2[mask] / I_total[mask]
        prop_I1[I_total == 0] = 1.0
        prop_I2[I_total == 0] = 0.0

        axC.plot(t, prop_I1, color='#1f77b4', lw=2.2, label=r'Strain 1 share ($I_1/I_{total}$)')
        axC.plot(t, prop_I2, color='#d62728', lw=2.2, label=r'Strain 2 share ($I_2/I_{total}$)')
        axC.fill_between(t, 0, prop_I2, color='#d62728', alpha=0.15)
        axC.fill_between(t, prop_I1, 1, color='#1f77b4', alpha=0.15)
        axC.set_xlim(t_start, t_end)
        axC.set_ylim(0, 1.0)
        axC.set_ylabel(r"Share of Active Infections")
        if row == n_rows - 1:
            axC.set_xlabel("Days")
        axC.set_title(f"({panel_letters[letter_idx]}) {label}: Strain Dominance", fontsize=13, pad=8)
        axC.legend(loc=legend_locs[row][2], framealpha=0.95, fontsize=9.5)
        axC.grid(alpha=0.3)
        letter_idx += 1

    fig.suptitle("The Shape of Competition: Time Series, Immune Landscape, and Dominance Across Three Cross-Immunity Regimes",
                 fontsize=17, weight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{filename}.{FIG_EXT}"), bbox_inches='tight')
    plt.close('all')

# =============================================================================
# 7. Dual Phase-Portrait Figure
# =============================================================================
def generate_dual_phase_portraits(base_params, t_max, scenario_defs, tem_compare_list,
                                   tem_compare_scenario, filename="Figure3_PhasePortraits"):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 7.8))

    # ---- Left: scenario comparison ----
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    for (name, e1, e2, t_em), col in zip(scenario_defs, colors):
        p = base_params.copy()
        p['eta1'], p['eta2'] = e1, e2
        t, y = simulate_epidemic(p, t_em, t_max)
        I1 = y[1] / p['N']
        I2 = y[2] / p['N']
        axL.plot(I1, I2, color=col, lw=1.8, alpha=0.9, label=rf"{name} ($\eta_1$={e1}, $\eta_2$={e2})")
        axL.scatter(I1[0], I2[0], color=col, marker='o', s=45, zorder=5, edgecolor='black', linewidth=0.6)
        axL.scatter(I1[-1], I2[-1], color=col, marker='*', s=200, zorder=6, edgecolor='black', linewidth=0.6)
    axL.set_title(f"(A) Phase Portraits Across Cross-Immunity Regimes", fontsize=13, pad=8)
    axL.set_xlabel(r"Strain 1 Proportion ($I_1/N$)")
    axL.set_ylabel(r"Strain 2 Proportion ($I_2/N$)")
    axL.legend(loc='upper left', bbox_to_anchor=(0.0, -0.13), framealpha=0.92, fontsize=9.5, ncol=1)
    axL.grid(alpha=0.3)

    # ---- Right: emergence-time comparison, fixed cross-immunity ----
    e1_fixed, e2_fixed = tem_compare_scenario
    distinct_colors = ['#7f3c8d', '#11a579', '#3969ac', '#f2b701', '#e73f74']
    for tem, col in zip(tem_compare_list, distinct_colors):
        p = base_params.copy()
        p['eta1'], p['eta2'] = e1_fixed, e2_fixed
        t, y = simulate_epidemic(p, tem, t_max)
        I1 = y[1] / p['N']
        I2 = y[2] / p['N']
        axR.plot(I1, I2, color=col, lw=1.8, alpha=0.9, label=rf"$t_{{em}}={tem}$")
        axR.scatter(I1[-1], I2[-1], color=col, marker='*', s=200, zorder=6, edgecolor='black', linewidth=0.6)
    axR.set_title(f"(B) Phase Portraits Across Emergence Times ($\eta_1={e1_fixed}, \eta_2={e2_fixed}$)", fontsize=13, pad=8)
    axR.set_xlabel(r"Strain 1 Proportion ($I_1/N$)")
    axR.set_ylabel(r"Strain 2 Proportion ($I_2/N$)")
    axR.legend(loc='upper left', bbox_to_anchor=(0.0, -0.13), framealpha=0.92, fontsize=9.5, ncol=1)
    axR.grid(alpha=0.3)

    fig.suptitle("Endemic Attractors Are Set by Cross-Immunity, Not by Emergence Timing", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{filename}.{FIG_EXT}"), bbox_inches='tight')
    plt.close('all')

# =============================================================================
# 8. Representative Time-Series Figure at t_em = 25
# =============================================================================
def find_yellow_strip_coordinates(base_params, t_max, eta_range, t_em=25):
    n = len(eta_range)
    fix_mat = np.zeros((n, n))
    peak_mat = np.zeros((n, n))

    for i, e1 in enumerate(eta_range):
        for j, e2 in enumerate(eta_range):
            p = base_params.copy()
            p['eta1'], p['eta2'] = e1, e2
            t, y = simulate_epidemic(p, t_em, t_max)
            mets = extract_metrics(t, y, p['N'])
            fix_mat[i, j] = mets['Fixation_Time']
            peak_mat[i, j] = mets['P2_count']

    idx_fix = np.unravel_index(np.argmax(fix_mat), fix_mat.shape)
    idx_peak = np.unravel_index(np.argmax(peak_mat), peak_mat.shape)

    result = {
        'max_fixation': (float(eta_range[idx_fix[0]]), float(eta_range[idx_fix[1]]), float(fix_mat[idx_fix])),
        'max_peaks': (float(eta_range[idx_peak[0]]), float(eta_range[idx_peak[1]]), float(peak_mat[idx_peak])),
        'fix_mat': fix_mat, 'peak_mat': peak_mat,
    }
    print(f"  [Verified] True max fixation time on grid: eta1={result['max_fixation'][0]:.3f}, "
          f"eta2={result['max_fixation'][1]:.3f}, fixation={result['max_fixation'][2]:.0f} days")
    print(f"  [Verified] True max Strain 2 peak count on grid: eta1={result['max_peaks'][0]:.3f}, "
          f"eta2={result['max_peaks'][1]:.3f}, peaks={result['max_peaks'][2]:.0f}")
    return result

def generate_representative_timeseries_tem25(base_params, t_max, yellow_strip_coords,
                                              filename="Figure4_RepresentativeTimeSeries_tem25"):
    e1_fix, e2_fix, fix_val = yellow_strip_coords['max_fixation']
    e1_peak, e2_peak, peak_val = yellow_strip_coords['max_peaks']

    coords = [
        (f"Yellow Strip (Max Fixation Time)\n$\eta_1={e1_fix:.2f}$, $\eta_2={e2_fix:.2f}$, $t_{{em}}=25$", e1_fix, e2_fix),
        (f"Yellow Strip (Max Peak Count)\n$\eta_1={e1_peak:.2f}$, $\eta_2={e2_peak:.2f}$, $t_{{em}}=25$", e1_peak, e2_peak),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.8))

    for ax, (label, e1, e2) in zip(axes, coords):
        p = base_params.copy()
        p['eta1'], p['eta2'] = round(e1, 4), round(e2, 4)
        t, y = simulate_epidemic(p, 25, t_max)
        mets = extract_metrics(t, y, p['N'])
        N = p['N']
        I1, I2 = y[1] / N, y[2] / N
        t_start, t_end = compute_plot_xlim(t, y[1], y[2], mets['Fixation_Time'], N)

        ax.plot(t, I1, color='#1f77b4', lw=2.0, label=r'Strain 1 ($I_1/N$)')
        ax.plot(t, I2, color='#d62728', lw=2.0, label=r'Strain 2 ($I_2/N$)')
        ax.scatter(t[mets['peaks1_idx']], I1[mets['peaks1_idx']], color='black', marker='x', s=45, zorder=5)
        ax.scatter(t[mets['peaks2_idx']], I2[mets['peaks2_idx']], color='black', marker='x', s=45, zorder=5)
        if mets['Fixation_Time'] <= t_end:
            ax.axvline(mets['Fixation_Time'], color='green', ls='--', lw=1.6,
                       label=f"Fixation (Day {int(mets['Fixation_Time'])})")
        ax.set_xlim(t_start, t_end)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Days")
        ax.set_ylabel(r"Proportion ($I_i/N$)")
        ax.set_title(label, fontsize=12.5, pad=8)
        ax.legend(loc='upper right', framealpha=0.92, fontsize=9.5)
        ax.grid(alpha=0.3)

    fig.suptitle("Representative Yellow-Strip Time Series at the Heatmap's Fixed Emergence Time", y=1.05, fontsize=15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{filename}.{FIG_EXT}"), bbox_inches='tight')
    plt.close('all')

# =============================================================================
# 9. Appendix Figure: Raw (non-proportional-competition) Phase Portrait
# =============================================================================
def generate_appendix_raw_phase_portrait(t, y, metrics, scenario_name, params, t_em,
                                          filename_prefix="Appendix"):
    N = params['N']
    I1, I2 = y[1] / N, y[2] / N
    plt.figure(figsize=(7.5, 7.5))
    plt.plot(I1, I2, color='purple', lw=1, alpha=0.7)
    plt.scatter(I1[0], I2[0], color='green', label='Start (Phase 1 end)', zorder=5, s=60)
    plt.scatter(metrics['Endemic_I1_prop'], metrics['Endemic_I2_prop'], color='red',
                marker='*', s=220, label='Endemic Attractor', zorder=5)
    plt.title(f"{scenario_name}: Phase Portrait (Appendix)\n$\eta_1={params['eta1']}$, $\eta_2={params['eta2']}$, $t_{{em}}={t_em}$",
              weight='bold', fontsize=12, pad=8)
    plt.xlabel(r"Strain 1 Proportion ($I_1/N$)")
    plt.ylabel(r"Strain 2 Proportion ($I_2/N$)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_{scenario_name.replace(' ', '_')}_PhasePortrait.{FIG_EXT}"),
                bbox_inches='tight')
    plt.close('all')

# =============================================================================
# 10A. SECTION: Fixation Time Sweeps
# =============================================================================
def run_fixation_time_section(base_params, t_max):
    print("=== FIXATION TIME SECTION ===")
    eta_range = np.linspace(0.1, 2.0, 30)
    X, Y = np.meshgrid(eta_range, eta_range)

    print("Computing fixation-time sweep across emergence thresholds...")
    t_em_list = [25, 50, 100, 150, 200, 250]
    all_fix_mats = []
    for t_em in t_em_list:
        print(f"  --> t_em = {t_em}")
        fix_mat = np.zeros((30, 30))
        for i, e1 in enumerate(eta_range):
            for j, e2 in enumerate(eta_range):
                p = base_params.copy()
                p['eta1'], p['eta2'] = e1, e2
                t, y = simulate_epidemic(p, t_em, t_max)
                mets = extract_metrics(t, y, p['N'])
                fix_mat[i, j] = mets['Fixation_Time']
        all_fix_mats.append(fix_mat)

    global_fix_max = max(np.max(m) for m in all_fix_mats)

    fig, axes = plt.subplots(2, 3, figsize=(19, 12.5))
    fig.suptitle(rf"Fixation Time Across Emergence Thresholds ($\beta_1={base_params['beta1']}$, $\beta_2={base_params['beta2']}$)", y=1.03, fontsize=16)
    for idx, (t_em, fix_mat) in enumerate(zip(t_em_list, all_fix_mats)):
        ax = axes.flatten()[idx]
        c = ax.pcolormesh(X, Y, fix_mat, shading='auto', cmap='viridis', vmin=0, vmax=global_fix_max)
        ax.axhline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax.axvline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax.set_title(rf"$t_{{em}} = {t_em}$", pad=6)
        ax.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        ax.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")
    fig.colorbar(c, ax=axes.ravel().tolist(), label='Days to Fixation', shrink=0.95)
    fig.savefig(os.path.join(output_dir, f"Figure5_FixationTime_EmergenceSweep.{FIG_EXT}"), bbox_inches='tight')
    plt.close('all')

    print("Computing fixation-time sweep across beta2/beta1 ratios (t_em=25)...")
    beta2_list = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    all_fix_mats_beta = []
    for b2 in beta2_list:
        print(f"  --> beta2 = {b2}")
        fix_mat = np.zeros((30, 30))
        for i, e1 in enumerate(eta_range):
            for j, e2 in enumerate(eta_range):
                p = base_params.copy()
                p['beta2'], p['eta1'], p['eta2'] = b2, e1, e2
                t, y = simulate_epidemic(p, 25, t_max)
                mets = extract_metrics(t, y, p['N'])
                fix_mat[i, j] = mets['Fixation_Time']
        all_fix_mats_beta.append(fix_mat)

    global_fix_max_beta = max(np.max(m) for m in all_fix_mats_beta)

    fig, axes = plt.subplots(2, 4, figsize=(23, 12.5))
    fig.suptitle(rf"Fixation Time Across $\beta_2/\beta_1$ Ratios (Fixed $t_{{em}}=25, \beta_1={base_params['beta1']}$)", y=1.03, fontsize=16)
    for idx, (b2, fix_mat) in enumerate(zip(beta2_list, all_fix_mats_beta)):
        ax = axes.flatten()[idx]
        c = ax.pcolormesh(X, Y, fix_mat, shading='auto', cmap='viridis', vmin=0, vmax=global_fix_max_beta)
        ax.axhline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax.axvline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax.set_title(rf"$\beta_2$={b2} (Ratio {b2/base_params['beta1']:.2f})", pad=6)
        if idx >= 4: ax.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        if idx % 4 == 0: ax.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")
    fig.colorbar(c, ax=axes.ravel().tolist(), label='Days to Fixation', shrink=0.95)
    fig.savefig(os.path.join(output_dir, f"Figure6_FixationTime_BetaRatioSweep.{FIG_EXT}"), bbox_inches='tight')
    plt.close('all')

    return all_fix_mats, all_fix_mats_beta, t_em_list, beta2_list, eta_range

# =============================================================================
# 10B. SECTION: Peak Count Sweeps
# =============================================================================
def run_peak_count_section(base_params, t_max):
    print("=== PEAK COUNT SECTION ===")
    eta_range = np.linspace(0.1, 2.0, 30)
    X, Y = np.meshgrid(eta_range, eta_range)

    print("Computing peak-count sweep across emergence thresholds...")
    t_em_list = [25, 50, 100, 150, 200, 250]
    all_peak_mats = []
    for t_em in t_em_list:
        print(f"  --> t_em = {t_em}")
        peak_mat = np.zeros((30, 30))
        for i, e1 in enumerate(eta_range):
            for j, e2 in enumerate(eta_range):
                p = base_params.copy()
                p['eta1'], p['eta2'] = e1, e2
                t, y = simulate_epidemic(p, t_em, t_max)
                mets = extract_metrics(t, y, p['N'])
                peak_mat[i, j] = mets['P2_count']
        all_peak_mats.append(peak_mat)

    global_peak_max = max(np.max(m) for m in all_peak_mats)

    fig, axes = plt.subplots(2, 3, figsize=(19, 12.5))
    fig.suptitle(rf"Strain 2 Peak Counts Across Emergence Thresholds ($\beta_1={base_params['beta1']}$, $\beta_2={base_params['beta2']}$)", y=1.03, fontsize=16)
    for idx, (t_em, peak_mat) in enumerate(zip(t_em_list, all_peak_mats)):
        ax = axes.flatten()[idx]
        c = ax.pcolormesh(X, Y, peak_mat, shading='auto', cmap='magma', vmin=0, vmax=global_peak_max)
        ax.axhline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax.axvline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax.set_title(rf"$t_{{em}} = {t_em}$", pad=6)
        ax.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        ax.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")
    fig.colorbar(c, ax=axes.ravel().tolist(), label='Number of Strain 2 Peaks', shrink=0.95)
    fig.savefig(os.path.join(output_dir, f"Figure7_PeakCount_EmergenceSweep.{FIG_EXT}"), bbox_inches='tight')
    plt.close('all')

    print("Computing peak-count sweep across beta2/beta1 ratios (t_em=25)...")
    beta2_list = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    all_peak_mats_beta = []
    for b2 in beta2_list:
        print(f"  --> beta2 = {b2}")
        peak_mat = np.zeros((30, 30))
        for i, e1 in enumerate(eta_range):
            for j, e2 in enumerate(eta_range):
                p = base_params.copy()
                p['beta2'], p['eta1'], p['eta2'] = b2, e1, e2
                t, y = simulate_epidemic(p, 25, t_max)
                mets = extract_metrics(t, y, p['N'])
                peak_mat[i, j] = mets['P2_count']
        all_peak_mats_beta.append(peak_mat)

    global_peak_max_beta = max(np.max(m) for m in all_peak_mats_beta)

    fig, axes = plt.subplots(2, 4, figsize=(23, 12.5))
    fig.suptitle(rf"Strain 2 Peak Counts Across $\beta_2/\beta_1$ Ratios (Fixed $t_{{em}}=25, \beta_1={base_params['beta1']}$)", y=1.03, fontsize=16)
    for idx, (b2, peak_mat) in enumerate(zip(beta2_list, all_peak_mats_beta)):
        ax = axes.flatten()[idx]
        c = ax.pcolormesh(X, Y, peak_mat, shading='auto', cmap='magma', vmin=0, vmax=global_peak_max_beta)
        ax.axhline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax.axvline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax.set_title(rf"$\beta_2$={b2} (Ratio {b2/base_params['beta1']:.2f})", pad=6)
        if idx >= 4: ax.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        if idx % 4 == 0: ax.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")
    fig.colorbar(c, ax=axes.ravel().tolist(), label='Number of Strain 2 Peaks', shrink=0.95)
    fig.savefig(os.path.join(output_dir, f"Figure8_PeakCount_BetaRatioSweep.{FIG_EXT}"), bbox_inches='tight')
    plt.close('all')

    return all_peak_mats, all_peak_mats_beta, t_em_list, beta2_list, eta_range

# =============================================================================
# 10C. SECTION: Peak Intensity & Dominance -- Replaces "Endemic Composition"
# =============================================================================
def run_peak_intensity_section(base_params, t_max, t_em_list=(25, 180)):
    """
    REDESIGNED: Replaces the flat 'Endemic Composition' grid with a robust analysis
    of Peak Intensity (Outbreak Size) and which strain generates the largest peak.
    This creates deeply informative visualizations that show where cross-immunity
    protects against severe secondary waves versus amplifying them.
    """
    print("=== PEAK INTENSITY & DOMINANCE SECTION ===")
    eta_range = np.linspace(0.1, 2.0, 30)
    X, Y = np.meshgrid(eta_range, eta_range)

    for t_em in t_em_list:
        print(f"  --> Peak Intensity at t_em = {t_em}")
        p1_max_mat = np.zeros((30, 30))
        p2_max_mat = np.zeros((30, 30))
        share_mat = np.zeros((30, 30))

        for i, e1 in enumerate(eta_range):
            for j, e2 in enumerate(eta_range):
                p = base_params.copy()
                p['eta1'], p['eta2'] = e1, e2
                t, y = simulate_epidemic(p, t_em, t_max)
                p1_max = np.max(y[1]) / p['N']
                p2_max = np.max(y[2]) / p['N']
                p1_max_mat[i, j] = p1_max
                p2_max_mat[i, j] = p2_max
                total_max = p1_max + p2_max
                share_mat[i, j] = p2_max / total_max if total_max > 0 else 0.5

        fig, axes = plt.subplots(1, 4, figsize=(30, 7.4))
        fig.suptitle(f"Transient Peak Intensity & Dominance Across the Cross-Immunity Grid ($t_{{em}}={t_em}$)",
                     y=1.06, fontsize=16)

        peak_vmax = max(p1_max_mat.max(), p2_max_mat.max())

        ax0 = axes[0]
        c0 = ax0.pcolormesh(X, Y, p2_max_mat, shading='auto', cmap='Reds', vmin=0, vmax=peak_vmax)
        ax0.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
        ax0.axvline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
        ax0.set_title(r"(A) Maximum Peak Size, Strain 2 (max $I_2/N$)", pad=6)
        ax0.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        ax0.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")
        fig.colorbar(c0, ax=ax0, label=r'max $I_2/N$', shrink=0.85)

        ax1 = axes[1]
        c1 = ax1.pcolormesh(X, Y, p1_max_mat, shading='auto', cmap='Blues', vmin=0, vmax=peak_vmax)
        ax1.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
        ax1.axvline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
        ax1.set_title(r"(B) Maximum Peak Size, Strain 1 (max $I_1/N$)", pad=6)
        ax1.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        ax1.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")
        fig.colorbar(c1, ax=ax1, label=r'max $I_1/N$', shrink=0.85)

        ax2 = axes[2]
        c2 = ax2.pcolormesh(X, Y, share_mat, shading='auto', cmap='RdBu_r', vmin=0, vmax=1.0)
        ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
        ax2.axvline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
        ax2.set_title(r"(C) Peak Dominance (Strain 2 Share of Max Peaks)", pad=6)
        ax2.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        ax2.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")
        fig.colorbar(c2, ax=ax2, label=r'max $I_2$ / (max $I_1$ + max $I_2$)', shrink=0.85)

        ax3 = axes[3]
        j_low = 0
        j_mid = np.argmin(np.abs(eta_range - 1.0))
        ax3.plot(eta_range, share_mat[:, j_low], color='#e73f74', lw=2.4,
                 label=rf"$\eta_2={eta_range[j_low]:.2f}$ (Strong $R_1$ immunity)")
        ax3.plot(eta_range, share_mat[:, j_mid], color='#3969ac', lw=2.0, ls='--',
                 label=rf"$\eta_2={eta_range[j_mid]:.2f}$ (Neutral $R_1$ immunity)")
        ax3.axhline(0.5, color='gray', lw=1.2, ls=':', label='Peak co-dominance (0.5)')
        ax3.set_ylim(0, 1.05)
        ax3.set_xlabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")
        ax3.set_ylabel(r"Strain 2 Peak Share")
        ax3.set_title("(D) Peak Dominance Profile\nAlong Selected $\eta_2$ Trajectories", pad=6, fontsize=12.5)
        ax3.legend(loc='best', framealpha=0.92, fontsize=9.5)
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"Figure9_PeakIntensity_tem{t_em}.{FIG_EXT}"), bbox_inches='tight')
        plt.close('all')

# =============================================================================
# 10D. SECTION: Robustness / Sensitivity Summary Figure
# =============================================================================
def run_sensitivity_summary_figure(base_params, t_max, filename="Figure10_SensitivitySummary"):
    print("=== SENSITIVITY SUMMARY FIGURE ===")
    eta_diag = np.linspace(0.2, 2.0, 20)
    fix_vals, peak_vals, share_vals = [], [], []
    for e in eta_diag:
        p = base_params.copy()
        p['eta1'], p['eta2'] = e, e
        t, y = simulate_epidemic(p, 180, t_max)
        mets = extract_metrics(t, y, p['N'])
        
        p1_max = np.max(y[1]) / p['N']
        p2_max = np.max(y[2]) / p['N']
        total_max = p1_max + p2_max
        peak_share = p2_max / total_max if total_max > 0 else 0.5
        
        fix_vals.append(mets['Fixation_Time'])
        peak_vals.append(mets['P2_count'])
        share_vals.append(peak_share)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    axes[0].plot(eta_diag, fix_vals, marker='o', color='#1f77b4', lw=2)
    axes[0].set_title(f"(A) Fixation Time Along the Symmetric Diagonal", pad=6, fontsize=12.5)
    axes[0].set_xlabel(r"$\eta_1 = \eta_2$")
    axes[0].set_ylabel("Days to Fixation")
    axes[0].grid(alpha=0.3)

    axes[1].plot(eta_diag, peak_vals, marker='s', color='#d62728', lw=2)
    axes[1].set_title(f"(B) Strain 2 Peak Count Along the Symmetric Diagonal", pad=6, fontsize=12.5)
    axes[1].set_xlabel(r"$\eta_1 = \eta_2$")
    axes[1].set_ylabel("Number of Strain 2 Peaks")
    axes[1].grid(alpha=0.3)

    axes[2].plot(eta_diag, share_vals, marker='^', color='#2ca02c', lw=2)
    axes[2].axhline(0.5, color='gray', ls=':', lw=1.2)
    axes[2].set_title(f"(C) Strain 2 Peak Share Along the Symmetric Diagonal", pad=6, fontsize=12.5)
    axes[2].set_xlabel(r"$\eta_1 = \eta_2$")
    axes[2].set_ylabel(r"Strain 2 Peak Share")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)

    fig.suptitle(r"Sensitivity Summary Along the Symmetric Cross-Immunity Diagonal ($\eta_1=\eta_2, t_{em}=180$)", y=1.06, fontsize=16)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{filename}.{FIG_EXT}"), bbox_inches='tight')
    plt.close('all')

# =============================================================================
# 11. Main Execution Pipeline
# =============================================================================
if __name__ == "__main__":
    t_maximum = 1825

    base_params = {
        'N': 100000,
        'beta1': 0.30, 'gamma1': 0.1,
        'beta2': 0.45, 'gamma2': 0.1,
        'eta1': 1.0,   'eta2': 1.0,
        'delta': 1 / 180
    }

    # -------------------------------------------------------------------
    # SECTION 1: The Shape of Competition -- 9-panel composite figure
    # -------------------------------------------------------------------
    scenario_defs = [
        ("Adversarial", 0.4, 0.4, 180),
        ("Cooperative", 1.6, 1.6, 180),
        ("Asymmetric Dominance", 0.5, 1.8, 180),
    ]

    scenario_runs = []
    with open(os.path.join(output_dir, "Comprehensive_Metrics_Report.txt"), "w") as report:
        report.write("MULTI-STRAIN TRANSIENT DYNAMICS REPORT\n")
        report.write("=" * 80 + "\n\n")

        for name, e1, e2, t_em in scenario_defs:
            print(f"Simulating baseline scenario: {name} (eta1={e1}, eta2={e2}, t_em={t_em})")
            params = base_params.copy()
            params['eta1'], params['eta2'] = e1, e2
            t, y = simulate_epidemic(params, t_em, t_maximum)
            metrics = extract_metrics(t, y, params['N'])
            scenario_runs.append((name, t, y, metrics, params, t_em))

            generate_appendix_raw_phase_portrait(t, y, metrics, name, params, t_em)

            report.write(f"SCENARIO: {name}\n")
            report.write(f"Parameters: beta1={params['beta1']}, beta2={params['beta2']}, gamma1={params['gamma1']}, "
                         f"gamma2={params['gamma2']}, delta=1/{round(1/params['delta'])}, N={params['N']}, "
                         f"eta1={e1}, eta2={e2}, t_em={t_em}\n")
            report.write("-" * 80 + "\n")
            report.write(f"Strain 1: {metrics['P1_count']} Peaks | Sizes (I1/N): {metrics['P1_sizes']} | Avg Dist: {metrics['P1_dist']:.1f} days\n")
            report.write(f"Strain 2: {metrics['P2_count']} Peaks | Sizes (I2/N): {metrics['P2_sizes']} | Avg Dist: {metrics['P2_dist']:.1f} days\n")
            report.write(f"Fixation Achieved: Day {metrics['Fixation_Time']:.0f}\n")
            report.write(f"Endemic Equilibria: I1*/N = {metrics['Endemic_I1_prop']:.4f}, I2*/N = {metrics['Endemic_I2_prop']:.4f}\n")
            report.write(f"Endemic Composition: Strain 2 share of combined endemic burden = {metrics['Endemic_I2_share']:.4f}\n\n")

    print("Generating 9-panel composite Figure 2 (Shape of Competition)...")
    generate_nine_panel_composite(scenario_runs, base_params)

    # -------------------------------------------------------------------
    # SECTION 1b: Dual phase-portrait figure (incl. t_em=50)
    # -------------------------------------------------------------------
    print("Generating dual phase-portrait Figure 3...")
    tem_compare_list = [25, 50, 100, 180, 250]
    tem_compare_scenario = (0.4, 0.4)
    generate_dual_phase_portraits(base_params, t_maximum,
                                   scenario_defs=[(n, e1, e2, 180) for n, e1, e2, _ in scenario_defs],
                                   tem_compare_list=tem_compare_list,
                                   tem_compare_scenario=tem_compare_scenario)

    # -------------------------------------------------------------------
    # SECTION 1c: TRUE argmax coordinates, representative time series
    # -------------------------------------------------------------------
    print("Computing TRUE Yellow Strip argmax coordinates at t_em=25 (full 30x30 grid)...")
    eta_range_full = np.linspace(0.1, 2.0, 30)
    yellow_strip_coords = find_yellow_strip_coordinates(base_params, t_maximum, eta_range_full, t_em=25)

    with open(os.path.join(output_dir, "Comprehensive_Metrics_Report.txt"), "a") as report:
        for tag, (e1, e2, val) in [("Max Fixation Time", yellow_strip_coords['max_fixation']),
                                    ("Max Peak Count", yellow_strip_coords['max_peaks'])]:
            p = base_params.copy()
            p['eta1'], p['eta2'] = round(e1, 4), round(e2, 4)
            t, y = simulate_epidemic(p, 25, t_maximum)
            mets = extract_metrics(t, y, p['N'])
            report.write(f"SCENARIO: Yellow Strip ({tag}) -- TRUE ARGMAX on 30x30 grid, t_em=25\n")
            report.write(f"Parameters: beta1={p['beta1']}, beta2={p['beta2']}, gamma1={p['gamma1']}, "
                         f"gamma2={p['gamma2']}, delta=1/{round(1/p['delta'])}, N={p['N']}, "
                         f"eta1={p['eta1']:.4f}, eta2={p['eta2']:.4f}, t_em=25\n")
            report.write("-" * 80 + "\n")
            report.write(f"Strain 1: {mets['P1_count']} Peaks | Sizes (I1/N): {mets['P1_sizes']} | Avg Dist: {mets['P1_dist']:.1f} days\n")
            report.write(f"Strain 2: {mets['P2_count']} Peaks | Sizes (I2/N): {mets['P2_sizes']} | Avg Dist: {mets['P2_dist']:.1f} days\n")
            report.write(f"Fixation Achieved: Day {mets['Fixation_Time']:.0f}\n")
            report.write(f"Endemic Equilibria: I1*/N = {mets['Endemic_I1_prop']:.4f}, I2*/N = {mets['Endemic_I2_prop']:.4f}\n")
            report.write(f"Endemic Composition: Strain 2 share of combined endemic burden = {mets['Endemic_I2_share']:.4f}\n\n")

    print("Generating representative Yellow Strip time series at t_em=25 (Figure 4)...")
    generate_representative_timeseries_tem25(base_params, t_maximum, yellow_strip_coords)

    # -------------------------------------------------------------------
    # SECTION 2: Fixation Time 
    # -------------------------------------------------------------------
    run_fixation_time_section(base_params, t_maximum)

    # -------------------------------------------------------------------
    # SECTION 3: Peak Count 
    # -------------------------------------------------------------------
    run_peak_count_section(base_params, t_maximum)

    # -------------------------------------------------------------------
    # SECTION 4: Peak Intensity & Dominance (Redesigned)
    # -------------------------------------------------------------------
    run_peak_intensity_section(base_params, t_maximum, t_em_list=(25, 180))

    # -------------------------------------------------------------------
    # SECTION 5: Sensitivity summary
    # -------------------------------------------------------------------
    run_sensitivity_summary_figure(base_params, t_maximum)

    print(f"\nPipeline completely executed. Output directory localized at '{output_dir}'.")
    print("\nFigure manifest:")
    print("  Figure2_ShapeOfCompetition_9Panel       -- 3x3 composite (Section: Shape of Competition)")
    print("  Figure3_PhasePortraits                  -- scenario comparison + emergence-time comparison")
    print("  Figure4_RepresentativeTimeSeries_tem25  -- Yellow Strip time series at t_em=25")
    print("  Figure5_FixationTime_EmergenceSweep     -- Section: Fixation Time")
    print("  Figure6_FixationTime_BetaRatioSweep     -- Section: Fixation Time")
    print("  Figure7_PeakCount_EmergenceSweep        -- Section: Peak Count")
    print("  Figure8_PeakCount_BetaRatioSweep        -- Section: Peak Count")
    print("  Figure9_PeakIntensity_tem{25,180}       -- Section: Peak Intensity & Dominance")
    print("  Figure10_SensitivitySummary             -- NEW: 1D diagonal sensitivity summary")
    print("  Appendix_*_PhasePortrait                -- Appendix")
