import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from datetime import datetime

# =============================================================================
# 1. System Setup and Directory Management
# =============================================================================
# Generates a unique timestamped directory to ensure previous runs are never overwritten.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"model_outputs_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print(f"Initializing Comprehensive Multi-Scenario Epidemic Pipeline...")
print(f"All outputs will be safely routed to: ./{output_dir}/\n")

# =============================================================================
# 2. Core ODE System (The Cyclic Reinfection "Inner Loop")
# =============================================================================
def two_strain_inner_loop(t, y, beta1, beta2, gamma1, gamma2, eta1, eta2, delta, N):
    """
    Evaluates the continuous deterministic differential equations for the two-strain model.
    Waning immunity routes directly to naive susceptibility (S), bypassing any isolated
    waned compartment (W) to force biological competition between waning and cross-infection.
    """
    S, I1, I2, R1, R2 = y

    # Frequency-dependent forces of infection
    lambda1 = beta1 * I1 / N
    lambda2 = beta2 * I2 / N

    # Compartmental flows
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
    """
    Simulates the epidemic in two chronologically distinct phases to mathematically
    replicate the emergence of a novel variant into an already primed immune landscape.
    """
    N = params['N']
    y0_phase1 = [N - 10, 10, 0, 0, 0]

    ode_args = (params['beta1'], params['beta2'], params['gamma1'],
                params['gamma2'], params['eta1'], params['eta2'],
                params['delta'], N)

    # Phase 1: Establish Strain 1 in pure isolation
    sol1 = solve_ivp(two_strain_inner_loop, [0, t_em], y0_phase1, args=ode_args,
                     method='Radau', dense_output=True, max_step=1.0)

    # Phase 2: Inject Variant 2 Seed
    y0_phase2 = sol1.y[:, -1].copy()
    seed_size = 10

    # Dynamically extract seed cases from non-infected pools to conserve N perfectly
    if y0_phase2[0] >= seed_size:
        y0_phase2[0] -= seed_size
    elif y0_phase2[3] >= seed_size:
        y0_phase2[3] -= seed_size
    else:
        # Ultimate fail-safe to prevent negative compartmental populations
        y0_phase2[0] = max(0, y0_phase2[0] - seed_size)

    y0_phase2[2] += seed_size

    # Integrate Phase 2 using stiff solver up to maximum temporal horizon
    sol2 = solve_ivp(two_strain_inner_loop, [t_em, t_max], y0_phase2, args=ode_args,
                     method='Radau', dense_output=True, max_step=1.0)

    t_combined = np.concatenate((sol1.t, sol2.t[1:]))
    y_combined = np.concatenate((sol1.y, sol2.y[:, 1:]), axis=1)

    # Absolute Mathematical Conservation Check
    total_population = np.sum(y_combined, axis=0)
    if not np.allclose(total_population, N, atol=1e-5):
        print(f"CRITICAL WARNING: Population conservation violated at eta1={params['eta1']}, eta2={params['eta2']}")

    return t_combined, y_combined

# =============================================================================
# 4. High-Fidelity Metric Extraction
# =============================================================================
def extract_metrics(t, y, N, epsilon=1e-5, window=30):
    """
    Mathematically extracts oscillatory peak counts, peak incidence sizes,
    average temporal distances, and formal epidemiological fixation time.

    FIXATION TIME - CORRECTED DEFINITION
    -------------------------------------
    The original implementation scanned forward and returned the FIRST rolling
    30-day window in which both |dI1/dt| and |dI2/dt| dropped below epsilon*N.
    That is wrong for this model: the "phantom ocean" dynamic frequently produces
    a temporary trough (a lull between waves) that is quiet enough to satisfy the
    threshold, long before the system has actually finished oscillating. The old
    code would latch onto that trough, report a spuriously early fixation day, and
    never check whether a later wave violated the condition again.

    The correct definition of fixation is a ONE-SIDED, PERMANENT condition: the
    earliest day after which the derivative condition holds for the REST of the
    simulation, with no subsequent violation. Equivalently: find the last day on
    which |dI1/dt| or |dI2/dt| exceeds epsilon*N, and report fixation as the very
    next day. This is automatically robust to any number of intermediate troughs,
    requires no forward window-verification loop, and is a single O(n) backward
    scan instead of the old O(n * window) nested loop.
    """
    I1, I2 = y[1], y[2]

    # Utilizing strict prominence filters to reject ODE numerical artifacts
    peaks1, _ = find_peaks(I1, prominence=N * 0.0005, distance=14)
    peaks2, _ = find_peaks(I2, prominence=N * 0.0005, distance=14)

    avg_dist_1 = np.mean(np.diff(t[peaks1])) if len(peaks1) > 1 else 0
    avg_dist_2 = np.mean(np.diff(t[peaks2])) if len(peaks2) > 1 else 0

    # Store peak sizes as clean, standard Python floats (avoiding ugly numpy wrappers in text)
    p1_sizes = [float(round(val / N, 4)) for val in I1[peaks1]]
    p2_sizes = [float(round(val / N, 4)) for val in I2[peaks2]]

    # ---------------------------------------------------------------
    # Corrected, O(n) Fixation Time: backward scan for last violation
    # ---------------------------------------------------------------
    dI1 = np.abs(np.diff(I1))
    dI2 = np.abs(np.diff(I2))
    threshold = epsilon * N
    violating = (dI1 >= threshold) | (dI2 >= threshold)  # length n-1, step i covers t[i]->t[i+1]

    violation_idx = np.nonzero(violating)[0]
    if len(violation_idx) == 0:
        # The system never violates the tolerance at all (e.g. fully flat from the start)
        fixation_time = t[0]
    else:
        last_violation = violation_idx[-1]
        # Fixation is declared the step immediately after the final violation, but we
        # additionally require that a full `window`-day quiet stretch actually exists
        # after that point (guards against reporting fixation one day before t_max on
        # a simulation that was simply truncated mid-oscillation).
        candidate_i = last_violation + 1
        if candidate_i >= len(t):
            fixation_time = t[-1]
        else:
            remaining_days = t[-1] - t[candidate_i]
            if remaining_days < window:
                # Not enough runway left to confirm a stable window; report t_max as
                # a conservative (non-converged) fixation estimate.
                fixation_time = t[-1]
            else:
                fixation_time = t[candidate_i]

    # Proportional Endemic Equilibria mapping (Averaged over final 50 days to ensure flat stability)
    endemic_I1_prop = float(np.mean(I1[-50:]) / N)
    endemic_I2_prop = float(np.mean(I2[-50:]) / N)

    return {
        'P1_count': len(peaks1), 'P1_sizes': p1_sizes, 'P1_dist': avg_dist_1,
        'P2_count': len(peaks2), 'P2_sizes': p2_sizes, 'P2_dist': avg_dist_2,
        'Fixation_Time': float(fixation_time),
        'Endemic_I1_prop': endemic_I1_prop, 'Endemic_I2_prop': endemic_I2_prop,
        'peaks1_idx': peaks1, 'peaks2_idx': peaks2
    }

# =============================================================================
# 5. Dedicated Scenario Visualizations
# =============================================================================
def generate_scenario_plots(t, y, metrics, scenario_name, params, t_em):
    """
    Renders the three core graphical analyses for a specific parameter profile:
    1. Truncated Time Series (Proportional Incidence)
    2. Two-Panel Immune Landscape Stackplots
    3. Endemic Convergence Phase Portraits
    """
    N = params['N']
    S, I1, I2, R1, R2 = y / N

    # Safe LaTeX parameter string utilizing raw f-string formulation
    param_str = rf"$\beta_1$={params['beta1']}, $\beta_2$={params['beta2']}, $\eta_1$={params['eta1']}, $\eta_2$={params['eta2']}, $t_{{em}}$={t_em}"
    plot_limit = 800  # Visually isolates the transient waves

    # ---------------------------------------------------------
    # Plot 1: Active Infections (Proportional)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t, I1, label=r'Strain 1 ($I_1/N$)', color='#1f77b4', lw=2)
    plt.plot(t, I2, label=r'Strain 2 ($I_2/N$)', color='#d62728', lw=2)
    plt.scatter(t[metrics['peaks1_idx']], I1[metrics['peaks1_idx']], color='black', zorder=5, marker='x')
    plt.scatter(t[metrics['peaks2_idx']], I2[metrics['peaks2_idx']], color='black', zorder=5, marker='x')

    if metrics['Fixation_Time'] < plot_limit:
        plt.axvline(metrics['Fixation_Time'], color='green', ls='--', label=f"Fixation (Day {int(metrics['Fixation_Time'])})")

    plt.title(f"{scenario_name}: Active Infections\nParameters: {param_str}", weight='bold')
    plt.xlabel("Days")
    plt.ylabel(r"Proportion of Population ($I_i/N$)")
    plt.xlim(0, plot_limit)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{scenario_name.replace(' ', '_')}_TimeSeries.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

    # ---------------------------------------------------------
    # Plot 2: Two-Panel Immune Landscape ("The Phantom Ocean")
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top Panel: Absolute Background Population Proportions
    ax1.stackplot(t, I1, I2, S, R1, R2,
                  labels=[r'Active S1 ($I_1/N$)', r'Active S2 ($I_2/N$)', r'Naive ($S/N$)', r'Immune S1 ($R_1/N$)', r'Immune S2 ($R_2/N$)'],
                  colors=['#1f77b4', '#d62728', '#d3d3d3', '#aec7e8', '#ff9896'], alpha=0.8)
    ax1.set_title(f"{scenario_name}: Evolving Immune Landscape\nParameters: {param_str}", weight='bold')
    ax1.set_ylabel("Population Proportion")
    ax1.legend(loc='lower right', fontsize='small')
    ax1.set_xlim(0, plot_limit)
    ax1.set_ylim(0, 1.0)

    # Bottom Panel: Pure Variant Competition Scaling
    I_total = I1 + I2
    prop_I1 = np.zeros_like(I1)
    prop_I2 = np.zeros_like(I2)
    mask = I_total > 0
    prop_I1[mask] = I1[mask] / I_total[mask]
    prop_I2[mask] = I2[mask] / I_total[mask]

    # Pre-invasion boundary control to prevent NaN errors
    prop_I1[I_total == 0] = 1.0
    prop_I2[I_total == 0] = 0.0

    ax2.stackplot(t, prop_I1, prop_I2, labels=['Strain 1 Proportion', 'Strain 2 Proportion'],
                  colors=['#1f77b4', '#d62728'], alpha=0.9)
    ax2.set_title("Relative Competition: Proportion of Active Infections", weight='bold')
    ax2.set_xlabel("Days")
    ax2.set_ylabel(r"Variant Fraction ($I_i / I_{total}$)")
    ax2.legend(loc='lower right', fontsize='small')
    ax2.set_xlim(0, plot_limit)
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{scenario_name.replace(' ', '_')}_ImmuneLandscape_2Panel.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

    # ---------------------------------------------------------
    # Plot 3: Proportional Phase Portraits
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.plot(I1, I2, color='purple', lw=1, alpha=0.7)
    plt.scatter(I1[0], I2[0], color='green', label='Start (Phase 1)', zorder=5)
    plt.scatter(metrics['Endemic_I1_prop'], metrics['Endemic_I2_prop'], color='red', marker='*', s=200, label='Endemic Attractor', zorder=5)

    plt.title(f"{scenario_name}: Phase Portrait\nParameters: {param_str}", weight='bold')
    plt.xlabel(r"Strain 1 Proportion ($I_1/N$)")
    plt.ylabel(r"Strain 2 Proportion ($I_2/N$)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{scenario_name.replace(' ', '_')}_PhasePortrait.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

# =============================================================================
# 6A. Parameter Sweeps: Emergence Threshold Variance (2x3 Grids)
# =============================================================================
def run_heatmap_emergence_sweep(base_params, t_max):
    print("Initiating Multi-Emergence Heatmap Sweeps (2x3 grids). Iterating through chronologies...")
    t_em_list = [25, 50, 100, 150, 200, 250]
    eta_range = np.linspace(0.1, 2.0, 30)
    X, Y = np.meshgrid(eta_range, eta_range)

    all_fix_mats, all_peak_mats = [], []

    for t_em in t_em_list:
        print(f"  --> Processing Emergence Timing: t_em = {t_em}")
        fix_mat = np.zeros((30, 30))
        peak_mat = np.zeros((30, 30))

        for i, e1 in enumerate(eta_range):
            for j, e2 in enumerate(eta_range):
                p = base_params.copy()
                p['eta1'], p['eta2'] = e1, e2
                t, y = simulate_epidemic(p, t_em, t_max)
                mets = extract_metrics(t, y, p['N'])
                fix_mat[i, j] = mets['Fixation_Time']
                peak_mat[i, j] = mets['P2_count']

        all_fix_mats.append(fix_mat)
        all_peak_mats.append(peak_mat)

    global_fix_max = max([np.max(mat) for mat in all_fix_mats])
    global_peak_max = max([np.max(mat) for mat in all_peak_mats])

    # Constructing the 2x3 Fixation Time Grid
    fig_fix, axes_fix = plt.subplots(2, 3, figsize=(18, 11))
    fig_fix.suptitle(rf"Fixation Time Across Emergence Thresholds ($\beta_1$={base_params['beta1']}, $\beta_2$={base_params['beta2']})", fontsize=20, weight='bold')

    for idx, (t_em, fix_mat) in enumerate(zip(t_em_list, all_fix_mats)):
        ax_f = axes_fix.flatten()[idx]
        c_f = ax_f.pcolormesh(X, Y, fix_mat, shading='auto', cmap='viridis', vmin=0, vmax=global_fix_max)
        ax_f.axhline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_f.axvline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_f.set_title(rf"Emergence Time $t_{{em}} = {t_em}$")
        ax_f.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        ax_f.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")

    fig_fix.colorbar(c_f, ax=axes_fix.ravel().tolist(), label='Days to Fixation', shrink=0.95)
    fig_fix.savefig(os.path.join(output_dir, "Sweep_Emergence_FixationTime_2x3.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

    # Constructing the 2x3 Peak Count Grid
    fig_peak, axes_peak = plt.subplots(2, 3, figsize=(18, 11))
    fig_peak.suptitle(rf"Transient Oscillations Across Emergence Thresholds ($\beta_1$={base_params['beta1']}, $\beta_2$={base_params['beta2']})", fontsize=20, weight='bold')

    for idx, (t_em, peak_mat) in enumerate(zip(t_em_list, all_peak_mats)):
        ax_p = axes_peak.flatten()[idx]
        c_p = ax_p.pcolormesh(X, Y, peak_mat, shading='auto', cmap='magma', vmin=0, vmax=global_peak_max)
        ax_p.axhline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_p.axvline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_p.set_title(rf"Emergence Time $t_{{em}} = {t_em}$")
        ax_p.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        ax_p.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")

    fig_peak.colorbar(c_p, ax=axes_peak.ravel().tolist(), label='Number of S2 Peaks', shrink=0.95)
    fig_peak.savefig(os.path.join(output_dir, "Sweep_Emergence_PeakCount_2x3.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

# =============================================================================
# 6B. Parameter Sweeps: Beta Ratio Scaling (2x4 Grids)
# =============================================================================
def run_heatmap_beta_sweep(base_params, t_max, t_em_fixed=25):
    """
    Sweeps beta2 against the eta1/eta2 grid at a FIXED, locked emergence time
    (per the meeting transcripts: t_em=25 is the "worst-case" emergence scenario
    used specifically to map how the yellow strip migrates as a function of the
    beta1/beta2 ratio, independent of emergence timing).
    """
    print("Initiating Beta Ratio Heatmap Sweeps (2x4 grids). Mapping the asymmetry migration...")

    beta2_list = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    eta_range = np.linspace(0.1, 2.0, 30)
    X, Y = np.meshgrid(eta_range, eta_range)

    all_fix_mats, all_peak_mats = [], []

    for b2 in beta2_list:
        print(f"  --> Sweeping Beta 2 = {b2}...")
        fix_mat, peak_mat = np.zeros((30, 30)), np.zeros((30, 30))
        for i, e1 in enumerate(eta_range):
            for j, e2 in enumerate(eta_range):
                p = base_params.copy()
                p['beta2'], p['eta1'], p['eta2'] = b2, e1, e2

                t, y = simulate_epidemic(p, t_em_fixed, t_max)
                mets = extract_metrics(t, y, p['N'])
                fix_mat[i, j] = mets['Fixation_Time']
                peak_mat[i, j] = mets['P2_count']

        all_fix_mats.append(fix_mat)
        all_peak_mats.append(peak_mat)

    global_fix_max = max([np.max(mat) for mat in all_fix_mats])
    global_peak_max = max([np.max(mat) for mat in all_peak_mats])

    # 2x4 Fixation Time Migration Grid
    fig_fix, axes_fix = plt.subplots(2, 4, figsize=(22, 11))
    fig_fix.suptitle(rf"Migration of Fixation Delay via Relative Infectiousness (Fixed $t_{{em}}$={t_em_fixed}, $\beta_1$={base_params['beta1']})", fontsize=22, weight='bold')

    for idx, (b2, fix_mat) in enumerate(zip(beta2_list, all_fix_mats)):
        ax_f = axes_fix.flatten()[idx]
        c_f = ax_f.pcolormesh(X, Y, fix_mat, shading='auto', cmap='viridis', vmin=0, vmax=global_fix_max)
        ax_f.axhline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_f.axvline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_f.set_title(rf"$\beta_2$ = {b2} (Ratio: {b2/base_params['beta1']:.2f})")
        if idx >= 4: ax_f.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        if idx % 4 == 0: ax_f.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")

    fig_fix.colorbar(c_f, ax=axes_fix.ravel().tolist(), label='Days to Fixation', shrink=0.95)
    fig_fix.savefig(os.path.join(output_dir, "Sweep_BetaRatio_FixationTime_2x4.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

    # 2x4 Transient Oscillations Migration Grid
    fig_peak, axes_peak = plt.subplots(2, 4, figsize=(22, 11))
    fig_peak.suptitle(rf"Migration of Transient Oscillations via Relative Infectiousness (Fixed $t_{{em}}$={t_em_fixed}, $\beta_1$={base_params['beta1']})", fontsize=22, weight='bold')

    for idx, (b2, peak_mat) in enumerate(zip(beta2_list, all_peak_mats)):
        ax_p = axes_peak.flatten()[idx]
        c_p = ax_p.pcolormesh(X, Y, peak_mat, shading='auto', cmap='magma', vmin=0, vmax=global_peak_max)
        ax_p.axhline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_p.axvline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_p.set_title(rf"$\beta_2$ = {b2} (Ratio: {b2/base_params['beta1']:.2f})")
        if idx >= 4: ax_p.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        if idx % 4 == 0: ax_p.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")

    fig_peak.colorbar(c_p, ax=axes_peak.ravel().tolist(), label='Number of S2 Peaks', shrink=0.95)
    fig_peak.savefig(os.path.join(output_dir, "Sweep_BetaRatio_PeakCount_2x4.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

# =============================================================================
# 7. Main Execution Pipeline
# =============================================================================
if __name__ == "__main__":
    # Baseline temporal horizon for full convergence validation
    t_maximum = 1825

    # Universal biological parameters (eta1/eta2 here are placeholders; each
    # scenario below supplies its own eta1/eta2/t_em)
    base_params = {
        'N': 100000,
        'beta1': 0.30, 'gamma1': 0.1,
        'beta2': 0.45, 'gamma2': 0.1,
        'eta1': 1.0,   'eta2': 1.0,
        'delta': 1 / 180
    }

    # Defining target extremes to dissect the components of the "Yellow Strip".
    # t_em is now scenario-specific: the isolated time-series/landscape scenarios
    # use t_em=180 (matching Comprehensive_Metrics_Report.txt and the manuscript's
    # Table 1), while the beta-ratio sweep below independently locks t_em=25 as
    # its own fixed "worst-case emergence" condition, per the meeting transcripts.
    scenarios = {
        "Adversarial": {'eta1': 0.4, 'eta2': 0.4, 't_em': 180},
        "Cooperative": {'eta1': 1.6, 'eta2': 1.6, 't_em': 180},
        "Asymmetric Dominance": {'eta1': 0.5, 'eta2': 1.8, 't_em': 180},
        "Yellow Strip (Max Fixation Time)": {'eta1': 1.3, 'eta2': 0.1, 't_em': 180},
        "Yellow Strip (Max Peak Count)": {'eta1': 1.8, 'eta2': 0.3, 't_em': 180},
    }

    # Generates isolated graphical studies and the rigorous statistical log
    with open(os.path.join(output_dir, "Comprehensive_Metrics_Report.txt"), "w") as report:
        report.write("MULTI-STRAIN TRANSIENT DYNAMICS REPORT\n")
        report.write("=" * 80 + "\n\n")

        for name, cfg in scenarios.items():
            t_em = cfg['t_em']
            print(f"Generating high-resolution analytics for isolated scenario: {name} (t_em={t_em})")

            params = base_params.copy()
            params['eta1'], params['eta2'] = cfg['eta1'], cfg['eta2']

            # Execute Model
            t, y = simulate_epidemic(params, t_em, t_maximum)
            metrics = extract_metrics(t, y, params['N'])

            # Export Visualizations
            generate_scenario_plots(t, y, metrics, name, params, t_em)

            # Document Exact Metrics
            report.write(f"SCENARIO: {name}\n")
            report.write(f"Parameters: beta1={params['beta1']}, beta2={params['beta2']}, eta1={cfg['eta1']}, eta2={cfg['eta2']}, t_em={t_em}\n")
            report.write("-" * 80 + "\n")
            report.write(f"Strain 1: {metrics['P1_count']} Peaks | Sizes (I1/N): {metrics['P1_sizes']} | Avg Dist: {metrics['P1_dist']:.1f} days\n")
            report.write(f"Strain 2: {metrics['P2_count']} Peaks | Sizes (I2/N): {metrics['P2_sizes']} | Avg Dist: {metrics['P2_dist']:.1f} days\n")
            report.write(f"Fixation Achieved: Day {metrics['Fixation_Time']:.0f}\n")
            report.write(f"Endemic Equilibria: I1*/N = {metrics['Endemic_I1_prop']:.4f}, I2*/N = {metrics['Endemic_I2_prop']:.4f}\n\n")

    # Proceed to intensive computational multi-dimensional sweeps
    run_heatmap_emergence_sweep(base_params, t_maximum)
    run_heatmap_beta_sweep(base_params, t_maximum, t_em_fixed=25)

    print(f"\nPipeline completely executed. Output directory localized at '{output_dir}'.")
