import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from datetime import datetime

# =============================================================================
# 1. System Setup and Directory Management
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"model_outputs_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print(f"Initializing Comprehensive Multi-Scenario Epidemic Pipeline...")
print(f"All outputs will be saved to: ./{output_dir}/\n")

# =============================================================================
# 2. Core ODE System
# =============================================================================
def two_strain_inner_loop(t, y, beta1, beta2, gamma1, gamma2, eta1, eta2, delta, N):
    S, I1, I2, R1, R2 = y
    
    # Forces of infection
    lambda1 = beta1 * I1 / N
    lambda2 = beta2 * I2 / N
    
    # Differential Equations
    dSdt  = -lambda1 * S - lambda2 * S + delta * R1 + delta * R2
    dI1dt = lambda1 * S + eta1 * lambda1 * R2 - gamma1 * I1
    dI2dt = lambda2 * S + eta2 * lambda2 * R1 - gamma2 * I2
    dR1dt = gamma1 * I1 - eta2 * lambda2 * R1 - delta * R1
    dR2dt = gamma2 * I2 - eta1 * lambda1 * R2 - delta * R2
    
    return [dSdt, dI1dt, dI2dt, dR1dt, dR2dt]

# =============================================================================
# 3. Two-Phase Simulation & Mathematical Conservation Check
# =============================================================================
def simulate_epidemic(params, t_em, t_max):
    N = params['N']
    y0_phase1 = [N - 10, 10, 0, 0, 0] 
    
    ode_args = (params['beta1'], params['beta2'], params['gamma1'], 
                params['gamma2'], params['eta1'], params['eta2'], 
                params['delta'], N)
    
    # Phase 1: Establish Strain 1
    sol1 = solve_ivp(two_strain_inner_loop, [0, t_em], y0_phase1, args=ode_args, 
                     method='Radau', dense_output=True, max_step=1.0)
    
    # Phase 2: Seed Variant 2
    y0_phase2 = sol1.y[:, -1].copy()
    seed_size = 10
    if y0_phase2[0] > seed_size:
        y0_phase2[0] -= seed_size 
    else:
        y0_phase2[3] -= seed_size 
    y0_phase2[2] += seed_size
    
    # Ensure long run time to capture full mathematical convergence
    sol2 = solve_ivp(two_strain_inner_loop, [t_em, t_max], y0_phase2, args=ode_args, 
                     method='Radau', dense_output=True, max_step=1.0)
    
    t_combined = np.concatenate((sol1.t, sol2.t[1:]))
    y_combined = np.concatenate((sol1.y, sol2.y[:, 1:]), axis=1)
    
    # MATHEMATICAL CHECK: Prove N is strictly constant (dN/dt = 0)
    total_population = np.sum(y_combined, axis=0)
    if not np.allclose(total_population, N, atol=1e-5):
        print("WARNING: Population conservation violated!")
        
    return t_combined, y_combined

# =============================================================================
# 4. Rigorous Metric Extraction (Updated for Proportions)
# =============================================================================
def extract_metrics(t, y, N, epsilon=1e-5, window=30):
    I1, I2 = y[1], y[2]
    
    # Extract peaks
    peaks1, _ = find_peaks(I1, prominence=N*0.0005, distance=14)
    peaks2, _ = find_peaks(I2, prominence=N*0.0005, distance=14)
    
    avg_dist_1 = np.mean(np.diff(t[peaks1])) if len(peaks1) > 1 else 0
    avg_dist_2 = np.mean(np.diff(t[peaks2])) if len(peaks2) > 1 else 0
    
    # Calculate Fixation Time
    fixation_time = t[-1] 
    for i in range(window, len(t)):
        dI1 = np.abs(np.diff(I1[i-window:i]))
        dI2 = np.abs(np.diff(I2[i-window:i]))
        if np.all(dI1 < epsilon * N) and np.all(dI2 < epsilon * N):
            fixation_time = t[i]
            break
            
    # Calculate Endemic Equilibria as a Proportion of N
    endemic_I1_prop = np.mean(I1[-50:]) / N
    endemic_I2_prop = np.mean(I2[-50:]) / N
            
    return {
        'P1_count': len(peaks1), 'P1_max_prop': (np.max(I1)/N) if len(peaks1)>0 else 0, 'P1_dist': avg_dist_1,
        'P2_count': len(peaks2), 'P2_max_prop': (np.max(I2)/N) if len(peaks2)>0 else 0, 'P2_dist': avg_dist_2,
        'Fixation_Time': fixation_time,
        'Endemic_I1_prop': endemic_I1_prop, 'Endemic_I2_prop': endemic_I2_prop,
        'peaks1_idx': peaks1, 'peaks2_idx': peaks2
    }

# =============================================================================
# 5. Advanced Visualization Generation (Truncated Time & 2-Panel Landscape)
# =============================================================================
def generate_scenario_plots(t, y, metrics, scenario_name, params, t_em):
    N = params['N']
    # Convert all compartments to proportions of total population
    S, I1, I2, R1, R2 = y / N  
    
    param_str = f"Parameters: $\\eta_1$={params['eta1']}, $\\eta_2$={params['eta2']}, $t_{{em}}$={t_em}"
    plot_limit = 800 # Truncate x-axis to focus on transients
    
    # -------------------------------------------------------------------------
    # Plot 1: Time Series (Proportions)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t, I1, label='Strain 1 ($I_1/N$)', color='#1f77b4', lw=2)
    plt.plot(t, I2, label='Strain 2 ($I_2/N$)', color='#d62728', lw=2)
    plt.scatter(t[metrics['peaks1_idx']], I1[metrics['peaks1_idx']], color='black', zorder=5, marker='x')
    plt.scatter(t[metrics['peaks2_idx']], I2[metrics['peaks2_idx']], color='black', zorder=5, marker='x')
    if metrics['Fixation_Time'] < plot_limit:
        plt.axvline(metrics['Fixation_Time'], color='green', ls='--', label=f"Fixation (Day {int(metrics['Fixation_Time'])})")
    
    plt.title(f"{scenario_name}: Active Infections\n{param_str}", weight='bold')
    plt.xlabel("Days")
    plt.ylabel("Proportion of Population ($I_i/N$)")
    plt.xlim(0, plot_limit)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{scenario_name.replace(' ', '_')}_TimeSeries.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 2: Two-Panel Immune Landscape
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Panel 1: Stackplot of Population Proportions
    ax1.stackplot(t, I1, I2, S, R1, R2, 
                  labels=['Active S1 ($I_1/N$)', 'Active S2 ($I_2/N$)', 'Naive ($S/N$)', 'Immune S1 ($R_1/N$)', 'Immune S2 ($R_2/N$)'], 
                  colors=['#1f77b4', '#d62728', '#d3d3d3', '#aec7e8', '#ff9896'], alpha=0.8)
    ax1.set_title(f"{scenario_name}: Evolving Immune Landscape\n{param_str}", weight='bold')
    ax1.set_ylabel("Population Proportion")
    ax1.legend(loc='lower right', fontsize='small')
    ax1.set_xlim(0, plot_limit)
    ax1.set_ylim(0, 1.0)
    
    # Panel 2: Stackplot of Variant Replacements (I1/Itotal and I2/Itotal)
    I_total = I1 + I2
    prop_I1 = np.zeros_like(I1)
    prop_I2 = np.zeros_like(I2)
    mask = I_total > 0
    prop_I1[mask] = I1[mask] / I_total[mask]
    prop_I2[mask] = I2[mask] / I_total[mask]
    
    # Handle the initial Phase 1 correctly (100% Strain 1)
    prop_I1[I_total == 0] = 1.0
    prop_I2[I_total == 0] = 0.0

    ax2.stackplot(t, prop_I1, prop_I2, labels=['Strain 1 Proportion', 'Strain 2 Proportion'], 
                  colors=['#1f77b4', '#d62728'], alpha=0.9)
    ax2.set_title("Relative Competition: Proportion of Active Infections", weight='bold')
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Variant Fraction ($I_i / I_{total}$)")
    ax2.legend(loc='lower right', fontsize='small')
    ax2.set_xlim(0, plot_limit)
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{scenario_name.replace(' ', '_')}_ImmuneLandscape_2Panel.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 3: Phase Portrait (Proportions)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.plot(I1, I2, color='purple', lw=1, alpha=0.7)
    plt.scatter(I1[0], I2[0], color='green', label='Start (Phase 1)', zorder=5)
    plt.scatter(metrics['Endemic_I1_prop'], metrics['Endemic_I2_prop'], color='red', marker='*', s=200, label='Endemic Attractor', zorder=5)
    plt.title(f"{scenario_name}: Phase Portrait\n{param_str}", weight='bold')
    plt.xlabel("Strain 1 Proportion ($I_1/N$)")
    plt.ylabel("Strain 2 Proportion ($I_2/N$)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{scenario_name.replace(' ', '_')}_PhasePortrait.png"), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 6. Parameter Sweeps (2x2 Grids for varying Emergence Times)
# =============================================================================
def run_heatmap_emergence_sweep(base_params, t_max):
    print("Initiating Multi-Emergence Heatmap Sweeps (2x2 grids). This will take a moment...")
    
    t_em_list = [50, 100, 200, 250]
    eta_range = np.linspace(0.1, 2.0, 30) # 30x30 resolution per panel
    X, Y = np.meshgrid(eta_range, eta_range)
    
    fig_fix, axes_fix = plt.subplots(2, 2, figsize=(14, 12))
    fig_peak, axes_peak = plt.subplots(2, 2, figsize=(14, 12))
    
    fig_fix.suptitle("Fixation Time Across Emergence Thresholds", fontsize=18, weight='bold')
    fig_peak.suptitle("Transient Oscillations Across Emergence Thresholds", fontsize=18, weight='bold')
    
    for idx, t_em in enumerate(t_em_list):
        print(f"  --> Sweeping t_em = {t_em}...")
        fix_mat = np.zeros((30, 30))
        peak_mat = np.zeros((30, 30))
        
        for i, e1 in enumerate(eta_range):
            for j, e2 in enumerate(eta_range):
                p = base_params.copy()
                p['eta1'] = e1
                p['eta2'] = e2
                
                t, y = simulate_epidemic(p, t_em, t_max)
                mets = extract_metrics(t, y, base_params['N'])
                
                fix_mat[i, j] = mets['Fixation_Time']
                peak_mat[i, j] = mets['P2_count']
                
        # Fixation Time Panel
        ax_f = axes_fix.flatten()[idx]
        c_f = ax_f.pcolormesh(X, Y, fix_mat, shading='auto', cmap='viridis')
        ax_f.axhline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_f.axvline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_f.set_title(f"Emergence Time $t_{{em}} = {t_em}$")
        # FIXED LABELS: Changed S1/S2 to I1/I2 as requested
        ax_f.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        ax_f.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")
        fig_fix.colorbar(c_f, ax=ax_f, label='Days to Fixation')
        
        # Peak Count Panel
        ax_p = axes_peak.flatten()[idx]
        c_p = ax_p.pcolormesh(X, Y, peak_mat, shading='auto', cmap='magma')
        ax_p.axhline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_p.axvline(1.0, color='white', linestyle='--', linewidth=1.5)
        ax_p.set_title(f"Emergence Time $t_{{em}} = {t_em}$")
        # FIXED LABELS: Changed S1/S2 to I1/I2 as requested
        ax_p.set_xlabel(r"$\eta_2$ ($R_1$ vulnerability to $I_2$)")
        ax_p.set_ylabel(r"$\eta_1$ ($R_2$ vulnerability to $I_1$)")
        fig_peak.colorbar(c_p, ax=ax_p, label='Number of S2 Peaks')

    fig_fix.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_peak.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig_fix.savefig(os.path.join(output_dir, "Sweep_FixationTime_2x2.png"), dpi=300)
    fig_peak.savefig(os.path.join(output_dir, "Sweep_PeakCount_2x2.png"), dpi=300)
    plt.close('all')

# =============================================================================
# 7. Main Execution Flow
# =============================================================================
if __name__ == "__main__":
    t_emergence_primary = 180  
    t_maximum = 1825   
    
    base_params = {
        'N': 100000,
        'beta1': 0.30, 'gamma1': 0.1,  
        'beta2': 0.45, 'gamma2': 0.1,  
        'eta1': 1.0,   'eta2': 1.0,
        'delta': 1/180                 
    }
    
    scenarios = {
        "Adversarial": {'eta1': 0.4, 'eta2': 0.4},
        "Cooperative": {'eta1': 1.6, 'eta2': 1.6},
        "Asymmetric":  {'eta1': 0.5, 'eta2': 1.8}
    }
    
    # Generate Time Series, Phase Portraits, and 2-Panel Landscapes
    with open(os.path.join(output_dir, "Comprehensive_Metrics_Report.txt"), "w") as report:
        report.write("MULTI-STRAIN TRANSIENT DYNAMICS REPORT\n")
        report.write("="*50 + "\n\n")
        
        for name, etas in scenarios.items():
            print(f"Generating dynamic plots for scenario: {name}")
            params = base_params.copy()
            params.update(etas)
            t, y = simulate_epidemic(params, t_emergence_primary, t_maximum)
            metrics = extract_metrics(t, y, params['N'])
            
            generate_scenario_plots(t, y, metrics, name, params, t_emergence_primary)
            
            # Log exact metrics (Now natively saving proportions)
            report.write(f"SCENARIO: {name} (eta1={etas['eta1']}, eta2={etas['eta2']}, t_em={t_emergence_primary})\n")
            report.write("-" * 50 + "\n")
            report.write(f"Strain 1: {metrics['P1_count']} Peaks | Max Peak (I1/N): {metrics['P1_max_prop']:.4f}\n")
            report.write(f"Strain 2: {metrics['P2_count']} Peaks | Max Peak (I2/N): {metrics['P2_max_prop']:.4f}\n")
            report.write(f"Fixation Achieved: Day {metrics['Fixation_Time']:.0f}\n")
            report.write(f"Endemic Equilibria: I1*/N = {metrics['Endemic_I1_prop']:.4f}, I2*/N = {metrics['Endemic_I2_prop']:.4f}\n\n")

    # Generate the requested 2x2 Heatmap Grids
    run_heatmap_emergence_sweep(base_params, t_maximum)
    
    print(f"\nPipeline Complete! All outputs saved to '{output_dir}'.")
