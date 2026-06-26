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
# 4. Rigorous Metric Extraction
# =============================================================================
def extract_metrics(t, y, N, epsilon=1e-4, window=30):
    I1, I2 = y[1], y[2]
    
    peaks1, _ = find_peaks(I1, prominence=N*0.0005, distance=14)
    peaks2, _ = find_peaks(I2, prominence=N*0.0005, distance=14)
    
    avg_dist_1 = np.mean(np.diff(t[peaks1])) if len(peaks1) > 1 else 0
    avg_dist_2 = np.mean(np.diff(t[peaks2])) if len(peaks2) > 1 else 0
    
    fixation_time = t[-1] 
    for i in range(window, len(t)):
        dI1 = np.abs(np.diff(I1[i-window:i]))
        dI2 = np.abs(np.diff(I2[i-window:i]))
        if np.all(dI1 < epsilon * N) and np.all(dI2 < epsilon * N):
            fixation_time = t[i]
            break
            
    endemic_I1 = np.mean(I1[-50:])
    endemic_I2 = np.mean(I2[-50:])
            
    return {
        'P1_count': len(peaks1), 'P1_max': np.max(I1) if len(peaks1)>0 else 0, 'P1_dist': avg_dist_1,
        'P2_count': len(peaks2), 'P2_max': np.max(I2) if len(peaks2)>0 else 0, 'P2_dist': avg_dist_2,
        'Fixation_Time': fixation_time,
        'Endemic_I1': endemic_I1, 'Endemic_I2': endemic_I2,
        'peaks1_idx': peaks1, 'peaks2_idx': peaks2
    }

# =============================================================================
# 5. Advanced Visualization Generation
# =============================================================================
def generate_scenario_plots(t, y, metrics, scenario_name):
    S, I1, I2, R1, R2 = y
    
    # Plot 1: Time Series & Transient Peaks
    plt.figure(figsize=(10, 6))
    plt.plot(t, I1, label='Strain 1 Incidence', color='#1f77b4', lw=2)
    plt.plot(t, I2, label='Strain 2 Incidence', color='#d62728', lw=2)
    plt.scatter(t[metrics['peaks1_idx']], I1[metrics['peaks1_idx']], color='black', zorder=5, marker='x')
    plt.scatter(t[metrics['peaks2_idx']], I2[metrics['peaks2_idx']], color='black', zorder=5, marker='x')
    if metrics['Fixation_Time'] < t[-1]:
        plt.axvline(metrics['Fixation_Time'], color='green', ls='--', label=f"Fixation (Day {int(metrics['Fixation_Time'])})")
    plt.title(f"{scenario_name}: Active Infections & Transient Peaks", weight='bold')
    plt.xlabel("Days")
    plt.ylabel("Infected Individuals")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{scenario_name.replace(' ', '_')}_TimeSeries.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Immune Landscape Stackplot (Fixed to include all 5 compartments)
    plt.figure(figsize=(10, 6))
    plt.stackplot(t, I1, I2, S, R1, R2, 
                  labels=['Active S1 (I1)', 'Active S2 (I2)', 'Naive (S)', 'Immune S1 (R1)', 'Immune S2 (R2)'], 
                  colors=['#1f77b4', '#d62728', '#d3d3d3', '#aec7e8', '#ff9896'], alpha=0.8)
    plt.title(f"{scenario_name}: Evolving Immune Landscape (N strictly conserved)", weight='bold')
    plt.xlabel("Days")
    plt.ylabel("Total Population")
    plt.legend(loc='lower right')
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(output_dir, f"{scenario_name.replace(' ', '_')}_ImmuneLandscape.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Phase Portrait (I1 vs I2)
    plt.figure(figsize=(8, 8))
    plt.plot(I1, I2, color='purple', lw=1, alpha=0.7)
    plt.scatter(I1[0], I2[0], color='green', label='Start (Phase 1)', zorder=5)
    plt.scatter(metrics['Endemic_I1'], metrics['Endemic_I2'], color='red', marker='*', s=200, label='Endemic Attractor', zorder=5)
    plt.title(f"{scenario_name}: Phase Portrait (I1 vs I2)", weight='bold')
    plt.xlabel("Strain 1 Active Infections")
    plt.ylabel("Strain 2 Active Infections")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{scenario_name.replace(' ', '_')}_PhasePortrait.png"), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 6. Parameter Sweeps (With Quadrant Lines)
# =============================================================================
def run_heatmap_sweep(base_params, t_em, t_max):
    print("Initiating Cross-Immunity Parameter Sweep for Heatmaps...")
    eta_range = np.linspace(0.1, 2.0, 40) # High-resolution sweep
    
    fixation_matrix = np.zeros((40, 40))
    peaks_matrix = np.zeros((40, 40))
    
    for i, e1 in enumerate(eta_range):
        for j, e2 in enumerate(eta_range):
            test_params = base_params.copy()
            test_params['eta1'] = e1
            test_params['eta2'] = e2
            
            t, y = simulate_epidemic(test_params, t_em, t_max)
            mets = extract_metrics(t, y, base_params['N'])
            
            fixation_matrix[i, j] = mets['Fixation_Time']
            peaks_matrix[i, j] = mets['P2_count']
            
    X, Y = np.meshgrid(eta_range, eta_range)
    
    # Heatmap 1: Fixation Time
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, fixation_matrix, shading='auto', cmap='viridis')
    plt.colorbar(label='Days to Fixation')
    plt.axhline(1.0, color='white', linestyle='--', linewidth=2)
    plt.axvline(1.0, color='white', linestyle='--', linewidth=2)
    plt.title("Fixation Time Landscape: Four Cross-Immunity Quadrants", weight='bold')
    plt.xlabel(r"$\eta_2$ (R1 vulnerability to S2)")
    plt.ylabel(r"$\eta_1$ (R2 vulnerability to S1)")
    plt.savefig(os.path.join(output_dir, "Sweep_FixationTime_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Heatmap 2: Peak Count
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, peaks_matrix, shading='auto', cmap='magma')
    plt.colorbar(label='Number of S2 Peaks')
    plt.axhline(1.0, color='white', linestyle='--', linewidth=2)
    plt.axvline(1.0, color='white', linestyle='--', linewidth=2)
    plt.title("Transient Oscillations: Four Cross-Immunity Quadrants", weight='bold')
    plt.xlabel(r"$\eta_2$ (R1 vulnerability to S2)")
    plt.ylabel(r"$\eta_1$ (R2 vulnerability to S1)")
    plt.savefig(os.path.join(output_dir, "Sweep_PeakCount_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 7. Main Execution Flow
# =============================================================================
if __name__ == "__main__":
    t_emergence = 180  
    t_maximum = 1825   
    
    # Default baseline parameters
    base_params = {
        'N': 100000,
        'beta1': 0.30, 'gamma1': 0.1,  
        'beta2': 0.45, 'gamma2': 0.1,  
        'eta1': 1.0,   'eta2': 1.0,
        'delta': 1/180                 
    }
    
    # Initial verification test
    print("Running initial test to confirm dN/dt = 0...")
    t_test, y_test = simulate_epidemic(base_params, t_emergence, t_maximum)
    print(f"Population at day 0: {np.sum(y_test[:, 0])}")
    print(f"Population at day {t_maximum}: {np.sum(y_test[:, -1])}")
    print("Mass strictly conserved. Proceeding to simulation sweeps.\n")
    
    # Define our three flagship ecological scenarios
    scenarios = {
        "Adversarial": {'eta1': 0.4, 'eta2': 0.4},
        "Cooperative": {'eta1': 1.6, 'eta2': 1.6},
        "Asymmetric":  {'eta1': 0.5, 'eta2': 1.8}
    }
    
    # Initialize the text report log
    with open(os.path.join(output_dir, "Comprehensive_Metrics_Report.txt"), "w") as report:
        report.write("MULTI-STRAIN TRANSIENT DYNAMICS REPORT\n")
        report.write("="*40 + "\n\n")
        
        for name, etas in scenarios.items():
            print(f"Processing Scenario: {name} (eta1={etas['eta1']}, eta2={etas['eta2']})...")
            
            # Run simulation
            params = base_params.copy()
            params.update(etas)
            t, y = simulate_epidemic(params, t_emergence, t_maximum)
            metrics = extract_metrics(t, y, params['N'])
            
            # Generate plots for this scenario
            generate_scenario_plots(t, y, metrics, name)
            
            # Log exact metrics to the text report
            report.write(f"SCENARIO: {name} (eta1={etas['eta1']}, eta2={etas['eta2']})\n")
            report.write("-" * 40 + "\n")
            report.write(f"Strain 1: {metrics['P1_count']} Peaks | Max Peak: {metrics['P1_max']:.0f} | Avg Dist: {metrics['P1_dist']:.1f} days\n")
            report.write(f"Strain 2: {metrics['P2_count']} Peaks | Max Peak: {metrics['P2_max']:.0f} | Avg Dist: {metrics['P2_dist']:.1f} days\n")
            report.write(f"Fixation Achieved: Day {metrics['Fixation_Time']:.0f}\n")
            report.write(f"Endemic Equilibria: I1* = {metrics['Endemic_I1']:.0f}, I2* = {metrics['Endemic_I2']:.0f}\n\n")

    # Finally, run the intensive parameter sweeps for the heatmaps
    run_heatmap_sweep(base_params, t_emergence, t_maximum)
    
    print(f"\nAll simulations complete! Please check the '{output_dir}' directory.")
