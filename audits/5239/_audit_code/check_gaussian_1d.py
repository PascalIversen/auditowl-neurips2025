"""Reproduces the 1D-Gaussian core numerics of shrinkage_kde_gaussian.py without
the matplotlib/usetex plotting path, to (1) check Figure-2 fitted slopes and
(2) check the Figure-3 histogram statistics, comparing n=100 (paper caption)
vs n=200 (code's n_example). Supports findings fig3-n-mismatch and the
traceability rows for Figures 2 and 3. Read-only: copies the author's exact
functions; does not import/modify the repo file (it has usetex at import)."""
import numpy as np
import os
from scipy.stats import norm

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

mixture_params_list = [
    {'pi': 0.4, 'mu1': -2, 'sigma1': 0.5, 'mu2':  2, 'sigma2': 1.0},
    {'pi': 0.3, 'mu1': -2, 'sigma1': 0.4, 'mu2':  4, 'sigma2': 1.5},
    {'pi': 0.5, 'mu1':  0, 'sigma1': 0.4, 'mu2':  1.5, 'sigma2': 1.5},
]

# ---- verbatim copies of the author's functions (shrinkage_kde_gaussian.py) ----
def mixture_pdf(x, params):
    pi_ = params['pi']
    return pi_*norm.pdf(x, params['mu1'], params['sigma1']) \
         + (1-pi_)*norm.pdf(x, params['mu2'], params['sigma2'])

def sample_from_mixture(n, params):
    pi_ = params['pi']
    z = np.random.rand(n) < pi_
    x_samps = np.zeros(n)
    x_samps[z]  = np.random.normal(params['mu1'], params['sigma1'], size=z.sum())
    x_samps[~z] = np.random.normal(params['mu2'], params['sigma2'], size=(~z).sum())
    return x_samps

def score_function(x, params, noise_std=0.0):
    p_x = mixture_pdf(x, params)
    pi_ = params['pi']
    mu1, s1 = params['mu1'], params['sigma1']
    mu2, s2 = params['mu2'], params['sigma2']
    d_comp1 = pi_*norm.pdf(x, mu1, s1)*((mu1 - x)/(s1**2))
    d_comp2 = (1-pi_)*norm.pdf(x, mu2, s2)*((mu2 - x)/(s2**2))
    dp_dx   = d_comp1 + d_comp2
    base_score = dp_dx / (p_x + 1e-15)
    if noise_std>0:
        return base_score + np.random.normal(0, noise_std, size=x.shape)
    return base_score

def silverman_bandwidth(data):
    n = len(data)
    std_dev = np.std(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    sigma = min(std_dev, iqr / 1.34)
    return 0.9 * sigma * n**(-1/5)

def one_step_debiased_data(x, params, noise_std=0.0):
    n = len(x)
    std_dev = np.std(x)
    iqr     = np.percentile(x, 75) - np.percentile(x, 25)
    sigma   = min(std_dev, iqr / 1.34)
    h = 0.4*sigma*n**(-1/9)
    delta = (h**2)/2.0
    s_x = score_function(x, params, noise_std=noise_std)
    return x + delta*s_x, h

def kde_pdf_eval(x_points, data, bandwidth):
    M = x_points.size
    N = data.size
    z = (x_points.reshape(M,1)-data.reshape(1,N))/bandwidth
    pdf_mat = (1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*z**2)
    return pdf_mat.mean(axis=1)/bandwidth

def approximate_mise(data, bandwidth, params, x_min=-8, x_max=8, step=0.05):
    x_grid = np.arange(x_min, x_max+step, step)
    p_vals = mixture_pdf(x_grid, params)
    q_vals = kde_pdf_eval(x_grid, data, bandwidth)
    return np.sum((p_vals - q_vals)**2)*step
# -----------------------------------------------------------------------------

def fig3_stats(n_example, n_seeds=50):
    """Histogram of MISE(Silverman)-MISE(SD-KDE, std=0); mean/std and frac>0."""
    rows = []
    for i, params in enumerate(mixture_params_list):
        diffs = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            x_data = sample_from_mixture(n_example, params)
            h_silv = silverman_bandwidth(x_data)
            mise_silv = approximate_mise(x_data, h_silv, params)
            x_deb, h_deb = one_step_debiased_data(x_data, params, noise_std=0)
            mise_deb = approximate_mise(x_deb, h_deb, params)
            diffs.append(mise_silv - mise_deb)
        diffs = np.array(diffs)
        frac_better = float(np.mean(diffs > 0))
        rows.append((i+1, n_example, float(diffs.mean()), float(diffs.std()), frac_better))
    return rows

def fig2_slopes(n_list, n_seeds=50):
    """Fitted log-log slope of mean MISE vs n for Silverman and SD-KDE std=0,2,4."""
    out = {}
    for i, params in enumerate(mixture_params_list):
        avg = {k: [] for k in ['silv', 's0', 's2', 's4']}
        for n_data in n_list:
            vals = {k: [] for k in ['silv', 's0', 's2', 's4']}
            for seed in range(n_seeds):
                np.random.seed(seed)
                x = sample_from_mixture(n_data, params)
                hs = silverman_bandwidth(x)
                vals['silv'].append(approximate_mise(x, hs, params))
                for key, nl in [('s0', 0), ('s2', 2), ('s4', 4)]:
                    xd, hd = one_step_debiased_data(x, params, noise_std=nl)
                    vals[key].append(approximate_mise(xd, hd, params))
            for k in avg:
                avg[k].append(np.mean(vals[k]))
        log_n = np.log(n_list)
        out[i+1] = {k: float(np.polyfit(log_n, np.log(np.array(v)+1e-15), 1)[0]) for k, v in avg.items()}
    return out

if __name__ == "__main__":
    # Figure 3 statistics at the code's n_example=200 AND the paper caption's n=100
    print("=== Figure 3 histogram stats: mean / std / frac(SD-KDE better) ===")
    with open(os.path.join(OUT, "fig3_stats.csv"), "w") as f:
        f.write("mixture,n,mean_diff,std_diff,frac_sdkde_better\n")
        for n_ex in (200, 100):
            for (mix, n, m, s, fr) in fig3_stats(n_ex, n_seeds=50):
                line = f"{mix},{n},{m:.4f},{s:.4f},{fr:.3f}"
                print("  " + line)
                f.write(line + "\n")

    # Figure 2 slopes (use a reduced but representative n_list for speed)
    n_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    print("\n=== Figure 2 fitted slopes (n_list up to 5000, 50 seeds) ===")
    slopes = fig2_slopes(n_list, n_seeds=50)
    with open(os.path.join(OUT, "fig2_slopes.csv"), "w") as f:
        f.write("mixture,silv,sd0,sd2,sd4\n")
        for mix, d in slopes.items():
            line = f"{mix},{d['silv']:.3f},{d['s0']:.3f},{d['s2']:.3f},{d['s4']:.3f}"
            print("  mixture", line)
            f.write(line + "\n")
    print("\n(paper Fig.2 reported slopes: M1 silv -0.54 sd0 -0.85 sd2 -0.82 sd4 -0.72;"
          " M2 -0.36/-0.58/-0.50/-0.39; M3 -0.63/-0.93/-0.92/-0.91)")
