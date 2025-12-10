import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numba import njit, prange
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 500})

gamma = 0.57721566490153286

@njit
def D(x):
    n = int(x)
    if n < 1:
        return 0
    s = int(n**0.5)
    total = 0
    for i in range(1, s + 1):
        total += n // i
    return 2 * total - s * s

@njit(parallel=True)
def compute_delta(x_arr):
    n = len(x_arr)
    delta = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        xi = x_arr[i]
        if xi >= 2:
            delta[i] = D(xi) - xi * (np.log(xi) + 2*gamma - 1)
    return delta

X_START = 1e3
X_MAX = 1e15
N = 6_500_000
WINDOW_WIDTH = 0.11
MIN_SAMPLES = 5000

print(f"Computing Δ(x) from {X_START:.0e} to {X_MAX:.0e}...")
t = np.linspace(np.log(X_START), np.log(X_MAX), N)
x = np.exp(t)
delta = compute_delta(x)
print("Δ(x) computed.")

log_log_x = []
log_var = []
sample_counts = []
i = 0
while i < len(t) - MIN_SAMPLES:
    left = t[i]
    right = left + WINDOW_WIDTH
    mask = (t >= left) & (t < right)
    
    if mask.sum() >= MIN_SAMPLES:
        var = np.var(delta[mask])
        if var > 0:
            log_log_x.append(np.mean(t[mask]))
            log_var.append(np.log(var))
            sample_counts.append(mask.sum())
        idxs = np.where(mask)[0]
        i = idxs[-1] + 1 if len(idxs) > 0 else i + 500
    else:
        i += 500
    if i >= len(t) - MIN_SAMPLES:
        break
    
log_log_x = np.array(log_log_x)
log_var = np.array(log_var)
sample_counts = np.array(sample_counts)
print(f"Number of windows: {len(log_log_x)}")

np.random.seed(42)
idx = np.random.permutation(len(log_log_x))
log_log_x = log_log_x[idx]
log_var = log_var[idx]
sample_counts = sample_counts[idx]

X = sm.add_constant(log_log_x)
weights = sample_counts / sample_counts.max()
robust = sm.WLS(log_var, X, weights=weights).fit().get_robustcov_results(cov_type='HC3')

slope = robust.params[1]
H = slope / 2
se_H = robust.bse[1] / 2
df = robust.df_resid
t_crit = stats.t.ppf(0.975, df)
ci_low = H - t_crit * se_H
ci_high = H + t_crit * se_H

bp_test = het_breuschpagan(robust.resid, robust.model.exog)
bp_pvalue = bp_test[1]

print("\n" + "="*88)
print(f"{'DIRICHLET DIVISOR PROBLEM — FINAL RESULT':^88}")
print("="*88)
print(f"Slope (2H)              : {slope:.6f}")
print(f"Hurst exponent H        : {H:.6f}")
print(f"Standard error (SE)     : {se_H:.6f}")
print(f"95% robust CI for H     : [{ci_low:.6f}, {ci_high:.6f}]")
print(f"R² (weighted)           : {robust.rsquared:.6f}")
print(f"Windows                 : {len(log_log_x)}")
print(f"Durbin–Watson           : {sm.stats.stattools.durbin_watson(robust.resid):.4f}")
print(f"Breusch–Pagan p-value   : {bp_pvalue:.6f}")
print("="*88)

order = np.argsort(log_log_x)
x_plot = log_log_x[order]
y_plot = log_var[order]
X_plot = sm.add_constant(x_plot)
pred = robust.get_prediction(X_plot)
ci = pred.summary_frame(alpha=0.05)

plt.figure(figsize=(12, 7))
plt.scatter(x_plot, y_plot, c='tab:blue', s=40, alpha=0.7, edgecolor='k', linewidth=0.4)
plt.plot(x_plot, ci['mean'], 'red', lw=3, label=f'H = {H:.5f} ± {t_crit*se_H:.5f}')
plt.fill_between(x_plot, ci['mean_ci_lower'], ci['mean_ci_upper'], color='red', alpha=0.15)
plt.xlabel(r'$\ln(\ln x)$', fontsize=14)
plt.ylabel(r'$\ln\,\mathrm{Var}[\Delta(x)]$', fontsize=14)
plt.title(f'Dirichlet Divisor Error Term — Variance Scaling\n'
          f'H = {H:.5f} (95% CI: [{ci_low:.5f}, {ci_high:.5f}])', fontsize=15)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_variance_scaling.pdf", bbox_inches='tight')
plt.savefig("plot_variance_scaling.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(x_plot, robust.resid, c='tab:purple', alpha=0.7, edgecolor='k', linewidth=0.4)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.xlabel(r'$\ln(\ln x)$', fontsize=14)
plt.ylabel('Residuals (WLS + HC3)', fontsize=14)
plt.title(r'Residuals vs $\ln(\ln x)$' + '\n(Shuffled data – autocorrelation broken)', fontsize=15)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_residuals.pdf", bbox_inches='tight')
plt.savefig("plot_residuals.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 10))
sm.qqplot(robust.resid, line='45', fit=True,
          markersize=6, markeredgecolor='k', markeredgewidth=0.4, alpha=0.7)
plt.title(r'Q-Q Plot of WLS Residuals' + '\n' +
          f'(H = {H:.5f}, 95% CI [{ci_low:.5f}, {ci_high:.5f}])', fontsize=15)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_qq.pdf", bbox_inches='tight')
plt.savefig("plot_qq.png", bbox_inches='tight')
plt.show()
