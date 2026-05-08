import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from src import generate_scenarios

LAGS = 20
L = 13   # your chosen L
N_PATHS = 500

hist_returns = historical_returns['US Ten Stocks'].values
sc = generate_scenarios(
    {'strat': historical_returns['US Ten Stocks']},
    n_paths=N_PATHS, path_length=1260, L_mean=L, seed=42
)

# Historical ACF — stable single estimate
hist_acf    = acf(hist_returns,    nlags=LAGS)[1:]
hist_sq_acf = acf(hist_returns**2, nlags=LAGS)[1:]

# Simulated ACF — compute per path then summarise
sim_acfs    = np.array([acf(sc['paths']['strat'][p],    nlags=LAGS)[1:]
                        for p in range(N_PATHS)])
sim_sq_acfs = np.array([acf(sc['paths']['strat'][p]**2, nlags=LAGS)[1:]
                        for p in range(N_PATHS)])

sim_acf_mean    = sim_acfs.mean(axis=0)
sim_acf_p05     = np.quantile(sim_acfs,    0.05, axis=0)
sim_acf_p95     = np.quantile(sim_acfs,    0.95, axis=0)

sim_sq_acf_mean = sim_sq_acfs.mean(axis=0)
sim_sq_acf_p05  = np.quantile(sim_sq_acfs, 0.05, axis=0)
sim_sq_acf_p95  = np.quantile(sim_sq_acfs, 0.95, axis=0)

lags = np.arange(1, LAGS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, hist_vals, sim_mean, sim_lo, sim_hi, label in [
    (axes[0], hist_acf,    sim_acf_mean,    sim_acf_p05,    sim_acf_p95,
     'Returns ACF'),
    (axes[1], hist_sq_acf, sim_sq_acf_mean, sim_sq_acf_p05, sim_sq_acf_p95,
     'Squared Returns ACF (vol clustering)'),
]:
    # Historical — solid line
    ax.plot(lags, hist_vals, color='black', lw=2, marker='o',
            markersize=4, label='Historical')
    # Simulated — mean + 90% band across paths
    ax.plot(lags, sim_mean, color='steelblue', lw=2, ls='--',
            marker='o', markersize=4, label=f'Sim mean (L={L})')
    ax.fill_between(lags, sim_lo, sim_hi, color='steelblue', alpha=0.2,
                    label='Sim 5–95% band (path variability)')
    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(label)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle(f'Bootstrap validation: historical vs simulated ACF  (L={L}, {N_PATHS} paths)',
             fontsize=11)
plt.tight_layout()
plt.show()