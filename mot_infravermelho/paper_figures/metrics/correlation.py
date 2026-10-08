import numpy as np
from scipy import stats

def pearson_r(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p

def spearman_rho(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    rho, p = stats.spearmanr(x[mask], y[mask])
    return rho, p

def r_squared(x, y):
    r, _ = pearson_r(x, y)
    return r**2 if not np.isnan(r) else np.nan

def confidence_interval(x, y, alpha=0.05):
    """
    Retorna uma função que, dado um array de x, retorna (lower, upper)
    para o intervalo de confiança de 95% da reta de regressão.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 3:
        return None

    z = np.polyfit(x_clean, y_clean, 1)
    slope, intercept = z[0], z[1]
    y_pred = slope * x_clean + intercept
    residuals = y_clean - y_pred
    dof = len(x_clean) - 2
    t_val = stats.t.ppf(1 - alpha/2, dof)

    se = np.sqrt(np.sum(residuals**2) / dof)
    mean_x = np.mean(x_clean)
    Sxx = np.sum((x_clean - mean_x)**2)
    n = len(x_clean)

    def ci_lower_upper(x_new):
        x_new = np.asarray(x_new)
        y_new = slope * x_new + intercept
        se_pred = se * np.sqrt(1/n + (x_new - mean_x)**2 / Sxx)
        lower = y_new - t_val * se_pred
        upper = y_new + t_val * se_pred
        return lower, upper

    return ci_lower_upper