import pandas as pd
import numpy as np
import scipy.special as sc
from scipy.optimize import minimize

def safe_zscore(series: pd.Series) -> pd.Series:
    """
    Standardizes valid observations while preserving missing values.
    Constant-valued metrics (or those with microscopic float variance) receive a neutral Z-score of zero.
    """
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.notna()

    if valid.sum() < 2:
        return result

    values = series[valid].astype(float)
    std = values.std(ddof=1)

    # Numerical tolerance check for microscopic float variance
    if std < 1e-9:
        result.loc[valid] = 0.0
    else:
        result.loc[valid] = (values - values.mean()) / std

    return result

def estimate_beta_binomial_priors(df: pd.DataFrame, dismissals_col: str, opps_col: str):
    """
    Estimates Beta prior parameters (alpha, beta) using a Beta-Binomial 
    maximum-likelihood fit. If MLE fails, gracefully falls back to the Method of Moments (MoM)
    accounting for binomial sampling variance attenuation.
    """
    valid_df = df[df[opps_col] > 0].copy()
    if len(valid_df) < 2:
        return np.nan, np.nan
        
    counts = valid_df[[dismissals_col, opps_col]].to_numpy(dtype=float)

    if (
        not np.isfinite(counts).all()
        or (counts[:, 0] < 0).any()
        or (counts[:, 1] <= 0).any()
        or (counts[:, 0] > counts[:, 1]).any()
        or not np.equal(counts, np.floor(counts)).all()
    ):
        return np.nan, np.nan
        
    k = counts[:, 0]
    n = counts[:, 1]
    
    # 1. MLE Attempt
    def neg_log_lik(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return np.inf
        ll = np.sum(
            sc.gammaln(k + alpha) + sc.gammaln(n - k + beta) - sc.gammaln(n + alpha + beta)
            + sc.gammaln(alpha + beta) - sc.gammaln(alpha) - sc.gammaln(beta)
        )
        return -ll

    # Pooled mean for MLE initialization
    mu = k.sum() / n.sum() if n.sum() > 0 else 0.5
    a0, b0 = mu * 10, (1 - mu) * 10
    
    if 0 < mu < 1:
        res = minimize(
            neg_log_lik, 
            [a0, b0], 
            bounds=[(1e-5, None), (1e-5, None)], 
            method='L-BFGS-B'
        )
        
        if (
            res.success
            and np.isfinite(res.fun)
            and np.isfinite(res.x).all()
            and res.x[0] > 1e-5
            and res.x[1] > 1e-5
        ):
            print("Beta-Binomial Priors generated via MLE")
            return res.x[0], res.x[1]

    # 2. Method of Moments (MoM) Fallback (Corrected for Attenuation)
    rates = k / n
    # Player-level weighting to align with the variance calculation
    p_bar = np.mean(rates)
    v_obs = np.var(rates, ddof=1)
    
    q = np.mean(1.0 / n)
    
    # Guardrail 1: q must be strictly between 0 and 1
    if not (0 < q < 1):
        print("Beta-Binomial Priors failed: q is not strictly between 0 and 1")
        return np.nan, np.nan
        
    # Estimate true between-player variance (sigma_p_squared)
    sigma_p_sq = (v_obs - (p_bar * (1 - p_bar) * q)) / (1 - q)
    
    # Guardrail 2: True variance must be finite and strictly positive
    if not np.isfinite(sigma_p_sq) or sigma_p_sq <= 0:
        print(f"Beta-Binomial Priors failed: sigma_p_sq ({sigma_p_sq:.6f}) is non-positive or infinite")
        return np.nan, np.nan
        
    # Guardrail 3: True variance must be strictly less than the max Bernoulli variance
    if sigma_p_sq >= (p_bar * (1 - p_bar)):
        print("Beta-Binomial Priors failed: sigma_p_sq exceeds max Bernoulli variance")
        return np.nan, np.nan
        
    # Calculate Concentration Parameter (M)
    M = (p_bar * (1 - p_bar) / sigma_p_sq) - 1
    
    if M > 0:
        alpha_mom = p_bar * M
        beta_mom = (1 - p_bar) * M
        
        # Guardrail 4: Resulting alpha and beta must be finite and strictly positive
        if np.isfinite(alpha_mom) and np.isfinite(beta_mom) and alpha_mom > 0 and beta_mom > 0:
            print("Beta-Binomial Priors generated via MoM Fallback")
            return alpha_mom, beta_mom

    print("Beta-Binomial Priors generation failed (both MLE and MoM invalid)")
    return np.nan, np.nan


def apply_dynamic_eb_shrinkage(df: pd.DataFrame, voa_col: str, cons_col: str, n_col: str):
    """
    Applies Dynamic Empirical-Bayes Shrinkage. Safely clamps tau^2 to 0 if within-player 
    noise completely explains the population variance.
    """
    res = df.copy()
    
    if voa_col not in res or cons_col not in res or n_col not in res:
        return pd.Series(np.nan, index=res.index)
        
    var_i = res[cons_col] ** 2
    pooled_var = var_i.dropna().mean()

    if pd.isna(pooled_var):
        return pd.Series(np.nan, index=res.index)

    var_i_filled = var_i.fillna(pooled_var)
    sampling_var_i = pd.Series(np.nan, index=res.index)
    valid_n_mask = res[n_col].notna() & (res[n_col] > 0)

    sampling_var_i.loc[valid_n_mask] = (
        var_i_filled.loc[valid_n_mask] /
        res.loc[valid_n_mask, n_col]
    )

    estimation_mask = res[voa_col].notna() & sampling_var_i.notna()

    if estimation_mask.sum() < 2:
        return pd.Series(np.nan, index=res.index)

    total_var = res.loc[estimation_mask, voa_col].var(ddof=1)
    mean_sampling_var = sampling_var_i.loc[estimation_mask].mean()

    tau_sq = total_var - mean_sampling_var

    if pd.isna(tau_sq):
        return pd.Series(np.nan, index=res.index)

    # SAFEGUARD: Clamp tau_sq to 0
    if tau_sq <= 0:
        print(f"Diagnostic: tau_sq estimated as {tau_sq:.6f} <= 0. Clamping between-player variance to 0.")
        tau_sq = 0.0

    denominator = tau_sq + sampling_var_i
    W = pd.Series(np.nan, index=res.index)
    
    # Safe division for weights
    valid_den = denominator > 0
    W.loc[valid_den] = tau_sq / denominator[valid_den]
    
    # SAFEGUARD: Zero sampling variance AND Zero tau_sq
    zero_den = (denominator == 0)
    W.loc[zero_den] = 0.0

    pool_mean = res.loc[estimation_mask, voa_col].mean()
    shrunken_voa = W * res[voa_col] + (1 - W) * pool_mean
    return pd.Series(shrunken_voa, index=res.index)


def generate_final_rankings(batting_deltas: pd.DataFrame, bowling_deltas: pd.DataFrame):
    """
    Executes the final statistical ranking pipeline for batters and bowlers.
    """
    batters = batting_deltas.copy()
    bowlers = bowling_deltas.copy()

    # =========================================================================
    # 1. BATTING PERFORMANCE & CONSISTENCY
    # =========================================================================
    
    # A. Beta-Binomial Survival Smoothing via MLE/MoM
    alpha, beta = estimate_beta_binomial_priors(batters, 'pressure_dismissals', 'pressure_opportunities')
    
    if pd.isna(alpha) or pd.isna(beta):
        print("Diagnostic: Beta-Binomial priors unavailable. Applying neutral survival Z-score (0.0).")
        batters['smoothed_dismissal_prob'] = np.nan
        batters['z_survival'] = 0.0
        batters['survival_status'] = 'UNAVAILABLE_NEUTRAL'
    else:
        batters['smoothed_dismissal_prob'] = (
            (batters['pressure_dismissals'] + alpha) / 
            (batters['pressure_opportunities'] + alpha + beta)
        )
        # Invert Z-score: lower dismissal probability is better
        batters['z_survival'] = safe_zscore(batters['smoothed_dismissal_prob']) * -1
        batters['survival_status'] = 'ESTIMATED'

    # B. Batting VOA Dynamic Shrinkage
    batters['shrunken_batting_voa'] = apply_dynamic_eb_shrinkage(
        df=batters,
        voa_col='batting_voa',
        cons_col='batting_consistency',
        n_col='valid_batting_innings'
    )
    batters['z_batting_voa'] = safe_zscore(batters['shrunken_batting_voa'])

    # C. Batting Performance & Consistency Integration
    # If z_survival is 0.0, performance is driven purely by 60% VOA, with the remaining 40% acting as a neutral anchor
    batters['z_batting_performance'] = (0.60 * batters['z_batting_voa']) + (0.40 * batters['z_survival'])
    
    # Invert Consistency Z-score: lower SD is better
    batters['z_batting_consistency'] = safe_zscore(batters['batting_consistency']) * -1
    
    # D. Batting Final Score
    batters['Final_Score'] = (0.70 * batters['z_batting_performance']) + (0.30 * batters['z_batting_consistency'])

    # =========================================================================
    # 2. BOWLING PERFORMANCE & CONSISTENCY (PROVISIONAL)
    # =========================================================================
    
    # A. Bowling VOA Dynamic Shrinkage
    bowlers['shrunken_bowling_voa_PROVISIONAL'] = apply_dynamic_eb_shrinkage(
        df=bowlers,
        voa_col='bowling_voa_PROVISIONAL',
        cons_col='bowling_consistency_PROVISIONAL',
        n_col='valid_bowling_innings'
    )
    bowlers['z_bowling_performance_PROVISIONAL'] = safe_zscore(bowlers['shrunken_bowling_voa_PROVISIONAL'])

    # B. Bowling Consistency Integration
    # Invert Consistency Z-score: lower SD is better
    bowlers['z_bowling_consistency_PROVISIONAL'] = safe_zscore(bowlers['bowling_consistency_PROVISIONAL']) * -1
    
    # C. Bowling Final Score
    bowlers['Final_Score_PROVISIONAL'] = (0.70 * bowlers['z_bowling_performance_PROVISIONAL']) + (0.30 * bowlers['z_bowling_consistency_PROVISIONAL'])

    # =========================================================================
    # 3. FILTRATION & OUTPUT
    # =========================================================================
    
    final_batters = batters.dropna(subset=['Final_Score']).sort_values(by='Final_Score', ascending=False)
    final_bowlers = bowlers.dropna(subset=['Final_Score_PROVISIONAL']).sort_values(by='Final_Score_PROVISIONAL', ascending=False)
    
    top_10_batters = final_batters.head(10).reset_index(drop=True)
    top_10_bowlers = final_bowlers.head(10).reset_index(drop=True)

    return top_10_batters, top_10_bowlers