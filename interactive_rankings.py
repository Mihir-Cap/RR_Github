import sys
import os
import pandas as pd
import numpy as np

print("======================================================")
print(" SupeRR Selector Season 2: Interactive Ranking Tuner")
print("======================================================")
print("\nThis tool allows you to dynamically adjust the ranking engine weights.")
print("(Note: Data Scope and Eligibility thresholds remain safely frozen in upstream modules).")

# --- INTERACTIVE PROMPTS ---
print("\n[1] BATTING: Overall Scoring Weight (Performance vs Consistency)")
print("  Effect: A higher performance weight (e.g., 0.90) pushes explosive, hit-or-miss batters to the top.")
print("          A lower weight (e.g., 0.50) rewards steady, reliable run-accumulators.")
b_perf_input = input("  Enter Batting Performance weight (0.0 to 1.0) [Default: 0.70]: ").strip()
try:
    b_perf_w = float(b_perf_input) if b_perf_input else 0.70
except ValueError:
    b_perf_w = 0.70
b_cons_w = 1.0 - b_perf_w
print(f"  -> Batting Set to: {b_perf_w*100:.0f}% Performance / {b_cons_w*100:.0f}% Consistency")

print("\n[2] BATTING Sub-Weight: VOA (Run-Scoring) vs Survival (Wicket Preservation)")
print("  Effect: Higher VOA weight (e.g., 0.80) favors aggressive strikers who hit boundaries.")
print("          Lower VOA weight (e.g., 0.40) favors stable anchors who protect their wicket.")
voa_input = input("  Enter Batting VOA weight (0.0 to 1.0) [Default: 0.60]: ").strip()
try:
    b_voa_w = float(voa_input) if voa_input else 0.60
except ValueError:
    b_voa_w = 0.60
b_surv_w = 1.0 - b_voa_w
print(f"  -> Batting Sub-Weight Set to: {b_voa_w*100:.0f}% VOA / {b_surv_w*100:.0f}% Survival")

print("\n[3] BOWLING: Overall Scoring Weight (Performance vs Consistency)")
print("  Effect: A higher performance weight (e.g., 0.80) favors aggressive wicket-takers who might occasionally leak runs.")
print("          A lower weight (e.g., 0.40) favors economical, highly consistent bowlers who lock down the scoring rate.")
bw_perf_input = input("  Enter Bowling Performance weight (0.0 to 1.0) [Default: 0.70]: ").strip()
try:
    bw_perf_w = float(bw_perf_input) if bw_perf_input else 0.70
except ValueError:
    bw_perf_w = 0.70
bw_cons_w = 1.0 - bw_perf_w
print(f"  -> Bowling Set to: {bw_perf_w*100:.0f}% Performance / {bw_cons_w*100:.0f}% Consistency")

print("\n[4] Bypass Empirical Bayes Shrinkage? (y/n)")
print("  Effect: 'n' (Default) uses strict mathematical shrinkage. If the dataset lacks variance, performance scores clamp to 0.")
print("          'y' bypasses the safety nets and uses Raw VOA. This forces high variance and makes the leaderboards wildly sensitive to your weights.")
bypass_input = input("  Bypass EB Shrinkage and use Raw VOA? (y/n) [Default: n]: ").strip().lower()
bypass_eb = bypass_input == 'y'
if bypass_eb:
    print("  -> Set to: RAW VOA (High Variance - Leaderboards WILL shift based on weights!)")
else:
    print("  -> Set to: SHRUNKEN VOA (Strict Math - Leaderboards may remain static if dataset is underdispersed)")

# --- RUN BASE PIPELINE SILENTLY ---
print("\nExecuting the frozen pipeline in the background to gather baseline data. Please wait...")
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w', encoding='utf-8')

try:
    import main
    from src.ranking.ranker import apply_dynamic_eb_shrinkage, safe_zscore, estimate_beta_binomial_priors
finally:
    sys.stdout.close()
    sys.stdout = original_stdout

print("Baseline data loaded. Applying custom mathematical thresholds...")

# --- CUSTOM RANKING LOGIC ---
batters = main.batting_deltas.copy()
bowlers = main.bowling_deltas.copy()

# Batting Recalculation
alpha, beta = estimate_beta_binomial_priors(batters, 'pressure_dismissals', 'pressure_opportunities')
if pd.isna(alpha) or pd.isna(beta):
    batters['z_survival'] = 0.0
    batters['survival_status'] = 'UNAVAILABLE_NEUTRAL'
else:
    batters['smoothed_dismissal_prob'] = (batters['pressure_dismissals'] + alpha) / (batters['pressure_opportunities'] + alpha + beta)
    batters['z_survival'] = safe_zscore(batters['smoothed_dismissal_prob']) * -1
    batters['survival_status'] = 'ESTIMATED'

if bypass_eb:
    batters['z_batting_voa'] = safe_zscore(batters['batting_voa'])
else:
    batters['shrunken_batting_voa'] = apply_dynamic_eb_shrinkage(batters, 'batting_voa', 'batting_consistency', 'valid_batting_innings')
    batters['z_batting_voa'] = safe_zscore(batters['shrunken_batting_voa'])

# Apply Custom Batting Weights
batters['z_batting_performance'] = (b_voa_w * batters['z_batting_voa']) + (b_surv_w * batters['z_survival'])
batters['z_batting_consistency'] = safe_zscore(batters['batting_consistency']) * -1
batters['Final_Score'] = (b_perf_w * batters['z_batting_performance']) + (b_cons_w * batters['z_batting_consistency'])

# Bowling Recalculation
if bypass_eb:
    bowlers['z_bowling_performance_PROVISIONAL'] = safe_zscore(bowlers['bowling_voa_PROVISIONAL'])
else:
    bowlers['shrunken_bowling_voa_PROVISIONAL'] = apply_dynamic_eb_shrinkage(bowlers, 'bowling_voa_PROVISIONAL', 'bowling_consistency_PROVISIONAL', 'valid_bowling_innings')
    bowlers['z_bowling_performance_PROVISIONAL'] = safe_zscore(bowlers['shrunken_bowling_voa_PROVISIONAL'])

bowlers['z_bowling_consistency_PROVISIONAL'] = safe_zscore(bowlers['bowling_consistency_PROVISIONAL']) * -1
bowlers['Final_Score_PROVISIONAL'] = (bw_perf_w * bowlers['z_bowling_performance_PROVISIONAL']) + (bw_cons_w * bowlers['z_bowling_consistency_PROVISIONAL'])

# --- SORT & MAP NAMES ---
final_batters = batters.dropna(subset=['Final_Score']).sort_values(by='Final_Score', ascending=False).head(10).reset_index(drop=True)
final_bowlers = bowlers.dropna(subset=['Final_Score_PROVISIONAL']).sort_values(by='Final_Score_PROVISIONAL', ascending=False).head(10).reset_index(drop=True)

flags_df = main.flags_df
b_name_col = 'striker_name'
bw_name_col = 'bowler_name'

if b_name_col in flags_df.columns and 'striker_id' in flags_df.columns:
    b_map = dict(zip(flags_df['striker_id'].dropna().astype(float), flags_df.dropna(subset=['striker_id'])[b_name_col]))
    final_batters['Player_Name'] = final_batters['eligible_batter_id'].astype(float).map(b_map)
else:
    final_batters['Player_Name'] = "Unknown Batter"

if bw_name_col in flags_df.columns and 'bowler_id' in flags_df.columns:
    bw_map = dict(zip(flags_df['bowler_id'].dropna().astype(float), flags_df.dropna(subset=['bowler_id'])[bw_name_col]))
    final_bowlers['Player_Name'] = final_bowlers['bowler_id'].astype(float).map(bw_map)
else:
    final_bowlers['Player_Name'] = "Unknown Bowler"

# --- EXPORT & DISPLAY ---
batters_csv_cols = ['eligible_batter_id', 'Player_Name', 'Final_Score', 'z_batting_performance', 'z_batting_consistency', 'survival_status']
bowlers_csv_cols = ['bowler_id', 'Player_Name', 'Final_Score_PROVISIONAL', 'z_bowling_performance_PROVISIONAL', 'z_bowling_consistency_PROVISIONAL']

final_batters[batters_csv_cols].to_csv('top_10_batters_custom.csv', index=False)
final_bowlers[bowlers_csv_cols].to_csv('top_10_bowlers_custom.csv', index=False)

print("\n======================================================")
print("               CUSTOM LEADERBOARDS")
print("======================================================")
print("\n--- FULL TOP 10 BATTERS ---")
print(final_batters[['Player_Name', 'Final_Score', 'z_batting_performance', 'z_batting_consistency']].to_string(index=False))

print("\n--- FULL TOP 10 BOWLERS ---")
print(final_bowlers[['Player_Name', 'Final_Score_PROVISIONAL', 'z_bowling_performance_PROVISIONAL', 'z_bowling_consistency_PROVISIONAL']].to_string(index=False))

print("\n✅ Success! Custom results saved to 'top_10_batters_custom.csv' and 'top_10_bowlers_custom.csv'.")