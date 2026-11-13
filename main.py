import pandas as pd
from src.reconstruction.delivery_events import build_delivery_events
from src.reconstruction.match_state import build_match_state
from src.metrics.baselines import compute_baselines_and_eligibility
from src.metrics.pressure_deltas import calculate_pressure_deltas
from src.ranking.ranker import generate_final_rankings, estimate_beta_binomial_priors, apply_dynamic_eb_shrinkage, safe_zscore
from src.ranking.ranker import generate_final_rankings, estimate_beta_binomial_priors, apply_dynamic_eb_shrinkage, safe_zscore
import pandas as pd

import numpy as np



# =========================================================================
# 1. LOAD & MERGE RAW DATA (WITH SAFEGUARDS)
# =========================================================================
balls_path = 'data/raw/ball_by_ball.csv'
matches_path = 'data/raw/matches.csv'
comp_path = 'data/raw/competitions.csv'

print(f"Loading balls data from {balls_path}...")
delivery_df = pd.read_csv(balls_path)

print(f"Loading match metadata from {matches_path}...")
matches_df = pd.read_csv(matches_path)
comp_df = pd.read_csv(comp_path)

# --- COMPETITION SCOPE FILTER ---
full_matches = pd.merge(matches_df, comp_df, on='comp_id', how='left')

def is_valid_comp(row):
    fmt = str(row['match_format']).upper()
    name = str(row['comp_name']).upper()
    season_str = str(row['season'])
    
    # Strict word match for 'IPL' to prevent substring leakage
    is_valid_format = (fmt == 'T20I') or ('IPL' in name.split()) or ('INDIAN PREMIER LEAGUE' in name)
    try:
        is_valid_year = int(season_str[:4]) >= 2022
    except:
        is_valid_year = False
        
    return is_valid_format and is_valid_year

valid_match_ids = full_matches[full_matches.apply(is_valid_comp, axis=1)]['match_id']

# Filter the raw datasets before passing them downstream
raw_matches = matches_df[matches_df['match_id'].isin(valid_match_ids)].copy()
raw_balls = delivery_df[delivery_df['match_id'].isin(valid_match_ids)].copy()
# --------------------------------

print("\n--- DATA MERGE VALIDATION ---")

# Safeguard A: Null match_id checks
null_balls_ids = raw_balls['match_id'].isna().sum()
null_matches_ids = raw_matches['match_id'].isna().sum()
print(f"Null match_ids in ball_by_ball: {null_balls_ids}")
print(f"Null match_ids in matches.csv: {null_matches_ids}")
if null_balls_ids > 0 or null_matches_ids > 0:
    raise ValueError("CRITICAL: Null match_ids detected in source data!")

# Safeguard B: Confirm match_id uniqueness in matches.csv
is_unique = raw_matches['match_id'].is_unique
print(f"match_id unique in matches.csv: {is_unique}")
if not is_unique:
    duplicate_ids = raw_matches[raw_matches.duplicated('match_id')]['match_id'].unique()
    raise ValueError(f"CRITICAL: Duplicate match_ids found in matches.csv! IDs: {duplicate_ids}")

# Safeguard C: Report unmatched match IDs and halt if any exist
balls_ids = set(raw_balls['match_id'])
matches_ids = set(raw_matches['match_id'])
missing_in_meta = balls_ids - matches_ids

print(f"Distinct match IDs in ball_by_ball missing from matches.csv: {len(missing_in_meta)}")
if missing_in_meta:
    affected_rows = raw_balls['match_id'].isin(missing_in_meta).sum()
    print(f" -> Delivery rows belonging to unmatched matches: {affected_rows}")
    print(f" -> Orphaned IDs: {missing_in_meta}")
    raise ValueError("CRITICAL: Orphaned match IDs detected. Halting execution to investigate missing metadata.")

# Safeguard D: Preserve required match metadata and prevent unintended overwrites
common_cols = set(raw_balls.columns).intersection(raw_matches.columns) - {'match_id'}
required_dls_cols = {'revised_target', 'revised_overs'}

cols_to_drop = common_cols - required_dls_cols
cols_collision = common_cols.intersection(required_dls_cols)

if cols_to_drop:
    print(f"Dropping non-essential overlapping columns from matches.csv: {cols_to_drop}")
    raw_matches = raw_matches.drop(columns=list(cols_to_drop))

# Safeguard E: Many-to-One Merge and Row Count Validation
original_row_count = raw_balls.shape[0]

merged_df = raw_balls.merge(
    raw_matches, 
    on='match_id', 
    how='left', 
    validate='m:1'
)

print(f"\nOriginal ball_by_ball rows: {original_row_count}")
print(f"Merged dataframe rows:      {merged_df.shape[0]}")
assert original_row_count == merged_df.shape[0], "CRITICAL: Row count altered during merge!"

# Safeguard F: Address DLS Column Collisions Post-Merge
if cols_collision:
    for col in cols_collision:
        col_balls = f"{col}_x"      # Originated from ball_by_ball.csv
        col_matches = f"{col}_y"    # Originated from matches.csv
        
        # Check for conflicting non-null values
        both_present = merged_df[col_balls].notna() & merged_df[col_matches].notna()
        conflicts = both_present & (merged_df[col_balls] != merged_df[col_matches])
        
        if conflicts.any():
            raise ValueError(f"CRITICAL: Conflicting non-null values found for '{col}' between ball_by_ball and matches.csv! Halting pipeline.")
        
        print(f"Resolving collision for '{col}': Prioritizing matches.csv metadata and restoring original column name.")
        # matches.csv (_y) is the authoritative metadata source. Fill NA with ball_by_ball (_x) just in case.
        merged_df[col] = merged_df[col_matches].combine_first(merged_df[col_balls])
        merged_df = merged_df.drop(columns=[col_balls, col_matches])

# =========================================================================
# 2. EXECUTE PIPELINE
# =========================================================================

# --- STEP 1: DELIVERY EVENTS ---
print("\n========================================")
print("Executing Delivery Events v1.0...")
delivery_df = build_delivery_events(merged_df)

print("\n--- DELIVERY STATUS REPORT ---")
print(delivery_df['delivery_status'].value_counts(dropna=False))

print("\n--- SPECIFIC VALIDATION ISSUES ---")
issues = delivery_df['validation_issues'].explode().dropna().value_counts()
if issues.empty:
    print("No validation issues found! Clean dataset.")
else:
    print(issues)

# --- STEP 2: MATCH STATE ---
print("\n========================================")
print("Executing Match State v1.0...")
match_state_df = build_match_state(delivery_df)

# Helper to safely check list-based issues
def has_issue(issue_name):
    return match_state_df['match_state_issues'].apply(
        lambda x: issue_name in x if isinstance(x, list) else False
    )

print("\n--- MATCH STATE STATUS REPORT ---")
print(match_state_df['match_state_status'].value_counts(dropna=False))

print("\n--- SPECIFIC MATCH STATE ISSUES ---")
state_issues = match_state_df['match_state_issues'].explode().dropna().value_counts()
if state_issues.empty:
    print("No match state issues found! Clean progression.")
else:
    print(state_issues)

# BLOCKED Contamination Diagnostics
print("\n--- BLOCKED CONTAMINATION PROPAGATION ---")
is_contaminated_state = has_issue('CUMULATIVE_STATE_CONTAMINATED_BY_PRIOR_BLOCK')
is_contaminated_target = has_issue('TARGET_DERIVED_FROM_BLOCKED_INNINGS_1')
missing_inn1_target = has_issue('MISSING_FIRST_INNINGS_TARGET')

print(f"Deliveries w/ contaminated pre-delivery states: {is_contaminated_state.sum()}")
print(f"Innings 2 deliveries w/ targets blocked by Innings 1: {is_contaminated_target.sum()}")
print(f"Innings 2 deliveries w/ missing Innings 1 entirely: {missing_inn1_target.sum()}")

# Reverse contamination check
inn2_blocked_matches = match_state_df[(match_state_df['innings'] == 2) & (match_state_df['match_state_status'] == 'BLOCKED')]['match_id'].unique()
print(f"\nMatches containing a BLOCKED Innings 2 delivery: {len(inn2_blocked_matches)}")

inn1_in_affected_matches = match_state_df[(match_state_df['match_id'].isin(inn2_blocked_matches)) & (match_state_df['innings'] == 1)]
if not inn1_in_affected_matches.empty:
    inn1_blocked = (inn1_in_affected_matches['match_state_status'] == 'BLOCKED').sum()
    inn1_contaminated = has_issue('CUMULATIVE_STATE_CONTAMINATED_BY_PRIOR_BLOCK')[inn1_in_affected_matches.index].sum()
    print(f" -> Innings 1 deliveries in these matches: {len(inn1_in_affected_matches)}")
    print(f" -> Innings 1 deliveries organically BLOCKED: {inn1_blocked}")
    print(f" -> Innings 1 deliveries showing contamination: {inn1_contaminated}")
    print("    (Note: Any Innings 1 contamination here stems purely from an Innings 1 block, not Innings 2)")

# DLS Diagnostics
print("\n--- DLS DETECTION & TARGET DIAGNOSTICS ---")
inn2_mask = match_state_df['innings'] == 2
is_dls = has_issue('UNVERIFIED_DLS_INTERVENTION_DETECTED')

dls_matches = match_state_df.loc[is_dls, 'match_id'].unique()
print(f"Distinct matches with DLS intervention flags: {len(dls_matches)}")

if len(dls_matches) > 0:
    dls_inn2_deliveries = match_state_df[is_dls & inn2_mask]
    
    print(f"DLS Innings 2 deliveries: {len(dls_inn2_deliveries)}")
    print(f" -> With PROVISIONAL status: {(dls_inn2_deliveries['match_state_status'] == 'PROVISIONAL').sum()}")
    print(f" -> With blanked standard allocations: {dls_inn2_deliveries['innings_balls_allocation'].isna().sum()}")
    print(f" -> With blanked targets: {dls_inn2_deliveries['target_runs_PROVISIONAL'].isna().sum()}")
else:
    print("No DLS matches found in the dataset.")

# Clarified Missing-Target Categories
is_missing_target = inn2_mask & match_state_df['target_runs_PROVISIONAL'].isna()
is_dls_target_issue = inn2_mask & is_dls
is_blocked_target_issue = inn2_mask & (is_contaminated_target | missing_inn1_target)

missing_total = is_missing_target.sum()
missing_dls_only = (is_missing_target & is_dls_target_issue & ~is_blocked_target_issue).sum()
missing_blocked_only = (is_missing_target & is_blocked_target_issue & ~is_dls_target_issue).sum()
missing_overlap = (is_missing_target & is_dls_target_issue & is_blocked_target_issue).sum()
missing_other = (is_missing_target & ~is_dls_target_issue & ~is_blocked_target_issue).sum()

print(f"\n--- INNINGS 2 MISSING TARGET BREAKDOWN ---")
print(f"Total Innings 2 deliveries with missing targets: {missing_total}")
print(f"  - Missing due to DLS provisional handling (Exclusive): {missing_dls_only}")
print(f"  - Missing due to BLOCKED/missing Inn1 (Exclusive): {missing_blocked_only}")
print(f"  - Overlap (Both DLS and BLOCKED/missing Inn1): {missing_overlap}")
print(f"  - Missing due to other/unknown reasons: {missing_other}")



from src.pressure.flags import apply_pressure_flags

# --- STEP 3: PRESSURE CLASSIFICATION ---
print("\n========================================")
print("Executing Pressure Flags v1.0...")
original_row_count = match_state_df.shape[0]
flags_df = apply_pressure_flags(match_state_df)

# 1. Row-Count Integrity Check
print(f"\nInput match_state_df rows: {original_row_count}")
print(f"Output flags_df rows:      {flags_df.shape[0]}")
assert original_row_count == flags_df.shape[0], "CRITICAL: Row count altered during pressure flag classification!"

# 2. Overall Pressure Distribution
print("\n--- PRESSURE CLASSIFICATION REPORT ---")
if 'is_pressure' not in flags_df.columns:
    raise KeyError("CRITICAL: 'is_pressure' column missing from pressure classification output!")
print(flags_df['is_pressure'].value_counts(dropna=False))

# 3. Blocked Data Leakage Check
# We must prove that ZERO blocked deliveries were accidentally classified as True for pressure
blocked_pressure_leakage = flags_df[(flags_df['match_state_status'] == 'BLOCKED') & (flags_df['is_pressure'] == True)]
leakage_count = len(blocked_pressure_leakage)
print(f"\nBLOCKED deliveries incorrectly flagged as pressure: {leakage_count}")
assert leakage_count == 0, f"CRITICAL: {leakage_count} BLOCKED deliveries leaked into pressure classification!"

# 4. Provisional Data Check
provisional_pressure = flags_df[(flags_df['match_state_status'] == 'PROVISIONAL') & (flags_df['is_pressure'] == True)]
print(f"PROVISIONAL deliveries flagged as pressure: {len(provisional_pressure)}")





# --- STEP 4: BASELINES & ELIGIBILITY ---
print("\n========================================")
print("Executing Baselines & Eligibility v1.0...")

# The frozen module returns 4 distinct objects: 2 lists, 1 dataframe, 1 dictionary.
eligible_batters, eligible_bowlers, comp_benchmarks_PROVISIONAL, global_priors = compute_baselines_and_eligibility(flags_df)

# 1. Delivery-Level Integrity
print(f"\nInput flags_df rows consumed: {flags_df.shape[0]}")
print("Note: baselines.py outputs aggregated metadata, not a delivery-level dataframe. Input dataframe is preserved.")

# 2 & 6. Exhaustive Eligibility Diagnostics & Threshold Verification
print("\n--- ELIGIBILITY REPORT (>= 5 Pressure Innings) ---")
print(f"Total Eligible Batters: {len(eligible_batters)}")
print(f"Total Eligible Bowlers: {len(eligible_bowlers)}")

# Isolate valid pressure deliveries using the exact column required by the frozen module
status_col = 'pressure_status'
valid_pressure_df = flags_df[(flags_df['is_pressure'] == True) & (flags_df[status_col] != 'BLOCKED')].copy()

# A. Exhaustive Batter Verification (Independent Actions)
# 1. Faced actions (credited to striker)
faced_mask = (valid_pressure_df['is_batter_ball_faced'] == True) & valid_pressure_df['striker_id'].notna()
faced_actions = valid_pressure_df[faced_mask].copy()
faced_actions['eligible_batter_id'] = faced_actions['striker_id']

# 2. Dismissal actions (credited to dismissed player)
dismissal_mask = (valid_pressure_df['is_batter_dismissal'] == True) & valid_pressure_df['dismissed_player_id'].notna()
dismissal_actions = valid_pressure_df[dismissal_mask].copy()
dismissal_actions['eligible_batter_id'] = dismissal_actions['dismissed_player_id']

# 3. Combine to ensure no actions are lost
pdf_bat_actions = pd.concat([faced_actions, dismissal_actions])

batter_inns_count = pdf_bat_actions.groupby(['eligible_batter_id', 'match_id', 'innings']).size().reset_index().groupby('eligible_batter_id').size()

failed_batters = [b for b in eligible_batters if batter_inns_count.get(b, 0) < 5]
print(f"Batters exhaustively verified against threshold: {len(eligible_batters)}")
if failed_batters:
    print(f"Batters failing >= 5 threshold: {len(failed_batters)} (IDs: {failed_batters[:5]}...)")
else:
    print("Batters failing >= 5 threshold: 0")
assert len(failed_batters) == 0, f"CRITICAL: {len(failed_batters)} batters failed the >= 5 innings threshold!"

# B. Exhaustive Bowler Verification
pdf_bowl = valid_pressure_df.dropna(subset=['bowler_id', 'is_legal_ball']).copy()
bowler_inns_count = pdf_bowl[pdf_bowl['is_legal_ball'] == True].groupby(['bowler_id', 'match_id', 'innings']).size().reset_index().groupby('bowler_id').size()

failed_bowlers = [b for b in eligible_bowlers if bowler_inns_count.get(b, 0) < 5]
print(f"Bowlers exhaustively verified against threshold: {len(eligible_bowlers)}")
if failed_bowlers:
    print(f"Bowlers failing >= 5 threshold: {len(failed_bowlers)} (IDs: {failed_bowlers[:5]}...)")
else:
    print("Bowlers failing >= 5 threshold: 0")
assert len(failed_bowlers) == 0, f"CRITICAL: {len(failed_bowlers)} bowlers failed the >= 5 innings threshold!"

# 3. Adversarial Blocked Data Leakage Check
print("\n--- ADVERSARIAL BLOCKED LEAKAGE CHECK ---")
# Create a test copy exclusively of BLOCKED records
blocked_df = flags_df[flags_df[status_col] == 'BLOCKED'].copy()
# Adversarial injection: Force is_pressure = True to simulate upstream classification failure
blocked_df['is_pressure'] = True

leak_batters, leak_bowlers, leak_comp, _ = compute_baselines_and_eligibility(blocked_df)

print(f"Eligible Batters extracted from adversarial BLOCKED data: {len(leak_batters)}")
print(f"Eligible Bowlers extracted from adversarial BLOCKED data: {len(leak_bowlers)}")
assert len(leak_batters) == 0, "CRITICAL: BLOCKED deliveries leaked into batter eligibility despite status filter!"
assert len(leak_bowlers) == 0, "CRITICAL: BLOCKED deliveries leaked into bowler eligibility despite status filter!"

# 4 & 5. Competition-Level Benchmarks & Missing Values
print("\n--- COMPETITION BENCHMARKS REPORT ---")
if not comp_benchmarks_PROVISIONAL.empty:
    print(f"Total Competitions Processed: {len(comp_benchmarks_PROVISIONAL)}")
    
    sr_col = 'comp_avg_pressure_sr'
    econ_col = 'comp_avg_pressure_econ'
    
    if sr_col in comp_benchmarks_PROVISIONAL.columns and econ_col in comp_benchmarks_PROVISIONAL.columns:
        missing_sr = comp_benchmarks_PROVISIONAL[sr_col].isna().sum()
        missing_econ = comp_benchmarks_PROVISIONAL[econ_col].isna().sum()
        print(f"Competitions missing '{sr_col}' (Undefined/NaN): {missing_sr}")
        print(f"Competitions missing '{econ_col}' (Undefined/NaN): {missing_econ}")
    else:
        print(f"Warning: Expected columns '{sr_col}' and '{econ_col}' not found.")
else:
    print("Warning: Competition benchmarks dataframe is empty. Check if 'comp_id' or 'competition_id' exist in raw data.")

# 7. Global Priors Report
print("\n--- GLOBAL PRIORS REPORT ---")
print(f"Alpha: {global_priors.get('alpha')}")
print(f"Beta: {global_priors.get('beta')}")
print(f"Global BPD (Balls Per Dismissal): {global_priors.get('global_bpd')}")




# --- STEP 5: PRESSURE DELTAS (VOA) ---
print("\n========================================")
print("Executing Pressure Deltas (VOA) v1.0...")

# The frozen module expects the full pressure_df and the two eligibility lists
batting_deltas, bowling_deltas = calculate_pressure_deltas(flags_df, eligible_batters, eligible_bowlers)

# 1. Output Coverage Integrity (Set-Based ID Verification & Duplicates)
print("\n--- DELTA COVERAGE REPORT ---")
out_batters = set(batting_deltas['eligible_batter_id']) if not batting_deltas.empty else set()
missing_batters = set(eligible_batters) - out_batters
extra_batters = out_batters - set(eligible_batters)
b_duplicates = batting_deltas.duplicated(subset=['eligible_batter_id']).sum() if not batting_deltas.empty else 0

print(f"Eligible Batters Input:  {len(eligible_batters)}")
print(f"Batting Deltas Output:   {len(out_batters)}")
if missing_batters: print(f" -> Missing Batters: {len(missing_batters)}")
if extra_batters: print(f" -> Unexpected Batters: {len(extra_batters)}")
print(f" -> Duplicate Batter Rows: {b_duplicates}")

assert len(missing_batters) == 0, f"CRITICAL: {len(missing_batters)} eligible batters dropped!"
assert len(extra_batters) == 0, f"CRITICAL: {len(extra_batters)} unexpected batters found!"
assert b_duplicates == 0, f"CRITICAL: {b_duplicates} duplicate batters found in output!"

out_bowlers = set(bowling_deltas['bowler_id']) if not bowling_deltas.empty else set()
missing_bowlers = set(eligible_bowlers) - out_bowlers
extra_bowlers = out_bowlers - set(eligible_bowlers)
bw_duplicates = bowling_deltas.duplicated(subset=['bowler_id']).sum() if not bowling_deltas.empty else 0

print(f"Eligible Bowlers Input:  {len(eligible_bowlers)}")
print(f"Bowling Deltas Output:   {len(out_bowlers)}")
if missing_bowlers: print(f" -> Missing Bowlers: {len(missing_bowlers)}")
if extra_bowlers: print(f" -> Unexpected Bowlers: {len(extra_bowlers)}")
print(f" -> Duplicate Bowler Rows: {bw_duplicates}")

assert len(missing_bowlers) == 0, f"CRITICAL: {len(missing_bowlers)} eligible bowlers dropped!"
assert len(extra_bowlers) == 0, f"CRITICAL: {len(extra_bowlers)} unexpected bowlers found!"
assert bw_duplicates == 0, f"CRITICAL: {bw_duplicates} duplicate bowlers found in output!"

# 2. VOA & Missing Value Validation (Batting)
print("\n--- BATTING VOA DIAGNOSTICS ---")
b_missing_voa = batting_deltas['batting_voa'].isna().sum()
b_missing_cons = batting_deltas['batting_consistency'].isna().sum()
print(f"Batters missing VOA (due to invalid LOO comps): {b_missing_voa}")
print(f"Batters missing Consistency (requires >= 2 valid innings): {b_missing_cons}")

# 3. VOA & Missing Value Validation (Bowling)
print("\n--- BOWLING VOA DIAGNOSTICS ---")
bw_missing_voa = bowling_deltas['bowling_voa_PROVISIONAL'].isna().sum()
bw_missing_cons = bowling_deltas['bowling_consistency_PROVISIONAL'].isna().sum()
print(f"Bowlers missing VOA PROV (due to invalid LOO comps): {bw_missing_voa}")
print(f"Bowlers missing Consistency PROV (requires >= 2 valid innings): {bw_missing_cons}")

# 4. Adversarial Blocked Leakage Verification
print("\n--- ADVERSARIAL BLOCKED LEAKAGE VERIFICATION (STEP 5) ---")
# Passing the strictly adversarial blocked_df from Step 4
leak_bat_deltas, leak_bowl_deltas = calculate_pressure_deltas(blocked_df, eligible_batters, eligible_bowlers)

# The frozen function structurally guarantees empty DataFrames when all inputs are BLOCKED
print(f"Batting BLOCKED Leakage Output Rows: {len(leak_bat_deltas)}")
assert leak_bat_deltas.empty, "CRITICAL: Batting adversarial leakage DataFrame is not empty!"

print(f"Bowling BLOCKED Leakage Output Rows: {len(leak_bowl_deltas)}")
assert leak_bowl_deltas.empty, "CRITICAL: Bowling adversarial leakage DataFrame is not empty!"

print("\nStep 5 diagnostics complete.")



# --- STEP 6: RANKING ENGINE ---
print("\n========================================")
print("Executing Final Rankings v1.0...")

# 1. Derive Actual Valid-Score Population (Mirroring full frozen ranking logic)
alpha, beta = estimate_beta_binomial_priors(batting_deltas, 'pressure_dismissals', 'pressure_opportunities')
print("\n--- BAYESIAN SMOOTHING PRIORS ---")
print(f"Estimated Alpha: {alpha}")
print(f"Estimated Beta:  {beta}")

# A. Batters Valid Population
if pd.isna(alpha) or pd.isna(beta):
    # If priors fail, the patched ranker applies exactly 0.0
    z_survival = pd.Series(0.0, index=batting_deltas.index)
else:
    smoothed_dp = (batting_deltas['pressure_dismissals'] + alpha) / (batting_deltas['pressure_opportunities'] + alpha + beta)
    z_survival = safe_zscore(smoothed_dp) * -1

b_shrunken_voa = apply_dynamic_eb_shrinkage(batting_deltas, 'batting_voa', 'batting_consistency', 'valid_batting_innings')
z_batting_voa = safe_zscore(b_shrunken_voa)
z_batting_cons = safe_zscore(batting_deltas['batting_consistency']) * -1

# b_valid_mask now safely passes z_survival for all players if it was set to the neutral 0.0
b_valid_mask = z_batting_voa.notna() & z_survival.notna() & z_batting_cons.notna()
b_available = b_valid_mask.sum()
b_excluded = len(batting_deltas) - b_available

# B. Bowlers Valid Population
bw_shrunken_voa = apply_dynamic_eb_shrinkage(bowling_deltas, 'bowling_voa_PROVISIONAL', 'bowling_consistency_PROVISIONAL', 'valid_bowling_innings')
z_bowling_voa = safe_zscore(bw_shrunken_voa)
z_bowling_cons = safe_zscore(bowling_deltas['bowling_consistency_PROVISIONAL']) * -1

bw_valid_mask = z_bowling_voa.notna() & z_bowling_cons.notna()
bw_available = bw_valid_mask.sum()
bw_excluded = len(bowling_deltas) - bw_available

print(f"Batters excluded due to unresolved metrics/Z-scores: {b_excluded}")
print(f"Bowlers excluded due to unresolved metrics/Z-scores: {bw_excluded}")

# Generate Final Leaderboards
top_10_batters, top_10_bowlers = generate_final_rankings(batting_deltas, bowling_deltas)

# 2 & 7. Expected Leaderboard Counts
print("\n--- LEADERBOARD DIAGNOSTICS ---")
expected_b_count = min(10, b_available)
expected_bw_count = min(10, bw_available)

print(f"Top 10 Batters Output Rows: {len(top_10_batters)} (Expected: {expected_b_count}, Valid Population: {b_available})")
print(f"Top 10 Bowlers Output Rows: {len(top_10_bowlers)} (Expected: {expected_bw_count}, Valid Population: {bw_available})")

assert len(top_10_batters) == expected_b_count, f"CRITICAL: Batter leaderboard count mismatch! Expected {expected_b_count}, got {len(top_10_batters)}"
assert len(top_10_bowlers) == expected_bw_count, f"CRITICAL: Bowler leaderboard count mismatch! Expected {expected_bw_count}, got {len(top_10_bowlers)}"

# 3. Model Integration & Missing Value Checks
if not top_10_batters.empty:
    assert top_10_batters['Final_Score'].notna().all(), "CRITICAL: Missing Final_Score in batter leaderboard!"
    assert top_10_batters['shrunken_batting_voa'].notna().all(), "CRITICAL: Missing shrunken_batting_voa for ranked batters!"
    assert top_10_batters['z_batting_voa'].notna().all(), "CRITICAL: Missing z_batting_voa for ranked batters!"
    assert top_10_batters['z_survival'].notna().all(), "CRITICAL: Missing z_survival for ranked batters!"
    assert top_10_batters['z_batting_consistency'].notna().all(), "CRITICAL: Missing z_batting_consistency for ranked batters!"
    assert 'survival_status' in top_10_batters.columns, "CRITICAL: Missing survival_status column!"
    
    if pd.isna(alpha) or pd.isna(beta):
        assert (top_10_batters['survival_status'] == 'UNAVAILABLE_NEUTRAL').all(), "CRITICAL: survival_status incorrect when priors fail!"
    else:
        assert top_10_batters['smoothed_dismissal_prob'].notna().all(), "CRITICAL: Missing smoothed_dismissal_prob for ranked batters!"

if not top_10_bowlers.empty:
    assert top_10_bowlers['Final_Score_PROVISIONAL'].notna().all(), "CRITICAL: Missing Final_Score_PROVISIONAL in bowler leaderboard!"
    assert top_10_bowlers['shrunken_bowling_voa_PROVISIONAL'].notna().all(), "CRITICAL: Missing shrunken_bowling_voa_PROVISIONAL for ranked bowlers!"
    assert top_10_bowlers['z_bowling_performance_PROVISIONAL'].notna().all(), "CRITICAL: Missing z_bowling_performance_PROVISIONAL for ranked bowlers!"
    assert top_10_bowlers['z_bowling_consistency_PROVISIONAL'].notna().all(), "CRITICAL: Missing z_bowling_consistency_PROVISIONAL for ranked bowlers!"

# 4. Duplicate ID Check
if not top_10_batters.empty:
    b_dupes = top_10_batters.duplicated(subset=['eligible_batter_id']).sum()
    assert b_dupes == 0, f"CRITICAL: {b_dupes} duplicate batter IDs in top 10!"
    
if not top_10_bowlers.empty:
    bw_dupes = top_10_bowlers.duplicated(subset=['bowler_id']).sum()
    assert bw_dupes == 0, f"CRITICAL: {bw_dupes} duplicate bowler IDs in top 10!"

# 5. Non-Increasing Order Check (Descending)
if not top_10_batters.empty:
    b_sorted = top_10_batters['Final_Score'].is_monotonic_decreasing
    assert b_sorted, "CRITICAL: Batter leaderboard is not in non-increasing order by Final_Score!"

if not top_10_bowlers.empty:
    bw_sorted = top_10_bowlers['Final_Score_PROVISIONAL'].is_monotonic_decreasing
    assert bw_sorted, "CRITICAL: Bowler leaderboard is not in non-increasing order by Final_Score_PROVISIONAL!"

# 6. Eligible Population Leakage Check
valid_b_ids = set(eligible_batters['eligible_batter_id']) if isinstance(eligible_batters, pd.DataFrame) else set(eligible_batters)
valid_bw_ids = set(eligible_bowlers['bowler_id']) if isinstance(eligible_bowlers, pd.DataFrame) else set(eligible_bowlers)

if not top_10_batters.empty:
    rogue_batters = set(top_10_batters['eligible_batter_id']) - valid_b_ids
    assert len(rogue_batters) == 0, f"CRITICAL: Rogue batters in leaderboard: {rogue_batters}"

if not top_10_bowlers.empty:
    rogue_bowlers = set(top_10_bowlers['bowler_id']) - valid_bw_ids
    assert len(rogue_bowlers) == 0, f"CRITICAL: Rogue bowlers in leaderboard: {rogue_bowlers}"

print("\nStep 6 diagnostics complete. Rankings successfully generated and validated.")
print("\nTOP 3 BATTERS:")
if not top_10_batters.empty:
    print(top_10_batters[['eligible_batter_id', 'Final_Score', 'z_batting_performance', 'z_batting_consistency', 'survival_status']].head(3).to_string(index=False))

print("\nTOP 3 BOWLERS:")
if not top_10_bowlers.empty:
    print(top_10_bowlers[['bowler_id', 'Final_Score_PROVISIONAL', 'z_bowling_performance_PROVISIONAL', 'z_bowling_consistency_PROVISIONAL']].head(3).to_string(index=False))



# --- STEP 7: FINAL REPORTING & NAME MAPPING ---
print("\n========================================")
print("SupeRR Selector Season 2: Final Leaderboards")

# 1. Exact Column Mapping based on raw data schema
b_name_col = 'striker_name'
bw_name_col = 'bowler_name'

if b_name_col in flags_df.columns and 'striker_id' in flags_df.columns:
    b_map = dict(zip(flags_df['striker_id'].dropna().astype(float), flags_df.dropna(subset=['striker_id'])[b_name_col]))
    top_10_batters['Player_Name'] = top_10_batters['eligible_batter_id'].astype(float).map(b_map)
else:
    top_10_batters['Player_Name'] = "Unknown Batter"

if bw_name_col in flags_df.columns and 'bowler_id' in flags_df.columns:
    bw_map = dict(zip(flags_df['bowler_id'].dropna().astype(float), flags_df.dropna(subset=['bowler_id'])[bw_name_col]))
    top_10_bowlers['Player_Name'] = top_10_bowlers['bowler_id'].astype(float).map(bw_map)
else:
    top_10_bowlers['Player_Name'] = "Unknown Bowler"

# 2. Format final outputs to STRICTLY exclude IDs and internal Z-scores
final_batters_display = top_10_batters[['Player_Name', 'Final_Score']].copy()
final_bowlers_display = top_10_bowlers[['Player_Name', 'Final_Score_PROVISIONAL']].copy()

print("\n🏆 TOP 3 BATTERS 🏆")
if not final_batters_display.empty:
    print(final_batters_display.head(3).to_string(index=False))

print("\n🏆 TOP 3 BOWLERS 🏆")
if not final_bowlers_display.empty:
    print(final_bowlers_display.head(3).to_string(index=False))

print("\n--- FULL TOP 10 BATTERS ---")
if not final_batters_display.empty:
    print(final_batters_display.to_string(index=False))

print("\n--- FULL TOP 10 BOWLERS ---")
if not final_bowlers_display.empty:
    print(final_bowlers_display.to_string(index=False))

# --- FINAL VALIDATION CHECKLIST ---
print("\n========================================")
print("FINAL VALIDATION: PATCH VERIFICATION & ASSERTIONS")

# Re-read raw matches to check what was filtered out
raw_matches = pd.read_csv('data/raw/matches.csv')
raw_comps = pd.merge(raw_matches, comp_df, on='comp_id', how='left')

# 1. Confirm excluded competitions contribute zero deliveries
comp_2670_matches = raw_comps[raw_comps['comp_id'] == 2670]['match_id']
comp_3289_matches = raw_comps[raw_comps['comp_id'] == 3289]['match_id']

deliv_2670 = delivery_df[delivery_df['match_id'].isin(comp_2670_matches)].shape[0]
deliv_3289 = delivery_df[delivery_df['match_id'].isin(comp_3289_matches)].shape[0]

print(f"1. Deliveries from Comp 2670: {deliv_2670}")
print(f"   Deliveries from Comp 3289: {deliv_3289}")
assert deliv_2670 == 0, f"CRITICAL: Comp 2670 leaked {deliv_2670} deliveries!"
assert deliv_3289 == 0, f"CRITICAL: Comp 3289 leaked {deliv_3289} deliveries!"

# 2. Confirm no other out-of-scope competitions remain
# (Use merged_df, which contains the exact matches that actually ran through the pipeline)
active_matches = merged_df[['match_id', 'comp_id']].drop_duplicates()
remaining_comps = pd.merge(active_matches, comp_df, on='comp_id', how='left')
invalid_remaining = remaining_comps[~remaining_comps.apply(is_valid_comp, axis=1)]
invalid_count = invalid_remaining['comp_id'].nunique()

print(f"2. Out-of-scope competitions remaining: {invalid_count}")
assert invalid_count == 0, f"CRITICAL: {invalid_count} out-of-scope competitions remain!"

ipl_comps = remaining_comps[remaining_comps['comp_name'].str.upper().str.contains('IPL')]
print(f"   Unique IPL competitions retained: {ipl_comps['comp_name'].unique().tolist()}")

# 3. Confirm filtered counts
print(f"3. Filtered Match Count: {len(matches_df)}")
print(f"   Filtered Delivery Count: {len(delivery_df)}")

# 4. Confirm raw EB shrunken VOA Z-scores
assert 'z_batting_voa' in top_10_batters.columns, "CRITICAL: 'z_batting_voa' column missing!"

b_raw_z = top_10_batters['z_batting_voa'].head(3).tolist()
bw_raw_z = top_10_bowlers['z_bowling_performance_PROVISIONAL'].head(3).tolist()

print(f"4. Batters raw EB Z-scores exact values (Top 3): {b_raw_z}")
print(f"   Bowlers raw EB Z-scores exact values (Top 3): {bw_raw_z}")

assert (top_10_batters['z_batting_voa'] == 0.0).all(), "CRITICAL: Batter raw EB Z-scores are not exactly 0.0"
assert (top_10_bowlers['z_bowling_performance_PROVISIONAL'] == 0.0).all(), "CRITICAL: Bowler raw EB Z-scores are not exactly 0.0"

# 5. Validated synthetic tests
micro_variance = pd.Series([0.0, 2e-10, 4e-10])
std_micro = micro_variance.std(ddof=1)
z_micro = safe_zscore(micro_variance).tolist()

print(f"5a. Synthetic micro variance std: {std_micro:.3e}")
print(f"5b. Synthetic micro variance Z-score output: {z_micro}")
assert 0 < std_micro < 1e-9, "CRITICAL: Micro variance std is outside expected range!"
assert all(z == 0.0 for z in z_micro), "CRITICAL: Micro variance did not clamp to exactly 0.0!"

synthetic_normal = pd.Series([10.0, 20.0, 30.0])
z_normal = safe_zscore(synthetic_normal).tolist()
print(f"5c. Synthetic normal variance Z-score output: {z_normal}")
assert np.allclose(z_normal, [-1.0, 0.0, 1.0]), "CRITICAL: Normal variance Z-score logic failed!"

# 6. Confirm leaderboard completion status
b_na = top_10_batters[['Final_Score', 'z_batting_performance', 'z_batting_consistency']].isna().sum().sum()
bw_na = top_10_bowlers[['Final_Score_PROVISIONAL', 'z_bowling_performance_PROVISIONAL', 'z_bowling_consistency_PROVISIONAL']].isna().sum().sum()

print(f"6. Final Batters Leaderboard rows: {len(top_10_batters)} (Missing metrics: {b_na})")
print(f"   Final Bowlers Leaderboard rows: {len(top_10_bowlers)} (Missing metrics: {bw_na})")

assert len(top_10_batters) == 10, f"CRITICAL: Batter leaderboard has {len(top_10_batters)} rows, expected 10."
assert len(top_10_bowlers) == 10, f"CRITICAL: Bowler leaderboard has {len(top_10_bowlers)} rows, expected 10."
assert b_na == 0, "CRITICAL: Missing metrics in batters leaderboard!"
assert bw_na == 0, "CRITICAL: Missing metrics in bowlers leaderboard!"

print("\nALL ASSERTIONS PASSED. PIPELINE STATISTICALLY VALIDATED.")