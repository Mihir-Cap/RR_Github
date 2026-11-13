import pandas as pd
import numpy as np

def build_match_state(delivery_events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs the cumulative match context before each delivery.
    Propagates validation statuses to ensure corrupted raw deliveries 
    do not silently compute invalid targets or run rates.
    """
    df = delivery_events_df.copy()
    
    # ---------------------------------------------------------
    # 1. INITIALIZE STATUS & ISSUES
    # ---------------------------------------------------------
    df['match_state_issues'] = df['validation_issues'].copy()
    df['match_state_status'] = df['delivery_status'].copy()
    
    def add_state_issue(mask, issue_text, status_level):
        """Helper to append issues and escalate status without overwriting BLOCKED."""
        df.loc[mask, 'match_state_issues'] = df.loc[mask, 'match_state_issues'].apply(lambda x: x + [issue_text])
        current_blocked = df['match_state_status'] == 'BLOCKED'
        new_blocked = mask & (status_level == 'BLOCKED')
        new_prov = mask & (status_level == 'PROVISIONAL') & (~current_blocked)
        df.loc[new_blocked, 'match_state_status'] = 'BLOCKED'
        df.loc[new_prov, 'match_state_status'] = 'PROVISIONAL'

    # ---------------------------------------------------------
    # 2. DELIVERY SEQUENCE & COMPLETENESS VALIDATION
    # ---------------------------------------------------------
    # Sort securely using exact source columns or fail explicitly
    sort_cols = ['match_id', 'innings']
    if 'innings_ball_num' in df.columns:
        sort_cols.append('innings_ball_num')
    elif 'overid' in df.columns and 'over_ball_num' in df.columns:
        sort_cols.extend(['overid', 'over_ball_num'])
    else:
        raise ValueError("Unreliable delivery sequence: Neither 'innings_ball_num' nor ('overid', 'over_ball_num') columns found in the dataset.")
    
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    duplicate_mask = df.duplicated(subset=sort_cols, keep=False)
    add_state_issue(duplicate_mask, 'DUPLICATE_DELIVERY_SEQUENCE', 'BLOCKED')

    df['is_super_over'] = df['innings'] > 2

    # ---------------------------------------------------------
    # 3. PRE-DELIVERY CUMULATIVE STATE
    # ---------------------------------------------------------
    group_cols = ['match_id', 'innings']
    
    # Use skipna=False to ensure missing indicators don't silently evaluate to zero
    df['_running_wickets'] = df.groupby(group_cols)['is_team_wicket'].cumsum(skipna=False).astype('Int64')
    df['_running_legal_balls'] = df.groupby(group_cols)['is_legal_ball'].cumsum(skipna=False).astype('Int64')
    df['_running_runs_PROVISIONAL'] = df.groupby(group_cols)['calc_team_runs_PROVISIONAL'].cumsum(skipna=False).astype('Int64')
    
    # Track BLOCKED contamination
    df['_is_blocked_delivery'] = df['match_state_status'] == 'BLOCKED'
    df['_running_blocked'] = df.groupby(group_cols)['_is_blocked_delivery'].cumsum() > 0

    # Shift by 1 to get PRE-DELIVERY state
    df['cumulative_team_runs_PROVISIONAL'] = df.groupby(group_cols)['_running_runs_PROVISIONAL'].shift(1).astype('Int64')
    df['cumulative_team_wickets'] = df.groupby(group_cols)['_running_wickets'].shift(1).astype('Int64')
    df['cumulative_legal_balls'] = df.groupby(group_cols)['_running_legal_balls'].shift(1).astype('Int64')
    
    # The first ball of an innings definitively starts at 0
    is_first_ball = df.groupby(group_cols).cumcount() == 0
    df.loc[is_first_ball, 'cumulative_team_runs_PROVISIONAL'] = 0
    df.loc[is_first_ball, 'cumulative_team_wickets'] = 0
    df.loc[is_first_ball, 'cumulative_legal_balls'] = 0

    # Actively destroy data contaminated by a prior BLOCKED delivery
    pre_delivery_blocked = df.groupby(group_cols)['_running_blocked'].shift(1).eq(True)
    df.loc[pre_delivery_blocked, ['cumulative_team_runs_PROVISIONAL', 'cumulative_team_wickets', 'cumulative_legal_balls']] = pd.NA
    add_state_issue(pre_delivery_blocked, 'CUMULATIVE_STATE_CONTAMINATED_BY_PRIOR_BLOCK', 'BLOCKED')

    # ---------------------------------------------------------
    # 4. INNINGS ALLOCATION & DLS MATCH-LEVEL DETECTION
    # ---------------------------------------------------------
    df['innings_balls_allocation'] = pd.Series(120, index=df.index, dtype='Int64')
    is_inn2 = df['innings'] == 2
    
    # Dynamically detect any DLS fields in the raw dataset
    dls_cols = [col for col in df.columns if 'revised_overs' in col or 'revised_target' in col]
    
    if dls_cols:
        # Detect DLS interventions at the match level
        dls_match_ids = df.loc[df[dls_cols].notna().any(axis=1), 'match_id'].unique()
        is_dls_match_inn2 = df['match_id'].isin(dls_match_ids) & is_inn2
    else:
        is_dls_match_inn2 = pd.Series(False, index=df.index)
        
    if is_dls_match_inn2.any():
        add_state_issue(is_dls_match_inn2, 'UNVERIFIED_DLS_INTERVENTION_DETECTED', 'PROVISIONAL')
        # Erase standard allocations for every 2nd-innings delivery in affected matches
        df.loc[is_dls_match_inn2, 'innings_balls_allocation'] = pd.NA

    df['balls_remaining'] = df['innings_balls_allocation'] - df['cumulative_legal_balls']

    # ---------------------------------------------------------
    # 5. TARGET CALCULATION
    # ---------------------------------------------------------
    df['target_runs_PROVISIONAL'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    
    inn1_df = df[df['innings'] == 1]
    
    # Use skipna=False to explicitly fail the aggregate if any single delivery has missing runs
    inn1_totals = inn1_df.groupby('match_id')['calc_team_runs_PROVISIONAL'].apply(lambda x: x.sum(skipna=False))
    
    # Broadcast Target to Innings 2 WITHOUT fillna(0)
    df.loc[is_inn2, 'target_runs_PROVISIONAL'] = df.loc[is_inn2, 'match_id'].map(inn1_totals) + 1
    
    # Remove targets for DLS matches until verification is complete
    df.loc[is_dls_match_inn2, 'target_runs_PROVISIONAL'] = pd.NA

    # Block states missing a first-innings reference (including those destroyed by skipna=False)
    missing_inn1_mask = is_inn2 & df['target_runs_PROVISIONAL'].isna() & (~is_dls_match_inn2)
    add_state_issue(missing_inn1_mask, 'MISSING_FIRST_INNINGS_TARGET', 'BLOCKED')

    # Contaminate Innings 2 Target if Innings 1 was BLOCKED
    inn1_blocked = inn1_df[inn1_df['match_state_status'] == 'BLOCKED'].groupby('match_id').size() > 0
    inn2_with_blocked_target = is_inn2 & df['match_id'].map(inn1_blocked).eq(True)
    add_state_issue(inn2_with_blocked_target, 'TARGET_DERIVED_FROM_BLOCKED_INNINGS_1', 'BLOCKED')
    df.loc[inn2_with_blocked_target, 'target_runs_PROVISIONAL'] = pd.NA

    # ---------------------------------------------------------
    # 6. RUNS REQUIRED & RRR
    # ---------------------------------------------------------
    df['runs_required_PROVISIONAL'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    df['required_run_rate_PROVISIONAL'] = pd.Series(pd.NA, index=df.index, dtype='Float64')

    df.loc[is_inn2, 'runs_required_PROVISIONAL'] = (
        df.loc[is_inn2, 'target_runs_PROVISIONAL'] - df.loc[is_inn2, 'cumulative_team_runs_PROVISIONAL']
    )

    # Handle Chases that have already concluded
    chase_completed = is_inn2 & (df['runs_required_PROVISIONAL'] <= 0)
    add_state_issue(chase_completed, 'CHASE_ALREADY_COMPLETED', 'OK') 

    # Valid RRR Calculation
    valid_rrr = is_inn2 & (~chase_completed) & (df['balls_remaining'] > 0)
    df.loc[valid_rrr, 'required_run_rate_PROVISIONAL'] = (
        df.loc[valid_rrr, 'runs_required_PROVISIONAL'].astype('Float64') / 
        (df.loc[valid_rrr, 'balls_remaining'].astype('Float64') / 6)
    )

    zero_balls_remaining = is_inn2 & (~chase_completed) & (df['balls_remaining'] <= 0)
    add_state_issue(zero_balls_remaining, 'ZERO_BALLS_REMAINING_WITH_RUNS_REQUIRED', 'PROVISIONAL')

    df = df.drop(columns=['_running_runs', '_running_wickets', '_running_legal_balls', 
                          '_is_blocked_delivery', '_running_blocked'], errors='ignore')
    
    return df