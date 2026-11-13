import pandas as pd
import numpy as np

def apply_pressure_flags(match_state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies individual deliveries as pressure or non-pressure based on pre-delivery match states.
    Applies garbage-time exclusions, evaluates conditions logically, and propagates validation statuses.
    """
    df = match_state_df.copy()

    # Initialize output columns
    df['is_pressure'] = pd.Series(pd.NA, index=df.index, dtype='boolean')
    df['is_garbage_time'] = pd.Series(False, index=df.index, dtype='boolean')
    df['pressure_reason'] = pd.Series(pd.NA, index=df.index, dtype='object')
    df['garbage_time_reason'] = pd.Series(pd.NA, index=df.index, dtype='object')
    df['pressure_status'] = df['match_state_status'].copy()
    
    def escalate_status(mask, status_level):
        current_blocked = df['pressure_status'] == 'BLOCKED'
        new_blocked = mask & (status_level == 'BLOCKED')
        new_prov = mask & (status_level == 'PROVISIONAL') & (~current_blocked)
        df.loc[new_blocked, 'pressure_status'] = 'BLOCKED'
        df.loc[new_prov, 'pressure_status'] = 'PROVISIONAL'

    # ---------------------------------------------------------
    # 1. BASELINE CALCULATIONS
    # ---------------------------------------------------------
    df['current_run_rate_PROVISIONAL'] = pd.Series(pd.NA, index=df.index, dtype='Float64')
    valid_rr_mask = (df['cumulative_legal_balls'] > 0)
    df.loc[valid_rr_mask, 'current_run_rate_PROVISIONAL'] = (
        df.loc[valid_rr_mask, 'cumulative_team_runs_PROVISIONAL'].astype('Float64') / 
        (df.loc[valid_rr_mask, 'cumulative_legal_balls'].astype('Float64') / 6)
    )

    # First Innings Competition Average RR
    comp_col = 'comp_id' if 'comp_id' in df.columns else ('competition_id' if 'competition_id' in df.columns else None)
    df['comp_avg_rr_inn1'] = pd.Series(pd.NA, index=df.index, dtype='Float64')
    
    is_inn1 = df['innings'] == 1
    is_inn2 = df['innings'] == 2
    
    if comp_col:
        # SCOPE NOTE: This computes a full-competition benchmark utilizing all dataset matches 
        # for the competition (including future matches relative to the current delivery). 
        # It is explicitly NOT a historical/pre-match benchmark.
        
        # Exclude BLOCKED and unresolved records from the competition average
        valid_inn1_mask = ( is_inn1 & (df['match_state_status'] != 'BLOCKED') &df['calc_team_runs_PROVISIONAL'].notna() & df['is_legal_ball'].notna())
        
        inn1_agg = df[valid_inn1_mask].groupby(comp_col).agg(
            total_runs=('calc_team_runs_PROVISIONAL', 'sum'),
            total_legal_balls=('is_legal_ball', 'sum')
        )
        inn1_agg['avg_rr'] = inn1_agg['total_runs'] / (inn1_agg['total_legal_balls'] / 6)
        df.loc[is_inn1, 'comp_avg_rr_inn1'] = df.loc[is_inn1, comp_col].map(inn1_agg['avg_rr'])
        
        # Preserve provisional status if the benchmark relies on provisional runs
        escalate_status(is_inn1 & df['comp_avg_rr_inn1'].notna(), 'PROVISIONAL')
    else:
        escalate_status(is_inn1, 'PROVISIONAL')
        df.loc[is_inn1, 'match_state_issues'] = df.loc[is_inn1, 'match_state_issues'].apply(lambda x: x + ['MISSING_COMPETITION_BENCHMARK'])

    # Wickets in preceding 8 overs (48 legal balls)
    df['wickets_last_48_balls'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    
    wkt_map_df = df.groupby(['match_id', 'innings', 'cumulative_legal_balls'])['cumulative_team_wickets'].max().reset_index()
    wkt_map_df['lookup_balls'] = wkt_map_df['cumulative_legal_balls'] + 48
    
    lookup_series = df.merge(
        wkt_map_df[['match_id', 'innings', 'lookup_balls', 'cumulative_team_wickets']], 
        left_on=['match_id', 'innings', 'cumulative_legal_balls'], 
        right_on=['match_id', 'innings', 'lookup_balls'], 
        how='left'
    )['cumulative_team_wickets_y']
    
    df['wickets_last_48_balls'] = df['cumulative_team_wickets'] - lookup_series.astype('Int64')

    # ---------------------------------------------------------
    # 2. EVALUATE INDIVIDUAL CONDITIONS
    # ---------------------------------------------------------
    # Innings 2 Conditions
    df['cond_inn2_rrr_high'] = (df['required_run_rate_PROVISIONAL'] > 10).astype('boolean')
    df['cond_inn2_balls_low'] = (df['balls_remaining'] <= 36).astype('boolean')
    df['cond_inn2_wickets_high'] = (df['cumulative_team_wickets'] >= 4).astype('boolean')
    
    # Innings 1 Conditions
    df['cond_inn1_early_wickets'] = ((df['cumulative_team_wickets'] >= 3) & (df['cumulative_legal_balls'] < 60)).astype('boolean')
    df['cond_inn1_mid_wickets'] = ((df['cumulative_team_wickets'] >= 5) & (df['cumulative_legal_balls'] < 90)).astype('boolean')
    
    df['cond_inn1_low_rr'] = (
        (df['cumulative_legal_balls'] >= 60) & 
        (df['current_run_rate_PROVISIONAL'] < (df['comp_avg_rr_inn1'] * 0.75))
    ).astype('boolean')

    # Garbage Time Conditions
    df['cond_gt_inn2_low_rrr'] = (is_inn2 & (df['required_run_rate_PROVISIONAL'] < 4)).astype('boolean')
    
    df['cond_gt_inn2_impossible'] = pd.Series(False, index=df.index, dtype='boolean')
    valid_gt_balls = is_inn2 & (df['balls_remaining'] > 0) & df['runs_required_PROVISIONAL'].notna()
    df.loc[valid_gt_balls, 'cond_gt_inn2_impossible'] = (
        df.loc[valid_gt_balls, 'runs_required_PROVISIONAL'].astype(float) > 
        (df.loc[valid_gt_balls, 'balls_remaining'].astype(float) / 6) * 36
    )
    
    df['cond_gt_inn1_cruise'] = (
        is_inn1 & 
        (df['cumulative_legal_balls'] >= 90) & 
        (df['current_run_rate_PROVISIONAL'] > 12) & 
        (df['wickets_last_48_balls'] == 0)
    ).astype('boolean')

    # ---------------------------------------------------------
    # 3. APPLY CLASSIFICATION LOGIC
    # ---------------------------------------------------------
    # Process Garbage Time First
    df.loc[df['cond_gt_inn2_low_rrr'] == True, ['is_garbage_time', 'garbage_time_reason']] = [True, 'INN2: RRR < 4']
    df.loc[df['cond_gt_inn2_impossible'] == True, ['is_garbage_time', 'garbage_time_reason']] = [True, 'INN2: Required > 36/over']
    df.loc[df['cond_gt_inn1_cruise'] == True, ['is_garbage_time', 'garbage_time_reason']] = [True, 'INN1: RR > 12 & 0 wkts in last 8 overs']

    # Process Innings 2 Pressure (Needs >= 2 conditions)
    # pd.NA propagates through addition without fillna(False)
    inn2_pressure_score = (
        df['cond_inn2_rrr_high'].astype('Int64') + 
        df['cond_inn2_balls_low'].astype('Int64') + 
        df['cond_inn2_wickets_high'].astype('Int64')
    )
    
    df.loc[is_inn2 & (inn2_pressure_score >= 2).fillna(False), ['is_pressure', 'pressure_reason']] = [True, 'INN2: >= 2 Thresholds Met']
    df.loc[is_inn2 & (inn2_pressure_score < 2).fillna(False), ['is_pressure', 'pressure_reason']] = [False, 'INN2: < 2 Thresholds Met']
    
    unresolved_inn2 = is_inn2 & inn2_pressure_score.isna()
    df.loc[unresolved_inn2, 'pressure_reason'] = 'UNRESOLVED_CONDITIONS'
    escalate_status(unresolved_inn2, 'BLOCKED')

    # Process Innings 1 Pressure (Needs >= 1 condition)
    inn1_pressure_score = (
        df['cond_inn1_early_wickets'].astype('Int64') + 
        df['cond_inn1_mid_wickets'].astype('Int64') + 
        df['cond_inn1_low_rr'].astype('Int64')
    )
    
    df.loc[is_inn1 & (inn1_pressure_score >= 1).fillna(False), ['is_pressure', 'pressure_reason']] = [True, 'INN1: >= 1 Threshold Met']
    df.loc[is_inn1 & (inn1_pressure_score < 1).fillna(False), ['is_pressure', 'pressure_reason']] = [False, 'INN1: 0 Thresholds Met']
    
    unresolved_inn1 = is_inn1 & inn1_pressure_score.isna()
    df.loc[unresolved_inn1, 'pressure_reason'] = 'UNRESOLVED_CONDITIONS'
    escalate_status(unresolved_inn1, 'BLOCKED')

    # Apply Overrides (Garbage Time, BLOCKED states, Super Overs)
    df.loc[df['is_garbage_time'] == True, ['is_pressure', 'pressure_reason']] = [False, 'GARBAGE_TIME_OVERRIDE']
    
    blocked_mask = df['pressure_status'] == 'BLOCKED'
    df.loc[blocked_mask, ['is_pressure', 'is_garbage_time', 'pressure_reason', 'garbage_time_reason']] = [pd.NA, pd.NA, 'BLOCKED_MATCH_STATE', pd.NA]
    
    super_over_mask = df['is_super_over'] == True
    # Exclude from pressure but preserve existing BLOCKED/PROVISIONAL status
    df.loc[super_over_mask, ['is_pressure', 'pressure_reason']] = [pd.NA, 'SUPER_OVER_EXCLUDED']

    # Clean up temporary lookup columns
    df = df.drop(columns=['current_run_rate_PROVISIONAL', 'comp_avg_rr_inn1', 'wickets_last_48_balls'])

    return df