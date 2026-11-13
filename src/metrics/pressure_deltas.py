import pandas as pd
import numpy as np

def calculate_pressure_deltas(pressure_df: pd.DataFrame, eligible_batters: list, eligible_bowlers: list):
    """
    Calculates Value Over Average (VOA) and Consistency for eligible players.
    Uses strict Leave-One-Out (LOO) benchmarking per player-competition combination.
    Propagates PROVISIONAL status appropriately based on input verification.
    """
    df = pressure_df.copy()
    
    # 1. FILTER FOR VALID PRESSURE DELIVERIES
    valid_mask = (df['is_pressure'] == True) & (df['pressure_status'] != 'BLOCKED')
    pdf = df[valid_mask].copy()
    
    # Standardize competition column
    comp_col = 'comp_id' if 'comp_id' in pdf.columns else ('competition_id' if 'competition_id' in pdf.columns else None)
    if comp_col is None:
        raise ValueError("Competition ID is required for competition-specific LOO benchmarking.")

    # =========================================================================
    # PART A: BATTING DELTAS & VOA
    # =========================================================================
    
    # Structure valid batting actions (handling non-striker dismissals)
    bat_base = pdf.dropna(subset=['striker_id', 'calc_batter_runs', 'is_batter_ball_faced', 'is_batter_dismissal']).copy()
    
    # Strikers
    strikers = bat_base.copy()
    strikers['eligible_batter_id'] = strikers['striker_id']
    strikers['batter_runs'] = strikers['calc_batter_runs']
    strikers['balls_faced'] = strikers['is_batter_ball_faced'].astype(int)
    strikers['is_my_dismissal'] = (strikers['is_batter_dismissal'] & (strikers['dismissed_player_id'] == strikers['striker_id'])).astype(int)
    
    # Dismissal Opportunities = Balls Faced OR Dismissals (prevents double-counting a bowled out delivery)
    strikers['opportunity'] = (strikers['is_batter_ball_faced'] | (strikers['is_my_dismissal'] == 1)).astype(int)
    
    # Non-strikers dismissed
    ns_mask = bat_base['is_batter_dismissal'] & (bat_base['dismissed_player_id'] != bat_base['striker_id']) & bat_base['dismissed_player_id'].notna()
    non_strikers = bat_base[ns_mask].copy()
    non_strikers['eligible_batter_id'] = non_strikers['dismissed_player_id']
    non_strikers['batter_runs'] = 0
    non_strikers['balls_faced'] = 0
    non_strikers['is_my_dismissal'] = 1
    non_strikers['opportunity'] = 1 # 0 balls faced + 1 dismissal = 1 opportunity
    
    bat_actions = pd.concat([strikers, non_strikers], ignore_index=True)
    bat_actions = bat_actions[bat_actions['eligible_batter_id'].isin(eligible_batters)]
    
    # Competition-level totals (for LOO subtraction)
    comp_bat_totals = bat_base.groupby(comp_col).agg(
        comp_runs=('calc_batter_runs', 'sum'),
        comp_balls=('is_batter_ball_faced', 'sum')
    ).reset_index()
    
    # Player-Competition totals
    p_comp_bat = bat_actions.groupby(['eligible_batter_id', comp_col]).agg(
        p_runs=('batter_runs', 'sum'),
        p_balls=('balls_faced', 'sum'),
        pressure_dismissals=('is_my_dismissal', 'sum'),
        pressure_opportunities=('opportunity', 'sum')
    ).reset_index()
    
    # Calculate LOO Benchmark per player-competition
    p_comp_bat = p_comp_bat.merge(comp_bat_totals, on=comp_col, how='left')
    p_comp_bat['loo_balls'] = p_comp_bat['comp_balls'] - p_comp_bat['p_balls']
    p_comp_bat['loo_runs'] = p_comp_bat['comp_runs'] - p_comp_bat['p_runs']
    
    p_comp_bat['loo_sr'] = np.where(
        p_comp_bat['loo_balls'] > 0,
        (p_comp_bat['loo_runs'] / p_comp_bat['loo_balls']) * 100,
        np.nan # Strict rule: If 0 or negative LOO denominator, return NaN
    )
    
    # Expected runs via LOO to allow safe multi-competition aggregation
    p_comp_bat['expected_runs'] = p_comp_bat['p_balls'] * (p_comp_bat['loo_sr'] / 100)
    
    # Overall Batting VOA (Aggregated across competitions)
    batting_deltas = p_comp_bat.groupby('eligible_batter_id').agg(
        pressure_runs=('p_runs', 'sum'),
        pressure_balls_faced=('p_balls', 'sum'),
        pressure_dismissals=('pressure_dismissals', 'sum'),
        pressure_opportunities=('pressure_opportunities', 'sum'),
        total_expected_runs=('expected_runs', lambda x: x.sum(min_count=1)),
        total_comps=(comp_col, 'count'),
        valid_loo_comps=('loo_sr', 'count')
    ).reset_index()
    
    # If any competition has an invalid LOO benchmark, invalidate overall expected runs
    invalid_loo_mask_bat = batting_deltas['total_comps'] != batting_deltas['valid_loo_comps']
    batting_deltas.loc[invalid_loo_mask_bat, 'total_expected_runs'] = np.nan
    
    batting_deltas['batting_sr'] = (batting_deltas['pressure_runs'] / batting_deltas['pressure_balls_faced']) * 100
    batting_deltas['overall_loo_sr'] = (batting_deltas['total_expected_runs'] / batting_deltas['pressure_balls_faced']) * 100
    batting_deltas['batting_voa'] = batting_deltas['batting_sr'] - batting_deltas['overall_loo_sr']

    # Batting Consistency (Per-Innings VOA Standard Deviation)
    inn_bat = bat_actions.groupby(['eligible_batter_id', comp_col, 'match_id', 'innings']).agg(
        inn_runs=('batter_runs', 'sum'),
        inn_balls=('balls_faced', 'sum')
    ).reset_index()
    
    inn_bat = inn_bat[inn_bat['inn_balls'] > 0] # Exclude 0-ball innings
    inn_bat['inn_sr'] = (inn_bat['inn_runs'] / inn_bat['inn_balls']) * 100
    
    inn_bat = inn_bat.merge(p_comp_bat[['eligible_batter_id', comp_col, 'loo_sr']], on=['eligible_batter_id', comp_col], how='left')
    inn_bat['inn_voa'] = inn_bat['inn_sr'] - inn_bat['loo_sr']
    
    bat_cons = inn_bat.groupby('eligible_batter_id').agg(
        valid_batting_innings=('inn_voa', 'count'),
        batting_consistency=('inn_voa', lambda x: x.std(ddof=1) if len(x) >= 2 else np.nan)
    ).reset_index()
    
    batting_deltas = batting_deltas.merge(bat_cons, on='eligible_batter_id', how='left')
    batting_deltas = batting_deltas.drop(columns=['total_expected_runs', 'overall_loo_sr', 'total_comps', 'valid_loo_comps'])


    # =========================================================================
    # PART B: BOWLING DELTAS & VOA (PROVISIONAL PROPAGATION)
    # =========================================================================
    
    bowl_base = pdf.dropna(subset=['bowler_id', 'calc_bowler_runs_PROVISIONAL', 'is_legal_ball', 'is_bowler_wicket']).copy()
    bowl_actions = bowl_base[bowl_base['bowler_id'].isin(eligible_bowlers)].copy()
    
    comp_bowl_totals = bowl_base.groupby(comp_col).agg(
        comp_runs_PROV=('calc_bowler_runs_PROVISIONAL', 'sum'),
        comp_legal_balls=('is_legal_ball', 'sum')
    ).reset_index()
    
    p_comp_bowl = bowl_actions.groupby(['bowler_id', comp_col]).agg(
        p_runs_PROV=('calc_bowler_runs_PROVISIONAL', 'sum'),
        p_balls=('is_legal_ball', 'sum'),
        pressure_wickets=('is_bowler_wicket', 'sum')
    ).reset_index()
    
    p_comp_bowl = p_comp_bowl.merge(comp_bowl_totals, on=comp_col, how='left')
    p_comp_bowl['loo_balls'] = p_comp_bowl['comp_legal_balls'] - p_comp_bowl['p_balls']
    p_comp_bowl['loo_runs_PROV'] = p_comp_bowl['comp_runs_PROV'] - p_comp_bowl['p_runs_PROV']
    
    p_comp_bowl['loo_econ_PROVISIONAL'] = np.where(
        p_comp_bowl['loo_balls'] > 0,
        p_comp_bowl['loo_runs_PROV'] / (p_comp_bowl['loo_balls'] / 6),
        np.nan # Strict rule implementation
    )
    
    p_comp_bowl['expected_runs_PROV'] = (p_comp_bowl['p_balls'] / 6) * p_comp_bowl['loo_econ_PROVISIONAL']
    
    bowling_deltas = p_comp_bowl.groupby('bowler_id').agg(
        pressure_runs_conceded_PROVISIONAL=('p_runs_PROV', 'sum'),
        pressure_legal_balls=('p_balls', 'sum'),
        pressure_wickets=('pressure_wickets', 'sum'),
        total_expected_runs_PROV=('expected_runs_PROV', lambda x: x.sum(min_count=1)),
        total_comps=(comp_col, 'count'),
        valid_loo_comps=('loo_econ_PROVISIONAL', 'count')
    ).reset_index()
    
    # If any competition has an invalid LOO benchmark, invalidate overall expected runs
    invalid_loo_mask_bowl = bowling_deltas['total_comps'] != bowling_deltas['valid_loo_comps']
    bowling_deltas.loc[invalid_loo_mask_bowl, 'total_expected_runs_PROV'] = np.nan
    
    bowling_deltas['bowling_econ_PROVISIONAL'] = bowling_deltas['pressure_runs_conceded_PROVISIONAL'] / (bowling_deltas['pressure_legal_balls'] / 6)
    bowling_deltas['overall_loo_econ_PROVISIONAL'] = bowling_deltas['total_expected_runs_PROV'] / (bowling_deltas['pressure_legal_balls'] / 6)
    
    # Bowling VOA = LOO Expected Econ - Player Econ (Higher is better)
    bowling_deltas['bowling_voa_PROVISIONAL'] = bowling_deltas['overall_loo_econ_PROVISIONAL'] - bowling_deltas['bowling_econ_PROVISIONAL']
    
    # Bowling Consistency (Per-Innings VOA Standard Deviation)
    inn_bowl = bowl_actions.groupby(['bowler_id', comp_col, 'match_id', 'innings']).agg(
        inn_runs_PROV=('calc_bowler_runs_PROVISIONAL', 'sum'),
        inn_balls=('is_legal_ball', 'sum')
    ).reset_index()
    
    inn_bowl = inn_bowl[inn_bowl['inn_balls'] > 0]
    inn_bowl['inn_econ_PROV'] = inn_bowl['inn_runs_PROV'] / (inn_bowl['inn_balls'] / 6)
    
    inn_bowl = inn_bowl.merge(p_comp_bowl[['bowler_id', comp_col, 'loo_econ_PROVISIONAL']], on=['bowler_id', comp_col], how='left')
    inn_bowl['inn_voa_PROVISIONAL'] = inn_bowl['loo_econ_PROVISIONAL'] - inn_bowl['inn_econ_PROV']
    
    bowl_cons = inn_bowl.groupby('bowler_id').agg(
        valid_bowling_innings=('inn_voa_PROVISIONAL', 'count'),
        bowling_consistency_PROVISIONAL=('inn_voa_PROVISIONAL', lambda x: x.std(ddof=1) if len(x) >= 2 else np.nan)
    ).reset_index()
    
    bowling_deltas = bowling_deltas.merge(bowl_cons, on='bowler_id', how='left')
    bowling_deltas = bowling_deltas.drop(columns=['total_expected_runs_PROV', 'overall_loo_econ_PROVISIONAL', 'total_comps', 'valid_loo_comps'])
    
    return batting_deltas, bowling_deltas