import pandas as pd
import numpy as np

def compute_baselines_and_eligibility(pressure_df: pd.DataFrame):
    """
    Determines player eligibility based on the 5-pressure-innings evidence threshold.
    Calculates standard competition-level pressure benchmarks and global priors for VOA.
    """
    df = pressure_df.copy()
    
    # ---------------------------------------------------------
    # 1. FILTER FOR VALID PRESSURE DELIVERIES & SEPARATE DATASETS
    # ---------------------------------------------------------
    # Exclude any blocked match states or unresolved pressure flags
    valid_pressure_mask = (df['is_pressure'] == True) & (df['pressure_status'] != 'BLOCKED')
    pdf = df[valid_pressure_mask].copy()

    # Separate batting and bowling datasets so missing fields in one don't corrupt the other
    # Dropna only on core scoring/delivery indicators; dismissed_player_id can legitimately be NaN
    pdf_bat = pdf.dropna(subset=['striker_id', 'calc_batter_runs', 'is_batter_ball_faced', 'is_batter_dismissal']).copy()

    bowl_cols = ['bowler_id', 'calc_bowler_runs_PROVISIONAL', 'is_legal_ball', 'is_bowler_wicket']
    pdf_bowl = pdf.dropna(subset=bowl_cols).copy()

    # ---------------------------------------------------------
    # 2. ELIGIBILITY: 5 PRESSURE INNINGS THRESHOLD
    # ---------------------------------------------------------
    # Batters: Innings where they faced at least 1 valid pressure ball or were dismissed
    # 1. Faced actions (credited to the striker)
    faced_actions = pdf_bat[pdf_bat['is_batter_ball_faced'] == True].copy()
    faced_actions['eligible_batter_id'] = faced_actions['striker_id']
    
    # 2. Dismissal actions (credited to the dismissed player, handling non-striker run-outs)
    dismissal_actions = pdf_bat[pdf_bat['is_batter_dismissal'] == True].copy()
    dismissal_actions['eligible_batter_id'] = dismissal_actions['dismissed_player_id']
    
    # 3. Combine actions to ensure no striker activity is lost during a non-striker dismissal
    pdf_bat_actions = pd.concat([faced_actions, dismissal_actions])
    
    batter_inns = pdf_bat_actions.groupby(['eligible_batter_id', 'match_id', 'innings']).size().reset_index()
    batter_inns_count = batter_inns.groupby('eligible_batter_id').size()
    eligible_batters = batter_inns_count[batter_inns_count >= 5].index.tolist()

    # Bowlers: Innings where they bowled at least 1 valid legal pressure ball
    bowler_inns = pdf_bowl[pdf_bowl['is_legal_ball'] == True].groupby(['bowler_id', 'match_id', 'innings']).size().reset_index()
    bowler_inns_count = bowler_inns.groupby('bowler_id').size()
    eligible_bowlers = bowler_inns_count[bowler_inns_count >= 5].index.tolist()

    # ---------------------------------------------------------
    # 3. COMPETITION-LEVEL PRESSURE BENCHMARKS (PROVISIONAL)
    # ---------------------------------------------------------
    comp_col = 'comp_id' if 'comp_id' in pdf.columns else ('competition_id' if 'competition_id' in pdf.columns else None)
    
    if comp_col and not pdf.empty:
        comp_bat = pdf_bat.groupby(comp_col).agg(
            comp_pressure_batter_runs=('calc_batter_runs', 'sum'),
            comp_pressure_balls_faced=('is_batter_ball_faced', 'sum'),
            comp_pressure_batter_dismissals=('is_batter_dismissal', 'sum')
        )
        
        comp_bowl = pdf_bowl.groupby(comp_col).agg(
            comp_pressure_bowler_runs=('calc_bowler_runs_PROVISIONAL', 'sum'),
            comp_pressure_legal_balls=('is_legal_ball', 'sum'),
            comp_pressure_bowler_wickets=('is_bowler_wicket', 'sum')
        )
        
        # Merge batting and bowling benchmarks
        comp_benchmarks_PROVISIONAL = pd.merge(
            comp_bat, comp_bowl, left_index=True, right_index=True, how='outer'
        ).reset_index()
        
        # Calculate standard aggregate rates without fillna(0) to preserve undefined/NaN states
        comp_benchmarks_PROVISIONAL['comp_avg_pressure_sr'] = (
            comp_benchmarks_PROVISIONAL['comp_pressure_batter_runs'].astype('Float64') / 
            comp_benchmarks_PROVISIONAL['comp_pressure_balls_faced'].astype('Float64') * 100
        )
        
        comp_benchmarks_PROVISIONAL['comp_avg_pressure_econ'] = (
            comp_benchmarks_PROVISIONAL['comp_pressure_bowler_runs'].astype('Float64') / 
            (comp_benchmarks_PROVISIONAL['comp_pressure_legal_balls'].astype('Float64') / 6)
        )
    else:
        comp_benchmarks_PROVISIONAL = pd.DataFrame()

    # ---------------------------------------------------------
    # 4. GLOBAL PRIORS FOR BETA-BINOMIAL (DISMISSAL PROBABILITY)
    # ---------------------------------------------------------
    # Prior strength and the dismissal-opportunity definition will be finalized
    # and sensitivity-tested in the downstream smoothing module.
    global_priors = {
        'alpha': np.nan,
        'beta': np.nan,
        'global_bpd': np.nan
    }

    return eligible_batters, eligible_bowlers, comp_benchmarks_PROVISIONAL, global_priors