import pandas as pd
import numpy as np

def build_delivery_events(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Single source of truth for cricket delivery events.
    Appends standardized scoring, delivery, and dismissal fields.
    Preserves the raw dataframe immutably and tracks multiple validation issues.
    """
    df = raw_df.copy()
    
    # Initialize validation tracking
    df['validation_issues'] = [[] for _ in range(len(df))]
    df['delivery_status'] = 'OK'
    
    def add_issue(mask, issue_text, status_level):
        df.loc[mask, 'validation_issues'] = df.loc[mask, 'validation_issues'].apply(lambda x: x + [issue_text])
        # Escalate status: BLOCKED overrides PROVISIONAL which overrides OK
        current_blocked = df['delivery_status'] == 'BLOCKED'
        new_blocked = mask & (status_level == 'BLOCKED')
        new_prov = mask & (status_level == 'PROVISIONAL') & (~current_blocked)
        df.loc[new_blocked, 'delivery_status'] = 'BLOCKED'
        df.loc[new_prov, 'delivery_status'] = 'PROVISIONAL'

    # ---------------------------------------------------------
    # WICKET NORMALIZATION
    # ---------------------------------------------------------
    wicket_map = {
        '1': True, 1: True, 'true': True, 'True': True, 
        '0': False, 0: False, 'false': False, 'False': False
    }
    df['is_wicket_norm'] = df['is_wicket'].map(wicket_map).astype('boolean')
    
    unresolved_wicket = df['is_wicket'].notna() & df['is_wicket_norm'].isna()
    add_issue(unresolved_wicket, 'UNRESOLVED_WICKET_INDICATOR', 'BLOCKED')

    # ---------------------------------------------------------
    # DISMISSAL CLASSIFICATION
    # ---------------------------------------------------------
    df['dismissal_type_norm'] = df['dismissal_type'].astype(str).str.lower().str.strip()
    df['dismissal_type_norm'] = df['dismissal_type_norm'].replace(['nan', 'none', ''], pd.NA)

    bowler_credit_dismissals = ['bowled', 'caught', 'caught and bowled', 'lbw', 'stumped', 'hit wicket']
    non_bowler_dismissals = ['run out', 'retire out', 'obstructing the field', 'retire hurt']
    known_dismissals = bowler_credit_dismissals + non_bowler_dismissals

    unknown_dismissal = ((df['is_wicket_norm'] == True) &(df['dismissal_type_norm'].notna()) &(~df['dismissal_type_norm'].isin(known_dismissals)))
    add_issue(unknown_dismissal, 'UNKNOWN_DISMISSAL_TYPE', 'BLOCKED')

    missing_dismissal_type = ((df['is_wicket_norm'] == True) & (df['dismissal_type_norm'].isna()))
    add_issue(missing_dismissal_type, 'MISSING_DISMISSAL_TYPE', 'BLOCKED')

    missing_dismissed_player = ((df['is_wicket_norm'] == True) & (df['dismissed_player_id'].isna() |  (df['dismissed_player_id'].astype(str).str.strip() == '')))
    add_issue(missing_dismissed_player, 'MISSING_DISMISSED_PLAYER_ID', 'BLOCKED')

    missing_wicket_indicator = df['is_wicket'].isna()
    add_issue( missing_wicket_indicator, 'MISSING_WICKET_INDICATOR',  'BLOCKED')

    

    df['is_team_wicket'] = df['is_wicket_norm']
    
    # Nullable booleans to prevent unresolved rows from silently resolving to False
    df['is_batter_dismissal'] = pd.Series(pd.NA, index=df.index, dtype='boolean')
    df['is_bowler_wicket'] = pd.Series(pd.NA, index=df.index, dtype='boolean')

    valid_wicket_mask = ((df['is_team_wicket'] == True) &(~unknown_dismissal) &(~missing_dismissal_type))
    
    df.loc[valid_wicket_mask, 'is_batter_dismissal'] = (
        (df.loc[valid_wicket_mask, 'dismissed_player_id'].astype(str) == df.loc[valid_wicket_mask, 'striker_id'].astype(str)) & 
        (df.loc[valid_wicket_mask, 'dismissal_type_norm'] != 'retire hurt')
    )
    df.loc[valid_wicket_mask, 'is_bowler_wicket'] = df.loc[valid_wicket_mask, 'dismissal_type_norm'].isin(bowler_credit_dismissals)
    df.loc[df['is_team_wicket'] == False, ['is_batter_dismissal', 'is_bowler_wicket']] = False

    # ---------------------------------------------------------
    # EXTRAS NORMALIZATION
    # ---------------------------------------------------------
    df['extras_type_norm'] = df['extras_type'].astype(str).str.lower().str.strip()
    df['extras_type_norm'] = df['extras_type_norm'].replace(['nan', 'none', ''], pd.NA)
    
    known_extras = ['wide', 'overstep', 'legbyes', 'byes', 'penalty', 'other']
    unknown_extra = df['extras_type_norm'].notna() & ~df['extras_type_norm'].isin(known_extras)
    add_issue(unknown_extra, 'UNKNOWN_EXTRAS_TYPE', 'BLOCKED')

    # ---------------------------------------------------------
    # DELIVERY LOGIC
    # ---------------------------------------------------------
    df['is_legal_ball'] = pd.Series(pd.NA, index=df.index, dtype='boolean')
    df['is_batter_ball_faced'] = pd.Series(pd.NA, index=df.index, dtype='boolean')

    valid_delivery = ~unknown_extra
    
    # Legal ball: Ordinary delivery (NA) or explicitly not wide/overstep
    df.loc[valid_delivery, 'is_legal_ball'] = (
        df.loc[valid_delivery, 'extras_type_norm'].isna() | 
        ~df.loc[valid_delivery, 'extras_type_norm'].isin(['wide', 'overstep'])
    )
    
    # Batter ball faced: Ordinary delivery or explicitly overstep/legbyes/byes/penalty/other
    df.loc[valid_delivery, 'is_batter_ball_faced'] = (
        df.loc[valid_delivery, 'extras_type_norm'].isna() | 
        (df.loc[valid_delivery, 'extras_type_norm'] != 'wide')
    )

    # ---------------------------------------------------------
    # SCORING CALCULATIONS
    # ---------------------------------------------------------
    df['calc_batter_runs'] = pd.to_numeric(df['runs'], errors='coerce')
    df['raw_extras_2_runs'] = pd.to_numeric(df['extras_2_runs'], errors='coerce')
    
    add_issue(df['calc_batter_runs'].isna() & df['runs'].notna(), 'UNRESOLVED_BATTER_RUNS', 'BLOCKED')
    
    missing_extras_val = df['extras_type_norm'].notna() & df['raw_extras_2_runs'].isna()
    add_issue(missing_extras_val, 'MISSING_EXTRAS_VALUE', 'BLOCKED')

    df['calc_extras_runs_PROVISIONAL'] = df['raw_extras_2_runs']
    add_issue(df['extras_type_norm'].notna(), 'UNVERIFIED_EXTRAS_2_RUNS_MEANING', 'PROVISIONAL')

    df['calc_team_runs_PROVISIONAL'] = df['calc_batter_runs'] + df['calc_extras_runs_PROVISIONAL'].fillna(0)
    
    df['calc_bowler_runs_PROVISIONAL'] = df['calc_batter_runs'].copy()
    is_bowler_extra = df['extras_type_norm'].isin(['wide', 'overstep'])
    df.loc[is_bowler_extra, 'calc_bowler_runs_PROVISIONAL'] += df.loc[is_bowler_extra, 'calc_extras_runs_PROVISIONAL']

    negative_runs = (df['calc_batter_runs'] < 0) | (df['calc_extras_runs_PROVISIONAL'] < 0)
    add_issue(negative_runs, 'NEGATIVE_RUNS', 'BLOCKED')

    return df