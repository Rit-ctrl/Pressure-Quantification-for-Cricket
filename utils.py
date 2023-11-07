import pandas as pd

def process_df(df:pd.DataFrame,useful_columns = ['match_id', 'season', 'venue', 'innings', 'ball',
       'striker', 'non_striker', 'bowler',
       'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes',
       'penalty', 'wicket_type', 'player_dismissed', 'other_wicket_type',
       'other_player_dismissed']):

    extra_cols = ['wides','noballs','byes','legbyes','penalty']

    df_cleaned = df[useful_columns].copy()
    df_cleaned['wicket_type'].fillna(0,inplace=True)
    df_cleaned['wicket'] = 0
    df_cleaned.loc[df_cleaned.wicket_type !=0,'wicket'] = 1
    df_cleaned['wickets_left'] = 10 - df_cleaned.groupby(['innings'])['wicket'].cumsum()
#    df_cleaned.query('wicket == 1') 
    df_cleaned.loc[df_cleaned['wicket'] == 1,'wickets_left'] += 1

    df_cleaned['total_runs'] = df_cleaned['runs_off_bat'] + df_cleaned['extras']
    df_cleaned['runs_left'] = df_cleaned.groupby('innings')['total_runs'].cumsum() #let us accumulate runs grouped by innings in runs_left for now
    inn_1_total = df_cleaned.groupby('innings')['runs_left'].max().iloc[0] #gets total runs scored by first innings team (gets the target needed)
    target = inn_1_total + 1
    df_cleaned.loc[df_cleaned['innings'] == 1,'runs_left'] = inn_1_total - df_cleaned.loc[df_cleaned['innings'] == 1,'runs_left']
    df_cleaned.loc[df_cleaned['innings'] == 2,'runs_left'] = target - df_cleaned.loc[df_cleaned['innings'] == 2,'runs_left']

    df_cleaned.fillna(dict.fromkeys(extra_cols,0),inplace=True)
    df_cleaned = df_cleaned.loc[(df_cleaned.noballs == 0) & (df_cleaned.wides == 0)] #select only those that are not noballs or wides

    season = df_cleaned.season.unique()[0] 
    if season == '2020/21':
        df_cleaned['season'] = 2020
    elif season ==  '2007/08':
        df_cleaned['season'] = 2008
    elif season ==  '2009/10':
        df_cleaned['season'] = 2010


    return df_cleaned