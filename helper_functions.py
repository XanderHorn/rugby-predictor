import requests
import datetime
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import numpy as np
import os

def update_historical_performance():
    df = pd.read_csv('data/historical_results.csv')
    df = df.loc[df['date'] <= '2023-10-10']
    max_date = df['date'].max()

    from_ts = pd.to_datetime(df['date'].max()) + timedelta(days=1)
    matches = get_historical_matches_for_date_range(from_date = from_ts, to_date = datetime.now().strftime('%Y-%m-%d'))
    if matches.shape[0] > 0:
        matches = matches[['eventDateStart', 'rugbyTeam.home.name', 'rugbyTeam.away.name', 'score.total.home', 'score.total.away', 'tour.name', 'venue.venueName']]
        matches['city'] = None
        matches['country'] = np.where((matches['tour.name'] == 'Rugby World Cup') & (pd.to_datetime(matches['eventDateStart']).dt.year == 2023), 'France', None)
        matches['neutral'] = np.where((matches['tour.name'] == 'Rugby World Cup') & (pd.to_datetime(matches['eventDateStart']).dt.year == 2023) & (matches['rugbyTeam.home.name'] != 'France') &  (matches['rugbyTeam.away.name'] != 'France'), True, False)
        matches['world_cup'] = np.where((matches['tour.name'] == 'Rugby World Cup') & (pd.to_datetime(matches['eventDateStart']).dt.year == 2023), True, False)
        matches.rename(columns = {
            'eventDateStart': 'date',
            'rugbyTeam.home.name': 'home_team',
            'rugbyTeam.away.name': 'away_team',
            'score.total.home': 'home_score',
            'score.total.away': 'away_score',
            'tour.name': 'competition',
            'venue.venueName': 'stadium',
        }, inplace = True)
        matches['date'] = pd.to_datetime(matches['date']).dt.date
        matches['home_team'] = np.where(matches['home_team'] == 'Springboks', 'South Africa', matches['home_team'])
        matches['away_team'] = np.where(matches['away_team'] == 'Springboks', 'South Africa', matches['away_team'])
        matches = matches[df.columns]
        df = pd.concat([df, matches], axis=0)
        df.drop_duplicates(inplace=True)
        #df.to_csv('data/historical_results.csv', index=False)
    return df, max_date

def format_column_names(cols: list) -> list:
    new_cols = []
    for col in cols:
        col = ''.join(['_'+c.lower() if c.isupper() else c for c in col]).lstrip('_')
        col = col.replace('.', '_')
        col = col.replace(' ', '_')
        new_cols.append(col)
    return new_cols

def get_team_world_rankings() -> pd.DataFrame:
    out_df = pd.DataFrame()
    for year in range(2003,2024):
        print(year)
        for date in pd.date_range(f"{year}-12-20", f"{year}-12-31"):
            url = f"https://api.wr-rims-prod.pulselive.com/rugby/v3/rankings/mru?language=en&date={date.date()}"
            res = requests.get(url)
            if res.status_code == 200 and res.json():
                tmp_df = pd.json_normalize(res.json()['entries'])
                tmp_df['date'] = date
                out_df = pd.concat([out_df, tmp_df])

    out_df['year_month'] = out_df['date'].dt.strftime('%Y-%m')
    out_df['last_date_in_year'] = out_df.groupby('year_month')['date'].transform('max')
    out_df = out_df.loc[out_df['date'] == out_df['last_date_in_year']]
    out_df['year'] = out_df['date'].dt.year
    out_df.drop(['date', 'last_date_in_year', 'year_month'], axis=1, inplace=True)
    out_df.columns = [
        'points', 'position', 'previous_points', 'previous_position', 'team_id', 'team_alt_id',
        'team_name', 'team_short_name', 'team_country_code', 'team_annotations', 'year'
    ]
    return out_df

def get_historical_matches_for_date_range(from_date: str, to_date: str, nr_results_per_day: int = 100) -> pd.DataFrame:
    out_df = pd.DataFrame()
    for date in tqdm(pd.date_range(from_date, to_date, freq='5D')):
        
        from_date_unix = int(datetime.strptime(date.strftime('%Y-%m-%d'), '%Y-%m-%d').timestamp())
        to_date_unix = int(datetime.strptime((date + timedelta(5)).strftime('%Y-%m-%d'),'%Y-%m-%d').timestamp())

        url = f"https://supersport.com/apix/rugby/v4.1/feed/score/summary?top={nr_results_per_day}&eventStatusIds=3&entityTagIds=8278e65c-6c8a-4527-9941-cd001dca9382&startDate={from_date_unix}&endDate={to_date_unix}"
        res = requests.get(url)
        if res.status_code != 200:
            print(res.text)
            for i in range(5):
                print('Retry nr: ', i)
                res = requests.get(url)
                if res.status_code == 200:
                    break
            
        elif res.status_code == 200 and res.json():
            df = pd.json_normalize(res.json()['Summary'])
            out_df = pd.concat([out_df, df])

            out_df['row_number'] = out_df.groupby('feedId').cumcount() + 1
            out_df = out_df.loc[out_df['row_number'] == 1]
            del out_df['row_number']

    return out_df

def get_match_timeline(match_id: str) -> pd.DataFrame:
    res = requests.get(f"https://supersport.com/apix/rugby/v5/match/{match_id}/events")
    if res.status_code == 200 and res.json():
        df = pd.json_normalize(res.json()['events'])
        df['match_id'] = match_id
        out_cols = [
            'player', 'side', 'type', 'team.name', 'time.minutes', 'time.seconds', 'player.id', 'player.sportsPersonId', 'player.name',
            'player.firstName', 'player.lastName', 'player.positionName', 'match_id'
        ]
        df[out_cols]
        df.columns = format_column_names(df.columns)
        return df

def get_match_stats(match_id: str) -> pd.DataFrame:
    res = requests.get(f"https://supersport.com/apix/rugby/v5.1/match/{match_id}/stats")
    if res.status_code == 200 and res.json():
        df = pd.json_normalize(res.json()['data'])
        df['match_id'] = match_id
        df.columns = format_column_names(df.columns)
        return df
    
def format_match_lineup(json_data: dict, match_id: str) -> pd.DataFrame:
    out_df = pd.DataFrame()
    team_out_cols = [
        'position', 'shirtNumber', 'player.firstName', 'player.lastName',
        'player.fullName', 'player.shortName', 'player.sportsPersonId']
    for team in ['home', 'away']:
        df = pd.json_normalize(json_data['data']['lineups'][team]['players']['main'])[team_out_cols]
        df['is_starting_player'] = 1
        df = pd.concat([df, pd.json_normalize(json_data['data']['lineups'][team]['players']['substitutes'])[team_out_cols]])
        df['coach_name'] = json_data['data']['lineups'][team]['coach']['fullName']
        df['coach_id'] = json_data['data']['lineups'][team]['coach']['sportsPersonId']
        df['team'] = team
        for col in ['coach_name', 'coach_id', 'team']:
            if col not in df.columns:
                df[col] = None
        out_df = pd.concat([out_df, df], axis = 0)
    out_df['is_starting_player'].fillna(0, inplace=True)
    out_df['match_id'] = match_id
    out_df.columns = format_column_names(out_df.columns)
    return out_df
    
def get_match_lineup(match_id: str) -> pd.DataFrame:
    res = requests.get(f"https://supersport.com/apix/rugby/v5/match/{match_id}/lineups")
    if res.status_code == 200 and res.json():
        return format_match_lineup(res.json(), match_id)
    
def format_player_match_info(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'tries', 'conversion_goal_success_rate', 'lineout_success_rate', 'penalty_goal_success_rate', 'minutes_played', 'points',
        'try_assists', 'kicks_from_hand', 'carries_meters', 'carries_crossed_gain_line', 'clean_breaks', 'passes', 'offloads',
        'handling_errors', 'meters_carried', 'attacking_rucks', 'defensive_rucks', 'tackles', 'dominant_tackles', 'missed_tackles',
        'turn_over_won', 'penalties_conceded', 'yellow_cards', 'red_cards'
    ]
    df.columns = cols

    df['conversions'] = int(df['conversion_goal_success_rate'][0].split('/')[0])
    del df['conversion_goal_success_rate']
    df['lineouts_won'] = int(df['lineout_success_rate'][0].split('/')[0])
    del df['lineout_success_rate']
    df['penalty_goals'] = int(df['penalty_goal_success_rate'][0].split('/')[0])
    del df['penalty_goal_success_rate']
    return df

def get_player_match_info(match_id: str, player_id: str) -> pd.DataFrame:
    res = requests.get(f"https://supersport.com/apix/rugby/v5/match/{match_id}/player/{player_id}/stats")
    if res.status_code == 200 and res.json():
        df = format_player_match_info(pd.json_normalize(res.json()['data']))
        df['match_id'] = match_id
        df['player_id'] = player_id

        try:
            player_info = pd.DataFrame(get_player_info(player_id), index=[0])
        except:
            player_info = pd.DataFrame()
        df = pd.concat([df, player_info], axis = 1)
        return df
    
def get_player_info(player_id: str) -> dict:
    url = f"https://supersport.com/rugby/players/{player_id}"
    res = requests.get(url)

    soup = BeautifulSoup(res.text, 'html.parser')
    weight_kg = soup.find('div', string='Weight').parent.text
    weight_kg = int(re.sub('[^0-9]', '', weight_kg))

    height_cm = soup.find('div', string='Height').parent.text
    height_cm = int(re.sub('[^0-9]', '', height_cm))

    birth_date = soup.find('div',class_='text-neutral-300', string = 'Date of Birth').parent.text.replace('Date of Birth', '')
    birth_date = datetime.strptime(birth_date, '%d/%m/%Y').strftime('%Y-%m-%d')
    bmi = weight_kg / (height_cm/100)**2
    return {'weight_kg': weight_kg, 'height_cm': height_cm, 'birth_date': birth_date, 'bmi': bmi}

def get_team_rolling_performance(df: pd.DataFrame, past_n_matches:int=5) -> pd.DataFrame:

    df['date'] = pd.to_datetime(df['date']).dt.date
    df['home_rolling_performance'] = 0
    df['away_rolling_performance'] = 0
    for team in df['home_team'].unique():
        tmp_df = df.loc[(df['home_team'] == team) | (df['away_team'] == team)].copy().sort_values('date')
        tmp_df['performance'] = np.where(tmp_df['winner'] == team, 1, 
                                        np.where(tmp_df['loser'] == team, 0, 0.5))
        tmp_df['rolling_performance'] = tmp_df['performance'].shift(1).rolling(past_n_matches, min_periods=1).sum()
        tmp_df['team'] = team
        tmp_df['rolling_performance'] = np.where(tmp_df.reset_index().index == 0, 0, tmp_df['rolling_performance'])

        df = df.merge(tmp_df[['date','home_team','away_team','rolling_performance']], how = 'left', on = ['date','home_team','away_team'])
        df['home_rolling_performance'] = np.where(df['home_team'] == team, df['rolling_performance'], df['home_rolling_performance'])
        df['away_rolling_performance'] = np.where(df['away_team'] == team, df['rolling_performance'], df['away_rolling_performance'])
        del df['rolling_performance']
    return df

def get_rankings(df: pd.DataFrame) -> pd.DataFrame:

    teams = list(df['home_team'].unique()) + list(df['away_team'].unique())
    teams = list(set(teams))
    ranking_points = {team: 80 for team in set(teams)}

    for i, row in df.iterrows():

        home_team = row['home_team']
        away_team = row['away_team']
        
        df.at[i, 'ranking_points_home'] = ranking_points[home_team]
        df.at[i, 'ranking_points_away'] = ranking_points[away_team]
        if row['neutral'] == True:
            home_points = ranking_points[home_team]
        else:
            home_points = ranking_points[home_team] + 3
        away_points = ranking_points[away_team]
        gap = home_points - away_points
        if gap < -10:
            gap = -10
        elif gap > 10:
            gap = 10
        if row['winner'] == 'DRAW':
            core = gap*0.1
        elif row['winner'] == home_team:
            core = 1 - (gap*0.1)
        else:
            core = 1 + (gap*0.1)
            
        if np.abs(row['home_score'] - row['away_score']) > 15:
            core *= 1.5
            
        if row['world_cup'] == True:
            core *= 2
            
        if row['winner'] != 'DRAW':
            ranking_points[row['winner']] += core
            ranking_points[row['loser']] -= core
        else:
            ranking_points[home_team] -= core
            ranking_points[away_team] += core
    return df

def format_timeline_data(match_df):

    files = os.listdir('data/match_timeline')
    timeline_df = pd.DataFrame()
    for file in files:
        timeline_df = pd.concat([timeline_df, pd.read_csv(f'data/match_timeline/{file}')], axis=0)

    timeline_df['team_name'] = np.where(timeline_df['team_name'].str.lower() == 'springboks', 'South Africa', timeline_df['team_name'])
    timeline_df = timeline_df.groupby(['match_id', 'time_minutes','team_name','type']).size().reset_index(name='count')

    event_mappings = {
        'Try':5,
        'Conversion':2,
        'Penalty':3,
        'Drop Goal':3,
        'Penalty Try':5
    }
    timeline_df['score'] = timeline_df['type'].map(event_mappings)
    timeline_df['score'].fillna(0, inplace = True)
    timeline_df['yellow_card'] = np.where(timeline_df['type'].str.lower() == 'yellow card', 1, 0)
    timeline_df['red_card'] = np.where(timeline_df['type'].str.lower().isin(['red card','red-card']), 1, 0)

    timeline_df['time_bucket'] = pd.cut(timeline_df['time_minutes'], bins=[-1, 10, 20, 30, 40, 50, 60, 70, 80, 999], labels=[10,20,30,40,50,60,70,80,999])
    timeline_df['time_bucket'] = timeline_df['time_bucket'].astype(int)

    tmp_df = match_df[['match_id', 'rugby_team_home_name']].copy()
    tmp_df['is_home_team'] = 1
    timeline_df = timeline_df.merge(tmp_df, how='left', left_on=['match_id', 'team_name'], right_on=['match_id', 'rugby_team_home_name'])
    timeline_df['is_home_team'].fillna(0, inplace=True)

    out_df = pd.DataFrame({
        'match_id':timeline_df['match_id'].unique()
    })
    for is_home_team in [0,1]:
        for i in timeline_df['time_bucket'].unique():
            tmp_df = timeline_df.loc[(timeline_df['time_bucket'] <= i) & (timeline_df['is_home_team'] == is_home_team)].copy()
            tmp_df = tmp_df.groupby(['match_id']).agg({
                'score':'sum',
                'yellow_card':'sum',
                'red_card':'sum'
            }).reset_index()
            if is_home_team == 1:
                tmp_df.rename(columns = {
                    'score':f'timeline_{i}_home_score',
                    'yellow_card':f'timeline_{i}_home_yellow_cards',
                    'red_card':f'timeline_{i}_home_red_cards'
                }, inplace=True)
            else:
                tmp_df.rename(columns = {
                    'score':f'timeline_{i}_away_score',
                    'yellow_card':f'timeline_{i}_away_yellow_cards',
                    'red_card':f'timeline_{i}_away_red_cards'
                }, inplace=True)
            out_df = out_df.merge(tmp_df, how='left', on='match_id')

    return out_df

def get_roc_auc_score(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)

def get_classification_report(y_true, y_pred):
    from sklearn.metrics import classification_report
    class_report = classification_report(y_true, y_pred)
    print("\nClassification report:\n")
    print(class_report)

def train_and_tune_random_forest(train_x, train_y, test_x=None, test_y=None, cv_folds=3, tune_iter = 10, seed=1991):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, KFold
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import label_binarize
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier())
    ])

    param_dist = {
        'classifier__n_estimators': np.arange(50, 501, 50),
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__max_depth': np.arange(5, 31, 5),
        'classifier__min_samples_split': np.arange(2, 21, 2),
        'classifier__min_samples_leaf': np.arange(1, 21, 2),
        'classifier__bootstrap': [True, False],
        'classifier__class_weight': ['balanced', 'balanced_subsample', None]
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    random_search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=tune_iter, 
        scoring='f1_weighted', 
        n_jobs=-1, 
        cv=cv, 
        verbose=1, 
        random_state=seed
    )

    random_search.fit(train_x, train_y)

    pipeline = random_search.best_estimator_

    roc_auc_scores = []
    cv_fold = 1
    for train_index, test_index in cv.split(train_x):
        
        X_train, X_test = train_x.iloc[train_index], train_x.iloc[test_index]
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict_proba(X_test)
        y_true_bin = label_binarize(y_test, classes=np.unique(train_y))
        auc = np.round(roc_auc_score(y_true_bin, y_pred[:, 1]),2)
        print(f"ROC AUC FOR CV FOLD {cv_fold}: {auc}")
        cv_fold += 1
        roc_auc_scores.append(auc)

    print(f"\nMean ROC AUC: {np.mean(roc_auc_scores)}")
    y_pred = pipeline.predict_proba(train_x)

    if test_x.shape[0] > 0 and len(test_y) > 0:
        test_pred = pipeline.predict_proba(test_x)
        test_auc = np.round(roc_auc_score(label_binarize(test_y, classes=np.unique(test_y)), test_pred[:, 1]),2)
        print("Test ROC AUC: ", test_auc)

    feature_importances = pipeline.named_steps['classifier'].feature_importances_
    features_df = pd.DataFrame({
        'feature': train_x.columns,
        'importance': feature_importances
    })

    features_df = features_df.sort_values(by='importance', ascending=False)

    return pipeline, features_df

def get_team_stats(match_df, stats_df, team_list):

    team_df = pd.DataFrame()
    for team in team_list:
        tmp_df = match_df.loc[(match_df['rugby_team_home_name'] == team) | (match_df['rugby_team_away_name'] == team) ].copy()
        tmp_df['is_home_team'] = np.where(tmp_df['rugby_team_home_name'] == team, 1, 0)
        tmp_df['score'] = np.where(tmp_df['is_home_team'] == 1, tmp_df['score_total_home'], tmp_df['score_total_away'])
        tmp_df['won_match'] = np.where((tmp_df['is_home_team'] == 1) & (tmp_df['score_total_home'] > tmp_df['score_total_away']), 1, 0)
        tmp_df['won_match'] = np.where((tmp_df['is_home_team'] == 0) & (tmp_df['score_total_home'] < tmp_df['score_total_away']), 1, tmp_df['won_match'])
        tmp_df['score_difference'] = np.where(tmp_df['won_match'] == 1, abs(tmp_df['score_total_home'] - tmp_df['score_total_away']), abs(tmp_df['score_total_home'] - tmp_df['score_total_away']) * -1)

        for index, row in tmp_df.iterrows():
            match_id = row['match_id']
            if row['is_home_team']:
                tmp_stats = stats_df.loc[stats_df['match_id'] == match_id].copy().filter(like='home')
                new_cols = {col: col.replace('home_', '') for col in tmp_stats.columns if 'home' in col}
                tmp_stats = tmp_stats.rename(columns=new_cols)
            else:
                tmp_stats = stats_df.loc[stats_df['match_id'] == match_id].copy().filter(like='away')
                new_cols = {col: col.replace('away_', '') for col in tmp_stats.columns if 'away' in col}
                tmp_stats = tmp_stats.rename(columns=new_cols)
            tmp_stats['match_date'] = pd.to_datetime(row['event_date_start']).date()
            tmp_stats['match_id'] = match_id
            tmp_stats['team_id'] = team
            tmp_stats['stats_won_match'] = row['won_match']
            tmp_stats['stats_score'] = row['score']
            tmp_stats['stats_score_difference'] = row['score_difference']
            team_df = pd.concat([team_df, tmp_stats], axis=0)
    return team_df