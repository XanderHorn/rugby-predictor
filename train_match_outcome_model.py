import pandas as pd
import numpy as np
import uuid
from helper_functions import get_team_rolling_performance, get_rankings, get_historical_matches_for_date_range, train_and_tune_random_forest
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('data/historical_results.csv')

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
    df.to_csv('data/historical_results.csv', index=False)

df['match_id'] = df.apply(lambda x: uuid.uuid4(), axis=1)

df['winner'] = np.where(df['home_score'] > df['away_score'], df['home_team'], 
                        np.where(df['home_score'] < df['away_score'], df['away_team'], 'DRAW'))
df['loser'] = np.where(df['home_score'] < df['away_score'], df['home_team'], 
                        np.where(df['home_score'] > df['away_score'], df['away_team'], 'DRAW'))
df['home_margin'] = df['home_score'] - df['away_score']
df['result'] = df['home_margin'].apply(lambda x: 'HOME_WIN' if x > 0 else ('AWAY_WIN' if x < 0 else 'DRAW'))

df = get_team_rolling_performance(df, past_n_matches=5)
df = get_rankings(df)
mappings = {True:1, False:0}
df['neutral'] = df['neutral'].map(mappings)
df['world_cup'] = df['world_cup'].map(mappings)

df = df.loc[df['result'] != 'DRAW']
df['home_win'] = np.where(df['result'] == 'HOME_WIN', 1, 0)
df = df.loc[pd.to_datetime(df['date']) >= '1990-01-01'].copy()

df['home_score_bin'] = pd.cut(df['home_score'], bins=[0, 5, 10, 15, 20, 25, 30, 40, 50,1000], right=False).astype(str)
df['away_score_bin'] = pd.cut(df['away_score'], bins=[0, 5, 10, 15, 20, 25, 30, 40, 50,1000], right=False).astype(str)

x_feats = ['home_rolling_performance', 'away_rolling_performance',
            'ranking_points_home', 'ranking_points_away', 'neutral', 'world_cup'
        ]
y = 'home_win'

train, test = train_test_split(df, test_size=0.2, random_state=1991)

model, feats = train_and_tune_random_forest(train[x_feats], train[y], test[x_feats], test[y], cv_folds=5, tune_iter=50, seed=1991)

with open('models/baseline_model.pkl', 'wb') as f:
    pickle.dump(model, f)

last_entries = df.groupby(['home_team']).tail(1)
last_entries = last_entries[['home_team', 'date', 'ranking_points_home', 'home_rolling_performance']]
last_entries.rename(columns={'home_team':'team', 'ranking_points_home':'ranking_points', 'home_rolling_performance':'rolling_performance'}, inplace=True)
last_entries = pd.concat([last_entries, df.groupby(['away_team']).tail(1)[['away_team', 'date', 'ranking_points_away', 'away_rolling_performance']].rename(columns={'away_team':'team', 'ranking_points_away':'ranking_points', 'away_rolling_performance':'rolling_performance'})], axis=0)
last_entries = last_entries.loc[last_entries.groupby('team')['date'].transform(max) == last_entries['date']]
last_entries.to_csv('data/prod_team_performance.csv', index=False)