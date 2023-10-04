import pandas as pd
import numpy as np
import glob
from helper_functions import train_and_tune_random_forest
from sklearn.model_selection import train_test_split
import pickle

files = glob.glob('data/match_timeline/*.csv')
cols = ['match_id', 'type', 'side', 'time_minutes']
dfs = [pd.read_csv(file)[cols] for file in files]
df = pd.concat(dfs, axis=0)

matches = pd.read_csv('data/matches.csv')
matches['home_win'] = np.where(matches['score_total_home'] > matches['score_total_away'], 1, 0)
matches.rename(columns={'feed_id':'match_id'}, inplace=True)

ids = df.loc[df['time_minutes'] >= 80]['match_id'].unique()
df = df.loc[df['match_id'].isin(ids)]

df['time_minutes'] = df['time_minutes'].apply(lambda x: int(x/5))

event_mappings = {
    'Try':5,
    'Conversion':2,
    'Penalty':3,
    'Drop Goal':3,
    'Penalty Try':5
}
df['score'] = df['type'].map(event_mappings)
df['score'].fillna(0, inplace = True)
df['substitution'] = np.where(df['type'].str.lower() == 'substitution', 1, 0)
df['yellow_card'] = np.where(df['type'].str.lower().isin(['yellow card','yellow-card']), 1, 0)
df['red_card'] = np.where(df['type'].str.lower().isin(['red card','red-card']), 1, 0)

pivot_df = pd.pivot_table(df, values=['score', 'substitution', 'yellow_card', 'red_card'], index=['match_id','time_minutes'], columns=['side'], aggfunc='sum')
pivot_df.columns = pivot_df.columns.droplevel(0)
pivot_df.reset_index(inplace=True)
pivot_df.columns = [
    'match_id', 'time_minutes', 'away_red_cards', 'home_red_cards', 'away_score', 'home_score', 'away_substitutions', 'home_substitutions', 'away_yellow_cards', 'home_yellow_cards'
]
pivot_df.fillna(0, inplace = True)

tmp_df = pivot_df.groupby(['match_id']).cumsum()
del tmp_df['time_minutes']
tmp_df.columns = ['cumsum_' + col for col in tmp_df.columns]

pivot_df = pd.concat([pivot_df, tmp_df], axis=1)
pivot_df = pivot_df[list(['match_id','time_minutes']) + list(pivot_df.columns[pivot_df.columns.str.contains('cumsum')])]
pivot_df = pivot_df.merge(matches[['match_id', 'home_win']], on='match_id', how='inner')

x_feats = [
    'time_minutes',
    'cumsum_away_red_cards',
    'cumsum_home_red_cards', 'cumsum_away_score', 'cumsum_home_score',
    'cumsum_away_substitutions', 'cumsum_home_substitutions',
    'cumsum_away_yellow_cards', 'cumsum_home_yellow_cards'
    ]
y = 'home_win'

train, test = train_test_split(pivot_df, test_size=0.2, random_state=1991)

model, feats = train_and_tune_random_forest(train[x_feats], train[y], test[x_feats], test[y], cv_folds=3, tune_iter=50, seed=1991)
with open('models/timeline_model.pkl', 'wb') as f:
    pickle.dump(model, f)