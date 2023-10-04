import pandas as pd
import numpy as np
from helper_functions import *
from datetime import datetime
import os

rankings = get_team_world_rankings()
rankings.to_csv('data/world_rankings.csv', index=False)

try:
    matches_scraped = pd.read_csv('data/scraped_matches.csv')
except:
    matches_scraped = pd.DataFrame({'match_id': []})


match_cols = [
    'event_date_start', 'event_date_end', 'feed_id', 'score_total_home',
    'score_total_away', 'rugby_team_home_name', 'rugby_team_away_name'
]

if not os.path.exists('data/scraped_matches.csv'):
    matches = get_historical_matches_for_date_range(from_date = '2021-01-01', to_date = datetime.now().strftime('%Y-%m-%d'))
else:
    matches = pd.read_csv('data/matches_reduced.csv')
    from_date = (pd.to_datetime(matches['event_date_end']).max().date() + timedelta(1)).strftime('%Y-%m-%d')
    to_date = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
    new_matches = get_historical_matches_for_date_range(from_date, to_date)
    if new_matches.shape[0] > 0:
        new_matches.columns = format_column_names(new_matches.columns)
        new_matches.reset_index(inplace=True, drop=True)
        matches.reset_index(inplace=True, drop=True)
        matches = pd.concat([matches, new_matches[match_cols]], axis = 0)
matches.to_csv('data/matches_reduced.csv', index=False)

new_matches = matches.loc[~matches['feed_id'].isin(matches_scraped['match_id'])]['feed_id'].unique()
matches_scraped = pd.concat([matches_scraped, pd.DataFrame({'match_id': new_matches})], axis=0)
matches_scraped.to_csv('data/scraped_matches.csv', index=False)

folders = ['match_lineup', 'match_stats', 'match_timeline', 'player_stats']
for folder in folders:
    if not os.path.exists(f'data/{folder}'):
        os.makedirs(f'data/{folder}')

for match_id in tqdm(new_matches):

    tmp_df = get_match_timeline(match_id)
    if tmp_df is not None:
        tmp_df.to_csv(f'data/match_timeline/{match_id}.csv', index=False)

    tmp_df = get_match_stats(match_id)
    if tmp_df is not None:
        tmp_df.to_csv(f'data/match_stats/{match_id}.csv', index=False)

    tmp_df = get_match_lineup(match_id)
    if tmp_df is not None:
        tmp_df.to_csv(f'data/match_lineup/{match_id}.csv', index=False)

player_ids = pd.DataFrame()
for file in os.listdir('data/match_lineup'):
    tmp_df = pd.read_csv(f'data/match_lineup/{file}')
    player_ids = pd.concat([player_ids, tmp_df[['player_sports_person_id', 'match_id']]], axis=0)

timeline_df = pd.DataFrame()
for file in os.listdir('data/match_timeline'):
    tmp_df = pd.read_csv(f'data/match_timeline/{file}')
    player_ids = pd.concat([player_ids, tmp_df[['player_sports_person_id', 'match_id']]], axis=0)

player_ids.drop_duplicates(inplace=True)
player_ids = player_ids.loc[player_ids['player_sports_person_id'].notnull()]