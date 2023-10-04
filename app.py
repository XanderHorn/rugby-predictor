import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title='Rugby Predictor', page_icon='🏉', layout='wide')

df = pd.read_csv('data/prod_team_performance.csv')
max_date = df['date'].max()
del df['date']

with open('models/baseline_model.pkl', 'rb') as f:
    baseline_model = pickle.load(f)

with open('models/timeline_model.pkl', 'rb') as f:
    timeline_model = pickle.load(f)

mappings = {'Yes':1, 'No':0}

x_feats_baseline = [
    'home_rolling_performance', 'away_rolling_performance',
    'ranking_points_home', 'ranking_points_away', 'neutral', 'world_cup'
]


teams = df['team'].unique().tolist()

team_flags = {
    'South Africa': '🇿🇦',    'New Zealand': '🇳🇿',    'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',    'Wales': '🏴󠁧󠁢󠁷󠁬󠁳󠁿',
    'Ireland': '🇮🇪',    'France': '🇫🇷',    'Australia': '🇦🇺',    'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
    'Japan': '🇯🇵',    'Argentina': '🇦🇷',    'Fiji': '🇫🇯',    'Georgia': '🇬🇪',
    'Italy': '🇮🇹',    'Samoa': '🇼🇸',    'Tonga': '🇹🇴',    'Romania': '🇷🇴',
    'Uruguay': '🇺🇾',    'Namibia': '🇳🇦',    'Portugal': '🇵🇹',    'Argentina': '🇦🇷',
    'Chile': '🇨🇱'
}

st.sidebar.title('Select teams and match conditions:')

sb_col1, sb_col2, sb_col3 = st.sidebar.columns(3)

with sb_col1:
    sb_col1.write('### Team 1:')
    home_team = sb_col1.selectbox("Team 1:", teams, index=0, label_visibility='hidden')
with sb_col2:
    sb_col2.write('### VS')
with sb_col3:
    sb_col3.write('### Team 1:')
    away_team = sb_col3.selectbox("Team 2:", teams, index=1, label_visibility='hidden')


neutral_grounds = st.sidebar.selectbox("Is Neutral Ground:",['Yes','No'], )
st.sidebar.caption('Neutral ground is defined as a match played at a venue that is not the home ground of either team')
world_cup = st.sidebar.selectbox("Is World Cup Match:",['Yes','No'])
st.sidebar.caption('Is the match part of a Rugby World Cup tournament?')

st.write('Data last updated at: ', max_date)
st.write('#### Minutes passed:')
st.caption('Time elapsed since the start of the match')
time_minutes = st.slider('', min_value=0, max_value=90, value=0) // 5

result = [(key, value) for key, value in team_flags.items() if key == home_team]
home_team_flag = result[0][1]
result = [(key, value) for key, value in team_flags.items() if key == away_team]
away_team_flag = result[0][1]

col1, col2, col3 = st.columns(3)

col1.header(f'{home_team} {home_team_flag}')
col2.header('VS')
col3.header(f'{away_team} {away_team_flag}')

home_df = df.loc[df['team'] == home_team].copy()
home_df.columns = ['home_' + col for col in home_df.columns]
away_df = df.loc[df['team'] == away_team].copy()
away_df.columns = ['away_' + col for col in away_df.columns]

baseline_df = pd.concat([home_df.reset_index(drop=True), away_df.reset_index(drop=True)], axis=1)
baseline_df['neutral'] = np.where(neutral_grounds == 'Yes', 1, 0)
baseline_df['world_cup'] =np.where(world_cup == 'Yes', 1, 0)
baseline_df.rename(columns={'home_ranking_points':'ranking_points_home', 'away_ranking_points':'ranking_points_away'}, inplace=True)

baseline_home_prob_win = baseline_model.predict_proba(baseline_df[x_feats_baseline])[0][1]
baseline_away_prob_win = 1 - baseline_home_prob_win

baseline_df['baseline_home_prob_win'] = baseline_home_prob_win
baseline_df['baseline_away_prob_win'] = baseline_away_prob_win

with col1:
    col1.write('##### Score')
    col1.caption('The score of the match thus far')
    home_score = col1.slider('Team 1 Score', min_value=0, max_value=100, value=0, label_visibility='hidden')
    col1.write('##### Nr Yellow Cards')
    col1.caption('The number of yellow cards issued to the team thus far')
    home_yellow_cards = col1.slider("Team 1 # Yellow Cards:",  min_value=0, max_value=10, value=0, label_visibility='hidden')
    col1.write('##### Nr Red Cards')
    col1.caption('The number of red cards issued to the team thus far')
    home_red_cards = col1.slider("Team 1# Red Cards:",  min_value=0, max_value=10, value=0, label_visibility='hidden')
    col1.write('##### Nr Substitutions')
    col1.caption('The number of subsitutions made by the team thus far (Can re-subsitute players)')
    home_substitutions = col1.slider("Team 1# Substitutions:",  min_value=0, max_value=15, value=0, label_visibility='hidden')

with col3:
    col3.write('##### Score')
    col3.caption('The score of the match thus far')
    away_score = col3.slider('Score', min_value=0, max_value=100, value=0, label_visibility='hidden')
    col3.write('##### Nr Yellow Cards')
    col3.caption('The number of yellow cards issued to the team thus far')
    away_yellow_cards = col3.slider("# Yellow Cards:",  min_value=0, max_value=10, value=0, label_visibility='hidden')
    col3.write('##### Nr Red Cards')
    col3.caption('The number of red cards issued to the team thus far')
    away_red_cards = col3.slider("# Red Cards:",  min_value=0, max_value=10, value=0, label_visibility='hidden')
    col3.write('##### Nr Substitutions')
    col3.caption('The number of subsitutions made by the team thus far (Can re-subsitute players)')
    away_substitutions = col3.slider("Team 2# Substitutions:",  min_value=0, max_value=15, value=0, label_visibility='hidden')

timeline_dict = {
    'time_minutes': time_minutes,
    'cumsum_away_red_cards': away_red_cards,
    'cumsum_home_red_cards': home_red_cards,
    'cumsum_away_score': away_score,
    'cumsum_home_score': home_score,
    'cumsum_away_substitutions': away_substitutions,
    'cumsum_home_substitutions': home_substitutions,
    'cumsum_away_yellow_cards': away_yellow_cards,
    'cumsum_home_yellow_cards': home_yellow_cards
}

if time_minutes > 0:
    timeline_home_prob_win = timeline_model.predict_proba([list(timeline_dict.values())])[0][1]
    timeline_away_prob_win = 1 - timeline_home_prob_win
else:
    timeline_home_prob_win = 0
    timeline_away_prob_win = 0

timeline_dict['home_win_prob'] = timeline_home_prob_win
timeline_dict['away_win_prob'] = timeline_away_prob_win

if home_team == away_team:
    baseline_home_prob_win = 0
    baseline_away_prob_win = 0
    timeline_home_prob_win = 0
    timeline_away_prob_win = 0

with col1:
    col1.write(f'###  Pre-Match Win Probability: {round(baseline_home_prob_win * 100, 2)}%')
    col1.caption('The likelihood of the team winning the match before the match has started based on historical performance')
    col1.write(f'###  Match Thus Far Win Probability: {round(timeline_home_prob_win * 100, 2)}%')
    col1.caption('The likelihood of the team winning the match based on the match thus far')
    
with col3:
    col3.write(f'###  Pre-Match Win Probability: {round(baseline_away_prob_win * 100, 2)}%')
    col3.caption('The likelihood of the team winning the match before the match has started based on historical performance')
    col3.write(f'###  Match Thus Far Win Probability: {round(timeline_away_prob_win * 100, 2)}%')
    col3.caption('The likelihood of the team winning the match based on the match thus far')
    
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write('For any questions reach out at: xanderhorn@hotmail.com')