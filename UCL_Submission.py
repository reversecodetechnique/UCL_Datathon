import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import f1_score, log_loss, accuracy_score, classification_report
from sklearn.feature_selection import RFECV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib

import requests
from bs4 import BeautifulSoup
import time
import os

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV


# Check scikit-learn version for metadata routing
if sklearn.__version__ >= '1.3':
    sklearn.set_config(enable_metadata_routing=True)

# -------------------------------
# Unified Team Name Cleaning Function
# -------------------------------
def clean_team_name(name):
    """Standardize team names across different data sources"""
    if not isinstance(name, str):
        return ""

    name = name.strip().lower()

    replacements = {
        "manchester united": "man utd",
        "paris saint-germain": "psg",
        "porto": "fc porto",
        "inter milan": "inter",
        "internazionale": "inter",
        "tottenham hotspur": "tottenham",
        "basel": "fc basel 1893",
        "manchester city": "man city",
        "spurs": "tottenham",
        "olympique lyonnais": "lyon",
        "olympique marseille": "marseille",
        "fc bayern münchen": "bayern munich",
        "fc bayern munich": "bayern munich",
        "bayern münchen": "bayern munich",
        "borussia dortmund": "dortmund",
        "bvb": "dortmund",
        "sporting lisbon": "sporting cp",
        "sporting clube de portugal": "sporting cp",
        "fc porto": "porto",
        "fc barcelona": "barcelona",
        "real madrid cf": "real madrid",
        "atletico madrid": "atletico",
        "club atlético de madrid": "atletico",
        "juventus fc": "juventus",
        "ac milan": "milan",
        "as roma": "roma",
        "ajax amsterdam": "ajax",
        "psv eindhoven": "psv",
        "besiktas jk": "besiktas",
        "fc shakhtar donetsk": "shakhtar donetsk"
    }

    for old, new in replacements.items():
        if old in name:
            name = name.replace(old, new)

    name = name.replace(" fc", "").replace(" cf", "").replace(" afc", "").replace(" ssc", "").replace(".", "").replace("-", " ").strip()
    name = ' '.join(name.split())

    return name

# -------------------------------
# Data Loading and Initial Cleaning
# -------------------------------

# Load full dataset (all European matches)
full_data = pd.read_csv('/Users/gunin/Desktop/IIT Madras/UCL Predictor/Full_Dataset.csv')

# Rename columns in full_data to match expected schema
full_data.rename(columns={
    'Date': 'date',
    'Team': 'home_team',
    'Opponent': 'away_team',
    'Team_Score': 'home_team_goal',
    'Opponent_Score': 'away_team_goal'
}, inplace=True)

# Load training data (CL knockout matches from 2004 to 2017)
with open('/Users/gunin/Desktop/IIT Madras/UCL Predictor/train.json') as f:
    train_data = json.load(f)

# Flatten train JSON into a dataframe
knockout_matches = []
for season, rounds in train_data.items():
    for stage, matches in rounds.items():
        for match in matches:
            match_flat = {
                "season": season,
                "stage": stage,
                "date": match["date"],
                "team_1": match["team_1"],
                "team_2": match["team_2"],
                "winner": match["winner"]
            }
            knockout_matches.append(match_flat)
train_df = pd.DataFrame(knockout_matches)

# Convert date fields to datetime
train_df['date'] = pd.to_datetime(train_df['date'], format="mixed", errors='coerce',dayfirst=True)

# Assign weights
train_df['weight'] = 1.0

# Ensure the correct column is used for date parsing in full_data
if 'date' not in full_data.columns:
    print("Available columns in full_data:", full_data.columns)
    raise KeyError("Expected 'date' column not found in Full_Dataset.csv after renaming. Please check the correct column name.")

# Convert date fields to datetime for full_data
full_data['date'] = pd.to_datetime(full_data['date'], format='mixed', errors='coerce',dayfirst=True)

# Apply the unified clean_team_name function to all relevant columns
full_data['home_team'] = full_data['home_team'].apply(clean_team_name)
full_data['away_team'] = full_data['away_team'].apply(clean_team_name)
train_df['team_1'] = train_df['team_1'].apply(clean_team_name)
train_df['team_2'] = train_df['team_2'].apply(clean_team_name)
train_df['winner'] = train_df['winner'].apply(clean_team_name)

# -------------------------------
# Web Scraping Functions for xG Data (with caching)
# -------------------------------

def fetch_page_content(url):
    """Fetches the HTML content of a given URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        return res.content
    except requests.exceptions.RequestException as e:
        print(f"❌ Network or HTTP error fetching {url}: {e}")
        return None

def parse_html_for_scripts(html_content):
    """Parses HTML content and returns all script tags."""
    if html_content is None:
        return []
    soup = BeautifulSoup(html_content, 'lxml')
    return soup.find_all('script')

def find_json_script_content(scripts, data_variable_name):
    """Identifies and extracts the raw string content of the script tag containing the specified JavaScript variable."""
    for el in scripts:
        if data_variable_name in str(el):
            return str(el).strip()
    return ''

def decode_and_load_json(json_string):
    """Decodes the Unicode escape sequences and loads the JSON string into a Python object."""
    if not json_string:
        return None
    try:
        ind_start = json_string.index("('") + 2
        ind_end = json_string.index("')")
        clean_json_data = json_string[ind_start:ind_end]
        clean_json_data = clean_json_data.encode('utf8').decode('unicode_escape')
        return json.loads(clean_json_data)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing JSON: {e}")
        return None

def get_team_xg_data(league_name, season_year):
    url = f"https://understat.com/league/{league_name}/{season_year}"
    headers = {'User-Agent': 'Mozilla/50'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Network or HTTP error fetching {url}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')
    scripts = soup.find_all('script')

    data = None
    for script in scripts:
        if 'teamsData' in script.text:
            try:
                raw_json = script.text.split("('")[1].split("')")[0]
                raw_json = bytes(raw_json, 'utf-8').decode('unicode_escape')
                data = json.loads(raw_json)
                break
            except (IndexError, json.JSONDecodeError) as e:
                print(f"Error parsing teamsData script for {league_name} {season_year}: {e}")
                return pd.DataFrame()
    else:
        print(f"⚠️ No teamsData found for {league_name} {season_year}")
        return pd.DataFrame()

    all_data = []
    for team_name_key, team_info in data.items():
        for match in team_info['history']:
            match_data = {
                'team_name': team_info['title'],
                'opponent': match.get('opponent'),
                'match_date': match.get('date'),
                'xG_scored': match.get('xG'),
                'xG_conceded': match.get('xGA'),
                'scored': match.get('scored'),
                'conceded': match.get('missed'),
                'is_home': match.get('isHome'),
                'season': season_year,
                'league': league_name
            }
            all_data.append(match_data)

    return pd.DataFrame(all_data)

def scrape_and_cache_xg_data(leagues, seasons, file_path):
    """
    Scrapes team-level xG data and caches it to a CSV file.
    Loads from cache if the file exists.
    """
    if os.path.exists(file_path):
        print(f"✅ Loading xG data from cache: {file_path}")
        xg_data = pd.read_csv(file_path)
        xg_data['match_date'] = pd.to_datetime(xg_data['match_date'])
        xg_data['team_name'] = xg_data['team_name'].apply(clean_team_name)
        xg_data['opponent'] = xg_data['opponent'].apply(clean_team_name)
        return xg_data
    else:
        print("Starting xG data scraping (no cache found)...")
        all_dataframes = []
        for league in leagues:
            for season in seasons:
                print(f"Scraping team xG data for {league} {season}...")
                df = get_team_xg_data(league, season)
                if not df.empty:
                    all_dataframes.append(df)
                time.sleep(2)
        
        xg_data = pd.concat(all_dataframes, ignore_index=True) if all_dataframes else pd.DataFrame()

        if not xg_data.empty:
            xg_data['team_name'] = xg_data['team_name'].apply(clean_team_name)
            xg_data['opponent'] = xg_data['opponent'].apply(clean_team_name)
            xg_data['match_date'] = pd.to_datetime(xg_data['match_date'])
            
            xg_data.to_csv(file_path, index=False)
            print(f"✨ xG data scraped and saved to cache: {file_path}")
        else:
            print("Warning: No xG data was scraped.")
            
        return xg_data

# Call the xG caching function
xg_data_cache_path = '/Users/gunin/Desktop/IIT Madras/UCL Predictor/xg_data_cache.csv'
leagues_to_scrape = ['EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1']
seasons_to_scrape = [str(year) for year in range(2010, 2024)]

xg_data = scrape_and_cache_xg_data(leagues_to_scrape, seasons_to_scrape, xg_data_cache_path)

if xg_data.empty:
    print("WARNING: xG data is empty. Predictions might rely on default values.")

# -------------------------------
# Feature Engineering Functions with xG Integration
# -------------------------------

def get_team_xg_stats(team_name, match_date, xg_df):
    """Get xG statistics for a team up to a given date"""
    if xg_df.empty:
        return {
            'xg_for_avg': 1.5,
            'xg_against_avg': 1.2,
            'xg_diff': 0.3,
            'npxg': 1.4
        }

    team_matches = xg_df[
        (xg_df['team_name'] == team_name) &
        (xg_df['match_date'] < match_date)
    ].sort_values('match_date', ascending=False)

    if team_matches.empty:
        return {
            'xg_for_avg': 1.5,
            'xg_against_avg': 1.2,
            'xg_diff': 0.3,
            'npxg': 1.4
        }

    last_5_matches = team_matches.head(5)

    return {
        'xg_for_avg': last_5_matches['xG_scored'].mean() if not last_5_matches.empty else 1.5,
        'xg_against_avg': last_5_matches['xG_conceded'].mean() if not last_5_matches.empty else 1.2,
        'xg_diff': (last_5_matches['xG_scored'] - last_5_matches['xG_conceded']).mean() if not last_5_matches.empty else 0.3,
        'npxg': last_5_matches['xG_scored'].mean() * 0.9 if not last_5_matches.empty else 1.4
    }

def calculate_h2h_stats(team1, team2, match_date, data):
    """Calculate head-to-head statistics between two teams"""
    h2h_matches = data[
        (((data['home_team'] == team1) & (data['away_team'] == team2)) |
         ((data['home_team'] == team2) & (data['away_team'] == team1))) &
        (data['date'] < match_date)
    ].sort_values('date', ascending=False)

    if h2h_matches.empty:
        return {
            'h2h_total_matches': 0,
            'h2h_team1_wins': 0,
            'h2h_team2_wins': 0,
            'h2h_draws': 0,
            'h2h_team1_win_rate': 0.5,
            'h2h_avg_goals': 2.5,
            'h2h_recent_form': 0
        }

    team1_wins = 0
    team2_wins = 0
    draws = 0
    total_goals = []
    recent_results = []

    for _, match in h2h_matches.iterrows():
        if match['home_team'] == team1:
            team1_goals = match['home_team_goal']
            team2_goals = match['away_team_goal']
        else:
            team1_goals = match['away_team_goal']
            team2_goals = match['home_team_goal']

        total_goals.extend([team1_goals, team2_goals])

        if team1_goals > team2_goals:
            team1_wins += 1
            recent_results.append(1)
        elif team2_goals > team1_goals:
            team2_wins += 1
            recent_results.append(-1)
        else:
            draws += 1
            recent_results.append(0)

    total_matches = len(h2h_matches)

    return {
        'h2h_total_matches': total_matches,
        'h2h_team1_wins': team1_wins,
        'h2h_team2_wins': team2_wins,
        'h2h_draws': draws,
        'h2h_team1_win_rate': team1_wins / total_matches if total_matches > 0 else 0.5,
        'h2h_avg_goals': np.mean(total_goals) if total_goals else 2.5,
        'h2h_recent_form': np.mean(recent_results[:3]) if recent_results else 0
    }

def calculate_team_strength(team_name, match_date, data):
    """Calculate team strength metrics based on historical performance"""
    one_year_ago = match_date - pd.Timedelta(days=365)
    team_matches = data[
        ((data['home_team'] == team_name) | (data['away_team'] == team_name)) &
        (data['date'] >= one_year_ago) &
        (data['date'] < match_date)
    ]

    if team_matches.empty:
        return {
            'strength_rating': 1500,
            'avg_goals_for': 1.5,
            'avg_goals_against': 1.5,
            'clean_sheet_rate': 0.3,
            'big_win_rate': 0.1
        }

    goals_for = []
    goals_against = []
    clean_sheets = 0
    big_wins = 0
    total_matches = len(team_matches)

    for _, match in team_matches.iterrows():
        if match['home_team'] == team_name:
            gf = match['home_team_goal']
            ga = match['away_team_goal']
        else:
            gf = match['away_team_goal']
            ga = match['home_team_goal']

        goals_for.append(gf)
        goals_against.append(ga)

        if ga == 0:
            clean_sheets += 1
        if gf >= ga + 2:
            big_wins += 1

    avg_goal_diff = np.mean(np.array(goals_for) - np.array(goals_against))
    strength_rating = 1500 + (avg_goal_diff * 100)

    return {
       'strength_rating': strength_rating,
        'avg_goals_for': np.mean(goals_for) if goals_for else 1.5,
        'avg_goals_against': np.mean(goals_against) if goals_against else 1.5,
        'clean_sheet_rate': clean_sheets / total_matches,
        'big_win_rate': big_wins / total_matches
    }

def calculate_team_form(team_name, match_date, data, xg_df, n_matches=10, lookback_days=90):
    """Enhanced team form calculation with xG data and various lookback periods."""
    team_matches = data[
        ((data['home_team'] == team_name) | (data['away_team'] == team_name)) &
        (data['date'] < match_date) &
        (data['date'] >= (match_date - pd.Timedelta(days=lookback_days)))
    ].sort_values('date', ascending=False)

    xg_stats = get_team_xg_stats(team_name, match_date, xg_df)

    if team_matches.empty:
        return {
            'win_rate_5': 0.5,
            'win_rate_10': 0.5,
            'goal_diff_5': 0,
            'goal_diff_10': 0,
            'home_win_rate': 0.5,
            'away_win_rate': 0.5,
            'run_in_win_rate': 0.5,
            'recent_goals_scored': 1.2,
            'recent_goals_conceded': 1.2,
            **xg_stats
        }

    results = []
    for _, row in team_matches.iterrows():
        if row['home_team'] == team_name:
            scored = row['home_team_goal']
            conceded = row['away_team_goal']
            is_home = True
        else:
            scored = row['away_team_goal']
            conceded = row['home_team_goal']
            is_home = False

        result = {
            'date': row['date'],
            'scored': scored,
            'conceded': conceded,
            'is_home': is_home,
            'win': scored > conceded,
            'draw': scored == conceded,
            'loss': scored < conceded,
            'goal_diff': scored - conceded
        }
        results.append(result)

    results_df = pd.DataFrame(results)

    last_5 = results_df.head(min(5, len(results_df)))
    win_rate_5 = last_5['win'].mean() if not last_5.empty else 0.5
    goal_diff_5 = last_5['goal_diff'].mean() if not last_5.empty else 0

    last_10 = results_df.head(min(10, len(results_df)))
    win_rate_10 = last_10['win'].mean() if not last_10.empty else 0.5
    goal_diff_10 = last_10['goal_diff'].mean() if not last_10.empty else 0

    home_matches = results_df[results_df['is_home']]
    away_matches = results_df[~results_df['is_home']]

    home_win_rate = home_matches['win'].mean() if not home_matches.empty else 0.5
    away_win_rate = away_matches['win'].mean() if not away_matches.empty else 0.5

    run_in_matches = results_df[results_df['date'].dt.month.isin([2, 3, 4, 5])]
    run_in_win_rate = run_in_matches['win'].mean() if not run_in_matches.empty else 0.5

    recent_goals_scored = results_df['scored'].mean() if not results_df.empty else 1.2
    recent_goals_conceded = results_df['conceded'].mean() if not results_df.empty else 1.2

    return {
        'win_rate_5': win_rate_5,
        'win_rate_10': win_rate_10,
        'goal_diff_5': goal_diff_5,
        'goal_diff_10': goal_diff_10,
        'home_win_rate': home_win_rate,
        'away_win_rate': away_win_rate,
        'run_in_win_rate': run_in_win_rate,
        'recent_goals_scored': recent_goals_scored,
        'recent_goals_conceded': recent_goals_conceded,
        **xg_stats
    }

# -------------------------------
# Feature Generation for Training Data
# -------------------------------

team_1_features = []
team_2_features = []
extra_features = []

for idx, row in train_df.iterrows():
    # Subset full_data and xG data to ensure no future data leakage
    training_data_subset = full_data[full_data['date'] < row['date']]
    xg_data_subset = xg_data[xg_data['match_date'] < row['date']]

    # Calculate team-specific features
    form_stats_1 = calculate_team_form(row['team_1'], row['date'], data=training_data_subset, xg_df=xg_data_subset, n_matches=10, lookback_days=90)
    form_stats_2 = calculate_team_form(row['team_2'], row['date'], data=training_data_subset, xg_df=xg_data_subset, n_matches=10, lookback_days=90)
    strength_stats_1 = calculate_team_strength(row['team_1'], row['date'], data=training_data_subset)
    strength_stats_2 = calculate_team_strength(row['team_2'], row['date'], data=training_data_subset)
    h2h_stats = calculate_h2h_stats(row['team_1'], row['team_2'], row['date'], data=training_data_subset)

    # Aggregate features for each team and match
    team_1_all_features = {**form_stats_1, **strength_stats_1}
    team_2_all_features = {**form_stats_2, **strength_stats_2}

    team_1_features.append(team_1_all_features)
    team_2_features.append(team_2_all_features)

    extra_features_dict = {
        'match_month': row['date'].month,
        'is_winter': 1 if row['date'].month in [12, 1, 2] else 0,
        **h2h_stats
    }
    extra_features.append(extra_features_dict)

# Convert feature lists to DataFrames and prefix columns
team_1_df = pd.DataFrame(team_1_features)
team_2_df = pd.DataFrame(team_2_features)
extra_df = pd.DataFrame(extra_features)

team_1_df.columns = [f'team_1_{col}' for col in team_1_df.columns]
team_2_df.columns = [f'team_2_{col}' for col in team_2_df.columns]

# Merge all generated features into the main training DataFrame
train_df = pd.concat([train_df.reset_index(drop=True), team_1_df, team_2_df, extra_df], axis=1)

# Add interaction features
train_df['goal_avg_diff'] = train_df['team_1_recent_goals_scored'] - train_df['team_2_recent_goals_conceded']
train_df['win_rate_diff'] = train_df['team_1_win_rate_10'] - train_df['team_2_win_rate_10']
train_df['strength_rating_diff'] = train_df['team_1_strength_rating'] - train_df['team_2_strength_rating']
train_df['xg_diff_combined'] = train_df['team_1_xg_diff'] - train_df['team_2_xg_diff']
train_df['h2h_win_rate_diff'] = train_df['h2h_team1_win_rate'] - (1 - train_df['h2h_team1_win_rate'])

# Define the target variable
train_df['label'] = (train_df['team_1'] == train_df['winner']).astype(int)

# --------------------------------------------------------------------------------------
# Model Training (Revised for LightGBM)
# --------------------------------------------------------------------------------------

# Prepare data for modeling by separating features and target
non_feature_columns = [
    'label',
    'season',
    'stage',
    'date',
    'team_1',
    'team_2',
    'winner',
    'weight'
]

feature_cols = [col for col in train_df.columns if col not in non_feature_columns]

X = train_df[feature_cols]
y = train_df['label']
weights = train_df['weight']

print(f"Total features generated: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:5]} ... {feature_cols[-5:]}")

# Define TimeSeriesSplit for consistent cross-validation
tscv = TimeSeriesSplit(n_splits=10)

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) for LightGBM
print("\n🔍 Performing Recursive Feature Elimination with Cross-Validation (RFECV) for LightGBM...")

estimator_for_rfe = lgb.LGBMClassifier(objective='binary', metric='f1', random_state=42, class_weight='balanced')
estimator_for_rfe.set_fit_request(sample_weight=True)

selector = RFECV(
    estimator=estimator_for_rfe,
    step=1,
    cv=tscv,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    min_features_to_select=1
)

selector.fit(X, y, sample_weight=weights)

print(f"Optimal number of features selected by RFECV : {selector.n_features_}")
selected_features_mask = selector.support_
selected_features_names = X.columns[selected_features_mask].tolist()
print(f"Selected features ({len(selected_features_names)}): {selected_features_names[:5]} ... {selected_features_names[-5:]}")

X_selected = X[selected_features_names]

# Perform Randomized Search for Hyperparameter Tuning for LightGBM on selected features
print("\n⚙️ Performing Randomized Search for Hyperparameter Tuning for LightGBM on selected features...")

param_dist_lgbm = {
    'num_leaves': [20, 31, 40, 50, 60],
    'max_depth': [5, 8, 10, 12, 15, -1],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'n_estimators': [100, 200, 300, 500, 700, 1000],
    'reg_alpha': [0, 0.1, 0.5, 1, 2],
    'reg_lambda': [0, 0.1, 0.5, 1, 2],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'boosting_type': ['gbdt'],
    'objective': ['binary'],
    'metric': ['f1'],
    'is_unbalance': [True, False],
    'random_state': [42],
    'n_jobs': [-1],
}

lgbm_estimator_rs = lgb.LGBMClassifier()
lgbm_estimator_rs.set_fit_request(sample_weight=True)

random_search_lgbm = RandomizedSearchCV(
    estimator=lgbm_estimator_rs,
    param_distributions=param_dist_lgbm,
    n_iter=100,
    scoring='f1_weighted',
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    error_score='raise'
)

random_search_lgbm.fit(X_selected, y, sample_weight=weights)

print("\nBest Hyperparameters from Randomized Search (LightGBM):", random_search_lgbm.best_params_)
print(f"Best CV Score (F1-Weighted) from Randomized Search (LightGBM): {random_search_lgbm.best_score_:.4f}")

best_params_rs_lgbm = random_search_lgbm.best_params_

# Retrain best model from RandomizedSearch with early stopping on the last split for evaluation
print("\n--- Final Model Evaluation on Out-of-Time Test Set (Last CV Fold) with Early Stopping ---")

train_idx_final, test_idx_oot = list(tscv.split(X_selected, y))[tscv.n_splits - 1]

X_train_early_stop = X_selected.iloc[train_idx_final]
y_train_early_stop = y.iloc[train_idx_final]
weights_train_early_stop = weights.iloc[train_idx_final]

X_oot_test = X_selected.iloc[test_idx_oot]
y_oot_test = y.iloc[test_idx_oot]

# Create a small validation set from the training data for early stopping
split_point = int(len(X_train_early_stop) * 0.8)
X_train_es, X_val_es = X_train_early_stop.iloc[:split_point], X_train_early_stop.iloc[split_point:]
y_train_es, y_val_es = y_train_early_stop.iloc[:split_point], y_train_early_stop.iloc[split_point:]
weights_train_es, weights_val_es = weights_train_early_stop.iloc[:split_point], weights_train_early_stop.iloc[split_point:]

# Final model with best parameters and early stopping
final_lgbm_model = lgb.LGBMClassifier(**best_params_rs_lgbm)

# Use early stopping
final_lgbm_model.fit(X_train_es, y_train_es,
                     sample_weight=weights_train_es,
                     eval_set=[(X_val_es, y_val_es)],
                     eval_metric='logloss',
                     callbacks=[lgb.early_stopping(100, verbose=False)],
                     eval_sample_weight=[weights_val_es]
                    )

y_pred_oot = final_lgbm_model.predict(X_oot_test)
y_pred_proba_oot = final_lgbm_model.predict_proba(X_oot_test)

print("Accuracy (OOT):", accuracy_score(y_oot_test, y_pred_oot))
print("F1-Weighted (OOT):", f1_score(y_oot_test, y_pred_oot, average='weighted'))
print("Log Loss (OOT):", log_loss(y_oot_test, y_pred_proba_oot))
print("Classification Report (OOT):\n", classification_report(y_oot_test, y_pred_oot))

# Calibrate probabilities
print("\n📏 Calibrating model probabilities...")
calibrated_model = CalibratedClassifierCV(final_lgbm_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_val_es, y_val_es)

y_pred_proba_oot_calibrated = calibrated_model.predict_proba(X_oot_test)
print("Log Loss (OOT, Calibrated):", log_loss(y_oot_test, y_pred_proba_oot_calibrated))

# Retrain model on ALL AVAILABLE DATA for deployment
print("\n✅ Retraining final LightGBM model on ALL available data with selected features...")

final_n_estimators = final_lgbm_model.best_iteration_ if final_lgbm_model.best_iteration_ is not None else best_params_rs_lgbm['n_estimators']

model_for_deployment = lgb.LGBMClassifier(**{**best_params_rs_lgbm, 'n_estimators': final_n_estimators})
model_for_deployment.fit(X_selected, y, sample_weight=weights)

print("Model retrained successfully on all data with selected features.")
model = model_for_deployment

# Visualize Feature Importance
print("\n📊 Visualizing Feature Importances...")
feature_importances = model.feature_importances_
feature_names = X_selected.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
top_features = importance_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(14, 8))
interaction_keywords = ['_diff', 'h2h_', 'elo_', 'xg_']
colors = ['green' if any(s in feat for s in interaction_keywords) else 'blue' for feat in top_features['Feature']]
sns.barplot(x='Importance', y='Feature', data=top_features, palette=colors)
plt.title(f"Top {len(top_features)} Feature Importances (Interaction/H2H/Elo/xG in green)", fontsize=16)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()

bottom_features = importance_df.sort_values(by='Importance', ascending=True).head(10)
print("\n🔍 Least Important Features (among selected):")
print(bottom_features)

all_initial_features = set(X.columns)
removed_features = list(all_initial_features - set(selected_features_names))
print(f"\n🗑️ Features initially generated but removed by RFECV ({len(removed_features)}):")
print(removed_features)

# -------------------------------
# Prediction Functions
# -------------------------------

def predict_winner(team_1, team_2, match_date, historical_data, xg_data_current):
    # Clean team names for consistent lookup
    t1_clean = clean_team_name(team_1)
    t2_clean = clean_team_name(team_2)
    date = pd.to_datetime(match_date)
    # Subset historical data up to the prediction date to prevent data leakage
    historical_data_subset = historical_data[historical_data['date'] < date].copy()
    xg_data_subset = xg_data_current[xg_data_current['match_date'] < date].copy()

    # Calculate various team and match features
    form_stats_1 = calculate_team_form(t1_clean, date, data=historical_data_subset, xg_df=xg_data_subset, n_matches=3, lookback_days=90)
    form_stats_2 = calculate_team_form(t2_clean, date, data=historical_data_subset, xg_df=xg_data_subset, n_matches=3, lookback_days=90)
    strength_stats_1 = calculate_team_strength(t1_clean, date, data=historical_data_subset)
    strength_stats_2 = calculate_team_strength(t2_clean, date, data=historical_data_subset)
    h2h_stats = calculate_h2h_stats(t1_clean, t2_clean, date, data=historical_data_subset)

    # Add date-based features
    date_features = {
        'match_month': date.month,
        'is_winter': 1 if date.month in [12, 1, 2] else 0
    }

    # Combine all features for team 1, team 2, and extra features
    features_dict = {
        **{f'team_1_{k}': v for k, v in {**form_stats_1, **strength_stats_1}.items()},
        **{f'team_2_{k}': v for k, v in {**form_stats_2, **strength_stats_2}.items()},
        **date_features,
        **h2h_stats
    }

    # Add interaction features, ensuring names match trained model's expectations
    features_dict['goal_avg_diff'] = features_dict['team_1_recent_goals_scored'] - features_dict['team_2_recent_goals_conceded']
    features_dict['win_rate_diff'] = features_dict['team_1_win_rate_10'] - features_dict['team_2_win_rate_10']
    features_dict['strength_rating_diff'] = features_dict['team_1_strength_rating'] - features_dict['team_2_strength_rating']
    features_dict['xg_diff_combined'] = features_dict['team_1_xg_diff'] - features_dict['team_2_xg_diff']
    features_dict['h2h_win_rate_diff'] = features_dict['h2h_team1_win_rate'] - (1 - features_dict['h2h_team1_win_rate'])

    features_df = pd.DataFrame([features_dict])

    # Ensure all columns required by the model are present and in correct order
    expected_features = list(model.feature_names_in_)
    for col_name in expected_features:
        if col_name not in features_df.columns:
            features_df[col_name] = 0.0

    # Reorder columns and handle potential NaNs or infinities
    features = features_df[expected_features].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0.0, inplace=True)

    # Predict the winner using the trained model
    pred = model.predict(features)[0]
    return team_1 if pred == 1 else team_2

# Helper function to resolve winner names from previous rounds based on matchup labels
def resolve_winner_name(label, match_list, winner_list):
    label = label.strip()
    if label.startswith("Winner of"):
        if "QF" in label:
            try:
                index = int(label.split("QF")[1]) - 1
                if 0 <= index < len(winner_list):
                    return winner_list[index]
            except (ValueError, IndexError):
                pass
        elif "SF" in label:
            try:
                index = int(label.split("SF")[1]) - 1
                if 0 <= index < len(winner_list):
                    return winner_list[index]
            except (ValueError, IndexError):
                pass

        teams_str = label.replace("Winner of", "").strip()
        if " vs " in teams_str:
            teams = [clean_team_name(t) for t in teams_str.split(" vs ")]
            if len(teams) == 2:
                for match_idx, match_data in enumerate(match_list):
                    m_t1 = clean_team_name(match_data["team_1"])
                    m_t2 = clean_team_name(match_data["team_2"])
                    if (m_t1 == teams[0] and m_t2 == teams[1]) or \
                       (m_t1 == teams[1] and m_t2 == teams[0]):
                        if 0 <= match_idx < len(winner_list):
                            return winner_list[match_idx]
    return label

# Simulate the entire tournament bracket for a given season
def simulate_tournament(season, matchups, historical_data_for_prediction, xg_data_for_prediction):
    bracket = {}

    def predict_round_winners(matches_in_round, prev_round_matches=None, prev_round_winners=None):
        round_winners = []
        for match in matches_in_round:
            match_date_dt = pd.to_datetime(match["date"], format='mixed',dayfirst=True)

            current_match_historical_data = historical_data_for_prediction[historical_data_for_prediction['date'] < match_date_dt].copy()
            current_match_xg_data = xg_data_for_prediction[xg_data_for_prediction['match_date'] < match_date_dt].copy()

            if prev_round_matches and prev_round_winners:
                team_1_resolved = resolve_winner_name(match["team_1"], prev_round_matches, prev_round_winners)
                team_2_resolved = resolve_winner_name(match["team_2"], prev_round_matches, prev_round_winners)
            else:
                team_1_resolved = match["team_1"]
                team_2_resolved = match["team_2"]

            winner = predict_winner(team_1_resolved, team_2_resolved, match_date_dt,
                                    historical_data=current_match_historical_data,
                                    xg_data_current=current_match_xg_data)
            round_winners.append(winner)
        return round_winners

    # Predict Round of 16 winners
    r16_matches = matchups["round_of_16_matchups"]
    r16_winners = predict_round_winners(r16_matches)
    bracket["round_of_16"] = r16_winners

    # Predict Quarterfinals winners
    qf_matches = matchups["quarter_finals_matchups"]
    qf_winners = predict_round_winners(qf_matches, r16_matches, r16_winners)
    bracket["quarter_finals"] = qf_winners

    # Predict Semifinals winners
    sf_matches = matchups["semi_finals_matchups"]
    sf_winners = predict_round_winners(sf_matches, qf_matches, qf_winners)
    bracket["semi_finals"] = sf_winners

    # Predict Final winner
    final_match = matchups["final_matchup"]
    final_winner = predict_round_winners([final_match], sf_matches, sf_winners)
    bracket["final"] = final_winner

    return bracket

# -------------------------------
# Generate Submission File
# -------------------------------

# Load test matchups for submission generation
with open('/Users/gunin/Desktop/IIT Madras/UCL Predictor/test_matchups.json') as f:
    test_matchups = json.load(f)

structured_submissions = []

# Loop through each season in the test matchups to simulate and format predictions
for i, (season, matchups) in enumerate(test_matchups.items()):
    # Simulate the tournament for the current season using the trained model and data
    bracket = simulate_tournament(season, matchups,
                                  historical_data_for_prediction=full_data,
                                  xg_data_for_prediction=xg_data)

    full_structure = {
        "round_of_16": [],
        "quarter_finals": [],
        "semi_finals": [],
        "final": []
    }

    # Populate Round of 16 predictions
    r16_matches = matchups["round_of_16_matchups"]
    r16_winners = bracket["round_of_16"]
    full_structure["round_of_16"] = [
        {"team_1": m["team_1"], "team_2": m["team_2"], "winner": w}
        for m, w in zip(r16_matches, r16_winners)
    ]

    # Populate Quarterfinals predictions, resolving team names from previous round
    qf_matches = matchups["quarter_finals_matchups"]
    qf_winners = bracket["quarter_finals"]
    full_structure["quarter_finals"] = []
    for match, winner in zip(qf_matches, qf_winners):
        team_1 = resolve_winner_name(match["team_1"], r16_matches, r16_winners)
        team_2 = resolve_winner_name(match["team_2"], r16_matches, r16_winners)
        full_structure["quarter_finals"].append({
            "team_1": team_1,
            "team_2": team_2,
            "winner": winner
        })

    # Populate Semifinals predictions, resolving team names from previous round
    sf_matches = matchups["semi_finals_matchups"]
    sf_winners = bracket["semi_finals"]
    full_structure["semi_finals"] = []
    for match, winner in zip(sf_matches, sf_winners):
        team_1 = resolve_winner_name(match["team_1"], qf_matches, qf_winners)
        team_2 = resolve_winner_name(match["team_2"], qf_matches, qf_winners)
        full_structure["semi_finals"].append({
            "team_1": team_1,
            "team_2": team_2,
            "winner": winner
        })

    # Populate Final prediction, resolving team names from previous round
    final_match = matchups["final_matchup"]
    final_winner = bracket["final"][0]
    team_1 = resolve_winner_name(final_match["team_1"], sf_matches, sf_winners)
    team_2 = resolve_winner_name(final_match["team_2"], sf_matches, sf_winners)
    full_structure["final"].append({
        "team_1": team_1,
        "team_2": team_2,
        "winner": final_winner
    })

    structured_submissions.append({
        "id": i,
        "season": season,
        "predictions": json.dumps(full_structure, ensure_ascii=False, separators=(',', ':'))
    })

submission_df = pd.DataFrame(structured_submissions, columns=["id", "season", "predictions"])
submission_path = '/Users/gunin/Desktop/IIT Madras/UCL Predictor/submission_final.csv'
submission_df.to_csv(submission_path, index=False)
print(f"✅ Final submission written exactly like sample_submission.csv: {submission_path}")


# -------------------------------
# Scoring Function
# -------------------------------

def score_predictions(predicted_bracket, true_bracket):
    score = 0
    round_weights = {
        "round_of_16": 1,
        "quarter_finals": 3,
        "semi_finals": 8,
        "final": 20
    }

    for round_key, weight in round_weights.items():
        pred_winners = predicted_bracket.get(round_key, [])
        true_winners = true_bracket.get(round_key, [])
        print(f"\n🔎 {round_key.upper()}")
        for i, (pred, actual) in enumerate(zip(pred_winners, true_winners)):
            result = "✅ Correct" if clean_team_name(pred) == clean_team_name(actual) else "❌ Incorrect"
            print(f"Match {i+1}: Predicted = {pred}, Actual = {actual} → {result}")
            if result == "✅ Correct":
                score += weight

    print(f"\n🎯 Total Score (weighted for this season): {score}/56")
    return score

# -------------------------------
# Evaluate Predictions on Test Seasons
# -------------------------------

# Initialize overall score tracking
overall_total_score = 0
total_possible_score_per_season = 56

# Define true winners for each season
true_winners_2017_18 = {
    "round_of_16": ["Juventus", "Manchester City", "Liverpool", "Sevilla", "Real Madrid", "Roma", "Barcelona", "Bayern Munich"],
    "quarter_finals": ["Real Madrid", "Liverpool", "Bayern Munich", "Roma"],
    "semi_finals": ["Liverpool", "Real Madrid"],
    "final": ["Real Madrid"]
}

if "2017-18" in test_matchups:
    predicted_2017_18 = simulate_tournament(
        "2017-18",
        test_matchups["2017-18"],
        historical_data_for_prediction=full_data,
        xg_data_for_prediction=xg_data
    )
    print("\n--- Evaluating 2017-18 Season ---")
    season_score = score_predictions(predicted_2017_18, true_winners_2017_18)
    overall_total_score += season_score

true_winners_2018_19 = {
    "round_of_16": ["Porto", "Manchester United", "Tottenham Hotspur", "Ajax", "Barcelona", "Liverpool", "Juventus", "Manchester City"],
    "quarter_finals": ["Ajax", "Tottenham Hotspur", "Barcelona", "Liverpool"],
    "semi_finals": ["Tottenham Hotspur", "Liverpool"],
    "final": ["Liverpool"]
}

if "2018-19" in test_matchups:
    predicted_2018_19 = simulate_tournament(
        "2018-19",
        test_matchups["2018-19"],
        historical_data_for_prediction=full_data,
        xg_data_for_prediction=xg_data
    )
    print("\n--- Evaluating 2018-19 Season ---")
    season_score = score_predictions(predicted_2018_19, true_winners_2018_19)
    overall_total_score += season_score

true_winners_2019_20 = {
    "round_of_16": ["Paris Saint-Germain", "Manchester City", "Atalanta", "Atletico Madrid", "Bayern Munich", "Lyon", "RB Leipzig", "Barcelona"],
    "quarter_finals": ["Paris Saint-Germain", "RB Leipzig", "Lyon", "Bayern Munich"],
    "semi_finals": ["Paris Saint-Germain", "Bayern Munich"],
    "final": ["Bayern Munich"]
}

if "2019-20" in test_matchups:
    predicted_2019_20 = simulate_tournament(
        "2019-20",
        test_matchups["2019-20"],
        historical_data_for_prediction=full_data,
        xg_data_for_prediction=xg_data
    )
    print("\n--- Evaluating 2019-20 Season ---")
    season_score = score_predictions(predicted_2019_20, true_winners_2019_20)
    overall_total_score += season_score

true_winners_2020_21 = {
    "round_of_16": ["Liverpool", "Paris Saint-Germain", "Porto", "Borussia Dortmund", "Chelsea", "Real Madrid", "Manchester City"],
    "quarter_finals": ["Chelsea", "Paris Saint-Germain", "Real Madrid", "Manchester City"],
    "semi_finals": ["Chelsea", "Manchester City"],
    "final": ["Chelsea"]
}

if "2020-21" in test_matchups:
    predicted_2020_21 = simulate_tournament(
        "2020-21",
        test_matchups["2020-21"],
        historical_data_for_prediction=full_data,
        xg_data_for_prediction=xg_data
    )
    print("\n--- Evaluating 2020-21 Season ---")
    season_score = score_predictions(predicted_2020_21, true_winners_2020_21)
    overall_total_score += season_score

true_winners_2021_22 = {
    "round_of_16": ["Real Madrid", "Manchester City", "Bayern Munich", "Liverpool", "Chelsea", "Villarreal", "Atletico Madrid", "Benfica"],
    "quarter_finals": ["Real Madrid", "Manchester City", "Villarreal", "Liverpool"],
    "semi_finals": ["Liverpool", "Real Madrid"],
    "final": ["Real Madrid"]
}

if "2021-22" in test_matchups:
    predicted_2021_22 = simulate_tournament(
        "2021-22",
        test_matchups["2021-22"],
        historical_data_for_prediction=full_data,
        xg_data_for_prediction=xg_data
    )
    print("\n--- Evaluating 2021-22 Season ---")
    season_score = score_predictions(predicted_2021_22, true_winners_2021_22)
    overall_total_score += season_score

true_winners_2022_23 = {
    "round_of_16": ["Manchester City", "Benfica", "Real Madrid", "AC Milan", "Napoli", "Chelsea", "Inter Milan", "Bayern Munich"],
    "quarter_finals": ["Inter Milan", "AC Milan", "Manchester City", "Real Madrid"],
    "semi_finals": ["Inter Milan", "Manchester City"],
    "final": ["Manchester City"]
}

if "2022-23" in test_matchups:
    predicted_2022_23 = simulate_tournament(
        "2022-23",
        test_matchups["2022-23"],
        historical_data_for_prediction=full_data,
        xg_data_for_prediction=xg_data
    )
    print("\n--- Evaluating 2022-23 Season ---")
    season_score = score_predictions(predicted_2022_23, true_winners_2022_23)
    overall_total_score += season_score

true_winners_2023_24 = {
    "round_of_16": ["Manchester City", "Real Madrid", "Bayern Munich", "Paris Saint-Germain", "Atletico Madrid", "Borussia Dortmund", "Arsenal","Barcelona"],
    "quarter_finals": ["Bayern Munich", "Borussia Dortmund", "Paris Saint-Germain", "Real Madrid"],
    "semi_finals": ["Real Madrid", "Borussia Dortmund"],
    "final": ["Real Madrid"]
}

if "2023-24" in test_matchups:
    predicted_2023_24 = simulate_tournament(
        "2023-24",
        test_matchups["2023-24"],
        historical_data_for_prediction=full_data,
        xg_data_for_prediction=xg_data
    )
    print("\n--- Evaluating 2023-24 Season ---")
    season_score = score_predictions(predicted_2023_24, true_winners_2023_24)
    overall_total_score += season_score

# Calculate and print overall model performance across all evaluated seasons
num_evaluated_seasons = 0
for season_key in ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]:
    if season_key in test_matchups:
        num_evaluated_seasons += 1

max_overall_score = num_evaluated_seasons * total_possible_score_per_season

print(f"\n==============================================")
print(f"🏆 OVERALL MODEL PERFORMANCE ACROSS ALL SEASONS")
print(f"==============================================")
print(f"Total Score (weighted, combined across seasons): {overall_total_score}/{max_overall_score}")
print(f"Average Score per Season: {overall_total_score / num_evaluated_seasons:.2f}/{total_possible_score_per_season}")
print(f"Overall Accuracy Percentage: {(overall_total_score / max_overall_score) * 100:.2f}%")
print(f"==============================================")