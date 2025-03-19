import streamlit as st
import pandas as pd
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics._regression import UndefinedMetricWarning
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from playersAPI import *
from player import player
from nba_api.stats.endpoints import *
import matplotlib.pyplot as plt






warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# App Title
st.title("🏀 NBA Player Career Stats Viewer")
st.write("Enter an NBA player's name to view their career stats.")

# Input for player name
player_name = st.text_input("Player Name", placeholder="e.g., LeBron James")

def plot_ppg_with_confidence_interval(df, predicted_ppg, confidence_interval):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['SEASON_ID'], df['PPG'], label="PPG per Season", color='blue', marker='o')
    ax.plot(df['SEASON_ID'].max() + 1, predicted_ppg, marker='o', color='red', label="Predicted PPG")
    ax.axvline(x=df['SEASON_ID'].max() + 1, color='red', linestyle='--', label="Next Season Prediction")
    ax.legend()
    ax.set_title("Points Per Game (PPG) with Confidence Interval")
    ax.set_xlabel("Season")
    ax.set_ylabel("PPG")
    st.pyplot(fig)


def predict_next_season_ppg(stats_df, degree_range=(1, 5)):
    if len(stats_df) < 2:
        return "Insufficient data to predict PPG."
    
    # Extract features and target variable
    stats_df['SEASON_ID'] = stats_df['SEASON_ID'].astype(str).str[:4].astype(int)
    X = stats_df[['SEASON_ID', 'MIN', 'FG_PCT', 'PLAYER_AGE']].values
    y = stats_df['PPG'].values
    
    # Try different polynomial degrees and pick the one with best cross-validation score
    best_degree = max(degree_range, key=lambda d: np.mean(cross_val_score(LinearRegression(), PolynomialFeatures(d).fit_transform(X), y, cv=5)))
    poly = PolynomialFeatures(degree=best_degree)
    X_poly = poly.fit_transform(X)
    
    # Train model with or without weights
    weights = stats_df.get('Weight', np.ones(len(stats_df)))  # Default weight to 1 if not present
    model = LinearRegression()
    model.fit(X_poly, y, sample_weight=weights)

    # Predict next season's PPG
    next_season_data = np.array([[X[-1, 0] + 1, X[-1, 1], X[-1, 2], X[-1, 3] + 1]])
    next_season_poly = poly.transform(next_season_data)
    predicted_ppg = model.predict(next_season_poly)[0]

    # Calculate confidence interval (using standard error of predictions)
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred, sample_weight=weights)
    standard_error = np.sqrt(mse)
    confidence_interval = 1.96 * standard_error

    return predicted_ppg, confidence_interval





def fetch_player_stats(name):
    players = get_active_players()
    
    # Try to find the player directly without unnecessary object creation
    for PLAYER in players:
        first_last = f"{PLAYER['first_name'].lower()} {PLAYER['last_name'].lower()}"
        if name.lower() in first_last:
            id = str(PLAYER['id'])
            career = playercareerstats.PlayerCareerStats(player_id=id)
            career_df = career.get_data_frames()[0]
            career_df = career_df[career_df['TEAM_ID'] != 'TOT']
            # Drop unnecessary columns
            career_df = career_df.drop(columns=['PLAYER_ID', 'LEAGUE_ID', 'TEAM_ID'])

            # Calculate per game stats and round to the tenth place
            career_df['PPG'] = (career_df['PTS'] / career_df['GP'].replace(0, np.nan)).round(1)
            career_df['RPG'] = ((career_df['OREB'] + career_df['DREB']) / career_df['GP'].replace(0, np.nan)).round(1)
            career_df['APG'] = (career_df['AST'] / career_df['GP'].replace(0, np.nan)).round(1)
            career_df['SPG'] = (career_df['STL'] / career_df['GP'].replace(0, np.nan)).round(1)
            career_df['BPG'] = (career_df['BLK'] / career_df['GP'].replace(0, np.nan)).round(1)

            # Apply higher weights to the latest seasons
            career_df['SEASON_ID'] = career_df['SEASON_ID'].str[:4].astype(int)
            max_season = career_df['SEASON_ID'].max()
            career_df['Weight'] = 1.0 + (career_df['SEASON_ID'] - (max_season - 3)).clip(lower=0) * 2.0

            return career_df
    return None  # If player not found

    
if player_name:
    
    stats_df = fetch_player_stats(player_name)
    if stats_df is not None:
        predicted_ppg, confidence_interval = predict_next_season_ppg(stats_df)
        selected_columns = ['Weight']
        stats_df = stats_df.drop(columns=selected_columns)
        st.success(f"Showing career stats for {player_name}")
        st.dataframe(stats_df, use_container_width=True)
        plot_ppg_with_confidence_interval(stats_df, predicted_ppg, confidence_interval)
        st.write(f"📈 Predicted PPG for Next Season: **{predicted_ppg:.2f}** ± {confidence_interval:.2f}")
        # Downloadable CSV
        csv = stats_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{player_name}_career_stats.csv",
            mime="text/csv"
        )
    else:
        st.error(f"Player {player_name} not found. Please try again.")

        
