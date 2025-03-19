import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from playersAPI import *
from player import player
from nba_api.stats.endpoints import *




# App Title
st.title("🏀 NBA Player Career Stats Viewer")
st.write("Enter an NBA player's name to view their career stats.")

# Input for player name
player_name = st.text_input("Player Name", placeholder="e.g., LeBron James")

def predict_next_season_ppg(stats_df, degree=3):
    # Ensure sufficient data points
    if len(stats_df) < 2:
        return "Insufficient data to predict PPG."

    # Extract Features and Target
    stats_df['SEASON_ID'] = stats_df['SEASON_ID'].str[:4].astype(int)
    X = stats_df[['SEASON_ID', 'MIN', 'FG_PCT', 'PLAYER_AGE']].values
    y = stats_df['PPG'].values

    # Apply Polynomial Features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Train Model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict for Next Season
    next_season_data = np.array([[X[-1, 0] + 1, X[-1, 1], X[-1, 2], X[-1, 3] + 1]])
    next_season_poly = poly.transform(next_season_data)
    predicted_ppg = model.predict(next_season_poly)[0]

    # Confidence Interval Calculation
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    standard_error = np.sqrt(mse)
    confidence_interval = 1.96 * standard_error

    return predicted_ppg, confidence_interval



def fetch_player_stats(name):
    players = []
    for PLAYER in get_active_players():
        athlete = player(PLAYER['first_name'],PLAYER['last_name'],PLAYER['id'])
        players.append(athlete)
    for athlete in players:
        first_last = f"{athlete.first_name.lower()} {athlete.last_name.lower()}"
        if (name.lower() in athlete.last_name.lower()) or (name.lower() in first_last):
            id = str(athlete.id)
            career = playercareerstats.PlayerCareerStats(player_id=id)
            career_df = career.get_data_frames()[0]
            selected_columns = ['PLAYER_ID','LEAGUE_ID','TEAM_ID']  # Example columns
            career_df = career_df.drop(columns=selected_columns)
            career_df['PPG'] = career_df['PTS'] / career_df['GP'].replace(0, np.nan)
             
            df = pd.DataFrame(career_df)
            return df
    
if player_name:
    stats_df = fetch_player_stats(player_name)
    if stats_df is not None:
        st.success(f"Showing career stats for {player_name}")
        st.dataframe(stats_df,use_container_width=True)

        # Plot PTS over seasons

        st.subheader("📊 PPG Per Season")
        st.line_chart(stats_df[['SEASON_ID', 'PPG']].set_index('SEASON_ID'))
        # Predict Next Season's PPG
        predicted_ppg, confidence_interval = predict_next_season_ppg(stats_df)
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
        print("failed")
