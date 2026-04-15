import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

# NBA API Imports
import nba_api.stats.static.players as players
import nba_api.stats.static.teams as find_team
import nba_api.stats.endpoints.playergamelog as game_logs
from nba_api.stats.endpoints import leagueleaders
import nba_on_court as noc

# --- INITIALIZATION & CACHING ---
st.set_page_config(page_title="NBA Shot Chart Medley", layout="wide")

# Fixing numpy compatibility for nba_on_court
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# Load all players for the dropdown
@st.cache_resource
def get_player_list():
    all_players = players.get_players()
    return {p['full_name']: p['id'] for p in all_players}

player_dict = get_player_list()

# --- HELPER FUNCTIONS (FROM NOTEBOOK) ---

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    if ax is None:
        ax = plt.gca()
    ax.set_facecolor('#D2B48C')
    # Court Elements (Hoop, Backboard, Paint, etc.)
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color, linestyle='dashed')
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color)

    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc, center_inner_arc]
    if outer_lines:
        outer_lines_rect = Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False)
        court_elements.append(outer_lines_rect)

    for element in court_elements:
        ax.add_patch(element)
    return ax

@st.cache_data(show_spinner="Downloading NBA Data...")
def load_prep_data(player_id, year):
    # Fetch player log to get game IDs
    headers = {"User-Agent": "Mozilla/5.0"}
    log = game_logs.PlayerGameLog(str(player_id), headers=headers, season=str(year)).get_data_frames()[0]
    log = log.rename(columns={'Game_ID': 'GAME_ID'})
    log['GAME_ID'] = log['GAME_ID'].astype(int)
    
    # Load play-by-play
    noc.load_nba_data(seasons=year, data='nbastats', untar=True)
    pbp_file = f'nbastats_{year}.csv'
    play_by_play = pd.read_csv(pbp_file)
    os.remove(pbp_file)

    # Load shot detail
    noc.load_nba_data(seasons=year, data='shotdetail', untar=True)
    shot_file = f'shotdetail_{year}.csv'
    shots = pd.read_csv(shot_file)
    os.remove(shot_file)

    # Filter for games player actually played
    game_ids = log['GAME_ID'].unique()
    plays = play_by_play[play_by_play['GAME_ID'].isin(game_ids)]

    # Simplified On-Court Logic for Streamlit Performance
    # In a real app, you might want to pre-process this data
    on_court_plays_list = []
    for g_id in game_ids:
        game_data = plays[plays['GAME_ID'] == g_id].reset_index(drop=True)
        try:
            df = noc.players_on_court(game_data)
            on_court_plays_list.append(df)
        except:
            continue
    
    if not on_court_plays_list:
        return pd.DataFrame()

    on_court_df = pd.concat(on_court_plays_list, ignore_index=True)
    on_court_df = on_court_df.rename(columns={'EVENTNUM': 'GAME_EVENT_ID'})
    
    merged = on_court_df.merge(shots, on=['GAME_ID', 'GAME_EVENT_ID'], how='inner')
    return merged

# --- APP LAYOUT ---

st.title("🏀 NBA Shot Chart Medley")
st.markdown("Visualize how a team's shot profile changes when a specific player is on or off the floor.")

with st.sidebar:
    st.header("Settings")
    selected_player_name = st.selectbox("Select Player", options=list(player_dict.keys()), index=list(player_dict.keys()).index("LeBron James"))
    selected_player_id = player_dict[selected_player_name]
    
    selected_year = st.number_input("Season (Start Year)", min_value=1996, max_value=2023, value=2022)
    chart_type = st.radio("Chart Type", ["Scatter Chart", "Hexbin Heatmap", "Player Involvement"])

# Data Loading
df = load_prep_data(selected_player_id, selected_year)

if df.empty:
    st.error("No data found for this selection. Try a different year or player.")
else:
    # Filter On/Off Court
    on_cols = ['AWAY_PLAYER1', 'AWAY_PLAYER2', 'AWAY_PLAYER3', 'AWAY_PLAYER4', 'AWAY_PLAYER5',
               'HOME_PLAYER1', 'HOME_PLAYER2', 'HOME_PLAYER3', 'HOME_PLAYER4', 'HOME_PLAYER5']
    on_mask = df[on_cols].eq(selected_player_id).any(axis=1)
    
    df_on = df[on_mask]
    df_off = df[~on_mask]

    col1, col2 = st.columns(2)

    def create_plot(data, title, is_hex=False, involvement=False):
        fig, ax = plt.subplots(figsize=(10, 8))
        draw_court(ax, outer_lines=True)
        
        if involvement:
            # Special logic for involvement
            made = data[data['SHOT_MADE_FLAG'] == 1]
            ax.scatter(made[made['PLAYER1_ID'] == selected_player_id]['LOC_X'], 
                       made[made['PLAYER1_ID'] == selected_player_id]['LOC_Y'], color='red', label='Made by Player')
            ax.scatter(made[made['PLAYER2_ID'] == selected_player_id]['LOC_X'], 
                       made[made['PLAYER2_ID'] == selected_player_id]['LOC_Y'], color='blue', label='Assisted by Player')
            ax.legend()
        elif is_hex:
            hb = ax.hexbin(data['LOC_X'], data['LOC_Y'], gridsize=25, bins='log', cmap='inferno', mincnt=1)
            plt.colorbar(hb, ax=ax, label='Frequency (log)')
        else:
            ax.scatter(data[data['SHOT_MADE_FLAG'] == 1]['LOC_X'], data[data['SHOT_MADE_FLAG'] == 1]['LOC_Y'], color='red', s=10, alpha=0.5, label="Made")
            ax.scatter(data[data['SHOT_MADE_FLAG'] == 0]['LOC_X'], data[data['SHOT_MADE_FLAG'] == 0]['LOC_Y'], color='blue', s=10, alpha=0.3, label="Missed")
            ax.legend()

        ax.set_xlim(-250, 250)
        ax.set_ylim(-50, 420)
        ax.set_title(title)
        ax.axis('off')
        return fig

    with col1:
        st.subheader("On Court")
        st.pyplot(create_plot(df_on, f"Team Shots with {selected_player_name} ON", is_hex=(chart_type == "Hexbin Heatmap"), involvement=(chart_type == "Player Involvement")))

    with col2:
        st.subheader("Off Court")
        if chart_type == "Player Involvement":
            st.info("Involvement chart is only applicable for 'On Court' play.")
        else:
            st.pyplot(create_plot(df_off, f"Team Shots with {selected_player_name} OFF", is_hex=(chart_type == "Hexbin Heatmap")))

    # Similarity Score
    st.divider()
    if not df_on.empty and not df_off.empty:
        x_bins = np.linspace(-250, 250, 25)
        y_bins = np.linspace(-50, 470, 25)
        hist_on, _, _ = np.histogram2d(df_on['LOC_X'], df_on['LOC_Y'], bins=[x_bins, y_bins])
        hist_off, _, _ = np.histogram2d(df_off['LOC_X'], df_off['LOC_Y'], bins=[x_bins, y_bins])
        score = cosine_similarity(hist_on.flatten().reshape(1, -1), hist_off.flatten().reshape(1, -1))[0][0]
        st.metric("Shot Profile Similarity Score", f"{score:.4f}")
        st.caption("A higher score means the team plays similarly whether the player is on or off the court.")
