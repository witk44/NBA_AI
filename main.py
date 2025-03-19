from playersAPI import *
from player import player
from nba_api.stats.endpoints import *
players = []
for PLAYER in get_active_players():
    players.append(player(PLAYER['first_name'],PLAYER['last_name'],PLAYER['id']))


for athlete in players:
    if "James" in athlete.last_name:
        print(athlete.id)
        career = playercareerstats.PlayerCareerStats(player_id='2544')
        print(career.get_data_frames()[0])

        # json
        # print(career.get_json())

        # dictionary
        # print(career.get_dict())