import pandas as pd
import requests
from time import sleep
from bs4 import BeautifulSoup

HLTV_URL = "https://www.hltv.org"


def scrap_tournament(tournament_url):
    tournament_page = requests.get(tournament_url)
    tournament_soup = BeautifulSoup(tournament_page.content, "html.parser")
    teams_div = tournament_soup.find("div", class_="placements")
    event_id = tournament_url.split("/")[4]
    teams = []
    for team_div in teams_div.find_all(class_="team"):
        team_href = team_div.a["href"].split("/")
        teams.append((team_href[2], team_href[3]))

    data = []
    for team_id, team_name in teams:
        sleep(1)
        players = get_team(f"{HLTV_URL}/stats/teams/players/{team_id}/{team_name}?event={event_id}")
        data.extend(players)

    return pd.DataFrame(data)


def get_player_data(url):
    player_page = requests.get(url)
    player_soup = BeautifulSoup(player_page.content, "html.parser")
    stats_div = player_soup.find_all(class_="summaryStatBreakdown")
    data = {}
    for div in stats_div:
        name_div = div.find(class_="summaryStatBreakdownSubHeader")
        name = name_div.text.split("\n")[0]
        value_div = div.find(class_="summaryStatBreakdownDataValue")
        value = float(value_div.text.replace("%", "e-2"))
        data[name] = value
    return data


def get_team(team_url):
    team_page = requests.get(team_url)
    team_soup = BeautifulSoup(team_page.content, "html.parser")
    team_table = team_soup.find("table", class_="stats-table player-ratings-table")
    players = []
    for row in team_table.tbody.find_all("tr"):
        player = {}
        for j, col in enumerate(row.find_all("td")):
            if "playerCol" in col["class"]:
                player["Player"] = col.text
                player_url = col.a["href"]
                player_data = get_player_data(f"{HLTV_URL}{player_url}")
                sleep(1)
                player.update(player_data)
            if "statsDetail" in col["class"] and j == 4:
                player["K/D"] = float(col.text)
        players.append(player)
    return players


if __name__ == "__main__":
    data = scrap_tournament("https://www.hltv.org/events/6345/blast-premier-spring-final-2022")
    data.to_csv("spring_data.csv", index=False)
