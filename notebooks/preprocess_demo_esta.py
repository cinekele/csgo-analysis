import json
import lzma

import pandas as pd


def read_parsed_demo(filename):
    with lzma.LZMAFile(filename, "rb") as f:
        d = json.load(f)
    return d


def get_team_data(frame, team, mapping):
    team_frame = frame[team]
    team_data = {team + 'Name': team_frame['teamName'], team + 'EqVal': team_frame['teamEqVal'],
                 team + 'AlivePlayers': team_frame['alivePlayers'], team + 'TotalUtility': team_frame['totalUtility']}

    for player in team_frame['players']:
        mapped_player = mapping[player['steamID']]
        team_data[f"{team}{mapped_player}_ID"] = player['steamID']
        for key_player in player:
            if key_player not in ['inventory', 'steamID', 'name', 'team', 'side', 'flashGrenades', 'smokeGrenades',
                                  'heGrenades', 'fireGrenades', 'totalUtility']:
                team_data[f'{team}{mapped_player}_{key_player}'] = player[key_player]
            elif key_player == 'inventory':
                team_data[f"{team}{mapped_player}_SmokeGrenade"] = 0
                team_data[f"{team}{mapped_player}_Flashbang"] = 0
                team_data[f"{team}{mapped_player}_DecoyGrenade"] = 0
                team_data[f"{team}{mapped_player}_fireGrenades"] = 0
                team_data[f"{team}{mapped_player}_HEGrenade"] = 0
                if player[key_player] is None:
                    team_data[f'{team}{mapped_player}_mainWeapon'] = ''
                    team_data[f'{team}{mapped_player}_secondaryWeapon'] = ''
                else:
                    for weapon in player[key_player]:
                        if weapon['weaponClass'] == 'Pistols':
                            team_data[f'{team}{mapped_player}_secondaryWeapon'] = weapon['weaponName']
                        elif weapon['weaponClass'] == 'Grenade':
                            if weapon['weaponName'] in {"Molotov", "Incendiary Grenade"}:
                                team_data[f"{team}{mapped_player}_fireGrenades"] = weapon['ammoInMagazine'] + \
                                                                                   weapon['ammoInReserve']
                            else:
                                team_data[f"{team}{mapped_player}_{weapon['weaponName'].replace(' ', '')}"] = \
                                    weapon['ammoInMagazine'] + weapon['ammoInReserve']
                        else:
                            team_data[f'{team}{mapped_player}_mainWeapon'] = weapon['weaponName']
                    if f'{team}{mapped_player}_mainWeapon' not in team_data and \
                            f'{team}{mapped_player}_secondaryWeapon' not in team_data:
                        team_data[f'{team}{mapped_player}_mainWeapon'] = ''
                    elif f'{team}{mapped_player}_mainWeapon' not in team_data:
                        team_data[f'{team}{mapped_player}_mainWeapon'] = \
                            team_data[f'{team}{mapped_player}_secondaryWeapon']
    return team_data


def get_frame_data(frame, mapping):
    frame_data = {**get_team_data(frame, 'ct', mapping), **get_team_data(frame, 't', mapping),
                  'bombPlanted': frame['bombPlanted'], 'bombsite': frame['bombsite'], 'tick': frame['tick'],
                  'seconds': frame['seconds'], 'clockTime': frame['clockTime']}
    bomb_data = frame['bomb']
    for key in bomb_data:
        frame_data[f"bomb_{key}"] = bomb_data[key]
    return frame_data


def create_mapping(round_):
    ct_players = round_['ctSide']
    map_steam_id = {}
    for i, player in enumerate(ct_players['players']):
        map_steam_id[player['steamID']] = f'Player_{i + 1}'

    t_players = round_['tSide']
    for i, player in enumerate(t_players['players']):
        map_steam_id[player['steamID']] = f'Player_{i + 1}'

    return map_steam_id


def preprocess_round_score(parsed_demo: dict) -> pd.DataFrame:
    data = [
        (round_["roundNum"], round_['ctScore'], round_['tScore'])
        for round_ in parsed_demo["gameRounds"]
    ]
    round_df = pd.DataFrame.from_records(data, columns=["roundNum", "ctScore", "tScore"])
    round_df["demoId"] = parsed_demo["demoId"]
    round_df["mapName"] = parsed_demo["mapName"]
    return round_df


def preprocess_kill_values(parsed_demo: dict) -> pd.DataFrame:
    counter = dict()
    data = []
    for round_ in parsed_demo["gameRounds"]:
        for kill in round_["kills"]:
            counter[kill["attackerSteamID"]] = counter.get(kill["attackerSteamID"], 0) + 1
        for key in counter:
            data.append((round_["roundNum"], key, counter[key]))

    kill_df = pd.DataFrame.from_records(data, columns=["roundNum", "attackerID", "kills"])
    kill_df["demoId"] = parsed_demo["demoId"]
    kill_df["mapName"] = parsed_demo["mapName"]
    return kill_df


def get_match_data(data):
    data_list = []
    round_ = data['gameRounds'][1] if data['gameRounds'][0]['ctSide']['players'] is None else data['gameRounds'][0]
    mapping = create_mapping(round_)
    for round_ in data['gameRounds']:
        for frame in round_['frames']:
            if (frame["ct"]["players"] is not None) & (frame["t"]["players"] is not None) & (
                    frame["clockTime"] != "00:00") & (frame["t"]["alivePlayers"] >= 0) & (
                    frame["ct"]["alivePlayers"] >= 1):
                if (len(frame["ct"]["players"]) == 5) & (len(frame["t"]["players"]) == 5):
                    converted_vector = get_frame_data(frame, mapping)
                    converted_vector['roundNum'] = round_['roundNum']
                    converted_vector['winningSide'] = round_['winningSide']
                    data_list.append(converted_vector)
    res = pd.DataFrame(data_list)
    res['matchName'] = data['matchName']
    res["demoId"] = data["demoId"]
    res['mapName'] = data['mapName']
    res.fillna(method='ffill', inplace=True)
    return res


def preprocess_demo(filename):
    parsed_demo = read_parsed_demo(filename)
    match_df = get_match_data(parsed_demo)
    kill_df = preprocess_kill_values(parsed_demo)
    round_df = preprocess_round_score(parsed_demo)
    return {
        "frame": match_df,
        "kills": kill_df,
        "roundScore": round_df
    }
