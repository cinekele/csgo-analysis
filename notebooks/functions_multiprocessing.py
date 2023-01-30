from multiprocessing import Pool

import numpy as np
import pandas as pd
from awpy.analytics.nav import find_closest_area
from awpy.data import NAV, PLACE_DIST_MATRIX
from tqdm import tqdm


def get_last_positions(df):
    def _get_last_positions(map_name, are_alive, positions):
        new_positions = []
        for position, isAlive in zip(positions, are_alive):
            if isAlive:
                area_id = find_closest_area(map_name, position)['areaId']
                new_place = NAV[map_name][area_id]["areaName"]
                new_positions.append(new_place)
            else:
                new_positions.append(None)
        return new_positions

    parts = []
    for team in tqdm(['ct', 't'], leave=True):
        data = [
            _get_last_positions(map_name=map_name,
                                are_alive=[isAlive1, isAlive2, isAlive3, isAlive4, isAlive5],
                                positions=[[p1_x, p1_y, p1_z], [p2_x, p2_y, p2_z], [p3_x, p3_y, p3_z],
                                           [p4_x, p4_y, p4_z], [p5_x, p5_y, p5_z]])
            for map_name, isAlive1, isAlive2, isAlive3, isAlive4, isAlive5,
            p1_x, p1_y, p1_z, p2_x, p2_y, p2_z, p3_x, p3_y, p3_z, p4_x, p4_y, p4_z, p5_x, p5_y, p5_z in
            zip(
                df[f'mapName'],
                df[f'{team}Player_1_isAlive'],
                df[f'{team}Player_2_isAlive'],
                df[f'{team}Player_3_isAlive'],
                df[f'{team}Player_4_isAlive'],
                df[f'{team}Player_5_isAlive'],
                df[f'{team}Player_1_x'],
                df[f'{team}Player_1_y'],
                df[f'{team}Player_1_z'],
                df[f'{team}Player_2_x'],
                df[f'{team}Player_2_y'],
                df[f'{team}Player_2_z'],
                df[f'{team}Player_3_x'],
                df[f'{team}Player_3_y'],
                df[f'{team}Player_3_z'],
                df[f'{team}Player_4_x'],
                df[f'{team}Player_4_y'],
                df[f'{team}Player_4_z'],
                df[f'{team}Player_5_x'],
                df[f'{team}Player_5_y'],
                df[f'{team}Player_5_z'])
        ]
        part = pd.DataFrame(data, columns=[f'{team}_Player_1_lastPlaceName', f'{team}_Player_2_lastPlaceName',
                                           f'{team}_Player_3_lastPlaceName', f'{team}_Player_4_lastPlaceName',
                                           f'{team}_Player_5_lastPlaceName'])
        parts.append(part)
    return pd.concat(parts, axis='columns')


def get_dist(df):
    def _get_dist(map_name, spots):
        new_positions = list(filter(lambda x: x is not None, spots))

        if len(new_positions) <= 1:
            return 9000, 9000
        else:
            try:
                dist = [
                    PLACE_DIST_MATRIX[map_name][position_x][position_y]['geodesic']['median_dist']
                    for i, position_x in enumerate(new_positions)
                    for j, position_y in enumerate(new_positions)
                    if i != j
                ]
            except:
                print(new_positions)
                print(map_name)
            dist_arr = np.array(dist)
            dist_arr = dist_arr[np.isfinite(dist_arr)]
            if dist_arr.size == 0:
                return 9000, 9000
            return np.mean(dist_arr), np.min(dist_arr)

    dist_parts = []
    for team in tqdm(['ct', 't'], leave=True, position=0):
        data = [
            _get_dist(mapName, [spot1, spot2, spot3, spot4, spot5])
            for mapName, spot1, spot2, spot3, spot4, spot5, in zip(
                df['mapName'],
                df[f'{team}_Player_1_lastPlaceName'],
                df[f'{team}_Player_2_lastPlaceName'],
                df[f'{team}_Player_3_lastPlaceName'],
                df[f'{team}_Player_4_lastPlaceName'],
                df[f'{team}_Player_5_lastPlaceName'])
        ]
        part = pd.DataFrame(data, columns=[f'{team}_meanDist', f'{team}_minDist'])
        dist_parts.append(part)
    return pd.concat(dist_parts, axis='columns')


def get_grouped_players(df):
    def _get_grouped_players(spots):
        counter = {}
        for spot in spots:
            if spot is not None:
                counter[spot] = counter.get(spot, 0) + 1
        maximum = 0 if len(counter.values()) == 0 else max(counter.values())
        return maximum

    groupped_parts = []
    for team in tqdm(['ct', 't'], leave=True, position=0):
        data = [_get_grouped_players(spots=[spot1, spot2, spot3, spot4, spot5])
                for spot1, spot2, spot3, spot4, spot5 in zip(
                df[f'{team}_Player_1_lastPlaceName'],
                df[f'{team}_Player_2_lastPlaceName'],
                df[f'{team}_Player_3_lastPlaceName'],
                df[f'{team}_Player_4_lastPlaceName'],
                df[f'{team}_Player_5_lastPlaceName'])
                ]
        part = pd.DataFrame(data, columns=[f'{team}_grouppedPlayers'])
        groupped_parts.append(part)

    return pd.concat(groupped_parts, axis='columns')


def get_bombsite_dist(df):
    def _get_bombsite_dist(map_name, position):
        if position is not None:
            dist_to_a = PLACE_DIST_MATRIX[map_name][position]['BombsiteA']['geodesic']['median_dist']
            dist_to_b = PLACE_DIST_MATRIX[map_name][position]['BombsiteB']['geodesic']['median_dist']
        else:
            dist_to_a = None
            dist_to_b = None
        return dist_to_a, dist_to_b

    all_parts = []
    for team in tqdm(['ct', 't'], leave=True, position=0):
        for i in tqdm(range(1, 6), leave=True, position=1):
            data = [
                _get_bombsite_dist(isAlive, lastPlace)
                for isAlive, lastPlace in
                zip(df[f'mapName'],
                    df[f'{team}_Player_{i}_lastPlaceName'])
            ]
            part = pd.DataFrame.from_records(data, columns=[f'{team}Player_{i}_distToA', f'{team}Player_{i}_distToB'])
            all_parts.append(part)
    return pd.concat(all_parts, axis="columns")


def get_spotted_players(df):
    def _get_spotted_players(*args):
        spotted = set()
        for arg in args:
            spotted.update(set(arg))
        return len(spotted)

    spotters_parts = []
    for team in tqdm(['ct', 't'], leave=True, position=0):
        data = [
            _get_spotted_players(spot1, spot2, spot3, spot4, spot5)
            for spot1, spot2, spot3, spot4, spot5 in
            zip(df[f'{team}Player_1_spotters'],
                df[f'{team}Player_2_spotters'],
                df[f'{team}Player_3_spotters'],
                df[f'{team}Player_4_spotters'],
                df[f'{team}Player_5_spotters'])
        ]
        part = pd.DataFrame(data, columns=[f'{team}_spottedPlayers'])
        spotters_parts.append(part)
    return pd.concat(spotters_parts, axis='columns')


def conquer_map(df):
    data = []
    for mapName, roundNum, lastRoundNum, \
            ct_spot_1, ct_spot_2, ct_spot_3, ct_spot_4, ct_spot_5, \
            t_spot_1, t_spot_2, t_spot_3, t_spot_4, t_spot_5 in tqdm(
        zip(df['mapName'], df['roundNum'], df['roundNum'].shift(fill_value=0),
            df['ct_Player_1_lastPlaceName'],
            df['ct_Player_2_lastPlaceName'],
            df['ct_Player_3_lastPlaceName'],
            df['ct_Player_4_lastPlaceName'],
            df['ct_Player_5_lastPlaceName'],
            df['t_Player_1_lastPlaceName'],
            df['t_Player_2_lastPlaceName'],
            df['t_Player_3_lastPlaceName'],
            df['t_Player_4_lastPlaceName'],
            df['t_Player_5_lastPlaceName'])):
        if lastRoundNum != roundNum:
            ct_areas = set()
            t_areas = set()

        curr_ct_areas = {
            ct_spot_1, ct_spot_2, ct_spot_3, ct_spot_4, ct_spot_5
        }
        curr_t_areas = {
            t_spot_1, t_spot_2, t_spot_3, t_spot_4, t_spot_5
        }
        for area in curr_ct_areas:
            if area not in curr_t_areas:
                ct_areas.add(area)
            if area in t_areas:
                t_areas.remove(area)

        for area in curr_t_areas:
            if area not in curr_ct_areas:
                t_areas.add(area)
            if area in ct_areas:
                ct_areas.remove(area)
        ct_areas.discard(None)
        t_areas.discard(None)
        ct_percentage = len(ct_areas) / len(PLACE_DIST_MATRIX[mapName].keys())
        t_percentage = len(t_areas) / len(PLACE_DIST_MATRIX[mapName].keys())
        data.append((ct_percentage, t_percentage))
    return pd.DataFrame.from_records(data, columns=['ct_conquerMap', 't_conquerMap'])


def parallelize_dataframe(df, func, n_cores):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df.reset_index(drop=True)
