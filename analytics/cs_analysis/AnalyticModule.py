import json
import logging
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from awpy.analytics.nav import find_closest_area
from awpy.data import PLACE_DIST_MATRIX, NAV
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine
from lightgbm import LGBMClassifier


class AnalyticModule:
    KILL_GAIN = 0.3
    DAMAGE_GAIN = 0.3
    FLASH_GAIN = 0.2
    TRADE_GAIN = 0.2
    FLASH_PENALTY = 0.2
    KILL_DAMAGE_PENALTY = 0.8
    DIFF_IMPORTANT_MOMENTS = 0.1
    FRAMES_IMPORTANT_MOMENTS = 10
    __slots__ = ["input_data", "model", "log", "map_encoder", "bomb_encoder"]

    def __init__(self, model):
        self.model = model
        self.log = logging.getLogger(__name__)
        self.map_encoder = OneHotEncoder(
            categories=[['de_ancient', 'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_overpass', 'de_vertigo']],
            sparse=False,
            handle_unknown='ignore')
        self.bomb_encoder = OneHotEncoder(categories=[['A', 'B', 'not_planted']], sparse=False,
                                          handle_unknown='ignore')

    def transform_data(self, input_data, score, kills):
        corrected_data = self._correct_last_place(input_data)
        sum_columns = self._get_summed_columns(corrected_data,
                                               ["hp", "armor", "hasHelmet", "DecoyGrenade", "Flashbang", "HEGrenade",
                                                "SmokeGrenade", "fireGrenades", "isBlinded"],
                                               [("ct", "hasDefuse"), ("t", "hasBomb"), ("ct", "isDefusing"),
                                                ("t", "isPlanting")])
        dist_bomb_cols = self._get_distance_to_bombsites(corrected_data)
        dist_players_cols = self._get_distance_players(corrected_data)
        spotters_cols = self._get_spotters(corrected_data)
        active_weap_cols = self._convert_active_weapons(corrected_data, 'activeWeapon')
        main_weap_cols = self._convert_weapons(corrected_data, 'mainWeapon')
        sec_weap_cols = self._convert_sec_weapons(corrected_data, 'secondaryWeapon')
        bombsite_col = np.where(corrected_data['bombsite'] == '', 'not_planted', corrected_data['bombsite'])
        map_cols = pd.DataFrame(self.map_encoder.fit_transform(corrected_data[['mapName']]),
                                columns=['de_ancient', 'de_dust2', 'de_inferno', 'de_mirage',
                                         'de_nuke', 'de_overpass', 'de_vertigo'])
        other_cols = corrected_data[["seconds", "ctEqVal", "tEqVal", "ctAlivePlayers", "tAlivePlayers", "roundNum"
                                     ]]
        bombsite_cols = pd.DataFrame(self.bomb_encoder.fit_transform(bombsite_col.reshape(-1, 1)),
                                     columns=['bombsite_A', 'bombsite_B', 'bombsite_not_planted'])
        kills_cols = self._prepare_kills(corrected_data, kills)
        groupped_cols = self._get_grouped_players(corrected_data)
        conquer_cols = self._conquer_map(corrected_data)
        result = pd.concat([sum_columns, dist_bomb_cols, dist_players_cols, spotters_cols, active_weap_cols,
                            main_weap_cols, sec_weap_cols, map_cols, bombsite_cols, other_cols, kills_cols,
                            groupped_cols, conquer_cols],
                           axis=1)
        result = result.merge(score, left_on='roundNum', right_on='roundNum')
        return result[self.model.feature_name_]

    def get_predictions(self, input_data, transformed_data):
        # Round Not Ended
        in_game = (input_data.gameEnded == 0)
        ticks = input_data.tick[in_game].reset_index(drop=True)
        prediction_data = transformed_data[in_game]

        # Round Ended
        other_data = input_data[~in_game].reset_index(drop=True)
        other_data["probaCtWin"] = np.where(other_data.winningSide == "CT", 1, 0)

        predictions = self.model.predict_proba(prediction_data)[:, 1]
        tick_predictions = pd.concat([ticks, pd.Series(predictions, name="probaCtWin")], axis=1)
        tick_predictions = pd.concat([tick_predictions, other_data[["tick", "probaCtWin"]]], axis=0)
        sorted_ticks = tick_predictions.sort_values(by="tick", ignore_index=True)
        last_proba = np.where(input_data.winningSide.tail(1) == "CT", 1, 0).item()
        sorted_ticks["nextProbaCtWin"] = sorted_ticks.probaCtWin.shift(-1, fill_value=last_proba)
        return sorted_ticks

    def _prepare_kills(self, input_data, kills):
        prepared_kills = kills.groupby(["attackerID", "roundNum"]).size().reset_index(name="kills")
        kills_windowed = prepared_kills[["attackerID", "kills"]].groupby(["attackerID"]).expanding().sum().reset_index(
            drop=True)
        growing_kills = pd.concat([prepared_kills.drop(columns="kills"), kills_windowed], axis=1)
        growing_kills["roundNum"] += 1
        states = input_data.copy()
        for team in ['ct', 't']:
            for i in range(1, 6):
                states = states.merge(growing_kills, how='left', left_on=['roundNum', f'{team}Player_{i}_ID'],
                                      right_on=['roundNum', 'attackerID'],
                                      suffixes=(None, f"_{team}_{i}"))
        states.rename(columns={'kills': 'kills_ct_1'}, inplace=True)
        states.loc[states.roundNum == 1, [f'kills_{team}_{i}' for team in ['ct', 't'] for i in range(1, 6)]] = 0
        for team in ['ct', 't']:
            for i in range(1, 6):
                states[f'kills_{team}_{i}'] = states[f'kills_{team}_{i}'].fillna(method='ffill')
        states['ctMeanKills'] = (states['kills_ct_1'] * states['ctPlayer_1_isAlive'] + states['kills_ct_2'] *
                                 states['ctPlayer_2_isAlive'] + states['kills_ct_3'] * states['ctPlayer_3_isAlive'] +
                                 states['kills_ct_4'] * states['ctPlayer_4_isAlive'] +
                                 states['kills_ct_5'] * states['ctPlayer_5_isAlive']) / states[
                                    [f'ctPlayer_{i}_isAlive' for i in range(1, 6)]].sum(axis=1)
        states['tMeanKills'] = (states['kills_t_1'] * states['tPlayer_1_isAlive'] + states['kills_t_2'] *
                                states['tPlayer_2_isAlive'] +
                                states['kills_t_3'] * states['tPlayer_3_isAlive'] +
                                states['kills_t_4'] * states['tPlayer_4_isAlive'] +
                                states['kills_t_5'] * states['tPlayer_5_isAlive']) / states[
                                   [f'tPlayer_{i}_isAlive' for i in range(1, 6)]].sum(axis=1)
        states['tMeanKills'] = states['tMeanKills'].fillna(-1)
        states['ctMeanKills'] = states['ctMeanKills'].fillna(-1)
        return states[['ctMeanKills', 'tMeanKills']]

    @staticmethod
    def _get_summed_columns(input_data, sum_columns: list, team_specific: list) -> pd.DataFrame:
        sum_cols_dict = {}
        for team in ["ct", "t"]:
            for column in sum_columns:
                summed_cols = [f"{team}Player_1_{column}", f"{team}Player_2_{column}", f"{team}Player_3_{column}",
                               f"{team}Player_4_{column}", f"{team}Player_5_{column}"]
                sum_cols_dict[f"{team}_{column}"] = input_data[summed_cols].sum(axis=1)

        for team, column in team_specific:
            summed_cols = [f"{team}Player_1_{column}", f"{team}Player_2_{column}", f"{team}Player_3_{column}",
                           f"{team}Player_4_{column}", f"{team}Player_5_{column}"]
            sum_cols_dict[f"{team}_{column}"] = input_data[summed_cols].sum(axis=1)

        return pd.DataFrame.from_dict(sum_cols_dict)

    @staticmethod
    def _conquer_map(input_data):
        data = []
        for mapName, roundNum, lastRoundNum, \
            ct_spot_1, ct_spot_2, ct_spot_3, ct_spot_4, ct_spot_5, \
            t_spot_1, t_spot_2, t_spot_3, t_spot_4, t_spot_5 in zip(input_data['mapName'], input_data['roundNum'],
                                                                    input_data['roundNum'].shift(fill_value=0),
                                                                    input_data['ctPlayer_1_lastPlaceName'],
                                                                    input_data['ctPlayer_2_lastPlaceName'],
                                                                    input_data['ctPlayer_3_lastPlaceName'],
                                                                    input_data['ctPlayer_4_lastPlaceName'],
                                                                    input_data['ctPlayer_5_lastPlaceName'],
                                                                    input_data['tPlayer_1_lastPlaceName'],
                                                                    input_data['tPlayer_2_lastPlaceName'],
                                                                    input_data['tPlayer_3_lastPlaceName'],
                                                                    input_data['tPlayer_4_lastPlaceName'],
                                                                    input_data['tPlayer_5_lastPlaceName']):
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

    @staticmethod
    def _correct_last_place(input_data):
        def _correct_place(is_alive, map_name, last_place, position):
            if is_alive:
                if last_place in PLACE_DIST_MATRIX[map_name]:
                    return last_place
                else:
                    area_id = find_closest_area(map_name, position)['areaId']
                    return NAV[map_name][area_id]["areaName"]
            else:
                return None

        for team in ['ct', 't']:
            for i in range(1, 6):
                data = [
                    _correct_place(isAlive, mapName, lastPlace, [x, y, z])
                    for isAlive, mapName, lastPlace, x, y, z in
                    zip(input_data[f'{team}Player_{i}_isAlive'],
                        input_data[f'mapName'],
                        input_data[f'{team}Player_{i}_lastPlaceName'],
                        input_data[f'{team}Player_{i}_x'],
                        input_data[f'{team}Player_{i}_y'],
                        input_data[f'{team}Player_{i}_z'])
                ]
                part = pd.Series(data, name=f'{team}Player_{i}_lastPlaceName')
                input_data[f'{team}Player_{i}_lastPlaceName'] = part
        return input_data

    @staticmethod
    def _get_distance_to_bombsites(input_data):
        def _get_bombsite_dist(map_name, position):
            if position is not None:
                dist_to_a = PLACE_DIST_MATRIX[map_name][position]['BombsiteA']['geodesic']['median_dist']
                dist_to_b = PLACE_DIST_MATRIX[map_name][position]['BombsiteB']['geodesic']['median_dist']
            else:
                dist_to_a = None
                dist_to_b = None
            return dist_to_a, dist_to_b

        all_parts = []
        for team in ['ct', 't']:
            for i in range(1, 6):
                data = [
                    _get_bombsite_dist(mapName, lastPlace)
                    for mapName, lastPlace in
                    zip(
                        input_data[f'mapName'],
                        input_data[f'{team}Player_{i}_lastPlaceName']
                    )
                ]
                part = pd.DataFrame.from_records(data,
                                                 columns=[f'{team}Player_{i}_distToA', f'{team}Player_{i}_distToB'])
                all_parts.append(part)

        dist_df = pd.concat(all_parts, axis=1)

        ct_dist_to_a = ["ctPlayer_1_distToA", "ctPlayer_2_distToA", "ctPlayer_3_distToA", "ctPlayer_4_distToA",
                        "ctPlayer_5_distToA"]
        ct_dist_to_b = ["ctPlayer_1_distToB", "ctPlayer_2_distToB", "ctPlayer_3_distToB", "ctPlayer_4_distToB",
                        "ctPlayer_5_distToB"]
        t_dist_to_a = ["tPlayer_1_distToA", "tPlayer_2_distToA", "tPlayer_3_distToA", "tPlayer_4_distToA",
                       "tPlayer_5_distToA"]
        t_dist_to_b = ["tPlayer_1_distToB", "tPlayer_2_distToB", "tPlayer_3_distToB", "tPlayer_4_distToB",
                       "tPlayer_5_distToB"]
        dist_df['ctMinDistToA'] = dist_df[ct_dist_to_a].min(axis=1)
        dist_df['ctMinDistToB'] = dist_df[ct_dist_to_a].min(axis=1)
        dist_df['tMinDistToA'] = dist_df[t_dist_to_a].min(axis=1)
        dist_df['tMinDistToB'] = dist_df[t_dist_to_b].min(axis=1)
        dist_df['ctMeanDistToA'] = dist_df[ct_dist_to_a].mean(axis=1)
        dist_df['ctMeanDistToB'] = dist_df[ct_dist_to_b].mean(axis=1)
        dist_df['tMeanDistToA'] = dist_df[t_dist_to_a].mean(axis=1)
        dist_df['tMeanDistToB'] = dist_df[t_dist_to_b].mean(axis=1)
        res_df = dist_df[
            ['ctMinDistToA', 'ctMinDistToB', 'tMinDistToA', 'tMinDistToB', 'ctMeanDistToA', 'ctMeanDistToB',
             'tMeanDistToA', 'tMeanDistToB']].copy()
        res_df = res_df.fillna(9000)
        res_df = res_df.replace(np.inf, 9000)
        return res_df

    @staticmethod
    def _get_distance_players(input_data):
        def _get_dist(map_name, spots):
            new_positions = list(filter(lambda x: x is not None, spots))
            if len(new_positions) <= 1:
                return 9000, 9000
            else:
                dist = [
                    PLACE_DIST_MATRIX[map_name][position_x][position_y]['geodesic']['median_dist']
                    for i, position_x in enumerate(new_positions)
                    for j, position_y in enumerate(new_positions)
                    if i != j
                ]
                dist_arr = np.array(dist)
                dist_arr = dist_arr[np.isfinite(dist_arr)]
                if dist_arr.size == 0:
                    return 9000, 9000
                return np.mean(dist_arr), np.min(dist_arr)

        dist_parts = []
        for team in ['ct', 't']:
            data = [
                _get_dist(mapName, [spot1, spot2, spot3, spot4, spot5])
                for mapName, spot1, spot2, spot3, spot4, spot5, in zip(
                    input_data['mapName'],
                    input_data[f'{team}Player_1_lastPlaceName'],
                    input_data[f'{team}Player_2_lastPlaceName'],
                    input_data[f'{team}Player_3_lastPlaceName'],
                    input_data[f'{team}Player_4_lastPlaceName'],
                    input_data[f'{team}Player_5_lastPlaceName'])
            ]
            part = pd.DataFrame(data, columns=[f'{team}_meanDist', f'{team}_minDist'])
            dist_parts.append(part)
        return pd.concat(dist_parts, axis='columns')

    @staticmethod
    def _get_grouped_players(input_data):
        def __get_grouped_players(spots):
            counter = {}
            for spot in spots:
                if spot is not None:
                    counter[spot] = counter.get(spot, 0) + 1
            maximum = 0 if len(counter.values()) == 0 else max(counter.values())
            return maximum

        groupped_parts = []
        for team in ['ct', 't']:
            data = [__get_grouped_players(spots=[spot1, spot2, spot3, spot4, spot5])
                    for spot1, spot2, spot3, spot4, spot5 in zip(
                    input_data[f'{team}Player_1_lastPlaceName'],
                    input_data[f'{team}Player_2_lastPlaceName'],
                    input_data[f'{team}Player_3_lastPlaceName'],
                    input_data[f'{team}Player_4_lastPlaceName'],
                    input_data[f'{team}Player_5_lastPlaceName'])
                    ]
            part = pd.DataFrame(data, columns=[f'{team}_grouppedPlayers'])
            groupped_parts.append(part)
        return pd.concat(groupped_parts, axis='columns')

    @staticmethod
    def _get_spotters(input_data):
        def get_spotted_players(*args):
            spotted = set()
            for arg in args:
                spotted.update(json.loads(arg))
            return len(spotted)

        spotters_parts = []
        for team in ['ct', 't']:
            data = [
                get_spotted_players(spot1, spot2, spot3, spot4, spot5)
                for spot1, spot2, spot3, spot4, spot5 in
                zip(input_data[f'{team}Player_1_spotters'],
                    input_data[f'{team}Player_2_spotters'],
                    input_data[f'{team}Player_3_spotters'],
                    input_data[f'{team}Player_4_spotters'],
                    input_data[f'{team}Player_5_spotters'])
            ]
            part = pd.DataFrame(data, columns=[f'{team}_spottedPlayers'])
            spotters_parts.append(part)
        return pd.concat(spotters_parts, axis=1)

    @staticmethod
    def _convert_weapons(input_data, col):
        pistols = {'Glock-18', 'USP-S', 'P2000', 'P250', 'Dual Berettas'}
        enhanced_pistols = {'CZ75 Auto', 'Five-SeveN', 'Tec-9', 'R8 Revolver'}
        deagle = 'Desert Eagle'
        shotguns = {'MAG-7', 'XM1014', 'Nova', 'Sawed-Off'}
        machine_guns = {'M249', 'Negev'}
        smgs = {'MP9', 'MP7', 'MP5-SD', 'MAC-10', 'UMP-45', 'PP-Bizon', 'P90'}
        weaker_rifles = {'Galil AR', 'SSG 08', 'FAMAS'}
        lunet_rifles = {'SG 553', 'AUG'}
        sniper_rifle = {'G3SG1', 'SCAR-20', 'AWP'}
        assault_rifle = {'M4A1', 'M4A4', 'AK-47'}
        others = {'Zeus x27', 'Knife', 'C4', 'Molotov', 'Incendiary Grenade',
                  'Smoke Grenade', 'Flashbang', 'Decoy Grenade', 'HE Grenade'}
        others.update(shotguns, machine_guns)
        weapon_dict = {}
        for team in ['ct', 't']:
            weapon_dict[f"{team}_{col}_Pistol"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_EnhancedPistols"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_Deagle"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_SMG"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_WeakAssaultRifle"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_LunetRifle"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_SniperRifle"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_AssaultRifle"] = np.zeros(len(input_data.index))
            for i in range(1, 6):
                weapon_dict[f"{team}_{col}_Pistol"] += input_data[f"{team}Player_{i}_{col}"].isin(pistols).astype(
                    int)
                weapon_dict[f"{team}_{col}_EnhancedPistols"] = input_data[f"{team}Player_{i}_{col}"].isin(
                    enhanced_pistols).astype(int)
                weapon_dict[f"{team}_{col}_Deagle"] += (input_data[f"{team}Player_{i}_{col}"] == deagle).astype(
                    int)
                weapon_dict[f"{team}_{col}_SMG"] += input_data[f"{team}Player_{i}_{col}"].isin(smgs).astype(int)
                weapon_dict[f"{team}_{col}_WeakAssaultRifle"] += input_data[f"{team}Player_{i}_{col}"].isin(
                    weaker_rifles).astype(int)
                weapon_dict[f"{team}_{col}_LunetRifle"] += input_data[f"{team}Player_{i}_{col}"].isin(
                    lunet_rifles).astype(int)
                weapon_dict[f"{team}_{col}_SniperRifle"] += input_data[f"{team}Player_{i}_{col}"].isin(
                    sniper_rifle).astype(int)
                weapon_dict[f"{team}_{col}_AssaultRifle"] += input_data[f"{team}Player_{i}_{col}"].isin(
                    assault_rifle).astype(int)
        return pd.DataFrame.from_dict(weapon_dict)

    @staticmethod
    def _convert_active_weapons(input_data, col):
        pistols = {'Glock-18', 'USP-S', 'P2000', 'P250', 'Dual Berettas'}
        enhanced_pistols = {'CZ75 Auto', 'Five-SeveN', 'Tec-9', 'R8 Revolver'}
        deagle = 'Desert Eagle'
        shotguns = {'MAG-7', 'XM1014', 'Nova', 'Sawed-Off'}
        machine_guns = {'M249', 'Negev'}
        smgs = {'MP9', 'MP7', 'MP5-SD', 'MAC-10', 'UMP-45', 'PP-Bizon', 'P90'}
        weaker_rifles = {'Galil AR', 'SSG 08', 'FAMAS'}
        lunet_rifles = {'SG 553', 'AUG'}
        sniper_rifle = {'G3SG1', 'SCAR-20', 'AWP'}
        assault_rifle = {'M4A1', 'M4A4', 'AK-47'}
        others = {'Zeus x27', 'Knife', 'C4', 'Molotov', 'Incendiary Grenade',
                  'Smoke Grenade', 'Flashbang', 'Decoy Grenade', 'HE Grenade'}
        others.update(shotguns, machine_guns)
        weapon_dict = {}
        for team in ['ct', 't']:
            weapon_dict[f"{team}_{col}_Pistol"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_EnhancedPistols"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_Deagle"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_SMG"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_WeakAssaultRifle"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_LunetRifle"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_SniperRifle"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_AssaultRifle"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_Others"] = np.zeros(len(input_data.index))
            for i in range(1, 6):
                weapon_dict[f"{team}_{col}_Pistol"] += input_data[f"{team}Player_{i}_{col}"].isin(pistols).astype(int)
                weapon_dict[f"{team}_{col}_EnhancedPistols"] = input_data[f"{team}Player_{i}_{col}"].isin(
                    enhanced_pistols).astype(int)
                weapon_dict[f"{team}_{col}_Deagle"] += (input_data[f"{team}Player_{i}_{col}"] == deagle).astype(int)
                weapon_dict[f"{team}_{col}_SMG"] += input_data[f"{team}Player_{i}_{col}"].isin(smgs).astype(int)
                weapon_dict[f"{team}_{col}_WeakAssaultRifle"] += input_data[f"{team}Player_{i}_{col}"].isin(
                    weaker_rifles).astype(int)
                weapon_dict[f"{team}_{col}_LunetRifle"] += input_data[f"{team}Player_{i}_{col}"].isin(
                    lunet_rifles).astype(int)
                weapon_dict[f"{team}_{col}_SniperRifle"] += input_data[f"{team}Player_{i}_{col}"].isin(
                    sniper_rifle).astype(int)
                weapon_dict[f"{team}_{col}_AssaultRifle"] += input_data[f"{team}Player_{i}_{col}"].isin(
                    assault_rifle).astype(int)
                weapon_dict[f"{team}_{col}_Others"] += input_data[f"{team}Player_{i}_{col}"].isin(others).astype(int)
        return pd.DataFrame.from_dict(weapon_dict)

    @staticmethod
    def _convert_sec_weapons(input_data, col):
        pistols = {'Glock-18', 'USP-S', 'P2000', 'P250', 'Dual Berettas'}
        enhanced_pistols = {'CZ75 Auto', 'Five-SeveN', 'Tec-9', 'R8 Revolver'}
        deagle = 'Desert Eagle'
        weapon_dict = {}
        for team in ['ct', 't']:
            weapon_dict[f"{team}_{col}_Pistol"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_EnhancedPistols"] = np.zeros(len(input_data.index))
            weapon_dict[f"{team}_{col}_Deagle"] = np.zeros(len(input_data.index))
            for i in range(1, 6):
                weapon_dict[f"{team}_{col}_Pistol"] += input_data[f"{team}Player_{i}_{col}"].isin(pistols).astype(
                    int)
                weapon_dict[f"{team}_{col}_EnhancedPistols"] = input_data[f"{team}Player_{i}_{col}"].isin(
                    enhanced_pistols).astype(int)
                weapon_dict[f"{team}_{col}_Deagle"] += (input_data[f"{team}Player_{i}_{col}"] == deagle).astype(
                    int)
        return pd.DataFrame.from_dict(weapon_dict)

    def get_gained_probability(self, kills_pred, damages):
        gain_prob = [None] * len(kills_pred.index)
        for id, round_num, tick, att_id, att_side, victim_id, victim_side, victim_blinded, \
            flash_id, flash_side, is_trade, trade_id, pred, next_pred in zip(kills_pred.index,
                                                                             kills_pred["roundNum"],
                                                                             kills_pred["tick_parsed"],
                                                                             kills_pred["attackerID"],
                                                                             kills_pred["attackerSide"],
                                                                             kills_pred["victimID"],
                                                                             kills_pred["victimSide"],
                                                                             kills_pred["victimBlinded"],
                                                                             kills_pred["flashThrowerID"],
                                                                             kills_pred["flashThrowerSide"],
                                                                             kills_pred["isTrade"],
                                                                             kills_pred["playerTradedID"],
                                                                             kills_pred["probaCtWin"],
                                                                             kills_pred["nextProbaCtWin"]):
            res = {
                "roundNum": round_num,
                "tick": tick,
                "attackerID": att_id,
                "attackerSide": att_side,
                "victimID": victim_id,
                "victimSide": victim_side,
                "prediction": pred,
                "next_prediction": next_pred
            }
            if victim_blinded == 1:
                res["flashThrowerID"] = flash_id
                res["flashSide"] = flash_side
            if is_trade == 1:
                res["playerTradedID"] = trade_id
            damages_local = damages.loc[(damages.roundNum == round_num) & (damages.victimID == victim_id)]
            damages_dict = dict()
            for attacker_id, damage_dealt, is_friendly, damage_tick in zip(damages_local["attackerID"],
                                                                           damages_local["hpDamageTaken"],
                                                                           damages_local["isFriendlyFire"],
                                                                           damages_local["tick_parsed"]):
                if attacker_id in damages_dict:
                    damages_dict[attacker_id]["hp_taken"] += damage_dealt
                else:
                    damages_dict[attacker_id] = {"hp_taken": damage_dealt, "is_friendly": is_friendly}
                if damage_tick == tick:
                    res["dealt_damage"] = res.get("dealt_damage", 0) + damage_dealt

            res["damages"] = damages_dict
            gain_prob[int(id)] = res
        return gain_prob

    def _clean_gained_prob(self, gained_prob, initial_data, transformed_data):
        new_gained_prob = [None] * len(gained_prob)
        map_col = {
            "isAlive": "AlivePlayers",
            "equipmentValue": "EqVal"
        }

        for i, (last_el, el) in enumerate(zip(gained_prob[:len(gained_prob) - 1], gained_prob[1:])):
            if last_el["tick"] == el["tick"]:
                chosen_info = initial_data.loc[initial_data.tick == last_el["tick"]].copy()
                chosen_state = transformed_data.iloc[chosen_info.index].copy()
                team = last_el['victimSide'].lower()
                for j in range(1, 6):
                    if chosen_info[f"{team}Player_{j}_ID"].values == last_el["victimID"]:
                        id = j
                        break
                for col in ["DecoyGrenade", "Flashbang", "HEGrenade", "SmokeGrenade",
                            "armor", "hp", "isAlive", "equipmentValue"]:
                    team_col = "_" + col if col not in map_col else map_col[col]
                    chosen_state[f"{team}{team_col}"] -= chosen_info[f"{team}Player_{id}_{col}"]
                prediction = self.model.predict_proba(chosen_state)[0, 1]
                last_el["next_prediction"] = el["prediction"] = prediction
            new_gained_prob[i] = last_el
        new_gained_prob[len(gained_prob) - 1] = gained_prob[-1]
        return new_gained_prob

    def _scale_rewards(self, action):
        if ("flashThrowerID" not in action and "playerTradedID" not in action) or (
                "playerTradedID" not in action and action["flashSide"] == action["victimSide"]):
            kill_reward = self.KILL_GAIN + self.KILL_GAIN / (self.KILL_GAIN + self.DAMAGE_GAIN) * (
                    self.FLASH_GAIN + self.TRADE_GAIN)
            damage_reward = self.DAMAGE_GAIN + self.DAMAGE_GAIN / (self.KILL_GAIN + self.DAMAGE_GAIN) * (
                    self.FLASH_GAIN + self.TRADE_GAIN)
            trade_reward = 0
            flash_reward = 0
        elif "flashThrowerID" not in action:
            kill_reward = self.KILL_GAIN + self.KILL_GAIN / (
                    self.KILL_GAIN + self.DAMAGE_GAIN + self.TRADE_GAIN) * self.FLASH_GAIN
            damage_reward = self.DAMAGE_GAIN + self.DAMAGE_GAIN / (
                    self.KILL_GAIN + self.DAMAGE_GAIN + self.TRADE_GAIN) * self.FLASH_GAIN
            trade_reward = self.TRADE_GAIN + self.TRADE_GAIN / (
                    self.KILL_GAIN + self.DAMAGE_GAIN + self.TRADE_GAIN) * self.FLASH_GAIN
            flash_reward = 0
        else:
            kill_reward = self.KILL_GAIN + self.KILL_GAIN / (
                    self.KILL_GAIN + self.DAMAGE_GAIN + self.FLASH_GAIN) * self.TRADE_GAIN
            damage_reward = self.DAMAGE_GAIN + self.DAMAGE_GAIN / (
                    self.KILL_GAIN + self.DAMAGE_GAIN + self.FLASH_GAIN) * self.TRADE_GAIN
            flash_reward = self.FLASH_GAIN + self.FLASH_GAIN / (
                    self.KILL_GAIN + self.DAMAGE_GAIN + self.FLASH_GAIN) * self.TRADE_GAIN
            trade_reward = 0
        return kill_reward, damage_reward, flash_reward, trade_reward

    def _scale_penalties(self, action):
        damages = action["damages"]
        sum_damages = sum(value["hp_taken"] for value in damages.values() if value["is_friendly"] == 1)
        damage_penalty = sum_damages / 100
        kill_penalty = 1 - damage_penalty
        flash_penalty = 0
        if "flashThrowerID" in action and action["flashSide"] == action["victimSide"]:
            flash_penalty = self.FLASH_PENALTY
            damage_penalty = self.KILL_DAMAGE_PENALTY * damage_penalty
            kill_penalty = self.KILL_DAMAGE_PENALTY * kill_penalty
        return kill_penalty, damage_penalty, flash_penalty

    def kill_reward(self, action, gained_value, kill_reward):
        reward = {"roundNum": action["roundNum"], "tick": action["tick"], "playerID": action["attackerID"],
                  "gainValue": gained_value * kill_reward, "type": "kill"}
        return reward

    def damage_reward(self, action, gained_value, damage_reward):
        damages = action["damages"]
        damage_rewards = []
        sum_damages = sum(value["hp_taken"] for value in damages.values() if value["is_friendly"] == 0)
        for player, value in damages.items():
            if value["is_friendly"] == 0:
                reward = {"roundNum": action["roundNum"], "tick": action["tick"], "playerID": player,
                          "gainValue": gained_value * (value["hp_taken"] / sum_damages) * damage_reward,
                          "type": "damage assist"}
                damage_rewards.append(reward)
        return damage_rewards

    def flash_reward(self, action, gained_value, flash_reward):
        reward = {"roundNum": action["roundNum"], "tick": action["tick"], "playerID": action["flashThrowerID"],
                  "gainValue": gained_value * flash_reward, "type": "flash assist"}
        return reward

    def trade_reward(self, action, gained_value, trade_reward):
        reward = {"roundNum": action["roundNum"], "tick": action["tick"], "playerID": action["playerTradedID"],
                  "gainValue": gained_value * trade_reward, "type": "traded"}
        return reward

    def kill_penalty(self, action, lost_val, kill_penalty):
        penalty = {"roundNum": action["roundNum"], "tick": action["tick"], "playerID": action["victimID"],
                   "gainValue": lost_val * kill_penalty, "type": "death"}
        return penalty

    def kill_teammate(self, action, lost_val, kill_penalty):
        penalty = {"roundNum": action["roundNum"], "tick": action["tick"], "playerID": action["attackerID"],
                   "gainValue": lost_val * kill_penalty, "type": "killed teammate"}
        return penalty

    def flash_penalty(self, action, lost_val, flash_penalty):
        penalty = {"roundNum": action["roundNum"], "tick": action["tick"], "playerID": action["flashThrowerID"],
                   "gainValue": lost_val * flash_penalty, "type": "flash assist teammate"}
        return penalty

    def damage_penalty(self, action, lost_val, damage_penalty):
        damages = action["damages"]
        damage_rewards = []
        sum_damages = sum(value["hp_taken"] for value in damages.values() if value["is_friendly"] == 1)
        for player, value in damages.items():
            if value["is_friendly"] == 1:
                reward = {"roundNum": action["roundNum"], "tick": action["tick"], "playerID": player,
                          "gainValue": lost_val * (value["hp_taken"] / sum_damages) * damage_penalty,
                          "type": "damage assist teammate"}
                damage_rewards.append(reward)
        return damage_rewards

    def share_proba(self, actions):
        action_desc = []
        for action in actions:
            gained_value = abs(action["next_prediction"] - action["prediction"])
            if action["attackerSide"] != action["victimSide"]:
                kill_gain_coef, damage_gain_coef, flash_gain_coef, trade_gain_coef = self._scale_rewards(action)
                action_desc.append(self.kill_reward(action, gained_value, kill_gain_coef))
                action_desc.extend(self.damage_reward(action, gained_value, damage_gain_coef))
                if flash_gain_coef != 0:
                    action_desc.append(self.flash_reward(action, gained_value, flash_gain_coef))
                if trade_gain_coef != 0:
                    action_desc.append(self.trade_reward(action, gained_value, trade_gain_coef))
                kill_pen_coef, damage_pen_coef, flash_pen_coef = self._scale_penalties(action)
                action_desc.append(self.kill_penalty(action, -gained_value, kill_pen_coef))
                if damage_pen_coef != 0:
                    action_desc.extend(self.damage_penalty(action, -gained_value, damage_pen_coef))
                if flash_pen_coef != 0:
                    action_desc.append(self.flash_penalty(action, -gained_value, flash_pen_coef))
            else:
                action_desc.extend(self.damage_reward(action, gained_value, 1))
                kill_pen_coef, damage_pen_coef, flash_pen_coef = self._scale_penalties(action)
                action_desc.append(self.kill_penalty(action, -gained_value, kill_pen_coef / 2))
                action_desc.append(self.kill_teammate(action, -gained_value, kill_pen_coef / 2))
                if damage_pen_coef != 0:
                    action_desc.extend(self.damage_penalty(action, -gained_value, damage_pen_coef))
                if flash_pen_coef != 0:
                    action_desc.append(self.flash_penalty(action, -gained_value, flash_pen_coef))
        return action_desc

    def get_ratings(self, initial_data, transformed_data, prediction, damages, kills):
        kills_filtered = kills.loc[kills.weapon != "C4"].copy()
        kills_pred = kills_filtered.merge(prediction, left_on=["tick_parsed"], right_on=["tick"])
        gained_prob = self.get_gained_probability(kills_pred, damages)
        cleaned_proba = self._clean_gained_prob(gained_prob, initial_data, transformed_data)
        rating = self.share_proba(cleaned_proba)
        return pd.DataFrame(rating)

    def get_accumulated_rating(self, rating, roundNum):
        summed_rating = rating.groupby("playerID").agg({"gainValue": "sum"}).reset_index()
        summed_rating["rating"] = summed_rating["gainValue"]/roundNum
        return summed_rating

    def find_important_moments(self, probablities, rounds):
        connected = pd.concat([probablities, rounds], axis=1)
        connected["difference"] = np.abs(connected["nextProbaCtWin"] - connected["probaCtWin"])
        connected["important_moments"] = connected.difference >= self.DIFF_IMPORTANT_MOMENTS
        indexes_true = np.argwhere(connected.important_moments.to_numpy()).flatten()
        indexes_round = connected.groupby("roundNum").indices
        for round_ in indexes_round:
            indexes_round[round_] = np.min(indexes_round[round_]), np.max(indexes_round[round_])

        for index in indexes_true:
            max_index = index + self.FRAMES_IMPORTANT_MOMENTS
            min_index = index - self.FRAMES_IMPORTANT_MOMENTS
            round_ = connected.loc[index, "roundNum"].item()
            if index == indexes_round[round_][1]:
                connected.loc[index, "important_moments"] = False
            else:
                if min_index < indexes_round[round_][0]:
                    min_index = indexes_round[round_][0]
                if max_index > indexes_round[round_][1]:
                    max_index = indexes_round[round_][1]
                connected.loc[min_index:max_index, "important_moments"] = True
        return connected


if __name__ == '__main__':
    db_con = create_engine('mysql+mysqlconnector://root:password@localhost/CSGOAnalysis?allow_local_infile=1')

    with db_con.connect() as connection:
        states = pd.read_sql("""SELECT g.mapName, f.*, 
                    CASE WHEN f.tick >= r.endTickCorrect THEN 1 
                         ELSE 0 END as gameEnded,
                        r.winningSide
                    FROM frame f
                    INNER JOIN round r ON f.matchID=r.matchID AND f.roundNum=r.roundNum
                    INNER JOIN game g ON g.ID=f.matchID
                    WHERE g.ID=1
            """, connection)

    with db_con.connect() as connection:
        score = pd.read_sql("""SELECT roundNum, ctScore, tScore
                                 FROM round
                                 WHERE matchID=1""", connection)

    with db_con.connect() as connection:
        damages = pd.read_sql("""SELECT *
                                FROM damage d
                                WHERE d.matchID=1
            """, connection)

    with db_con.connect() as connection:
        kills = pd.read_sql("""SELECT *
                                FROM elimination k
                                WHERE k.matchID=1
            """, connection)

    with open("model.pkl", "rb") as handle:
        model = pickle.load(handle)

    am = AnalyticModule(model)
    kills_prep = kills.loc[~kills.attackerID.isna(), ["roundNum", "attackerID"]].reset_index()
    transformed_data = am.transform_data(states, score, kills_prep)
    pred = am.get_predictions(states, transformed_data)
    scores = am.get_ratings(initial_data=states, transformed_data=transformed_data, prediction=pred, damages=damages,
                            kills=kills)
    acc_rating = am.get_accumulated_rating(scores, states.roundNum.max())
    moments = am.find_important_moments(pred, states.roundNum)
    moments = moments.loc[moments.roundNum == 8]
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(moments["tick"], moments["probaCtWin"], 'b-')
    ax = plt.gca()
    ax.fill_between(moments["tick"], moments["probaCtWin"], 1, interpolate=True, color='orange')
    ax.fill_between(moments["tick"], 0, moments["probaCtWin"], interpolate=True, color='blue')
    plt.xlim((moments["tick"].min(), moments["tick"].max()))
    plt.ylim((0, 1))
    plt.savefig("plot.png")
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(moments["tick"], moments["probaCtWin"], 'b-')
    ax = plt.gca()
    ax.fill_between(moments["tick"], moments["probaCtWin"], 1, interpolate=True, color='orange')
    ax.fill_between(moments["tick"], 0, moments["probaCtWin"], interpolate=True, color='blue')
    ax.fill_between(moments["tick"], 0, 1, where=~moments.important_moments, facecolor='gray', alpha=0.9)
    plt.xlim((moments["tick"].min(), moments["tick"].max()))
    plt.ylim((0, 1))
    plt.savefig("new_plot.png")
    plt.show()
    print(acc_rating)
    print(moments.important_moments.value_counts())
