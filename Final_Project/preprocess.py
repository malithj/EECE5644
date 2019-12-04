import pandas as pd
import numpy as np
import re
from sklearn import preprocessing


def get_preprocessed_data():
    """
    Preprocesses data read from the dataset. The steps involved are
        1. Removing rows with Nan values
        2. Converting player height from feet and inches to inches
        3. Removing user formatted data from "Windspeed" incluing "gusts up to" and range measurements. Also removed 'E' measurements
        4. Converting datetime values to POSIX timestamps
        5. Converting player birthdate to player age (No change in data frame Column name)
        6. Converting all string data to categorical values using sklearn
    :return: Preprocessed Pandas data frame
    """
    # read pandas data file
    train_file_path = '../../nfl-big-data-bowl-2020/train.csv'
    df = pd.read_csv(train_file_path, header=0)

    # drop rows with NaN values
    df = df.dropna()

    # convert player height to inches
    r = re.compile(r"([0-9]+)-([0-9]+)")

    def get_inches(el):
        m = r.match(el)
        if m is None:
            return float('NaN')
        else:
            return int(m.group(1)) * 12 + float(m.group(2))

    player_height = df.loc[:, 'PlayerHeight'].apply(lambda x: get_inches(x))
    df = df.assign(PlayerHeight=player_height)

    # remove MPH and mph from wind speed. Remove 'gusts up to' and '-' from speed. Replace with average wind speed
    r_gusts = re.compile(r"([0-9]+) gusts up to ([0-9]+)")

    def remove_windspeed_literals(el):
        if type(el) is float:
            return el
        m = r_gusts.match(el)
        if m is None:
            return el
        else:
            return (float(m.group(1)) + float(m.group(2))) / 2

    r_dash = re.compile(r"([0-9]+)-([0-9]+)")

    def remove_windspeed_range(el):
        if type(el) is float:
            return el
        m = r_dash.match(el)
        if m is None:
            return el
        else:
            return (float(m.group(1)) + float(m.group(2))) / 2

    wind_speed = df['WindSpeed'].map(lambda x: str(x).rstrip(' MPh').rstrip(' MPH').rstrip(' mph'))
    wind_speed = wind_speed.apply(lambda x: remove_windspeed_literals(x))
    wind_speed = wind_speed.apply(lambda x: remove_windspeed_range(x))
    df = df.assign(WindSpeed=wind_speed)
    # remove the E character rows
    df = df[df['WindSpeed'] != 'E']

    # creating labelEncoder
    le = preprocessing.LabelEncoder()

    # converting string labels into numbers.
    team = le.fit_transform(df.loc[:, 'Team'])
    display_name = le.fit_transform(df.loc[:, 'DisplayName'])
    possession_team = le.fit_transform(df.loc[:, 'PossessionTeam'])
    field_position = le.fit_transform(df.loc[:, 'FieldPosition'])
    offense_formation = le.fit_transform(df.loc[:, 'OffenseFormation'])
    offense_personnel = le.fit_transform(df.loc[:, 'OffensePersonnel'])
    defense_personnel = le.fit_transform(df.loc[:, 'DefensePersonnel'])
    play_direction = le.fit_transform(df.loc[:, 'PlayDirection'])
    player_college_name = le.fit_transform(df.loc[:, 'PlayerCollegeName'])
    position = le.fit_transform(df.loc[:, 'Position'])
    home_team_abbr = le.fit_transform(df.loc[:, 'HomeTeamAbbr'])
    visitor_team_abbr = le.fit_transform(df.loc[:, 'VisitorTeamAbbr'])
    stadium = le.fit_transform(df.loc[:, 'Stadium'])
    stadium_type = le.fit_transform(df.loc[:, 'StadiumType'])
    location = le.fit_transform(df.loc[:, 'Location'])
    turf = le.fit_transform(df.loc[:, 'Turf'])
    game_weather = le.fit_transform(df.loc[:, 'GameWeather'])
    wind_direction = le.fit_transform(df.loc[:, 'WindDirection'])

    # convert Timestamps to POSIX
    time_in_clock_format = pd.to_datetime(df['GameClock'], format="%H:%M:%S")
    game_clock = time_in_clock_format.dt.hour * 3600 + time_in_clock_format.dt.minute * 60 + time_in_clock_format.dt.second
    time_handoff = pd.to_datetime(df['TimeHandoff'], format="%Y-%m-%dT%H:%M:%S.%f").astype(np.int64)
    time_snap = pd.to_datetime(df['TimeSnap'], format="%Y-%m-%dT%H:%M:%S.%f").astype(np.int64)
    # convert player birth date to player age
    time_in_date_format = pd.to_datetime(df['PlayerBirthDate'], format="%m/%d/%Y")
    player_birth_date = (2019 - time_in_date_format.dt.year)

    # replace the numerical columns
    df = df.assign(Team=team, DisplayName=display_name, PossessionTeam=possession_team, FieldPosition=field_position,
                   OffenseFormation=offense_formation, OffensePersonnel=offense_personnel,
                   DefensePersonnel=defense_personnel,
                   PlayDirection=play_direction, PlayerCollegeName=player_college_name, Position=position,
                   HomeTeamAbbr=home_team_abbr, VisitorTeamAbbr=visitor_team_abbr, Stadium=stadium,
                   StadiumType=stadium_type,
                   Location=location, Turf=turf, GameWeather=game_weather, WindDirection=wind_direction)
    # replace timestamps
    df = df.assign(GameClock=game_clock, TimeHandoff=time_handoff, TimeSnap=time_snap,
                   PlayerBirthDate=player_birth_date)
    return df
