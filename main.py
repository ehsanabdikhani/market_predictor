from show import *
import datetime
import pandas as pd
import glob
import numpy as np
import tradingview_indicators as ta
import pandas_ta as pta
import talib


def check_movement(df, point, mv_value):

    entry_price = df.loc[point, 'Close']
    long_level = entry_price + mv_value
    short_level = entry_price - mv_value

    future_highs = df['High'].values[point + 1:]
    future_lows = df['Low'].values[point + 1:]

    long_hits = future_highs >= long_level
    short_hits = future_lows <= short_level

    if long_hits.any():
        long_idx = np.argmax(long_hits)

        if short_hits.any():
            short_idx = np.argmax(short_hits)

            if short_idx < long_idx:
                return 'Short'

        return 'Long'

    elif short_hits.any():
        return 'Short'

    return None


def find_potential_points_with_cooldown(
    df,
    overbought_limit,
    oversold_limit,
    overbought_cooldown_value=57,
    oversold_cooldown_value=43
):

    ob_points = []
    os_points = []

    waiting_for_cooldown_ob = False
    waiting_for_cooldown_os = False

    for i in range(1, len(df)):

        current_rsi = df['RSI'].iloc[i]
        prev_rsi = df['RSI'].iloc[i - 1]

        if (
            prev_rsi < overbought_limit
            and current_rsi > overbought_limit
            and not waiting_for_cooldown_ob
        ):
            ob_points.append(i)
            waiting_for_cooldown_ob = True

        if (
            prev_rsi > oversold_limit
            and current_rsi < oversold_limit
            and not waiting_for_cooldown_os
        ):
            os_points.append(i)
            waiting_for_cooldown_os = True

        if waiting_for_cooldown_ob and current_rsi < overbought_cooldown_value:
            waiting_for_cooldown_ob = False

        if waiting_for_cooldown_os and current_rsi > oversold_cooldown_value:
            waiting_for_cooldown_os = False

    return ob_points, os_points


def last_HTF_candle(df, p_time):

    temp = df[df['Date'] <= p_time]

    if not temp.empty and len(temp) >= 2:
        return (
            'Bullish'
            if temp['Close'].iloc[-2] > temp['Open'].iloc[-2]
            else 'Bearish'
        )

    return None


def last_HTF_value(df, p_time, column):

    temp = df[df['Date'] <= p_time]

    if not temp.empty and len(temp) >= 2:
        return temp[column].iloc[-2]

    return None


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

names = [
    'result\\EURUSD\\EURUSD',
    'result\\EURAUD\\EURAUD',
    'result\\EURCHF\\EURCHF',
    'result\\EURGBP\\EURGBP',
    'result\\EURJPY\\EURJPY',
    'result\\USDCAD\\USDCAD',
    'result\\USDCHF\\USDCHF',
    'result\\USDJPY\\USDJPY'
]

paths = [
    'merged_data\\EURUSD_M1.csv',
    'merged_data\\EURAUD_M1.csv',
    'merged_data\\EURCHF_M1.csv',
    'merged_data\\EURGBP_M1.csv',
    'merged_data\\EURJPY_M1.csv',
    'merged_data\\USDCAD_M1.csv',
    'merged_data\\USDCHF_M1.csv',
    'merged_data\\USDJPY_M1.csv'
]

paths_h1 = [
    'merged_data\\EURUSD_H1.csv',
    'merged_data\\EURAUD_H1.csv',
    'merged_data\\EURCHF_H1.csv',
    'merged_data\\EURGBP_H1.csv',
    'merged_data\\EURJPY_H1.csv',
    'merged_data\\USDCAD_H1.csv',
    'merged_data\\USDCHF_H1.csv',
    'merged_data\\USDJPY_H1.csv'
]


paths_h4 = [
    'merged_data\\EURUSD_H4.csv',
    'merged_data\\EURAUD_H4.csv',
    'merged_data\\EURCHF_H4.csv',
    'merged_data\\EURGBP_H4.csv',
    'merged_data\\EURJPY_H4.csv',
    'merged_data\\USDCAD_H4.csv',
    'merged_data\\USDCHF_H4.csv',
    'merged_data\\USDJPY_H4.csv'
]


movement_value = [
    0.0002,
    0.0003,
    0.0002,
    0.0001,
    0.02,
    0.0002,
    0.0003,
    0.03
]



for i in range(8):


    df = pd.read_csv(paths[i])
    df['Date'] = pd.to_datetime(df['Date'])

    df["RSI"] = ta.RSI(df["Close"], 100)


    df_h4 = pd.read_csv(paths_h4[i])
    df_h4['Date'] = pd.to_datetime(df_h4['Date'])


    df_h4['EMA200'] = (
        df_h4['Close']
        .ewm(span=200, adjust=False)
        .mean()
    )

    df_h4['EMA100'] = (
        df_h4['Close']
        .ewm(span=100, adjust=False)
        .mean()
    )

    df_h4['EMA50'] = (
        df_h4['Close']
        .ewm(span=50, adjust=False)
        .mean()
    )


    df_h4['EMA50_200'] = (
        (df_h4['EMA50'] / df_h4['EMA200'] - 1) * 100 
    )


    df_h4['EMA100_200'] = (
        (df_h4['EMA100'] / df_h4['EMA200'] - 1) * 100 
    )

    df_h4['EMA50_100'] = (
        (df_h4['EMA50'] / df_h4['EMA100'] - 1) * 100 
    )

    macd = ta.MACD(
        df_h4['Close'],
        fast_length=12,
        slow_length=26,
        signal_length=9
    )

    df_h4['MACD_Histogram'] = macd['histogram']

    df_h1 = pd.read_csv(paths_h1[i])
    df_h1['Date'] = pd.to_datetime(df_h1['Date'])

    macd = ta.MACD(
        df_h1['Close'],
        fast_length=12,
        slow_length=26,
        signal_length=9
    )

    df_h1['MACD_Histogram'] = macd['histogram']

    df_h1['EMA200'] = (
        df_h1['Close']
        .ewm(span=200, adjust=False)
        .mean()
    )

    df_h1['EMA100'] = (
        df_h1['Close']
        .ewm(span=100, adjust=False)
        .mean()
    )

    df_h1['EMA50'] = (
        df_h1['Close']
        .ewm(span=50, adjust=False)
        .mean()
    )


    df_h1['EMA50_200'] = (
        (df_h1['EMA50'] / df_h1['EMA200'] - 1) * 100
    )

    df_h1['EMA100_200'] = (
        (df_h1['EMA100'] / df_h1['EMA200'] - 1) * 100 
    )

    df_h1['EMA50_100'] = (
        (df_h1['EMA50'] / df_h1['EMA100'] - 1 ) * 100
    )



    df_h1['RSI14'] = ta.RSI(
        df_h1['Close'],
        14
    )


    df_h1['RSI14'] = df_h1['RSI14'] - 50


    df['Potential_type'] = None

    ob_points, os_points = find_potential_points_with_cooldown(
        df,
        58,
        42,
        56,
        44
    )

    df.loc[ob_points, 'Potential_type'] = 'overbought'
    df.loc[os_points, 'Potential_type'] = 'oversold'

    condition = df['Potential_type'].notna()


    df.loc[condition, 'Last_4h'] = (
        df.loc[condition, 'Date'].apply(
            lambda x: last_HTF_candle(df_h4, x)
        )
    )

    df.loc[condition, 'H4_EMA50_100'] = (
        df.loc[condition, 'Date'].apply(
            lambda x: last_HTF_value(
                df_h4,
                x,
                'EMA50_100'
            )
        )
    )


    df.loc[condition, 'H4_EMA50_200'] = (
        df.loc[condition, 'Date'].apply(
            lambda x: last_HTF_value(
                df_h4,
                x,
                'EMA50_200'
            )
        )
    )


    # H4 EMA200 distance
    df.loc[condition, 'H4_EMA100_200'] = (
        df.loc[condition, 'Date'].apply(
            lambda x: last_HTF_value(
                df_h4,
                x,
                'EMA100_200'
            )
        )
    )


    
    df.loc[condition, 'Last_1h'] = (
        df.loc[condition, 'Date'].apply(
            lambda x: last_HTF_candle(df_h1, x)
        )
    )


    df.loc[condition, 'H1_EMA50_100'] = (
        df.loc[condition, 'Date'].apply(
            lambda x: last_HTF_value(
                df_h1,
                x,
                'EMA50_100'
            )
        )
    )


    df.loc[condition, 'H1_EMA50_200'] = (
        df.loc[condition, 'Date'].apply(
            lambda x: last_HTF_value(
                df_h1,
                x,
                'EMA50_200'
            )
        )
    )


    df.loc[condition, 'H1_EMA100_200'] = (
        df.loc[condition, 'Date'].apply(
            lambda x: last_HTF_value(
                df_h1,
                x,
                'EMA100_200'
            )
        )
    )


    df.loc[condition, 'H1_MACD'] = (
        df.loc[condition, 'Date'].apply(
            lambda x: last_HTF_value(
                df_h1,
                x,
                'MACD_Histogram'
            )
        )
    )


    df.loc[condition, 'H4_MACD'] = (
        df.loc[condition, 'Date'].apply(
            lambda x: last_HTF_value(
                df_h4,
                x,
                'MACD_Histogram'
            )
        )
    )



    # H1 RSI14
    df.loc[condition, 'H1_RSI14'] = (
        df.loc[condition, 'Date'].apply(
            lambda x: last_HTF_value(
                df_h1,
                x,
                'RSI14'
            )
        )
    )


    for j in range(1, 21):

        df.loc[condition, 'Movement_Signal'] = (
            df.loc[condition].apply(
                lambda row: check_movement(
                    df,
                    row.name,
                    j * movement_value[i]
                ),
                axis=1
            )
        )

        output_columns = [

            'Date',

            # M1
            'Potential_type',

            # H1
            'H1_EMA50_100',
            'H1_EMA50_200',
            'H1_EMA100_200',
            'H1_RSI14',
            'H1_MACD',
            'Last_1h',


            # H4
            'Last_4h',
            'H4_EMA50_100',
            'H4_EMA50_200',
            'H4_EMA100_200',
            'H4_MACD',
                        
            # Target
            'Movement_Signal'
        ]

        df.loc[condition].dropna()[
            output_columns
        ].to_csv(
            names[i] + "_" + str(j) + ".csv",
            index=False
        )

        print(
            "saved to "
            + names[i]
            + "_"
            + str(j)
            + ".csv"
        )