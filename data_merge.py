
paths_1 = [
    '2026_data\\EURUSD_M1.csv',
    '2026_data\\EURAUD_M1.csv',
    '2026_data\\EURCHF_M1.csv',
    '2026_data\\EURGBP_M1.csv',
    '2026_data\\EURJPY_M1.csv',
    '2026_data\\USDCAD_M1.csv',
    '2026_data\\USDCHF_M1.csv',
    '2026_data\\USDJPY_M1.csv'
]

paths_h1_1 = [
    '2026_data\\EURUSD_H1.csv',
    '2026_data\\EURAUD_H1.csv',
    '2026_data\\EURCHF_H1.csv',
    '2026_data\\EURGBP_H1.csv',
    '2026_data\\EURJPY_H1.csv',
    '2026_data\\USDCAD_H1.csv',
    '2026_data\\USDCHF_H1.csv',
    '2026_data\\USDJPY_H1.csv'
]


paths_h4_1 = [
    '2026_data\\EURUSD_H4.csv',
    '2026_data\\EURAUD_H4.csv',
    '2026_data\\EURCHF_H4.csv',
    '2026_data\\EURGBP_H4.csv',
    '2026_data\\EURJPY_H4.csv',
    '2026_data\\USDCAD_H4.csv',
    '2026_data\\USDCHF_H4.csv',
    '2026_data\\USDJPY_H4.csv'
]



paths_2 = [
    'data\\EURUSD_M1.csv',
    'data\\EURAUD_M1.csv',
    'data\\EURCHF_M1.csv',
    'data\\EURGBP_M1.csv',
    'data\\EURJPY_M1.csv',
    'data\\USDCAD_M1.csv',
    'data\\USDCHF_M1.csv',
    'data\\USDJPY_M1.csv'
]

paths_h1_2 = [
    'data\\EURUSD_H1.csv',
    'data\\EURAUD_H1.csv',
    'data\\EURCHF_H1.csv',
    'data\\EURGBP_H1.csv',
    'data\\EURJPY_H1.csv',
    'data\\USDCAD_H1.csv',
    'data\\USDCHF_H1.csv',
    'data\\USDJPY_H1.csv'
]


paths_h4_2 = [
    'data\\EURUSD_H4.csv',
    'data\\EURAUD_H4.csv',
    'data\\EURCHF_H4.csv',
    'data\\EURGBP_H4.csv',
    'data\\EURJPY_H4.csv',
    'data\\USDCAD_H4.csv',
    'data\\USDCHF_H4.csv',
    'data\\USDJPY_H4.csv'
]



paths_3 = [
    'merged_data\\EURUSD_M1.csv',
    'merged_data\\EURAUD_M1.csv',
    'merged_data\\EURCHF_M1.csv',
    'merged_data\\EURGBP_M1.csv',
    'merged_data\\EURJPY_M1.csv',
    'merged_data\\USDCAD_M1.csv',
    'merged_data\\USDCHF_M1.csv',
    'merged_data\\USDJPY_M1.csv'
]

paths_h1_3 = [
    'merged_data\\EURUSD_H1.csv',
    'merged_data\\EURAUD_H1.csv',
    'merged_data\\EURCHF_H1.csv',
    'merged_data\\EURGBP_H1.csv',
    'merged_data\\EURJPY_H1.csv',
    'merged_data\\USDCAD_H1.csv',
    'merged_data\\USDCHF_H1.csv',
    'merged_data\\USDJPY_H1.csv'
]


paths_h4_3 = [
    'merged_data\\EURUSD_H4.csv',
    'merged_data\\EURAUD_H4.csv',
    'merged_data\\EURCHF_H4.csv',
    'merged_data\\EURGBP_H4.csv',
    'merged_data\\EURJPY_H4.csv',
    'merged_data\\USDCAD_H4.csv',
    'merged_data\\USDCHF_H4.csv',
    'merged_data\\USDJPY_H4.csv'
]



import pandas as pd


for i in range(8):

    a = pd.read_csv(paths_1[i])
    b = pd.read_csv(paths_2[i])

    a['Date'] = pd.to_datetime(a['Date'])
    b['Date'] = pd.to_datetime(b['Date'])

    combined = pd.concat([a, b], ignore_index=True)

    combined = combined.drop_duplicates(subset=['Date']).sort_values('Date')

    combined.to_csv(paths_3[i], index=False)



    a = pd.read_csv(paths_h1_1[i])
    b = pd.read_csv(paths_h1_2[i])

    a['Date'] = pd.to_datetime(a['Date'])
    b['Date'] = pd.to_datetime(b['Date'])

    combined = pd.concat([a, b], ignore_index=True)

    combined = combined.drop_duplicates(subset=['Date']).sort_values('Date')

    combined.to_csv(paths_h1_3[i], index=False)



    a = pd.read_csv(paths_h4_1[i])
    b = pd.read_csv(paths_h4_2[i])

    a['Date'] = pd.to_datetime(a['Date'])
    b['Date'] = pd.to_datetime(b['Date'])

    combined = pd.concat([a, b], ignore_index=True)

    combined = combined.drop_duplicates(subset=['Date']).sort_values('Date')

    combined.to_csv(paths_h4_3[i], index=False)
