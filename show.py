import plotly.graph_objects as go
import numpy as np
import pandas as pd

class block:
    def __init__(self , high , low , start_date , end_date):
        self.high = high
        self.low = low
        self.start_date = start_date
        self.end_date = end_date
        

class point:
    def __init__(self, date, price, symbol):
        self.date = date
        self.price = price
        self.symbol = symbol


'''
this function is used for showing candlesticks data as a chart
also you can pass blocks & points list to it for showing boxes & points on chart 
the chart saves as a html file and opens automatically

expected structure for candlestick dataframe : 
    'Date': candle.start_date,
    'Open': candle.open,
    'High': candle.high,
    'Low': candle.low,
    'Close': candle.close


parameters : 
    candles : dataframe of candlestick data,
    blocks : list of blocks
    points : list of points
'''

def show(df, interval ,blocks=None , points=None , file_name_extention = ''):

    df = df.sort_values('Date').reset_index(drop=True)

    df['Delta'] = df['Date'].diff()

    break_rows = df[df['Delta'] > interval]

    dt_breaks = [
        {
            "bounds": [
                df.loc[i-1, 'Date'] + interval,   # start of missing zone
                df.loc[i, 'Date']                 # next actual candle
            ]
        }
        for i in break_rows.index
    ]

    # Create the candlestick figure
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])


    # Add blocks if provided
    if blocks:
        for blk in blocks:
            fig.add_shape(
                type="rect",
                x0=blk.start_date,
                x1=blk.end_date,
                y0=blk.low,
                y1=blk.high,
                fillcolor="blue",
                opacity=0.3,
                line_width=0,
                layer="below"
            )
                

    if points:
        for pt in points:
            symbol = pt.symbol
            fig.add_annotation(
                x=pt.date,
                y=pt.price,
                text=symbol,
                showarrow=False,
                font=dict(size=16),
                align="center",
                yshift=10
            )

    # Update the x-axis and layout
    fig.update_xaxes(
        title='Time',
        rangeslider_visible=False,
        rangebreaks=dt_breaks
    )

    fig.update_layout(
        title='Custom Candle Chart with Blocks',
        yaxis_title='Price',
    )

    # Save and open the chart
    fig.write_html('charts\\chart' + str(file_name_extention) +'.html', config={"scrollZoom": True})



def split_show(df, interval, blocks=None, points=None):

    n = max(1, len(df) // 30000)
    dfs = np.array_split(df, n)

    file_name_extention = 0

    for d in dfs:
        # If np.array_split somehow returns ndarray, convert it back
        if not isinstance(d, pd.DataFrame):
            d = pd.DataFrame(d, columns=df.columns)

        temp_points = None
        temp_blocks = None

        if points:
            temp_points = []
            for p in points:
                if d['Date'].iloc[0] < p.date < d['Date'].iloc[-1]:
                    temp_points.append(p)

        if blocks:
            temp_blocks = []
            for b in blocks:
                if d['Date'].iloc[0] < b.start_date < d['Date'].iloc[-1]:
                    temp_blocks.append(b)  # <-- b, not p

        show(
            d,
            interval,
            blocks=temp_blocks,
            points=temp_points,
            file_name_extention=file_name_extention
        )

        file_name_extention += 1