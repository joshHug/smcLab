import pandas as pd
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

def read_iss_data(filename):
    df = pd.read_csv(filename)
    if "seconds_since_epoch" in df.columns:
        df = df.rename(columns = {"seconds_since_epoch": "dtime"})

    # convert datetime strings into actual datetimes
    if isinstance(df["dtime"].iloc[0], str):
        df["dtime"] = pd.to_datetime(df["dtime"])

    # data is stored in epoch seconds, so convert into a datetime accordingly
    if isinstance(df["dtime"].iloc[0], np.float64):
        df['dtime'] = pd.to_datetime(df['dtime'], unit = 's', origin = 'unix')
    return df

def find_gap_indices(df):
    one_percent_time = (df['dtime'].max() - df['dtime'].min()) / 100
    gapped = df['dtime'].sort_values().diff() > one_percent_time
    return gapped

def get_gap_times(df):
    gi = find_gap_indices(df)
    gi.index = df.index    
    gap_times = df[gi]["dtime"] - pd.to_timedelta('1 second')
    return list(gap_times)

def create_gap_ends(gap_times, df):
    # Determine the number of rows in the output dataframe
    N = len(gap_times)
    
    # Create an empty dataframe with the same columns as df
    output_df = pd.DataFrame(columns=df.columns, index=range(N))
    
    # Fill in the first column of the output dataframe with the datetime objects from lodo
    output_df.iloc[:, 0] = gap_times
    
    # Fill in the remaining columns of the output dataframe with None
    output_df.iloc[:, 1:] = None
    
    return output_df

def hide_gaps(df):
    """Adds a None value after any data gap of a day or more."""
    ge = create_gap_ends(list(get_gap_times(df)), df)
    df_no_gaps = pd.concat([df, ge], ignore_index = True).sort_values("dtime")
    return df_no_gaps

def downsample(df):
    df_downsampled = df.copy()
    df_downsampled = df_downsampled.resample('60min', on='dtime').mean()
    df_downsampled = df_downsampled.reset_index().dropna()    
    return df_downsampled

def hourly_average(df):
    df_downsampled = df.copy()
    df_downsampled = df_downsampled.resample('60min', on='dtime').mean()
    df_downsampled = df_downsampled.reset_index().dropna()    
    return df_downsampled

def histogram(df, x):
    fig = px.histogram(df[x])
    fig.update_layout(font_size = 16)
    if x == "altitude":
        fig.update_layout(xaxis_title="altitude (km)")
    fig.show()

def box_plot(df, y):
    fig = px.box(df[y])
    fig.update_layout(font_size = 16)
    if y == "altitude":
        fig.update_layout(yaxis_title="altitude (km)")    
    fig.show()


def line_plot(df, x, y):
    fig = px.line(hide_gaps(df), x, y = y)
    if y == "altitude":
        fig.update_layout(yaxis_title="altitude (km)")
    fig.update_layout(font_size = 16)
    fig.show()

def plot_quantity(df, quantities):
    if isinstance(quantities, str):
        quantities = [quantities]
    filt = df[df['description'].isin(quantities)]
    pivot = hide_gaps(filt.pivot_table(index = 'date_time', columns = 'description', values = 'value').reset_index())
    fig = px.line(pivot, x = "date_time", y = pivot.columns[1:])
    return fig


def plot_ypr(ypr, lower_date = '2010-01-01', upper_date = '2030-01-01', y = ["yaw", "pitch", "roll"]):
    assert bool(re.search(r'\d{4}-\d{2}-\d{2}', lower_date)) and bool(re.search(r'\d{4}-\d{2}-\d{2}', upper_date))
    filtered = ypr[(ypr['dtime'] > lower_date) & (ypr['dtime'] < upper_date)]
    if len(filtered) > 20000:
        filtered = downsample(filtered)
    return px.line(hide_gaps(filtered), x = "dtime", y = y)

def make_two_box_plots(df, x, y, left, right):
    df_left = df[df["description"].isin(left)]
    df_right = df[df["description"].isin(right)]
    df_not_qty = df[~df["description"].str.contains("Qty")]

    # Create a subplots object with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Box(x=df_left["description"], y=df_left["value"]),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(x=df_right["description"], y=df_right["value"]),
        row=1, col=2
    )

    fig.update_layout(font_size = 16)
    fig.show()

