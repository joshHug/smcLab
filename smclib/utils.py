import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas import Series, DataFrame, Timestamp
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

def read_iss_data(filename: str) -> DataFrame:
    df = pd.read_csv(filename)
    if "seconds_since_epoch" in df.columns:
        df = df.rename(columns={"seconds_since_epoch": "dtime"})

    # convert datetime strings into actual datetimes
    if isinstance(df["dtime"].iloc[0], str):
        df["dtime"] = pd.to_datetime(df["dtime"])

    # data is stored in epoch seconds, so convert into a datetime accordingly
    if isinstance(df["dtime"].iloc[0], np.float64):
        df['dtime'] = pd.to_datetime(df['dtime'], unit='s', origin='unix')
    return df

def find_gap_indices(df: DataFrame) -> Series:
    """
    Return a boolean pandas Series where each value indicates whether the corresponding 'dtime' entry in the
    input dataframe is the start of a significant gap in the data.

    A 'gap' is defined as a difference in 'dtime' that is greater than 1% of the total time range in the data.
    """

    # Calculate 1% of the total time range in the data
    one_percent_time = (df['dtime'].max() - df['dtime'].min()) / 100

    # Sort 'dtime' in ascending order, calculate the difference between each consecutive entry,
    # and compare each difference to one_percent_time
    gapped = df['dtime'].sort_values().diff() > one_percent_time

    # The result is a boolean Series where each value indicates whether the corresponding
    # 'dtime' is the start of a significant gap in the data.
    return gapped

def get_gap_times(df: DataFrame) -> list[Timestamp]:
    """
    Return a list of timestamps representing the times just before a significant gap in the data.

    A 'gap' is defined as a difference in 'dtime' that is greater than 1% of the total time range in the data.
    """

    # Use find_gap_indices to get a boolean Series where True values indicate the start of a data gap
    gi: Series = find_gap_indices(df)

    # Subset df to get the 'dtime' entries corresponding to the start of a data gap, and subtract one second
    gap_times = df[gi]["dtime"] - pd.to_timedelta('1 second')

    # Convert the result to a list and return
    return list(gap_times)

def create_gap_ends(gap_times: list[Timestamp], df: DataFrame) -> DataFrame:
    """
    Return a DataFrame with the same structure as `df`, filled with the timestamps from `gap_times`
    and `None` in the rest of the columns.

    The output DataFrame represents the end times of the data gaps (each timestamp is the time just before a data gap starts).
    """

    # Determine the number of rows in the output dataframe (same as the length of gap_times)
    N = len(gap_times)

    # Create an empty dataframe with the same columns as df and N rows
    output_df = DataFrame(columns=df.columns, index=range(N))

    # Fill in the first column of the output dataframe with the datetime objects from gap_times
    output_df.iloc[:, 0] = gap_times

    # Fill in the remaining columns of the output dataframe with None
    output_df.iloc[:, 1:] = None

    return output_df

def hide_gaps(df: DataFrame) -> DataFrame:
    """
    Return a DataFrame that combines `df` with a new DataFrame representing the end times of the data gaps.

    This function "hides" gaps in the data by inserting rows with `None` values at the times
    just before a data gap starts. The returned DataFrame is sorted by 'dtime'.
    """

    # Create a new DataFrame representing the end times of the data gaps
    ge = create_gap_ends(get_gap_times(df), df)

    # Combine the original DataFrame with the new DataFrame and sort the result by 'dtime'
    df_no_gaps = pd.concat([df, ge], ignore_index=True).sort_values("dtime")

    return df_no_gaps

def downsample(df: DataFrame) -> DataFrame:
    df_downsampled = df.copy()
    df_downsampled = df_downsampled.resample('60min', on='dtime').mean()
    df_downsampled = df_downsampled.reset_index().dropna()
    return df_downsampled

def hourly_average(df: DataFrame) -> DataFrame:
    df_downsampled = df.copy()
    df_downsampled = df_downsampled.resample('60min', on='dtime').mean()
    df_downsampled = df_downsampled.reset_index().dropna()
    return df_downsampled

def histogram(df: DataFrame, column_name: str) -> None:
    """
    Create and display a histogram of the specified column (`column_name`) from a pandas DataFrame (`df`).

    The histogram is created using Plotly Express and the font size of the plot is set to 16.
    If `column_name` is "altitude", the x-axis title is set to "altitude (km)".
    """

    # Create a histogram of the specified column using Plotly Express
    fig = px.histogram(df[column_name])

    # Update the layout to set the font size to 16
    fig.update_layout(font_size=16)

    # Add units to axis title
    if column_name == "altitude":
        fig.update_layout(xaxis_title="altitude (km)")

    # Display the plot
    fig.show()

def box_plot(df: DataFrame, column_name: str) -> None:
    fig = px.box(df[column_name])
    fig.update_layout(font_size=16)
    if column_name == "altitude":
        fig.update_layout(yaxis_title="altitude (km)")
    fig.show()

def line_plot(df: DataFrame, x: str, y: str) -> None:
    """
    Create and display a line plot with the specified x and y columns from a pandas DataFrame (`df`).

    The line plot is created using Plotly Express and the font size of the plot is set to 16.
    If `y` is "altitude", the y-axis title is set to "altitude (km)".
    """

    # Create a line plot using Plotly Express with specified x and y columns
    fig = px.line(hide_gaps(df), x, y=y)

    # Add units to axis title
    if y == "altitude":
        fig.update_layout(yaxis_title="altitude (km)")

    # Set the font size of the plot to 16
    fig.update_layout(font_size=16)

    # Display the plot
    fig.show()


def plot_quantity(df: DataFrame, quantities: str | list[str]) -> Figure:
    """
    Generate a line plot for one or more specified quantities from a pandas DataFrame (`df`).

    The quantities to be plotted are provided in `quantities`, which can be a string (for one quantity)
    or a list of strings (for multiple quantities).

    The plot is created using Plotly Express, with each quantity plotted as a separate line.
    Gaps in the data are hidden using the `hide_gaps` function.

    Returns a plotly Figure object representing the plot.
    """

    # If a single quantity is provided, convert it into a list
    if isinstance(quantities, str):
        quantities = [quantities]

    # Filter df to only include rows where the 'description' column is in quantities
    filtered_df: DataFrame = df[df['description'].isin(quantities)]

    # Pivot this filtered data into a new DataFrame where each quantity has its own column,
    # with datetime values as the index
    pivot = filtered_df.pivot_table(index='date_time', columns='description', values='value').reset_index()

    # Hide any gaps in this DataFrame
    pivot = hide_gaps(pivot)

    # Generate a line plot using Plotly Express, with each quantity plotted as a separate line
    fig = px.line(pivot, x="date_time", y=pivot.columns[1:])

    return fig


def plot_ypr(ypr: DataFrame, lower_date: str = '2010-01-01', upper_date: str = '2030-01-01', y: list[str] | None = None) -> Figure:
    """
    Generate a line plot for yaw, pitch, and roll from a pandas DataFrame (`ypr`) within a specified date range (`lower_date` to `upper_date`).

    The columns to be plotted are provided in `y`, which can be a list of strings (default is ["yaw", "pitch", "roll"] if None is passed).

    The plot is created using Plotly Express, with each column plotted as a separate line.
    Gaps in the data are hidden using the `hide_gaps` function.

    Returns a Plotly Express Figure object representing the plot.
    """

    # Avoid having a mutable default argument
    if y is None:
        y = ["yaw", "pitch", "roll"]

    # Ensure that the dates are in the correct format
    assert bool(re.search(r'\d{4}-\d{2}-\d{2}', lower_date)) and bool(re.search(r'\d{4}-\d{2}-\d{2}', upper_date))

    # Filter ypr to only include rows where the 'dtime' column is within the given date range
    filtered = ypr[(ypr['dtime'] > lower_date) & (ypr['dtime'] < upper_date)]

    # If the filtered data contains more than 20000 rows, downsample it
    if len(filtered) > 20000:
        filtered = downsample(filtered)

    # Generate a line plot using Plotly Express, with each column in y plotted as a separate line
    return px.line(hide_gaps(filtered), x="dtime", y=y)

def make_two_box_plots(df: DataFrame, x: str, y: str, left: list[str], right: list[str]) -> Figure:
    """
    Generate two box plots side-by-side for different descriptions from a pandas DataFrame (`df`).

    The first box plot represents the DataFrame filtered by the descriptions provided in `left`,
    and the second box plot represents the DataFrame filtered by the descriptions provided in `right`.

    The plot is created using Plotly's make_subplots function, which creates a subplot object with one row and two columns.

    Returns a Plotly Figure object representing the plot.
    """

    # Filter df by the descriptions provided in left and right
    df_left = df[df["description"].isin(left)]
    df_right = df[df["description"].isin(right)]

    # Create a subplots object with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2)

    # Add box plots to the subplots object
    fig.add_trace(
        go.Box(x=df_left["description"], y=df_left["value"]),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(x=df_right["description"], y=df_right["value"]),
        row=1, col=2
    )

    # Update the layout of the figure
    fig.update_layout(font_size=16)

    return fig
