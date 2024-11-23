#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_fig.py

Functions to produce visualizations of beacon data after it
has been ingested, cleaned, and standardized. 

Visualizations include: 
    map - a map of the iceberg track
    time - variables over time - temperature, displacement and velocity
    dist - statistical distributions - polar plot of direction, histogram of speed
    temp - temperature changes (used for checking if beacon is still on target)

Author: Adam Garbo, December 2022
Modified: Derek Mueller, July 2024
"""
import os
import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from collections import namedtuple


def nolog():
    """
    Create a logger instance that doesn't do anything.

    Used to allow logging or not in the code below

    Returns
    -------
    NoOpLogger
        A named tuple that mimics a log instance.

    """
    NoOpLogger = namedtuple(
        "NoOpLogger", ["debug", "info", "warning", "error", "critical"]
    )
    return NoOpLogger(*([lambda *args, **kwargs: None] * 5))


def plot_map(track, path_output=".", dpi=300, interactive=False, log=None):
    """
    Create a map of the iceberg track.

    Map is saved as a png, the star indicates the start

    Parameters
    ----------
    track : track object
        Standardized beacon track object.
    path_output : str, optional
        Path to save output. The default is ".".
    dpi : int, optional
        Resolution of the graph in dots per inch. The default is 300.


    Returns
    -------
    None.

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # quiet these loggers a bit..
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    log.info("Plotting map")

    # Add Natural Earth coastline
    coast = cfeature.NaturalEarthFeature(
        "physical", "land", "10m", edgecolor="black", facecolor="lightgray", lw=0.5
    )

    # Set map centres
    x = track.data["longitude"].median()
    y = track.data["latitude"].median()

    # Plot latitude and longitude
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    ax = plt.axes(projection=ccrs.Orthographic(x, y))
    ax.add_feature(coast)
    ax.set_adjustable("datalim")
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        color="black",
        alpha=0.25,
        linestyle="dotted",
        x_inline=False,
        y_inline=False,
    )
    gl.rotate_labels = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xpadding = 5

    sns.scatterplot(
        x="longitude",
        y="latitude",
        data=track.data,
        linewidth=1,
        edgecolor="black",
        legend=False,
        transform=ccrs.PlateCarree(),
    )

    # plot the very start of the track
    ax.plot(
        track.data.longitude.iloc[0],
        track.data.latitude.iloc[0],
        marker="*",
        ms=20,
        mfc="r",
        mec="k",
        transform=ccrs.PlateCarree(),
    )

    fig.suptitle(
        f"{track.beacon_id} map", fontweight="bold", fontsize=18
    )  # note use y= to adjust

    ax.text(
        0,
        -0.09,
        f"Start (*): {track.data_start:%Y-%m-%d %H:%M:%S} UTC, End: {track.data_end:%Y-%m-%d %H:%M:%S} UTC\n  \
        Duration: {track.duration:.2f} days, Distance: {track.distance:,.2f} km, Observations: {track.observations:,}",
        transform=ax.transAxes,
        color="black",
        fontsize=12,
        fontweight="regular",
    )

    if interactive:
        plt.show()

    else:
        plt.savefig(
            os.path.join(path_output, f"{track.beacon_id}_map.png"),
            dpi=dpi,
            transparent=False,
            bbox_inches="tight",
        )
        plt.close()

    log.info("Plotting map complete")


def plot_temp(track, path_output=".", dpi=300, interactive=False, log=None):
    """
    Create a graph of the beacon temperature.

    There are 3 panels:
        Temperature (of the air, surface or internal, depending on what is available)
        7 day rolling mean of temperature
        7 day rolling standard deviation of temperature

    This is useful for finding the start and end of the track:
        When beacons are deployed they may still be acclimatizing to ambient conditions
        When beacons fall off their target into the water, the standard deviation should
            decrease.  Temperature should approach water temperature (typically decreasing)

    Parameters
    ----------
    track : track object
        Standardized beacon track object.
    path_output : str, optional
        Path to save output. The default is ".".
    dpi : int, optional
        Resolution of the graph in dots per inch. The default is 300.

    Returns
    -------
    None.

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # quiet these loggers a bit..
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    log.info("Plotting temp")

    # get the temperature
    track.data["temperature"] = track.data["temperature_air"]
    temp_type = "Air"
    if track.data["temperature_air"].isnull().all():
        temp_type = "Surface"
        track.data["temperature"] = track.data["temperature_surface"]
        if track.data["temperature_surface"].isnull().all():
            temp_type = "Internal"
            track.data["temperature"] = track.data["temperature_internal"]
            if track.data["temperature_internal"].isnull().all():
                temp_type = "NA"
                track.data["temperature"] = 0

    track.data["Tmean"] = track.data.rolling(
        window="7d", on="datetime_data", center=True
    ).temperature.mean()
    track.data["Tstd"] = track.data.rolling(
        window="7d", on="datetime_data", center=True
    ).temperature.std()

    # plotting  - t temp, m mean, s std
    fig, (t, m, s, v) = plt.subplots(
        4, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )

    # Temperature plot
    t.grid(ls="dotted")
    sns.lineplot(
        ax=t,
        x="datetime_data",
        y="temperature",
        data=track.data,
        errorbar=None,
        color="b",
    )
    t.set(xlabel=None, ylabel="Temperature (°C)")

    # Rolling temperature mean plot
    m.grid(ls="dotted")
    sns.lineplot(
        ax=m, x="datetime_data", y="Tmean", data=track.data, errorbar=None, color="r"
    )
    m.set(xlabel=None, ylabel="Temp. rolling mean (°C)")

    # Rolling temperature std plot
    s.grid(ls="dotted")
    sns.lineplot(
        ax=s, x="datetime_data", y="Tstd", data=track.data, errorbar=None, color="k"
    )
    s.set(xlabel=None, ylabel="Temp. rolling std (°C)")

    v.grid(ls="dotted")
    sns.lineplot(
        ax=s, x="datetime_data", y="voltage", data=track.data, errorbar=None, color="k"
    )
    v.set(xlabel=None, ylabel="Battery (V)")
    plt.xticks(rotation=45, horizontalalignment="center")

    # plot the track trimming points so it can be verified before trimming.
    if not track.trimmed:
        if not pd.isnull(track.track_start):
            t.axvline(track.track_start, linestyle="dashdot", color="g")
            m.axvline(track.track_start, linestyle="dashdot", color="g")
            s.axvline(track.track_start, linestyle="dashdot", color="g")
        if not pd.isnull(track.track_end):
            t.axvline(track.track_end, linestyle="dashdot", color="g")
            m.axvline(track.track_end, linestyle="dashdot", color="g")
            s.axvline(track.track_end, linestyle="dashdot", color="g")

    if temp_type == "NA":
        fig.suptitle(
            f"{track.beacon_id} temperature not available",
            fontweight="bold",
            fontsize=18,
        )
    else:
        fig.suptitle(
            f"{track.beacon_id} {temp_type} temperature plot",
            fontweight="bold",
            fontsize=18,
        )

    s.text(
        0,
        -0.61,
        f"Start (*): {track.data_start:%Y-%m-%d %H:%M:%S} UTC, End: {track.data_end:%Y-%m-%d %H:%M:%S} UTC\n  \
        Duration: {track.duration:.2f} days, Distance: {track.distance:,.2f} km, Observations: {track.observations:,}",
        transform=s.transAxes,
        color="black",
        fontsize=12,
        fontweight="regular",
    )

    fig.align_ylabels()

    if interactive:
        plt.show()

    else:
        plt.savefig(
            os.path.join(path_output, f"{track.beacon_id}_temp.png"),
            dpi=dpi,
            transparent=False,
            bbox_inches="tight",
        )
        plt.close()

        log.info("Plotting temp complete")


def plot_dist(track, path_output=".", dpi=300, interactive=False, log=None):
    """
    Create a graph of the track's statistical distributions.

    Parameters
    ----------
    track : track object
        Standardized beacon track object.
    path_output : str, optional
        Path to save output. The default is ".".
    dpi : int, optional
        Resolution of the graph in dots per inch. The default is 300.

    Returns
    -------
    None.

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # quiet these loggers a bit..
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    log.info("Plotting dist")

    fig = plt.figure(figsize=(10, 5))  # , constrained_layout=True)
    h = plt.subplot(121)
    p = plt.subplot(122, projection="polar")

    # Histogram of the speed
    """ clean data -- this should not be needed, so commented out. (but kept here in case someone want to use it )
    unreasonably high >2.8 m/s Garbo, > 8 m/s Dalton  Dalton fastest was 2.3 m/s
    track.data.replace([np.inf, -np.inf], np.nan, inplace=True)
    track.data.loc[track.data["speed"] > 5] = np.nan  # there is no way an iceberg moves faster than 5 m/s

    Create histogram
    here we are scaling under the assumption that there will not be any speeds greater 
    than 2.5 m/s (i.e., more constrained)
    by setting the same lower and upper limit, the histograms are comparable
    """
    xlowerlim = 0
    xupperlim = 2.5
    nbins = int(10 * (xupperlim - xlowerlim) / 0.5)  # this sets the number of bins

    """ Here bins start with 0 to 0.05 m/s. That represents a displacement of 180 m so there is
    likely true motion within that bin.
    Dalton examined a stationary gps beacon and found mean + 1 SD = 0.04 m/s.
    """

    # TODO - calculate the number of observations to right edge to the histogram and print that out?
    out_of_range = sum(track.data.speed > xupperlim)
    # calcuate bins and values
    values, bins = np.histogram(
        track.data.speed[~np.isnan(track.data.speed)],
        range=(xlowerlim, xupperlim),
        bins=nbins,
    )
    # convert to fraction of all obs
    frac = values / len(track.data.speed[~np.isnan(track.data.speed)])
    # cumfrac = np.cumsum(frac)

    # here we set the y axis (left) limits - they represent fraction of obs within each bin
    ylowerlim = 0
    yupperlim = 1

    h.set_xlabel("Speed (m/s)")
    h.set_ylabel("Fraction of observations", color="red")
    h.hist(bins[:-1], bins, weights=frac, color="red")
    h.tick_params(axis="y", labelcolor="red")
    h.set_ylim([ylowerlim, yupperlim * 1.01])
    h.set_xlim([xlowerlim, xupperlim])

    # plot cummulative hist on right y axis
    h2 = h.twinx()

    h2.set_ylabel(
        "Cumulative fraction of observations", color="blue", rotation=270, labelpad=15
    )
    h2.hist(
        bins[:-1], bins, weights=frac, color="blue", histtype="step", cumulative=True
    )
    h2.tick_params(axis="y", labelcolor="blue")
    h2.set_ylim([ylowerlim, yupperlim * 1.01])

    # Polar plot of speed and direction
    p.plot(
        track.data.direction,
        track.data.speed,
        marker="o",
        markerfacecolor="b",
        markeredgecolor="k",
        linestyle="None",
    )
    p.grid(True)
    p.set_theta_direction(-1)  # turn clockwise
    p.set_theta_zero_location("N")  # north up
    p.set_rlabel_position(200.5)  # Move radial labels away from plotted line
    plt.text(0.05, -0.05, "Speed (m/s)", transform=p.transAxes)

    fig.suptitle(f"{track.beacon_id} distribution plot", fontweight="bold", fontsize=18)

    h.text(
        0.0,
        -0.275,
        f"Start (*): {track.data_start:%Y-%m-%d %H:%M:%S} UTC, End: {track.data_end:%Y-%m-%d %H:%M:%S} UTC\n  \
        Duration: {track.duration:.2f} days, Distance: {track.distance:,.2f} km, Observations: {track.observations:,}",
        transform=h.transAxes,
        color="black",
        fontsize=12,
        fontweight="regular",
    )

    plt.subplots_adjust(wspace=0.3)

    if interactive:
        plt.show()

    else:
        plt.savefig(
            os.path.join(path_output, f"{track.beacon_id}_dist.png"),
            dpi=dpi,
            transparent=False,
            bbox_inches="tight",
        )

        plt.close()

    log.info("Plotting dist complete")


def plot_time(track, path_output=".", dpi=300, interactive=False, log=None):
    """
    Create a graph of variables with respect to time.

    There are 3 panels:
        Temperature (of the air, surface or internal, depending on what is available)
        Distance (scatter plot)
        Speed and direction (quiver)

    TODO: Subsample to a daily timestep?

    Parameters
    ----------
    track : track object
        Standardized beacon track object.
    path_output : str, optional
        Path to save output. The default is ".".
    dpi : int, optional
        Resolution of the graph in dots per inch. The default is 300.

    Returns
    -------
    None.

    """

    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # quiet these loggers a bit..
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    log.info("Plotting timeseries")

    # get the temperature
    track.data["temperature"] = track.data["temperature_air"]
    temp_type = "Air"
    if track.data["temperature_air"].isnull().all():
        temp_type = "Surface"
        track.data["temperature"] = track.data["temperature_surface"]
        if track.data["temperature_surface"].isnull().all():
            temp_type = "Internal"
            track.data["temperature"] = track.data["temperature_internal"]
            if track.data["temperature_internal"].isnull().all():
                temp_type = "No"
                track.data["temperature"] = 0

    # plotting
    fig, (t, d, q) = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )

    # Temperature plot
    t.grid(ls="dotted")
    sns.lineplot(
        ax=t,
        x="datetime_data",
        y="temperature",
        data=track.data,
        errorbar=None,
        color="b",
    )
    t.set(xlabel=None, ylabel=f"{temp_type} temperature (°C)")
    if temp_type == "NA":
        t.set(xlabel=None, ylabel="No temperature available (°C)")

    # Distance plot
    sns.lineplot(
        ax=d, x="datetime_data", y="distance", data=track.data, errorbar=None, color="r"
    )
    d.set(xlabel=None, ylabel="Displacement (m)")

    # quiver plot
    u = track.data.speed * np.sin(np.radians(track.data.direction))
    v = track.data.speed * np.cos(np.radians(track.data.direction))

    q.quiver(track.data["datetime_data"], 0, u, v)
    q.set(xlabel=None, ylabel="Velocity (m/s)")
    fig.align_ylabels()
    plt.xticks(rotation=45, horizontalalignment="center")

    fig.suptitle(f"{track.beacon_id} time plot", fontweight="bold", fontsize=18)

    q.text(
        0,
        -0.61,
        f"Start (*): {track.data_start:%Y-%m-%d %H:%M:%S} UTC, End: {track.data_end:%Y-%m-%d %H:%M:%S} UTC\n  \
        Duration: {track.duration:.2f} days, Distance: {track.distance:,.2f} km, Observations: {track.observations:,}",
        transform=q.transAxes,
        color="black",
        fontsize=12,
        fontweight="regular",
    )

    if interactive:
        plt.show()

    else:
        plt.savefig(
            os.path.join(path_output, f"{track.beacon_id}_time.png"),
            dpi=dpi,
            transparent=False,
            bbox_inches="tight",
        )
        plt.close()

    log.info("Plotting timeseries complete")


"""
def main():
    '''Work from command line.'''
    # get parameters from command line:
    parser = argparse.ArgumentParser(description="Beacon track visualization functions")
    parser.add_argument("std_file", help="enter full path to the standard data file")
    parser.add_argument(
        "-p",
        "--path_output",
        help="enter path to store output png files ; \
                        defaults to the current directory",
    )
    parser.add_argument(
        "-g",
        "--graphs",
        nargs="+",
        choices={"map", "time", "temp", "dist"},
        help="list the graphs to produce: map time temp dist ; defaults to producing ALL graphs",
    )
    args = parser.parse_args()

    std_file = args.std_file
    path_output = args.path_output
    graphs = args.graphs

    # validate the parameters

    if not os.path.isfile(std_file):
        print(f"{std_file} not found, exiting...")
        sys.exit(1)

    if path_output:
        if not os.path.isdir(path_output):
            print(f"{path_output} not found, exiting... ")
            sys.exit(1)
    else:
        print(
            f"Output directory not requested, defaulting to current directory: {os.getcwd()}..."
        )
        path_output = os.getcwd()

    trk = Track(std_file)

    if graphs:
        if "map" in graphs:
            plot_map(trk, path_output=path_output)
        if "dist" in graphs:
            plot_dist(trk, path_output=path_output)
        if "temp" in graphs:
            plot_temp(trk, path_output=path_output)
        if "time" in graphs:
            plot_time(trk, path_output=path_output)

    else:  # if no graphs are requested, then produce them all
        plot_map(trk, path_output=path_output)
        plot_temp(trk, path_output=path_output)
        plot_dist(trk, path_output=path_output)
        plot_time(trk, path_output=path_output)

"""

"""
# TODO Add more tracks to plots 
# here is some pseudocode:
def plot(track, *other_tracks):
    fig, ax = plt.subplots()
    ax.plot(track.data, label=track.beacon_id)
    
    for track in other_tracks:
        ax.plot(track.data, label=track.beacon_id)

this will use matplotlibs colours to plot each one as a different colour
need to deal with the title and text.
"""


"""
if __name__ == "__main__":
    main()
"""


# std_file = "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/standardized_data/2010/300034012592660/2010_300034012592660.csv"

# output = "/home/dmueller/Desktop"
# path_output = output

# # Get standardized data output path
# path_output = Path(file).resolve().parents[0]

# # Get unique beacon ID
# filename = Path(file).stem


# # -----------------------------------------------------------------------------
# # Configure logging
# # -----------------------------------------------------------------------------

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter(
#     "%(asctime)s:%(msecs)d %(name)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
# )
