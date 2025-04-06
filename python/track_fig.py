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

Author: Derek Mueller, July 2024 to Jan 2025, modifying code from Adam Garbo from 2021
"""
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from collections import namedtuple


# turn on the copy-on-write functionality
pd.options.mode.copy_on_write = True


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
    ax = plt.axes(projection=ccrs.Orthographic(x, y), zorder=1)

    # add coast and grid
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

    # plot the point data
    sns.scatterplot(
        x="longitude",
        y="latitude",
        data=track.data,
        linewidth=0.75,
        edgecolor="black",
        color="royalblue",  # this is the closest match to the default colour
        legend=False,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    # plot the line data
    ax.plot(
        track.data["longitude"],
        track.data["latitude"],
        color="dimgrey",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    # plot the track trimming points so it can be verified before trimming.
    # green dot is the first good value, orange dot is the last good value

    if not track.trimmed:
        if not pd.isnull(track.trim_start):
            ax.plot(
                # you only want the first one...
                track.data.longitude.loc[
                    track.data.datetime_data <= track.trim_start
                ].iloc[-1],
                track.data.latitude.loc[
                    track.data.datetime_data <= track.trim_start
                ].iloc[-1],
                marker="o",
                ms=8,
                mfc="lime",
                mec="k",
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

        if not pd.isnull(track.trim_end):
            ax.plot(
                # you only want the last one...
                track.data.longitude.loc[
                    track.data.datetime_data >= track.trim_end
                ].iloc[0],
                track.data.latitude.loc[
                    track.data.datetime_data >= track.trim_end
                ].iloc[0],
                marker="o",
                ms=8,
                mfc="tab:orange",
                mec="k",
                transform=ccrs.PlateCarree(),
                zorder=6,
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
        zorder=4,
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


def plot_trim(track, path_output=".", dpi=300, interactive=False, log=None):
    """
    Create a graph of the beacon temperature and battery voltage.

    There are 3 panels:
        Temperature (of the air, surface or internal, depending on what is available)
        5 and 3 day rolling mean and standard deviation of temperature

    This is useful for finding the end of the track:
        When beacons fall off their target into the water, the standard deviation should
            decrease.  Temperature should approach water temperature (typically decreasing)

    Note when beacons are deployed they may still be acclimatizing to ambient conditions.
    In this scenario, the rolling mean and std are less helpful because their windows are
    positioned on the left to capture a sudden change.

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

    log.info("Plotting trim")

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

    # the following makes a left aligned window which should match well with any event
    # reverse the series, run the rolling operation and reverse again. Center false
    # keeps the window index on the right, reversing sets this to the left.
    track.data["Tmean5"] = (
        track.data[::-1]
        .rolling(window="5d", on="datetime_data", center=False)
        .temperature.mean()[::-1]
    )
    track.data["Tmean3"] = (
        track.data[::-1]
        .rolling(window="3d", on="datetime_data", center=False)
        .temperature.mean()[::-1]
    )

    track.data["Tstd5"] = (
        track.data[::-1]
        .rolling(window="5d", on="datetime_data", center=False)
        .temperature.std()[::-1]
    )
    track.data["Tstd3"] = (
        track.data[::-1]
        .rolling(window="3d", on="datetime_data", center=False)
        .temperature.std()[::-1]
    )

    # plotting  - t temp, m mean, s std
    fig, (t, r, v) = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )

    # Temperature plot  = t
    t.grid(ls="dotted")
    sns.lineplot(
        ax=t,
        x="datetime_data",
        y="temperature",
        label=f"{temp_type} temperature",
        data=track.data,
        errorbar=None,
        color="b",
    )
    t.set(xlabel=None, ylabel=f"{temp_type} temperature (°C)")
    t.get_legend().remove()

    if track.data["pressure"].notna().any():
        # plot pressure on y axis - only if there is data...
        t2 = t.twinx()
        t2.set_ylabel("Air pressure (hPa)", color="black", rotation=270, labelpad=15)

        sns.lineplot(
            ax=t2,
            x="datetime_data",
            y="pressure",
            label="pressure",
            data=track.data,
            errorbar=None,
            color="r",
        )

        handles_t1, labels_t1 = t.get_legend_handles_labels()
        handles_t2, labels_t2 = t2.get_legend_handles_labels()

        t2.legend(
            handles_t1 + handles_t2, labels_t1 + labels_t2, loc="upper center"
        )  # This position avoids overlap with beginning and end of track

    # Rolling temperature mean/std plot = r
    r.grid(ls="dotted")
    sns.lineplot(
        ax=r,
        x="datetime_data",
        y="Tmean5",
        label="5 day mean",
        data=track.data,
        errorbar=None,
        color="r",
        linestyle="-",
    )
    sns.lineplot(
        ax=r,
        x="datetime_data",
        y="Tmean3",
        label="3 day mean",
        data=track.data,
        errorbar=None,
        color="r",
        linestyle="--",
    )
    r.set(xlabel=None, ylabel="Temp. rolling mean (°C)")
    r.get_legend().remove()

    # second axis
    r2 = r.twinx()
    r2.set_ylabel("Temp. rolling std (°C)", color="black", rotation=270, labelpad=15)

    sns.lineplot(
        ax=r2,
        x="datetime_data",
        y="Tstd5",
        label="5 day std",
        data=track.data,
        errorbar=None,
        color="k",
        linestyle="-",
    )
    sns.lineplot(
        ax=r2,
        x="datetime_data",
        y="Tstd3",
        label="3 day std",
        data=track.data,
        errorbar=None,
        color="k",
        linestyle="--",
    )

    handles_r1, labels_r1 = r.get_legend_handles_labels()
    handles_r2, labels_r2 = r2.get_legend_handles_labels()

    r2.legend(
        handles_r1 + handles_r2, labels_r1 + labels_r2, loc="upper center"
    )  # This position avoids overlap with beginning and end of track

    # Voltage plot = v
    v.grid(ls="dotted")
    sns.lineplot(
        ax=v,
        x="datetime_data",
        y="voltage",
        data=track.data,
        label="battery",
        errorbar=None,
        color="g",
    )
    v.set(xlabel=None, ylabel="Battery (V)")

    v.get_legend().remove()

    if track.data.pitch.notna().any():
        # plot pitch and roll on y axis - only if there is data...
        v2 = v.twinx()

        v2.set_ylabel("Tilt (degrees)", color="black", rotation=270, labelpad=15)
        sns.lineplot(
            ax=v2,
            x="datetime_data",
            y="pitch",
            label="pitch",
            data=track.data,
            errorbar=None,
            color="m",
        )
        sns.lineplot(
            ax=v2,
            x="datetime_data",
            y="roll",
            label="roll",
            data=track.data,
            errorbar=None,
            color="c",
        )
        v2.tick_params(axis="y", labelcolor="black")
        v2.set_ylim([-185, 185])

        handles_v1, labels_v1 = v.get_legend_handles_labels()
        handles_v2, labels_v2 = v2.get_legend_handles_labels()

        v2.legend(
            handles_v1 + handles_v2, labels_v1 + labels_v2, loc="upper center"
        )  # This position avoids overlap with beginning and end of track

    # plot the track trimming points so it can be verified before trimming. zorder >> big
    if not track.trimmed:
        if not pd.isnull(track.trim_start):
            t.axvline(track.trim_start, linestyle="dashdot", color="lime", zorder=100)
            r.axvline(track.trim_start, linestyle="dashdot", color="lime", zorder=100)
            v.axvline(track.trim_start, linestyle="dashdot", color="lime", zorder=100)
        if not pd.isnull(track.trim_end):
            t.axvline(
                track.trim_end, linestyle="dashdot", color="tab:orange", zorder=100
            )
            r.axvline(
                track.trim_end, linestyle="dashdot", color="tab:orange", zorder=100
            )
            v.axvline(
                track.trim_end, linestyle="dashdot", color="tab:orange", zorder=100
            )

    fig.suptitle(
        f"{track.beacon_id} trim plot",
        fontweight="bold",
        fontsize=18,
    )

    # plt.xticks(rotation=45, horizontalalignment="center")
    v.tick_params(axis="x", rotation=45)
    v.text(
        0,
        -0.61,
        f"Start (*): {track.data_start:%Y-%m-%d %H:%M:%S} UTC, End: {track.data_end:%Y-%m-%d %H:%M:%S} UTC\n  \
        Duration: {track.duration:.2f} days, Distance: {track.distance:,.2f} km, Observations: {track.observations:,}",
        transform=v.transAxes,
        color="black",
        fontsize=12,
        fontweight="regular",
    )

    fig.align_ylabels()

    if interactive:
        plt.show()

    else:
        plt.savefig(
            os.path.join(path_output, f"{track.beacon_id}_trim.png"),
            dpi=dpi,
            transparent=False,
            bbox_inches="tight",
        )
        plt.close()

        log.info("Plotting trim complete")


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
    """ Deal with high speeds -- this should not be needed, so commented out. (but kept here in case someone want to use it )
    unreasonably high speeds >2.8 m/s Garbo thesis, > 8 m/s Dalton thesis 
    Dalton's fastest was 2.3 m/s
    # code to remove data
    track.data.replace([np.inf, -np.inf], np.nan, inplace=True)
    track.data.loc[track.data["speed"] > 5] = np.nan  # there is no way an iceberg moves faster than 5 m/s

    Create histogram
    here we are scaling under the assumption that there will not be any speeds greater 
    than 2.5 m/s (i.e., more constrained than above)
    by setting the same lower and upper limit, the histograms are comparable
    """
    xlowerlim = 0
    xupperlim = 2.5
    nbins = int(10 * (xupperlim - xlowerlim) / 0.5)  # this sets the number of bins

    """ Here bins start with 0 to 0.05 m/s. That represents a displacement of 180 m so there is
    likely true motion within that bin, but much of it is likely positional error.
    Dalton examined a stationary gps beacon and found mean + 1 SD = 0.04 m/s.
    """

    # calcuate bins and values
    values, bins = np.histogram(
        track.data.speed[~np.isnan(track.data.speed)],
        range=(xlowerlim, xupperlim),
        bins=nbins,
    )
    # convert to fraction of all obs
    frac = values / len(track.data.speed[~np.isnan(track.data.speed)])
    cumfrac = np.cumsum(frac)

    # here we set the y axis (left) limits - they represent fraction of obs within each bin
    ylowerlim = 0
    yupperlim = 100

    h.set_xlabel("Speed (m/s)")
    h.set_ylabel("Observations (%)", color="red")

    h.hist(bins[:-1], bins, weights=frac * 100, color="red")
    h.tick_params(axis="y", labelcolor="red")
    h.set_ylim([ylowerlim, yupperlim * 1.01])
    h.set_xlim([xlowerlim, xupperlim])

    # plot exceedence hist on right y axis
    h2 = h.twinx()

    h2.set_ylabel("Exceedence probability (%)", color="blue", rotation=270, labelpad=15)

    # since the cumulative fraction is descending, we need to use the stairs function.
    h2.stairs(
        100 - cumfrac * 100,
        bins,
        color="blue",
    )
    h2.tick_params(axis="y", labelcolor="blue")
    h2.set_ylim([ylowerlim, yupperlim * 1.01])

    # Calculate the number of observations to right edge to the histogram and print that
    out_of_range = sum(track.data.speed > xupperlim)
    if out_of_range > 0:
        plt.text(
            0.4,
            95,
            f"Warning: {out_of_range} values exceed {xupperlim} m/s",
        )

    # Polar plot of speed and direction
    p.plot(
        np.deg2rad(track.data.direction),
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
    d.grid(ls="dotted")
    sns.lineplot(
        ax=d, x="datetime_data", y="distance", data=track.data, errorbar=None, color="r"
    )
    d.set(xlabel=None, ylabel="Displacement (m)")
    d.set_ylim(
        0,
        track.data.distance.quantile(0.99),
    )
    # quiver plot
    q.grid(ls="dotted")
    u = track.data.speed * np.sin(np.radians(track.data.direction))
    v = track.data.speed * np.cos(np.radians(track.data.direction))

    time_numeric = mdates.date2num(track.data["datetime_data"])
    q_ax = q.quiver(
        time_numeric,
        np.zeros_like(time_numeric),
        u,
        v,
        scale=1,  # scale 1 will scale arrows the same as q.set_ylim()
        angles="uv",  # select uv since you are plotting along time (not space)
        scale_units="y",  # scale as per the q.set_ylim
        width=0.002,  # width of the arrow (keep small)
    )
    # this sets the y axis limits and therefore the scale of the arrows
    # setting this to 90th percentile
    q.set_ylim(-1 * track.data.speed.quantile(0.99), track.data.speed.quantile(0.99))

    q.set(xlabel=None, ylabel="Velocity (m/s)")

    # 2 reference arrows
    if track.data.speed.quantile(0.99) >= 1:
        ref_arrow = 0.2
    if track.data.speed.quantile(0.99) < 1:
        ref_arrow = 0.1
    if track.data.speed.quantile(0.99) < 0.1:
        ref_arrow = 0.02
    q.quiverkey(
        q_ax,
        X=0.9,
        Y=0.9,
        U=ref_arrow,
        label=f"{ref_arrow} m/s",
        color="grey",
        labelpos="E",
    )

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

    # plot the track trimming points so it can be verified before trimming.
    if not track.trimmed:
        if not pd.isnull(track.trim_start):
            t.axvline(track.trim_start, linestyle="dashdot", color="lime")
            d.axvline(track.trim_start, linestyle="dashdot", color="lime")
            q.axvline(track.trim_start, linestyle="dashdot", color="lime")
        if not pd.isnull(track.trim_end):
            t.axvline(track.trim_end, linestyle="dashdot", color="tab:orange")
            d.axvline(track.trim_end, linestyle="dashdot", color="tab:orange")
            q.axvline(track.trim_end, linestyle="dashdot", color="tab:orange")

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


if __name__ == "__main__":
    main()
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
