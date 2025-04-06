# -*- coding: utf-8 -*-
"""
track_collation.py

Batch process many tracks and/or collate many tracks into a larger database

Note you must have a track metadata file and a beacon model file to run this

The file structure of the raw data folders (scandir) is preserved. Some functions can
be used to move files into a different arrangement.

Author: Derek Mueller Jan 2025
"""

# import
import os
import sys
import shutil
import pandas as pd
from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# import IBTD modules
import track_process
from ibtd import Meta, Models

# turn on the copy-on-write functionality
pd.options.mode.copy_on_write = True


def combine_data(indir, outdir, run_name):
    """
    Walk through source folder looking for standard track data csv files, concatenate and output.

    Note that this function creates 3 files:
        1) a csv of all track points found
        2) a geopackage file of all the track points
        3) a geopackage file of all the track lines

    Assumes that all csv files in the lowest directories in the tree are standard track
    data files.  This will be the case after running the main part of this script.


    Parameters
    ----------
    indir : str
        Directory to scan for standard track data csv files.
    outdir : str
        Directory to put output files.
    run_name : str
        Basename of the output files.

    Returns
    -------
    alltrack_data : GeoPandas.GeoDataFrame
        A geodatafame of all the track points and attributes.

    Example
    -------
    Scan the processed data directory, combine all tracks, and save outputs with basename "ibtd_v1"
    combined_data = combine_data("/path/to/processed_tracks", "/path/to/output", "ibtd_v1")

    This will create:
    - /path/to/output/ibtd_v1.csv (all track points)
    - /path/to/output/ibtd_v1_pt.gpkg (geopackage of track points)
    - /path/to/output/ibtd_v1_ln.gpkg (geopackage of track lines)

    """
    alltrack_data = pd.DataFrame()
    for root, dirs, files in os.walk(indir):
        # only match files in the lowest part of the dir tree (where no subfolders exist)
        if not dirs:
            # Filter files by extension  - note that this assumes all csv files are standard track data!!
            matching_files = [f for f in files if f.endswith(".csv")]
            for f in matching_files:
                # Get the relative path from source folder
                trk_data = pd.read_csv(os.path.join(root, f))
                alltrack_data = pd.concat([alltrack_data, trk_data]).reset_index(
                    drop=True
                )

    # output the data
    alltrack_data.to_csv(
        (Path(outdir) / run_name).with_suffix(".csv"),
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S%:z",
    )

    trackpoints = gpd.GeoDataFrame(
        alltrack_data,
        geometry=gpd.points_from_xy(
            alltrack_data["longitude"], alltrack_data["latitude"]
        ),
    )

    # Set CRS
    trackpoints.crs = "EPSG:4326"

    # Convert to line
    tracklines = trackpoints.groupby(["beacon_id"])["geometry"].apply(
        lambda x: LineString(x.tolist())
    )
    # Set CRS
    tracklines.crs = "EPSG:4326"

    # output files
    trackpoints.to_file(
        (Path(outdir) / (run_name + "_pt")).with_suffix(".gpkg"), driver="GPKG"
    )
    tracklines.to_file(
        (Path(outdir) / (run_name + "_ln")).with_suffix(".gpkg"), driver="GPKG"
    )

    return alltrack_data


def file_mover(src, dest, ext, levelup=0, copy=True):
    """
    Move files of a certain type to a new directory, and optionally simplify the tree.

    Note that you cannot have the dest directory inside the src directory.

    Levelup is used to simpliy the directory structure:
    levelup =0 put files in the same dir structure as the source folder (<dest>/year/track/track_file)
    levelup =1 put files from all tracks together by year (<dest>/year/track_files)
    levelup =2 put files from all tracks in the same dest folder (<dest>/track_files)

    Warning: No clobber protection. Defaults to copy (not move) for safety.

    Parameters
    ----------
    src : str or Path
        Source directory path containing the files to be moved/copied
    dest : str or Path
        Destination directory path where files will be moved/copied to
    ext : str
        File extension to match (e.g., '.png', '.csv' or even 'map.png', etc)
    levelup : int, optional
        Number of directory levels to move up when recreating structure.
        The default is 0.
    copy : bool, optional
        If True, copy files instead of moving them. The default is True

    Returns
    -------
    None.

    """
    if Path(src) in Path(dest).parents:
        print("Select a directory that is not inside your source path (src)")
        sys.exit()

    # Walk through the source directory
    for root, dirs, files in os.walk(src):
        # only match files in the lowest part of the dir tree (where no subfolders exist)
        if not dirs:
            # Filter files by extension
            matching_files = [f for f in files if f.endswith(ext)]
            for f in matching_files:
                # Get the relative path from source folder
                if levelup == 0:
                    relative_path = Path(root).relative_to(src)
                else:
                    relative_path = Path(root).parents[levelup - 1].relative_to(src)

                # Create the same directory structure in destination
                new_dir = os.path.join(dest, relative_path)
                os.makedirs(new_dir, exist_ok=True)

                # Construct full paths
                f_src = os.path.join(root, f)
                f_dest = os.path.join(new_dir, f)

                # Move the file
                if copy:
                    shutil.copy(f_src, f_dest)
                else:
                    shutil.move(f_src, f_dest)


def database_stats(ibtd_df, metadata, modeldata, run_name, outdir):
    """
    Output summary statistics, tables and figure about the Database for reporting.

    In particular, draft (unformatted) tables are ouptut for the Documentation along with
    a map of all the tracks.

    -"Tracks contributed to the Database by organization"
    -"Tracks by source of iceberg in the Database"
    -"Beacon model characteristics and number of tracks"
    -"Iceberg drift tracks in the Iceberg Beacon Track Database version 1"

    Parameters
    ----------
    ibtd_df : Geopandas.GeoDataFrame
        The entire database - track point data.
    metadata : IBTD.Meta
        An instance of Meta, the track metadata class for the Database.
    modeldata : IBTD.Models
        An instance of Models class for the Database.
    run_name : str
        The name of the database run.
    outdir : str
        The output directory

    Returns
    -------
    None.

    """
    # check to see if the file you need is there:
    if not os.path.isfile(f"{Path(outdir) / run_name}_ln.gpkg"):
        print("\nYou need to run the combine_data function to create a trackline file")
        print("\nMake sure to use the same values for run_name and outdir")
        sys.exit(1)

    # First print out some generic info about the database

    # number of tracks and
    print(
        f"The IBTD contains {len(ibtd_df.beacon_id.unique()):,} tracks comprised of a total of {len(ibtd_df):,} positions"
    )
    print(
        f"Data span from {ibtd_df.datetime_data.min()} to {ibtd_df.datetime_data.max()}"
    )

    # number of groups and projects
    print(
        f"A total of {len(metadata.df.data_contributor.unique()):,} groups from government, \
    industry and academia contributed data from {len(metadata.df.project.unique()):,} different projects."
    )

    """ 
    Table for report.  "Tracks contributed to the Database by organization"
    
    Columns :  Organization, Tracks, Deployment years
    
    Note that the table can be reorganized by sector post-export (Gov't, University, Other) 
    
    """

    # # for unique data owners

    # table data contributor, number of beacons
    track_count = (
        metadata.df.groupby("data_contributor").agg("beacon_id").count().reset_index()
    )
    # table data contributor, list years
    unique_years = (
        metadata.df.groupby("data_contributor")["year"]
        .agg(lambda x: sorted(x.unique()))
        .reset_index()
    )

    # merge tables together
    table_tracks_contributed = track_count.merge(unique_years)

    # convert list to string
    table_tracks_contributed["year"] = table_tracks_contributed["year"].apply(
        lambda x: ",".join(map(str, map(int, x)))
    )

    # rename columns
    table_tracks_contributed.columns = ["Organization", "Tracks", "Deployment years"]

    # output table
    table_tracks_contributed.to_csv(
        f"{Path(outdir) / 'table_contributed_tracks.csv'}", index=False
    )

    """ 
    Table for report.  "Tracks by source of iceberg in the Database"
    
    Columns :  Iceberg source, Deployment years, Selected iceberg names, Tracks
    
    Note that the table rows should be sorted, select iceberg names, and give ranges for years 
    
    
   """

    # change from NA to "Unknown"
    source_df = metadata.df.copy()
    source_df["iceberg_source"] = source_df["iceberg_source"].fillna("Unknown")
    source_df["iceberg_name"] = source_df["iceberg_name"].fillna("Unknown")

    # grouping operation
    sources = (
        source_df.groupby("iceberg_source")
        .agg(
            years=("year", list),
            iceberg_name=("iceberg_name", list),
            tracks=("beacon_id", "count"),
        )
        .reset_index()
    )

    # convert lists to strings
    sources["iceberg_name"] = sources["iceberg_name"].apply(
        lambda x: ",".join(map(str, x))
    )
    sources["years"] = sources["years"].apply(lambda x: ",".join(map(str, map(int, x))))

    # rename columns
    sources.columns = [
        "Iceberg source",
        "Deployment years",
        "Selected iceberg names",
        "Tracks",
    ]

    # output
    sources.to_csv(f"{Path(outdir) / 'table_sources_tracks.csv'}", index=False)

    """ 
    Table for report.  "Beacon model characteristics and number of tracks"
    
    Columns :  Beacon make, Beacon model, Tracks, Transmitter, Air deployable, Buoyant, then 
    all the capabilities
    
    After export, remove 'Default' and format the column header text to be vertical
        
    """

    # work on a copy to be sure
    df = modeldata.df.copy()

    # set the deployment by air column
    df["Air deployable"] = 0.0
    df.loc[df["deployment"].str.contains("by air"), "Air deployable"] = 1.0

    # select columns
    cols = [
        "make",
        "model",
        "transmitter",
        "Air deployable",
        "buoyant",
        "temperature_int",
        "temperature_surface",
        "temperature_air",
        "voltage",
        "pressure",
        "pitch",
        "roll",
        "heading",
    ]
    df = df[cols]

    # convert all columns to string and make the true values checkmarks
    # the false and no data should be blank
    df = df.astype(str)
    df = df.replace("1.0", "✓")
    df = df.replace("nan", "")
    df = df.replace("0.0", "")

    # get the number of tracks per beacon model
    model_df = metadata.df.groupby("model").agg(number=("beacon_id", "count"))

    # prepare for merge
    model_df.reset_index(inplace=True)
    model_df.columns = ["model", "Tracks"]
    model_df.Tracks = model_df.Tracks.astype(int)

    df = df.merge(model_df, how="left", left_on="model", right_on="model")

    # sort rows
    df.sort_values(["make", "model"], inplace=True)

    # rename columns
    df.columns = [
        "Make",
        "Model",
        "Transmitter",
        "Air deployable",
        "Buoyant",
        "Internal temp.",
        "Surface temp.",
        "Air temp.",
        "Battery voltage",
        "Air pressure",
        "Pitch",
        "Roll",
        "Heading",
        "Tracks",
    ]

    # sort columns
    df = df[
        [
            "Make",
            "Model",
            "Tracks",
            "Transmitter",
            "Air deployable",
            "Buoyant",
            "Internal temp.",
            "Surface temp.",
            "Air temp.",
            "Battery voltage",
            "Air pressure",
            "Pitch",
            "Roll",
            "Heading",
        ]
    ]

    # output
    df.to_csv(f"{Path(outdir) / 'table_beacon.csv'}", index=False)

    """ 
    Figure for report.  "Iceberg drift tracks in the Iceberg Beacon Track Database version 1"
   
    """

    coast = cfeature.NaturalEarthFeature(
        "physical", "land", "10m", edgecolor="black", facecolor="lightgray", lw=0.5
    )

    # make a figure - square shape
    fig = plt.figure(figsize=(5, 5), constrained_layout=True)

    # polar stereo - generic - Canada up.
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(-100))
    # Orthographic centred on data
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.Orthographic(
            ibtd_df["longitude"].median(), ibtd_df["latitude"].median()
        ),
    )
    # plot the tracks, one at a time, with different colours
    i = 0  # counter for colour
    for track, trackdata in ibtd_df.groupby("beacon_id"):
        # print(f"{track}, index {i}, colour C{i % 10}")

        # plot the line data
        ax.plot(
            trackdata["longitude"],
            trackdata["latitude"],
            color=f"C{i % 10}",
            lw=0.3,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        i += 1  # counter add

    # set the extent to the data
    ax.set_extent(
        [
            ibtd_df["longitude"].min(),
            ibtd_df["longitude"].max(),
            ibtd_df["latitude"].min(),
            ibtd_df["latitude"].max(),
        ],
        ccrs.PlateCarree(),
    )

    # add coast
    ax.add_feature(coast, zorder=2)

    # this is needed to make the map centred and not too narrow and tall
    # ax.set_aspect("equal")  # try auto too?

    # add grid
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        color="black",
        alpha=0.25,
        linestyle="dotted",
        x_inline=False,
        y_inline=False,
        zorder=3,
    )
    gl.rotate_labels = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xpadding = 5

    # output
    fig.savefig(
        os.path.join(outdir, f"{run_name}_alldata_map.png"),
        dpi=600,
        transparent=False,
        bbox_inches="tight",
    )
    plt.close()


def main():
    """
    Run script with the following hardcoded parameters.

    Returns
    -------
    None.

    """
    # -------------------------------------------------------
    # Complete this to set up runtime parameters (all hard coded)

    # The run name will determine the folder name and the base name of the log and metadata files
    run_name = "20250406"

    # path to the metadata file
    meta_file = "/ibtd/metadata/track_metadata_raw.ods"

    # path to the model specs file
    spec_file = "/ibtd/metadata/beacon_specs.ods"

    # directory to look through for raw data
    scandir = "/ibtd/raw_data"

    # put the output data here
    outdir = "/ibtd/" + run_name

    # if true, the log file will be sent to the raw data folder, it will go to the outdir otherwise
    log2raw = True

    level = 2  # set to level 2, if you are working out of the raw data folder (files are 2 levels below the raw_data dir)

    # this just warns that data will be clobbered... last chance!
    ans = input(
        f"Note this will overwrite files in the folder {outdir} press y to continue, or any other key to quit: "
    )

    if ans.lower() != "y":
        print("Quitting at user request.... ")
        sys.exit()

    # set up the folders
    scandir = os.path.abspath(scandir)
    outdir = os.path.abspath(outdir)
    prefix = len(scandir) + len(os.path.sep)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # set up logging
    logfile = run_name
    log = track_process.tracklog(logfile, outdir, level="INFO")
    log.info("Starting run.....")

    # open the metadata files.
    metadata = Meta(meta_file, log)
    modeldata = Models(spec_file, log)

    # create an empty dataframe for all the metadata output for all tracks
    alltrack_meta = pd.DataFrame()

    # go down the tree..
    # at each level, recreate the same folder structure as we have in scandir but in the outdir
    # at each level, get basename of the files and compare with meta data file platform_id
    # if a match, then run the proceess.
    # put the output in the copied folder structure

    for root, dirs, files in os.walk(scandir, topdown=True):
        dirs.sort()  # this will keep the processing more ordered
        for d in dirs:
            if not os.path.isdir(os.path.join(outdir, root[prefix:], d)):
                os.mkdir(os.path.join(outdir, root[prefix:], d))

        files.sort()  # this will keep the processing more ordered
        for f in files:

            # ignore all log files
            if Path(f).suffix == ".log":
                continue

            else:
                # if this condition is met, the file is assumed to be data... proceed...
                if Path(f).stem in metadata.df.beacon_id.values:
                    print(f"\n\n......{f}.......")

                    # set up a log file for this track either in the outdir or the raw data folder
                    if log2raw:
                        log = track_process.tracklog(
                            Path(os.path.join(root, f)).stem,
                            os.path.join(outdir, root),
                            level="INFO",
                        )

                    else:
                        log = track_process.tracklog(
                            Path(os.path.join(root, f)).stem,
                            os.path.join(outdir, root[prefix:]),
                            level="INFO",
                        )

                    # process this track with the settings below.
                    trk_meta = track_process.process(
                        os.path.join(root, f),
                        os.path.join(outdir, root[prefix:]),
                        metadata=metadata,
                        specs=modeldata,
                        output_name=None,
                        output_types=["csv", "pt_kml", "ln_kml", "pt_gpkg", "ln_gpkg"],
                        output_plots=["trim", "map", "dist", "time"],
                        interactive=False,  # set to False unless you have nothing better to do today
                        raw_data=True,  # set to True for database collation
                        trim_check=False,  # set to False for database collation
                        meta_verbose=True,  # set to True for database collation
                        meta_export="json",
                    )

                    # add track metadata to the dataframe of all metadata
                    alltrack_meta = pd.concat([alltrack_meta, trk_meta]).reset_index(
                        drop=True
                    )

    # output the metadata
    alltrack_meta.to_csv(
        os.path.join(outdir, f"{run_name}_meta.csv"), index=False, na_rep="NA"
    )

    # combine data for the stats
    alldata = combine_data(Path(outdir), Path(outdir), run_name)

    # output summary stats
    database_stats(alldata, metadata, modeldata, run_name, outdir)

    file_mover(outdir, Path(outdir).parent / (run_name + "_data"), "csv", levelup=level)
    file_mover(
        outdir,
        Path(outdir).parent / (run_name + "_geodata/gpkg"),
        "gpkg",
        levelup=level,
    )
    file_mover(
        outdir, Path(outdir).parent / (run_name + "_geodata/kml"), "kml", levelup=level
    )
    file_mover(
        outdir,
        Path(outdir).parent / (run_name + "_figures/maps"),
        "map.png",
        levelup=level,
    )
    file_mover(
        outdir,
        Path(outdir).parent / (run_name + "_figures/time"),
        "time.png",
        levelup=level,
    )
    file_mover(
        outdir,
        Path(outdir).parent / (run_name + "_figures/dist"),
        "dist.png",
        levelup=level,
    )
    file_mover(
        outdir,
        Path(outdir).parent / (run_name + "_figures/trim"),
        "trim.png",
        levelup=level,
    )


if "__name__" == "__main__":
    main()
