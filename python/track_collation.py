# -*- coding: utf-8 -*-
"""
track_collation.py

Batch process many tracks and/or collate many tracks into a larger database

Note you must have a metadata file and a beacon specs file to run this

The file structure of the raw data folders (scandir) is preserved.  Some scripts below
(commented out for now) can be used to move files into a different arrangement.

Save plots, geospatial data, etc.
Save log/logs

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

# import ITBD modules
import track_processing
from itbd import Meta, Models

# turn on the copy-on-write functionality
pd.options.mode.copy_on_write = True


def combine_data(src, outname):
    """
    Walk through source folder looking for standard csv files, concatenate and output.

    Parameters
    ----------
    src : TYPE
        DESCRIPTION.
    outname : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    alltrack_data = pd.DataFrame()
    for root, dirs, files in os.walk(src):
        # only match files in the lowest part of the dir tree (where no subfolders exist)
        if not dirs:
            # Filter files by extension
            matching_files = [f for f in files if f.endswith(".csv")]
            for f in matching_files:
                # Get the relative path from source folder
                trk_data = pd.read_csv(os.path.join(root, f))
                alltrack_data = pd.concat([alltrack_data, trk_data]).reset_index(
                    drop=True
                )

    # output the data
    alltrack_data.to_csv(
        Path(outname).with_suffix(".csv"),
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

    trackpoints.to_file(Path(outname + "_pt").with_suffix(".gpkg"), driver="GPKG")
    tracklines.to_file(Path(outname + "_ln").with_suffix(".gpkg"), driver="GPKG")

    return alltrack_data


def file_jockey(src, dest, ext, levelup=0, copy=True):
    """
    Move files of a certain type to a new directory.

    Defaults to copy for safety
    levelup =0 for the same dir structure as the source folder
    levelup =1 for putting files from all tracks together by year
    levelup =2 for putting files from all tracks in the same folder

    Warning: No clobber protection

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
        0 = maintain exact structure
        1 = move files up one directory level
        2 = move files up two directory levels
        The default is 0.
    copy : bool, optional
        If True, copy files instead of moving them.
        The default is True.N. The default is True.

    Returns
    -------
    None.

    """
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


# def database_stats(itbd_df, metadata, modeldata):
#     """


#     Parameters
#     ----------
#     alltrack_data : TYPE
#         DESCRIPTION.
#     metadata_file : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """


# ###IMPORTANT - I changed this to read from a file so removed .df from metadata -
# ### we want the most up to date info (from the concatenated metadata not the input metadata)


#     # number of tracks
#     print(
#         f"The ITBD contains {len(itbd_df.beacon_id.unique()):,} tracks comprised of a total of {len(itbd_df):,} positions"
#     )
#     print(f"Data span from {itbd_df.datetime_data.min()} to {itbd_df.datetime_data.max()}")

#     # number of groups and projects
#     print(
#         f"A total of {len(metadata.data_owner.unique()):,} groups from government, \
#     industry and academia contributed data from {len(metadata.project.unique()):,} different projects."
#     )

#     # # for unique data owners
#     # table data owner, number of beacons, list years
#     track_count = metadata.groupby("data_owner").agg("beacon_id").count().reset_index()
#     unique_years = (
#         metadata.groupby("data_owner")["year"]
#         .agg(lambda x: sorted(x.unique()))
#         .reset_index()
#     )
#     table_data_owners = pd.concat([track_count, unique_years], axis=1)

#     table_data_owners["year"] = table_data_owners["year"].apply(
#         lambda x: ",".join(map(str, map(int, x)))
#     )

#     table_data_owners.to_csv(f"{Path(outdir) / 'table_data_owners.csv'}", index=False)

#     # # for named targets
#     # table target name, source, target type, number of beacons, year of deployments

#     # this table will need to be sorted out manually.  Need to remove named targets of little
#     # consequence and lump

#     targets = (
#         metadata.groupby("iceberg_name")
#         .agg(
#             years=("year", list),
#             sources=("iceberg_source", "first"),
#             beacons=("beacon_id", "count"),
#             type=("shape", list),
#             size=("size", list),
#         )
#         .reset_index()
#     )
#     targets["years"] = targets["years"].apply(lambda x: ",".join(map(str, map(int, x))))

#     targets.to_csv(f"{Path(outdir) / 'table_targets.csv'}", index=False)

#     # table tracks vs model / manufacturer
#     model_df = modeldata[["make", "model"]]
#     model_df = metadata.df.groupby("beacon_model").agg(number=("beacon_id", "count"))

#     pd.concat([track_count, years], axis=1).to_csv("table_data_owners.csv", index=False)


# -------------------------------------------------------
# Complete this to set up runtime parameters (all hard coded)

# The run name will be the name of the log and metadata file
run_name = "20250306"

# path to the metadata file
meta_file = (
    "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/database/metadata.ods"
)

# path to the model specs file
spec_file = "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/database/models.ods"

# directory to look through for data
scandir = "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/raw_data"

# put the output data here
outdir = "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/" + run_name

# log file name
logfile = run_name

level = 2  # set to level 2, if you are working out of the raw data folder (files are 2 levels below the raw_data dir)

ans = input(
    f"Note this will overwrite files in the folder {outdir} press y to continue, or any other key to quit: "
)

if ans.lower() != "y":
    print("Quitting at user request.... ")
    sys.exit()

# go down the tree..
# at each level, create the same folder structure as we have in scandir
# at each level, get basename of the files and compare with meta data file beacon_id
# if match, then run the proceess.
# put the output in the copied folder structure

scandir = os.path.abspath(scandir)
outdir = os.path.abspath(outdir)
prefix = len(scandir) + len(os.path.sep)

if not os.path.isdir(outdir):
    os.makedirs(outdir)

log = track_processing.tracklog(logfile, outdir, level="INFO")
log.info("Starting run.....")

metadata = Meta(meta_file, log)
modeldata = Models(spec_file, log)

alltrack_meta = pd.DataFrame()


for root, dirs, files in os.walk(scandir, topdown=True):
    dirs.sort()  # this will keep the processing more ordered
    for d in dirs:
        if not os.path.isdir(os.path.join(outdir, root[prefix:], d)):
            os.mkdir(os.path.join(outdir, root[prefix:], d))
    files.sort()  # this will keep the processing more ordered
    for f in files:

        if Path(f).stem in metadata.df.beacon_id.values:
            print(f"\n\n......{f}.......")

            log = track_processing.tracklog(
                Path(os.path.join(root, f)).stem,
                os.path.join(outdir, root[prefix:]),
                level="INFO",
            )

            trk_meta = track_processing.track_process(
                os.path.join(root, f),
                os.path.join(outdir, root[prefix:]),
                metadata=metadata,
                specs=modeldata,
                output_file=None,
                output_types=["csv", "pt_kml", "ln_kml", "pt_gpkg", "ln_gpkg"],
                output_plots=["trim", "map", "dist", "time"],
                interactive=False,
                raw_data=True,
                trim_check=False,
            )

            # add track metadata to the dataframe of all metadata
            alltrack_meta = pd.concat([alltrack_meta, trk_meta]).reset_index(drop=True)

# output the metadata

alltrack_meta.to_csv(os.path.join(outdir, f"{run_name}.csv"), index=False, na_rep="NA")
# TODO - add na_rep to data

os.makedirs(Path(outdir + "_tmp"))
os.makedirs(Path(outdir + "_tmp") / "data")
os.makedirs(Path(outdir + "_tmp") / "geodata")
os.makedirs(Path(outdir + "_tmp") / "figures")
os.makedirs(Path(outdir + "_tmp") / "figures/maps")
os.makedirs(Path(outdir + "_tmp") / "figures/time")
os.makedirs(Path(outdir + "_tmp") / "figures/trim")
os.makedirs(Path(outdir + "_tmp") / "figures/dist")


file_jockey(outdir, Path(outdir + "_tmp") / "data", "csv", levelup=level)
file_jockey(outdir, Path(outdir + "_tmp") / "geodata", "gpkg", levelup=level)
file_jockey(outdir, Path(outdir + "_tmp") / "figures/maps", "map.png", levelup=level)
file_jockey(outdir, Path(outdir + "_tmp") / "figures/time", "time.png", levelup=level)
file_jockey(outdir, Path(outdir + "_tmp") / "figures/dist", "dist.png", levelup=level)
file_jockey(outdir, Path(outdir + "_tmp") / "figures/trim", "trim.png", levelup=level)

# move data from tmp to actual folder
alldata = combine_data(Path(outdir + "_tmp") / "data", run_name)

# ----

# ### crashed so pick up where we left off:
# # read col 2 as string (that's datetime transmit btw)
# alldata = pd.read_csv(f"{Path(outdir) / 'alldata.csv'}", dtype={2: str})
# metadata = pd.read_csv(f"{Path(outdir) }/{run_name}.csv")
# metadata_file = metadata
# itbd_df = alldata
