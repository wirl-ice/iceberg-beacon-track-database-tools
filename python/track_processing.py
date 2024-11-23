#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_proceessing.py

Stand-alone script to process tracks for the ITBD.  

"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd

# custom modules
from itbd import Track, Meta, Models, nolog


def tracklog(beacon_id, path_output, level="DEBUG"):
    """
    Set up the logging.

    Parameters
    ----------
    beacon_id : str
        the beacon id - used for naming log
    path_output : str
        path to the output file
    level : str, optional
        the logging level for the file. The default is "DEBUG".

    Returns
    -------
    track_log : logging.Logger
        An instance of the logger class

    """
    # create a name for the file
    loggerFileName = f"{beacon_id}.log"

    # add full path here so it goes to the right place
    loggerFileName = os.path.join(path_output, loggerFileName)

    # assign the log level here, defaults to DEBUG, note there is no CRITICAL level
    match level.lower():
        case "debug":
            loglevel = logging.DEBUG
        case "info":
            loglevel = logging.INFO
        case "warning":
            loglevel = logging.WARNING
        case "error":
            loglevel = logging.ERROR
        case _:
            loglevel = logging.DEBUG

    # Create a logger instance here - it will be named after the module name (__main__?)
    track_log = logging.getLogger()

    # Remove all handlers associated with the logger (avoids duplicates)
    if track_log.hasHandlers():
        track_log.handlers.clear()

    track_log.setLevel(logging.DEBUG)  # sets the default level for this logger

    # Create handlers - these control output from the logger
    # stream handler - output to the console
    c_handler = logging.StreamHandler()
    # file handler - output to a file
    f_handler = logging.FileHandler(loggerFileName)

    # this sets the logging level for both handlers:
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(loglevel)

    # Create formatters and add them to handlers - this gives control over output
    c_format = logging.Formatter("%(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    track_log.addHandler(c_handler)
    track_log.addHandler(f_handler)

    return track_log


def read_args():
    """
    Read arguments from command line, checks them, reads certain files.

    This function facilitates the command line operation of the workflow. Note that most
    of the arguments are not required, since they have defaults. There are some choices:
        The use can specify the reader, beacon model, track start and end times for trimming,
        or leave them blank (which may limit what steps can be accomplished),
        or provide the path to the metadata file.  If the metdata file is present, the
        arguments mentioned above will be overwritten.

    Returns
    -------
    a list of the arguments.

    """
    prog_description = "Beacon track standardization and cleaning functions"
    parser = argparse.ArgumentParser(prog_description)

    # these are the required parameters - all others have default values
    parser.add_argument("data_file", help="enter the track data file")
    parser.add_argument(
        "output_path", help="enter the path to write the output file to"
    )
    # These keywords are not obligatory.
    parser.add_argument(
        "-r",
        "--reader",
        type=str,
        default="standard",
        help="provide the name of the reader function",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="Default",
        help="the beacon model name (must be exact)",
    )
    parser.add_argument(
        "-sf",
        "--spec_file",
        type=str,
        default=None,
        help="the path/name of the model specifications file",
    )
    parser.add_argument(
        "-s",
        "--track_start",
        type=str,
        default=None,
        help="the date-time of the track start in UTC (format: yyyy-mm-dd HH:MM:SS) for trimming",
    )
    parser.add_argument(
        "-e",
        "--track_end",
        type=str,
        default=None,
        help="the date-time of the track end in UTC (format: yyyy-mm-dd HH:MM:SS) for trimming",
    )
    parser.add_argument(
        "-mf",
        "--meta_file",
        type=str,
        default=None,
        help="the path/name of the metadata csv file. Note that the script will seek \
            reader, model, track_start, track_end in the meta_file",
    )
    parser.add_argument(
        "-of",
        "--output_file",
        type=str,
        default=None,
        help="the name of the standardized fully processed track file",
    )
    parser.add_argument(
        "-ot",
        "--output_types",
        type=str,
        default="csv",
        nargs="+",
        choices={"csv", "pt_kml", "ln_kml", "pt_gpkg", "ln_gpkg"},
        help="list the output types to produce:  ; defaults to producing csv only",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="set this to turn off logging to screen and file",
    )

    args = parser.parse_args()

    # all the arguments here:
    data_file = args.data_file
    output_path = args.output_path
    reader = args.reader
    model = args.model
    spec_file = args.spec_file
    track_start = args.track_start
    track_end = args.track_end
    meta_file = args.meta_file
    output_file = args.output_file
    output_types = args.output_types
    quiet = args.quiet

    # some attempt at error trapping early on....
    assert os.path.isfile(
        data_file
    ), f"Data file: {data_file} was not found. Please check and run again"
    assert os.path.isdir(
        output_path
    ), f"Output path: {output_path} was not found. Please check and run again"
    if spec_file:
        assert os.path.isfile(
            spec_file
        ), f"Spec file: {spec_file} was not found. Please check and run again"
    if meta_file:
        assert os.path.isfile(
            meta_file
        ), f"Meta file: {meta_file} was not found. Please check and run again"

    if os.path.basename(data_file) == output_file:
        if Path(data_file).parent == output_path:
            print(
                "The output file you specified will overwrite the raw data file, please fix and re-run"
            )
            sys.exit(1)

    # if there is a meta_file, then open it and replace the following parameters
    if meta_file:
        metadata = Meta(meta_file)
    else:
        metadata = None

    # read in the spec file
    if spec_file:
        specs = Models(spec_file)
    else:
        specs = None

    return [
        data_file,
        output_path,
        metadata,
        reader,
        model,
        specs,
        track_start,
        track_end,
        output_file,
        output_types,
        quiet,
    ]


def track_process(
    data_file,
    output_path,
    metadata=None,
    reader=None,
    model=None,
    specs=None,
    track_start=None,
    track_end=None,
    output_file=None,
    output_types=["csv"],
):
    """
        Process a raw track: standardize, clean, trim and output.

    #TODO write this properly
        Parameters
        ----------
        data_file : str
            path to the track data file.
        output_path : TYPE
            DESCRIPTION.
        reader : TYPE
            DESCRIPTION.
        model : TYPE
            DESCRIPTION.
        specs_df : TYPE
            DESCRIPTION.
        track_start : TYPE
            DESCRIPTION.
        track_end : TYPE
            DESCRIPTION.
        output_file : TYPE
            DESCRIPTION.
        output_types : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

    """
    log = logging.getLogger()
    log.info(f"\n~Processing {Path(data_file).stem}....\n")

    if metadata:
        trk = Track(data_file, metadata=metadata, logger=log)
    elif reader:
        trk = Track(data_file, reader=reader, logger=log)
    else:
        log.error(
            "You must specify a valid metadata object or reader function to read a track"
        )

    if specs:  # no point cleaning without the beacon specs
        trk.load_model_specs(specs)
        trk.clean()
    trk.sort()
    trk.speed()
    trk.speed_limit()
    # trk.trim()
    trk.output(output_types, path_output=output_path, file_output=output_file)
    trk_meta = trk.track_metadata("pandas")
    log.info("Completed track processing... \n")
    return trk_meta


def main():
    """
    Main function.

    """
    import pdb

    pdb.set_trace()
    # read in arguments from the command line.
    (
        data_file,
        output_path,
        metadata,
        reader,
        model,
        specs,
        track_start,
        track_end,
        output_file,
        output_types,
        quiet,
    ) = read_args()

    if quiet:
        log = nolog()
    else:
        log = tracklog(Path(data_file).stem, output_path, level="INFO")

    track_process(
        data_file,
        output_path,
        metadata,
        reader,
        model,
        specs,
        track_start,
        track_end,
        output_file,
        output_types,
    )


if __name__ == "__main__":
    main()


# # Problem IDs:
# 2009_300034012571050
# 2010_300034012592660
