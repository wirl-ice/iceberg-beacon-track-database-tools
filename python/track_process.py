#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_proceessing.py

Stand-alone script to process tracks for the IBTD (Iceberg Beacon Track Database)

--or--

If you have standardized data already, it can be used to make plots and output data

Author: Derek Mueller Jan 2025

"""
import os
import sys
import argparse
import logging
from pathlib import Path

# custom modules
from ibtd import Track, Meta, Specs, nolog


def tracklog(platform_id, path_output, level="INFO"):
    """
    Set up logging.

    The level is the standard logging levels with DEBUG being the most verbose, INFO
    being the recommended level and WARNING capturing only bad news.

    Parameters
    ----------
    platform_id : str
        the beacon id - used for naming log
    path_output : str
        path to the output file
    level : str, optional
        the logging level for the file. The default is "INFO".

    Returns
    -------
    track_log : logging.Logger
        An instance of the logger class

    """
    # name the logger after the platform_id
    track_log = logging.getLogger(platform_id)

    # if not track_log.handlers:
    #     # Create a name for the file
    loggerFileName = f"{platform_id}.log"
    loggerFileName = os.path.join(path_output, loggerFileName)

    # assign the log level here, defaults to INFO, note there is no ERROR or CRITICAL level
    match level.lower():
        case "debug":
            loglevel = logging.DEBUG
        case "warning":
            loglevel = logging.WARNING
        case _:
            loglevel = logging.INFO

    # Remove all handlers associated with the logger (avoids duplicates)
    if track_log.hasHandlers():
        track_log.handlers.clear()

    track_log.setLevel(logging.DEBUG)  # sets the default level for this logger

    # Create handlers - these control output from the logger
    # stream handler - output to the console
    c_handler = logging.StreamHandler()
    # file handler - output to a file
    # Note that mode = 'w' sets logger to overwrite (not append).
    # For some applications (testing speed_limit thresholds for example) you may want to change this
    f_handler = logging.FileHandler(loggerFileName, mode="w")

    # this sets the logging level for both handlers:
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(loglevel)

    # Create formatters and add them to handlers - this gives control over output
    c_format = logging.Formatter("%(message)s")
    f_format = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(name)s - %(message)s"
    )
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    track_log.addHandler(c_handler)
    track_log.addHandler(f_handler)

    # Prevent propagation to avoid duplicate logs
    track_log.propagate = False

    return track_log


def read_args():
    """
    Read arguments from command line, checks them, reads some files.

    This function facilitates the command line operation of the workflow. Note that most
    of the arguments are not required, since they have defaults. There are some choices:
        The user can specify the reader, beacon model, trim start and end times (for trimming),
        or leave them blank (which may limit what steps can be accomplished),
        or they can provide the path to the track metadata file. If this file is present, the
        keywords mentioned above will be overwritten.

    Returns
    -------
    list
    A list containing the following arguments in order:
    - data_file : str
        Path to the track data file.
    - output_path : str
        Directory path where output files will be written.
    - metadata : Meta object or None
        Metadata object containing track information if provided.
    - reader : str
        Name of the reader function to use.
    - model : str
        Beacon model name.
    - specs : Models object or None
        Object containing model specifications if provided.
    - trim_start : str or None
        Timestamp to trim the start of the track in UTC format.
    - trim_end : str or None
        Timestamp to trim the end of the track in UTC format.
    - output_name : str or None
        Name for the output file.
    - output_types : list or None
        List of output file types to produce.
    - output_plots : list or None
        List of plot types to generate.
    - interactive : bool
        Whether to create interactive plots.
    - trim_check : bool
        Whether to check trim points without applying trimming.
    - raw_data : bool
        Whether the input is raw data (not standardized).
    - meta_export : str
        Format for metadata export ('pandas', 'json', or 'both').
    - log : str
        chose 'quiet' for no loggging, 'info' for regular logging or 'debug' for verbose logging.


    """
    prog_description = """Beacon track processing functions
    
    For reading-in standard data files:
        - do not include the -rd (--raw_data) flag 
        - -r (--reader) will be ignored and reader listed in the meta_file will be ignored
        - -m (--model) will be ignored
        - -sf (--spec_file) will be ignored
        - track can be re-trimmed as required (be sure to add -s and -e values)
        - file and plot outputs can be requested
        
    For reading-in, standardizing and processing raw data: 
        - include the -rd (--raw_data) flag 
        - the reader must be specified (-r or listed in the track metadata file -mf)
        - trimming (trim_start and/or trim_end) must be listed (-s -e) or track metadata file
        - file and plot outputs can be requested
    
    Example: read standard data file 2021_300434065868240.csv, trim it to a 4 day sub-section 
        and output a map and a csv file in the current directory: 
        >python track_process.py 2021_300434065868240.csv . -s '2021-08-22 13:00:00' -e '2021-08-26 21:00:00' -op map -ot

    Example: read standard data file 2011_300234010031950.csv, output a map, timeseries 
    plot and kml in the parent directory:     
        >python track_process.py 2011_300234010031950.csv .. -op map time -ot ln_kml

    For more information see github. 
        
    """

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
        help="provide the name of the reader function (default is 'standard', which is used with processed data)",
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
        "--trim_start",
        type=str,
        default=None,
        help="timestamp to trim the start of the track in UTC (format: yyyy-mm-dd HH:MM:SS)",
    )
    parser.add_argument(
        "-e",
        "--trim_end",
        type=str,
        default=None,
        help="timestamp to trim the end of the track in UTC (format: yyyy-mm-dd HH:MM:SS)",
    )
    parser.add_argument(
        "-mf",
        "--meta_file",
        type=str,
        default=None,
        help="the path/name of the metadata csv file. Note that the script will OVERWRITE \
            arguments reader, model, trim_start, trim_end with values in this file",
    )
    parser.add_argument(
        "-of",
        "--output_name",
        type=str,
        default=None,
        help="the name of the standardized fully processed track file: defaults to platform_id",
    )
    parser.add_argument(
        "-ot",
        "--output_types",
        type=str,
        nargs="+",
        choices={"csv", "pt_kml", "ln_kml", "pt_gpkg", "ln_gpkg"},
        help="list the output types to produce",
    )
    parser.add_argument(
        "-op",
        "--output_plots",
        type=str,
        nargs="+",
        choices={"map", "time", "dist", "trim"},
        help="list the output plots to produce",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="create interactive plots; defaults to non-interactive plots",
    )
    parser.add_argument(
        "-t",
        "--trim_check",
        action="store_true",
        help="set -t to check where the trim points _would be_ on the map, time and \
            trim plot; defaults to false",
    )
    parser.add_argument(
        "-rd",
        "--raw_data",
        action="store_true",
        help="set -rd to work with raw data (not standard); defaults to false",
    )
    parser.add_argument(
        "-me",
        "--meta_export",
        help="Specify whether to export metadata in the current directory in 'pandas' \
            or 'json' format or 'both'. The default (None) does not export a file.",
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        choices={"quiet", "info", "debug", "warning"},
        default="info",
        help="chose 'quiet' for no loggging, 'info' for regular logging or 'debug' for verbose logging.",
    )

    args = parser.parse_args()

    # all the arguments here:
    data_file = args.data_file
    output_path = args.output_path
    reader = args.reader
    model = args.model
    spec_file = args.spec_file
    trim_start = args.trim_start
    trim_end = args.trim_end
    meta_file = args.meta_file
    output_name = args.output_name
    output_types = args.output_types
    output_plots = args.output_plots
    interactive = args.interactive
    trim_check = args.trim_check
    raw_data = args.raw_data
    meta_export = args.meta_export
    loglevel = args.loglevel

    # some attempt at error trapping early on....
    if not os.path.isfile(data_file):
        Exception(f"Data file: {data_file} was not found. Please check and run again")
        sys.exit(1)

    if not os.path.isdir(output_path):
        Exception(
            f"Output path: {output_path} was not found. Please check and run again"
        )
        sys.exit(1)

    if os.path.basename(data_file) == output_name:
        if Path(data_file).parent == output_path:
            Exception(
                "The output file you specified will overwrite the raw data file, please fix and re-run"
            )
            sys.exit(1)

    return [
        data_file,
        output_path,
        meta_file,
        reader,
        model,
        spec_file,
        trim_start,
        trim_end,
        output_name,
        output_types,
        output_plots,
        interactive,
        trim_check,
        raw_data,
        meta_export,
        loglevel,
    ]


def read_meta_files(spec_file, meta_file, log=None):
    """
    Read track metadata file and beacon spec file.

    Reads in the metadata and specs files

    Parameters
    ----------
    spec_file : str
        Full path to the beacon_specs file.
    meta_file : str
        Full path to the track_metadata file.
    log : logger instance
        A logger instance.  The default is None.

    Returns
    -------
    metadata : Meta object, optional
        Metadata object containing track information. The default is None.
    specs : Models object, optional
        Object containing model specifications. The default is None.

    """
    if spec_file:
        if not os.path.isfile(spec_file):
            log.error(
                f"Spec file: {spec_file} was not found. Please check and run again"
            )
            sys.exit(1)

    if meta_file:
        if not os.path.isfile(meta_file):
            log.error(
                f"Meta file: {meta_file} was not found. Please check and run again"
            )
            sys.exit(1)

    # if there is a meta_file, then open it and replace the following parameters
    if meta_file:
        metadata = Meta(meta_file, log)
    else:
        metadata = None

    # read in the spec file
    if spec_file:
        specs = Specs(spec_file, log)
    else:
        specs = None

    return metadata, specs


def process(
    data_file,
    output_path,
    metadata=None,
    reader=None,
    model=None,
    specs=None,
    trim_start=None,
    trim_end=None,
    output_name=None,
    output_types=None,
    output_plots=None,
    interactive=False,
    trim_check=False,
    raw_data=False,
    meta_export=None,
    meta_verbose=False,
    log=None,
):
    """
    Process a raw track: standardize, purge, trim and output.

    Parameters
    ----------
    data_file : str
        Path to the track data file to be processed.
    output_path : str
        Path where output files will be written.
    metadata : Meta object, optional
        Metadata object containing track information. The default is None.
    reader : str, optional
        Name of the reader function to use. The default is None.
    model : str, optional
        Beacon model name. The default is None.
    specs : Models object, optional
        Object containing model specifications. The default is None.
    trim_start : str, optional
        Timestamp to trim the start of the track in UTC format. The default is None.
    trim_end : str, optional
        Timestamp to trim the end of the track in UTC format. The default is None.
    output_name : str, optional
        Name for the output file. The default (None) will result in naming based on platform_id.
    output_types : list, optional
        List of output file types to produce. The default is None.
    output_plots : list, optional
        List of plot types to generate. The default is None.
    interactive : bool, optional
        Whether to create interactive plots. The default is False.
    trim_check : bool, optional
        Whether to check trim points without applying trimming. The default is False.
    raw_data : bool, optional
        Whether the input is raw data (not standardized). The default is False.
    meta_export : str, optional
        Format for metadata export ('pandas', 'json', or 'both'). The default is None.
    meta_verbose : bool, optional
        Whether to include all available metadata fields in the track metadata. If True,
        metadata categories like processing details and valid range information are included.
        The default is False.
    log : logger instance
        A logger instance.  The default is None.

    Returns
    -------
    trk_meta : pandas Dataframe
        Track metadata (one row) in a dataframe.

    """
    log.info(f"~Processing {Path(data_file).stem}....")

    log.debug(f"Data file: {data_file}")
    log.debug(f"Output path: {output_path}")
    if metadata:
        log.debug("Metadata file read-in")
    else:
        log.debug("No metadata file")
        log.debug("Reader: {reader}")
        log.debug("Model: {model}")

    if specs:
        log.debug("Beacon specs read-in")
    else:
        log.debug("No beacon specs available")

    log.debug(f"Trim start: {trim_start}")
    log.debug(f"Trim end: {trim_end}")
    log.debug(f"Output name: {output_name}")
    log.debug(f"Output types: {output_types}")
    log.debug(f"Output plots: {output_plots}")
    log.debug(f"Interactive: {interactive}")
    log.debug(f"Trim check: {trim_check}")
    log.debug(f"Raw data: {raw_data}")
    log.debug(f"Meta export: {meta_export}")
    log.debug(f"Meta verbose: {meta_verbose}")

    if metadata and raw_data:
        trk = Track(
            data_file, metadata=metadata, raw_data=raw_data, logger=log
        )  # get reader from metadata file
    elif reader and raw_data:
        trk = Track(
            data_file,
            reader=reader,
            model=model,
            trim_start=trim_start,
            trim_end=trim_end,
            raw_data=raw_data,
            logger=log,
        )  # get reader from user
    elif not raw_data:
        trk = Track(
            data_file,
            reader="standard",
            model=model,
            trim_start=trim_start,
            trim_end=trim_end,
            logger=log,
        )  # this is not raw data
    else:
        log.error(
            "You must specify a valid metadata object or reader function to read a raw data track"
        )

    # note that the steps purge, sort, speed, speed_limit and trim are for processing raw data
    # if you have standard data, it is possible to trim it to a specific period, as desired
    # the script runs sort, speed and speed_limit regardless (doesn't take much time/ can't hurt)
    # if you want to refine the speedlimit, then that can be done by adding a new limit here
    # like: trk.speed_limit(1.5)

    if (
        specs and raw_data
    ):  # no point purging bad data without the beacon specs, only raw data needs purging
        trk.load_model_specs(specs)
        trk.purge()
    trk.sort()
    # if you want to see where the trim points are, don't run trim
    if not trim_check:
        trk.trim()

    trk.speed()
    trk.speed_limit()

    # # if you want to see where the trim points are, don't run trim
    # if not trim_check:
    #     trk.trim()

    # output the track files
    trk.output(output_types, path_output=output_path, file_name=output_name)

    # generate figures
    if output_plots:
        if "map" in output_plots:
            trk.plot_map(interactive=interactive, path_output=output_path)
        if "trim" in output_plots:
            trk.plot_trim(interactive=interactive, path_output=output_path)
        if "time" in output_plots:
            trk.plot_time(interactive=interactive, path_output=output_path)
        if "dist" in output_plots:
            trk.plot_dist(interactive=interactive, path_output=output_path)

    # create a trk_meta pandas dataframe:
    trk_meta = trk.track_metadata(
        path_output=output_path, meta_export=meta_export, verbose=meta_verbose
    )

    # complete the run.
    log.info("Completed track processing... \n")
    return trk_meta


def main():
    """Run main function."""
    (
        data_file,
        output_path,
        meta_file,
        reader,
        model,
        spec_file,
        trim_start,
        trim_end,
        output_name,
        output_types,
        output_plots,
        interactive,
        trim_check,
        raw_data,
        meta_export,
        loglevel,
    ) = read_args()

    if loglevel == "quiet":
        log = nolog()
    else:
        log = tracklog(Path(data_file).stem, output_path, level=loglevel)

    metadata, specs = read_meta_files(spec_file, meta_file, log)

    process(
        data_file,
        output_path,
        metadata=metadata,
        reader=reader,
        model=model,
        specs=specs,
        trim_start=trim_start,
        trim_end=trim_end,
        output_name=output_name,
        output_types=output_types,
        output_plots=output_plots,
        interactive=interactive,
        raw_data=raw_data,
        trim_check=trim_check,
        meta_export=meta_export,
        meta_verbose=True,
        log=log,
    )


if __name__ == "__main__":
    main()
