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
from ibtd import Track, Meta, Models, nolog


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

    # Create a logger instance here - it will be named after the module name
    track_log = logging.getLogger()

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
        The user can specify the reader, beacon model, track start and end times for trimming,
        or leave them blank (which may limit what steps can be accomplished),
        or provide the path to the metadata file.  If the metdata file is present, the
        arguments mentioned above will be overwritten.

    Returns
    -------
    a list of the arguments in order

    """
    prog_description = """Beacon track processing functions
    
    For reading-in standard data files:
        - do not include the -rd (--raw_data) flag 
        - -r (--reader) will be ignored and reader listed in the meta_file will be ignored
        - -m (--model) will be ignored
        - -sf (--spec_file) will be ignored
        - track can be re-trimmed as required
        - file and plot outputs can be requested
        
    For reading-in, standardizing and cleaning raw data: 
        - include the -rd (--raw_data) flag 
        - the reader must be specified (-r or listed in the meta_file -mf)
        - trimming (trim_start and/or trim_end) must be listed or in meta_file
        - file and plot outputs can be requested
    
    Example: read standard data file 2021_300434065868240.csv and output a map in the current directory: 
        >python track_processing.py 2021_300434065868240.csv . -2011-08-08 13:00:00 2011-08-12 21:00:00 -op map

    Example: read standard data file 2011_300234010031950.csv, trim it and output a map, time plot and kml 
    in the parent directory:     
        >python track_processing.py 2011_300234010031950.csv .. -s '2011-08-08 13:00:00' -e '2011-08-12 21:00:00' -op map time -ot ln_kml

    For more info see github readme.
        
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
        help="provide the name of the reader function (default is 'standard' clean data)",
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
        "-op",
        "--output_plots",
        type=str,
        default="map",
        nargs="+",
        choices={"map", "time", "dist", "trim"},
        help="list the output plots to produce; defaults to producing map only",
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
        help="set -t to check where the trim points _would be_ on the map and trim plot; defaults to false",
    )
    parser.add_argument(
        "-rd",
        "--raw_data",
        action="store_true",
        help="set -rd to work with raw data; defaults to false",
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
    trim_start = args.trim_start
    trim_end = args.trim_end
    meta_file = args.meta_file
    output_file = args.output_file
    output_types = args.output_types
    output_plots = args.output_plots
    interactive = args.interactive
    trim_check = args.trim_check
    raw_data = args.raw_data
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
        trim_start,
        trim_end,
        output_file,
        output_types,
        output_plots,
        interactive,
        trim_check,
        raw_data,
        quiet,
    ]


def track_process(
    data_file,
    output_path,
    metadata=None,
    reader=None,
    model=None,
    specs=None,
    trim_start=None,
    trim_end=None,
    output_file=None,
    output_types=["csv"],
    output_plots=None,
    interactive=False,
    trim_check=False,
    raw_data=False,
    meta_verbose=False,
):
    """
         Process a raw track: standardize, purge, trim and output.

        Parameters
        ----------
        data_file : str
            DESCRIPTION.
        output_path : str
            DESCRIPTION.
        metadata : str, optional
            DESCRIPTION. The default is None.
        reader : TYPE, optional
            DESCRIPTION. The default is None.
        model : TYPE, optional
            DESCRIPTION. The default is None.
        specs : TYPE, optional
            DESCRIPTION. The default is None.
        trim_start : TYPE, optional
            DESCRIPTION. The default is None.
        trim_end : TYPE, optional
            DESCRIPTION. The default is None.
        output_file : TYPE, optional
            DESCRIPTION. The default is None.
        output_types : TYPE, optional
            DESCRIPTION. The default is ["csv"].
        output_plots : TYPE, optional
            DESCRIPTION. The default is None.
        interactive : TYPE, optional
            DESCRIPTION. The default is False.
        trim_check : TYPE, optional
            DESCRIPTION. The default is False.
        raw_data : TYPE, optional
            DESCRIPTION. The default is False.

    TODO add meta output
        Returns
        -------
        None.

    """
    log = logging.getLogger()
    log.info(f"\n~Processing {Path(data_file).stem}....\n")

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
            trim_start=trim_start,  # TODO not sure what to do here
            trim_end=trim_end,
            logger=log,
        )  # this is not raw data
    else:
        log.error(
            "You must specify a valid metadata object or reader function to read a raw data track"
        )

    # note that the steps purge, trim, sort, speed, and speed_limit are for raw data
    # if you have standard data, it is possible to trim it to a specific period, as desired
    # the script runs sort, speed and speed_limit regardless (doesn't take much time/ can't hurt)

    if (
        specs and raw_data
    ):  # no point purging bad data without the beacon specs, only raw data needs purging
        trk.load_model_specs(specs)
        trk.purge()
    trk.sort()
    trk.speed()
    trk.speed_limit()

    # if you want to see where the trim points are, don't run trim
    if not trim_check:
        trk.trim()

    # output the track files
    trk.output(output_types, path_output=output_path, file_output=output_file)

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

    # create a trk_meta object:
    trk_meta = trk.track_metadata(
        path_output=output_path, meta_export="json", verbose=meta_verbose
    )

    # complete the run.
    log.info("Completed track processing... \n")
    return trk_meta


def main():
    """Run main function."""
    (
        data_file,
        output_path,
        metadata,
        reader,
        model,
        specs,
        trim_start,
        trim_end,
        output_file,
        output_types,
        output_plots,
        interactive,
        raw_data,
        trim_check,
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
        trim_start,
        trim_end,
        output_file,
        output_types,
        output_plots,
        interactive,
        raw_data,
        trim_check,
    )


if __name__ == "__main__":
    main()
