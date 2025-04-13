# -*- coding: utf-8 -*-
"""
ibtd.py

Main module for the Iceberg Beacon Track Database (IBTD).

Defines the class:
    'Track' containing an individual iceberg track along with methods and properties.

    'Meta' containing the track metadata

    'Models' containing beacon model specifications, read in from a table

    'Specs' containing a single beacon model specifications, used for purging bad data

Note that creating an instance of a Track that _is_ in the standard format assumes the track
 has been processed, which means all the steps in a workflow for standardizing, purging, filtering
 and adding derived data are complete.

The functions to read the various raw_data formats and define the standard format are in track_readers.py
The functions to plot figures are in track_fig.py

The Database itself is created using this code base.  To (re-)create the Database use track_collate.py

Author: Derek Mueller Jul 2024-Apr 2025, with contribution from Adam Garbo's code
"""
# imports
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
import pyproj
from collections import namedtuple
import json
import copy
import datetime

# this functionality is in other modules.
import track_readers
from track_fig import plot_trim, plot_map, plot_dist, plot_time

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


def json_serialize(value):
    """
    Check to see if the value is a type that json can't serialize and, if so, convert it.

    In practice that means converting datetime to string and numpy boolean to regular T/F

    Parameters
    ----------
    value : Any variable
        A variable.

    Returns
    -------
    value
        a value in a format json can serialize

    """
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)


class Specs:
    """
    Class that holds info/specifications for all beacon models.

    Currently this is the valid range (min, max) of various sensors and some beacon attributes
    but this could be expanded to hold any data related to the model.


    """

    def __init__(self, specs_file, logger=None):
        """
        Read beacon specification file and create a dataframe.

        Parameters
        ----------
        model_file : str
            Full path to the model_file (*.ods or *.xls or *.xlsx).

        """
        if logger is None:
            logger = nolog()
        self.log = logger

        self.specs_file = specs_file
        try:
            df = pd.read_excel(specs_file)

        except:
            pass
            self.log.error(f"Beacon specifications file {self.specs_file} read error")
        self.df = df
        self.log.debug(f"Model specifications file {self.specs_file} read")


class Meta:
    """Class that reads a metadata file and stores all rows in a dataframe."""

    def __init__(self, meta_file, logger=None):
        """
        Read metadata file and create a dataframe.

        Parameters
        ----------
        meta_file : str
            Full path to the beacon metadata file. (*.xls, *.xlsx, *.ods)
        logger: instance of logger class
            Pass a logger here if you want

        Returns
        -------
        None.

        """
        if logger is None:
            logger = nolog()
        self.log = logger

        self.log.debug(f"Initializing Meta instance from {meta_file}")

        self.meta_file = meta_file

        try:
            df = pd.read_excel(meta_file)
        except:
            self.log.error(f"Failed to read {self.meta_file}, exiting... ")
            raise Exception(f"Failed to read {self.meta_file}")
        self.df = df
        self.log.info(f"Track metadata file {self.meta_file} read")


class Track:
    """
    Class representing an iceberg beacon track.

    This track could be a raw data file or a track that has been processed into a
    the standard format.

    Tracks have many properties and methods for data purging, filtering, inspection and analysis.
    """

    def __init__(
        self,
        data_file,
        reader="standard",
        model=None,
        trim_start=None,
        trim_end=None,
        metadata=None,
        raw_data=False,
        logger=None,
    ):
        """
        Read the track raw or standardized data.

        Note the default is to read standardized data (which assumes it was fully processed)
        If track metadata is provided, that info will be used; otherwise properties
        will be set from keywords here

        Parameters
        ----------
        data_file : str
            path to the raw/standard data file.
        reader : str, optional
            name of the reader function to use. The default is "standard".
        model : str, optional
            name of the beacon model. The default is None.
        trim_start : str, optional
            timestamp to trim the start of the track. The default is None.
        trim_end : str, optional
            timestamp to trim the end of the track. The default is None.
        metadata : Meta Class, optional
            An object of the Meta class. The default is None.
        raw_data : bool, optional
            Set to true if you are working with raw data.  The default is False.
        logger : logger instance, optional
            A logger instance to log to. The default is None.

        Returns
        -------
        None.

        """
        if logger == None:
            logger = nolog()
        self.log = logger  # a log instance is now part of the class

        self.log.debug(f"Initializing track instance from {data_file}")

        self.datafile = data_file
        self.platform_id = Path(self.datafile).stem
        self.year, self.id = self.platform_id.split("_")

        # raw_data flag
        self.raw_data = raw_data

        # if metadata is not given then set properties
        if metadata == None:
            self.reader = reader
            self.model = model
            self.trim_start = trim_start
            self.trim_end = trim_end

        else:
            self.load_metadata(metadata)

        # this simply overrides the reader column in the metadata
        if not self.raw_data:
            self.reader = "standard"

        self.log.info(
            f"Reading data from {self.datafile} using the {self.reader} reader"
        )

        reader_function = getattr(track_readers, self.reader)
        self.data = reader_function(self.datafile, self.log)

        self.log.info(f"Track rows: {len(self.data)} - after data file read")

        # at a minimum this step should be taken since a track must have all 3 of these
        # Drop all rows where timestamp, latitude or longitude is nan
        self.log.info(
            f'Track rows: {self.data[["timestamp", "latitude", "longitude"]].isnull().sum().sum()} rows removed. NAs found in timestamp or position'
        )
        self.data.dropna(subset=["timestamp", "latitude", "longitude"], inplace=True)

        self.log.info(f"Track rows: {len(self.data)} - after NAs removed")

        # check here to see if there are any data in the track.
        if len(self.data) < 1:
            self.log.error(
                f"No data read from {self.datafile} using the {self.reader} reader"
            )
            raise Exception(
                f"No data read from {self.datafile} using the {self.reader} reader"
            )

        # make sure these datetimes convert ok
        if self.trim_start:
            try:
                self.trim_start = pd.to_datetime(self.trim_start, utc=True)
            except:
                self.log.error("Unrecognized trim start format")
                raise Exception("Check trim start value")

        if self.trim_end:
            try:
                self.trim_end = pd.to_datetime(self.trim_end, utc=True)
            except:
                self.log.error("Unrecognized track end format")
                raise Exception("Check track end value")

        # This captures the data file start and end
        # note that this value *may* not match the actual file data since some reader functions
        # remove bad data beforehand
        # The data_start and _end won't change from here on.
        self.data_start = self.data.timestamp.min()
        self.data_end = self.data.timestamp.max()

        # in case the trim_start or trim_end is not set, set it to the widest range possible
        # report the difference between the data_start/end and the trim_start/end
        if self.trim_start == None:
            self.trim_start = self.data.timestamp.min()
            self.log.warning(
                "trim_start not specified, the track_start was set to the time of the first valid data point (data_start)"
            )
        else:
            delta = self.trim_start - self.data.timestamp.min()
            if delta.total_seconds() < 0:
                self.log.info(
                    f"trim_start was set to {abs(delta)} before the first valid data point (data_start)"
                )
            elif delta.total_seconds() == 0:
                self.log.info(
                    "trim_start was set exactly to the first valid data point (data_start)"
                )
            else:
                self.log.info(
                    f"trim_start was set to {delta} after the first valid data point (data_start)"
                )

        if self.trim_end == None:
            self.trim_end = self.data.timestamp.max()
            self.log.warning(
                "trim_end not specified, the track_end was set to the time of the last valid data point (data_end)"
            )
        else:
            delta = self.trim_end - self.data.timestamp.max()
            if delta.total_seconds() < 0:
                self.log.info(
                    f"trim_end was set to {abs(delta)} before the last valid data point (data_end)"
                )
            elif delta.total_seconds() == 0:
                self.log.info(
                    "trim_end was set exactly to the last valid data point (data_end)"
                )
            else:
                self.log.info(
                    f"trim_end was set to {delta} after the last valid data point (data_end)"
                )

        # These will be added later - after track is geod.
        self.trackpoints = None
        self.trackline = None

        # These properties record what has been done to the track.
        self.purged = False
        self.sorted = False
        self.speeded = False
        self.speedlimited = False
        self.trimmed = False
        self.geoed = False

        # if the reader is standard then assume this has been done previously.
        # if you want to re-process data set the reader to standard and the raw_data to True
        if self.reader == "standard" and not self.raw_data:
            self.purged = True
            self.sorted = True
            self.speeded = True
            self.speedlimited = True
            self.trimmed = True

        # refresh track limits and track stats.
        self.refresh_stats()

        # some outputs
        if self.raw_data:
            self.log.info(
                f"Raw data read-in with {self.observations} rows of valid data from {self.data_start} to {self.data_end}"
            )
        else:
            self.log.info(
                f"Standard data read-in with {self.observations} rows of valid data from {self.data_start} to {self.data_end}"
            )

    def refresh_stats(self, speed=True):
        """
        Refresh track limits, stats, speed calcs and geodata.

        Do this after any changes in the dataframe. If you are running this after the speed
        function set speed=False or it will loop forever.

        1) Ensure track_start and track_end represent the current range of valid track data.

        Also, in theory the trim_start should not be << data_start and the trim_end should
        not be >> data_end.  If that is so, warn about this in the log.

        This could be due to an operator error (trim value for the wrong track for example)
        But this does not necessarily mean there is an issue! This can happen when the
        beacon is deployed but not activated or when data at the start or end of the
        track is deemed non-valid by various purging/filtering functions.

        2) Rerun the speed method which calculates the speed_wrt_ground, displacement and course
            between positions.

        3) Recreate the geospatial tracklines and trackpoints for the track.

        4) Calculate track stats, whcih includes the observations, duration and distance travelled
            for the track, as well as the starting and ending latitude and longitude.

        Be sure to run this after all purging/filtering steps after sorting the data.

        Parameters
        ----------
        speed : bool, optional
            If true refresh_stats will recalculate speed. The default is True.

        Returns
        -------
        None.

        """
        self.log.debug("Refreshing track stats")

        # now set the track range
        self.track_start = self.data.timestamp.min()
        self.track_end = self.data.timestamp.max()

        # reset the index
        self.data.reset_index(drop=True, inplace=True)

        # recalculate the speed and displacement
        if self.speeded and speed:
            self.speed()
        # self.speed_limit() # Assuming this is not needed here but leaving comment to flag this step
        # if you have geospatial data, it must be recreated
        if self.geoed:
            self.geo()

        # populate some simple properties that all tracks have
        duration = self.track_end - self.track_start

        self.duration = round(duration.days + duration.seconds / (24 * 60 * 60), 2)
        self.observations = len(self.data.index)

        # this only works after some processing or if the data are standard
        # note this will change when trimmed too...
        if self.sorted:
            self.latitude_start = self.data.latitude.iloc[0]
            self.longitude_start = self.data.longitude.iloc[-1]
            self.latitude_end = self.data.latitude.iloc[0]
            self.longitude_end = self.data.longitude.iloc[-1]
        else:
            self.latitude_start = None
            self.longitude_start = None
            self.latitude_end = None
            self.longitude_end = None

        if self.speeded:
            self.distance_travelled = round(
                self.data["platform_displacement"].sum() / 1000, 2
            )
        else:
            self.distance_travelled = None

    def load_model_specs(self, Specs):
        """
        Load the specific specifications for this beacon and write them to track properties.

        Parameters
        ----------
        Specs : Specs class
            Specifications for all beacon models

        Returns
        -------
        None.

        """
        self.log.debug("Reading model specifications")

        if not self.model:
            self.log.error("Model not known. Check input")

        default_specs = Specs.df.loc[Specs.df.model == "Default"]
        beacon_specs = Specs.df.loc[Specs.df.model == self.model]

        if len(default_specs) != 1:
            msg = "Unkown Default model, check spelling -or- duplicate model retrieved"
            self.log.error(msg)
            raise Exception(msg)

        if len(beacon_specs) != 1:
            msg = f"Unkown model {self.model}, check spelling -or- duplicate model retrieved"
            self.log.error(msg)
            raise Exception(msg)

        # find which specs are meant to be 'default'and replace with default
        beacon_specs_ind = beacon_specs.loc[:].values == "?"
        beacon_specs_ind = np.argwhere(beacon_specs_ind.flatten()).tolist()
        beacon_specs_ind = [i for row in beacon_specs_ind for i in row]
        beacon_specs.iloc[0, beacon_specs_ind] = default_specs.iloc[0, beacon_specs_ind]

        # remove a few columns that are not needed
        beacon_specs = beacon_specs.drop(columns=["notes", "model"])

        # set to boolean
        beacon_specs = beacon_specs.astype(
            {
                "air_deployable": bool,
                "buoyant": bool,
                "has_internal_temperature": bool,
                "has_surface_temperature": bool,
                "has_air_temperature": bool,
                "has_air_pressure": bool,
                "has_platform_pitch": bool,
                "has_platform_roll": bool,
                "has_platform_orientation": bool,
                "has_voltage_battery_volts": bool,
            }
        )

        # put the contents of the specs df into the track properities
        for column in beacon_specs.columns:
            setattr(self, column, beacon_specs[column].iloc[0])

        # now coerce the data to floats, argo_position_accuracy are really integers, so try, but skip if NA
        specs = beacon_specs.filter(regex="_max|_min")
        specs = specs.astype(float)

        try:
            specs = specs.astype(
                {"argo_position_accuracy_max": int, "argo_position_accuracy_min": int}
            )
        except:
            pass
        # overwrite the properties above but with proper data types
        for column in specs.columns:
            setattr(self, column, specs[column].iloc[0])

    def load_metadata(self, Meta):
        """
        Read the metadata record for this track and overwrite track properties.

        Parameters
        ----------
        Meta : Meta object
            An instance of the class Meta representing a dataframe of metadata.

        """
        self.log.info("Reading track metadata")
        # filter records to find the data for this beacon
        record = Meta.df.loc[Meta.df.platform_id == self.platform_id]

        # check that one and only one record is returned
        if len(record) == 0:
            self.log.error("metadata not found, exiting....")
            raise Exception(f"The metadata for beacon {self.platform_id} was not found")

        if len(record) > 1:
            self.log.error("beacon metadata duplicated, exiting....")
            raise Exception(
                f"The metadata search for beacon {self.platform_id} returned duplicate records"
            )

        # load properties
        try:
            self.reader = record.reader.iloc[0]
        except:
            self.reader = "standard"

        try:
            self.model = record.model.iloc[0]
        except:
            self.model = "Default"

        try:
            self.trim_start = record.trim_start.iloc[0]
        except:
            self.trim_start = None

        try:
            self.trim_end = record.trim_end.iloc[0]
        except:
            self.trim_end = None

        self.log.info(f"Beacon model: {self.model}")
        self.log.info(
            f"trim_start: {self.trim_start} and trim_end: {self.trim_end} requested"
        )

        # convert to datetime and check for errors
        if self.trim_start:
            try:
                self.trim_start = pd.to_datetime(self.trim_start, utc=True)
            except:
                self.log.error(
                    "Unrecognized track start format - trimming will not work as expected"
                )

        if self.trim_end:
            try:
                self.trim_end = pd.to_datetime(self.trim_end, utc=True)
            except:
                self.log.error(
                    "Unrecognized track end format - trimming will not work as expected"
                )

        # next pull in all the remaining available metadata and store it as meta_dict
        record = record.drop(
            ["platform_id", "reader", "model", "trim_start", "trim_end"],
            axis=1,
            errors="ignore",
        )
        self.meta_dict = record.iloc[0].to_dict()

    def purge(self):
        """Purge bad data by assigning NaN to values that exceed the min/max range."""
        self.log.debug("Purging bad data from track")

        # this tests whether you have specs read in...
        if not self.make:
            self.log.error("No beacon specs are available, no data purging attempted")
            return

        # Latitude
        self.data.loc[
            (self.data["latitude"] > self.latitude_max)
            | (self.data["latitude"] < self.latitude_min),
            "latitude",
        ] = np.nan

        # Longitude
        self.data.loc[
            (self.data["longitude"] > self.longitude_max)
            | (self.data["longitude"] < self.longitude_min)
            | (self.data["longitude"] == 0),
            "longitude",
        ] = np.nan

        # Air temperature
        self.data.loc[
            (self.data["air_temperature"] > self.air_temperature_max)
            | (self.data["air_temperature"] < self.air_temperature_min),
            "air_temperature",
        ] = np.nan

        # Internal temperature
        self.data.loc[
            (self.data["internal_temperature"] > self.internal_temperature_max)
            | (self.data["internal_temperature"] < self.internal_temperature_min),
            "internal_temperature",
        ] = np.nan

        # Surface temperature
        self.data.loc[
            (self.data["surface_temperature"] > self.surface_temperature_max)
            | (self.data["surface_temperature"] < self.surface_temperature_min),
            "surface_temperature",
        ] = np.nan

        # Pressure
        self.data.loc[
            (self.data["air_pressure"] > self.air_pressure_max)
            | (self.data["air_pressure"] < self.air_pressure_min),
            "air_pressure",
        ] = np.nan

        # Pitch
        self.data.loc[
            (self.data["platform_pitch"] > self.platform_pitch_max)
            | (self.data["platform_pitch"] < self.platform_pitch_min),
            "platform_pitch",
        ] = np.nan

        # Roll
        self.data.loc[
            (self.data["platform_roll"] > self.platform_roll_max)
            | (self.data["platform_roll"] < self.platform_roll_min),
            "platform_roll",
        ] = np.nan

        # Heading
        self.data.loc[
            (self.data["platform_orientation"] > self.platform_orientation_max)
            | (self.data["platform_orientation"] < self.platform_orientation_min),
            "platform_orientation",
        ] = np.nan

        # Battery voltage
        self.data.loc[
            (self.data["voltage_battery_volts"] > self.voltage_battery_volts_max)
            | (self.data["voltage_battery_volts"] < self.voltage_battery_volts_min),
            "voltage_battery_volts",
        ] = np.nan

        # Drop data with poor accuracy (as specified in the specs)
        drop_index = self.data[
            (self.data["argo_position_accuracy"] > self.argo_position_accuracy_max)
            | (self.data["argo_position_accuracy"] < self.argo_position_accuracy_min)
        ].index

        if len(drop_index) > 0:
            self.data.drop(drop_index, inplace=True)
            self.log.info(
                f"Track rows: {len(drop_index)} rows ({len(drop_index)/len(self.data):.1%}) removed due to unacceptable argo_position_accuracy."
            )
            self.log.info(f"Track rows: {len(self.data)} - after argos location filter")

        # Drop all rows where timestamp, latitude or longitude is nan
        self.data.dropna(subset=["timestamp", "latitude", "longitude"], inplace=True)

        self.data = self.data.round(
            {
                "air_temperature": 2,
                "internal_temperature": 2,
                "surface_temperature": 2,
                "air_pressure": 2,
                "platform_pitch": 2,
                "platform_roll": 2,
                "platform_orientation": 2,
                "voltage_battery_volts": 2,
            }
        )

        self.purged = True
        self.log.info("Track bad data purged")

        # recalculate stats here since things may have changed
        self.refresh_stats()

    def sort(self):
        """
        Order the track chronologically and remove redundant entries.

        Some data formats are ordered in reverse, whereas others can have duplicates.
        This function takes care of these issues.

        """
        # sort by timestamp, and argo_position_accuracy if available. The best argo_position_accuracy is the highest number
        self.data.sort_values(["timestamp", "argo_position_accuracy"], inplace=True)
        # look for repeated values
        # sdf_dup = self.data.loc[self.data.duplicated(subset=["timestamp"], keep=False)] # all lines
        sdf_dup = self.data.loc[
            self.data.duplicated(subset=["timestamp"], keep="last")
        ]  # keep last dup
        if self.raw_data:
            self.log.info(
                f"Track rows: {len(sdf_dup)} rows removed due to duplicate timestamps"
            )
        # remove all rows with duplicate times, prefer the one with best location accuracy
        self.data.drop_duplicates(
            subset=["timestamp"], keep="last", inplace=True, ignore_index=True
        )

        # this should be true (check!)
        assert self.data[
            "timestamp"
        ].is_monotonic_increasing, "Issue with timestamps, sort data!"

        # recalculate stats here since things may have changed
        self.refresh_stats()

        self.sorted = True
        if self.raw_data:
            self.log.info(
                f"Track sorted, duplicates removed (if any). The track now has {self.observations} rows"
            )

    def speed(self):
        """Calculate displacement, speed and course bearing between iceberg positions.

        - platform_displacement : Distance from previous to current track position
        - platform_speed_wrt_ground : Speed from previous to current track position
        - platform_course :	Azimuth relative to true north from previous to current track position

        The speed is rounded to 3 decimal places and direction and distance are rounded
        to 0 decimal places which is plenty for sig figs. Since they are floating point values, they
        export like ##.0, implying a precision that does not exist.

        """
        # Ensure rows are sorted by datetime.
        assert self.data[
            "timestamp"
        ].is_monotonic_increasing, "Issue with timestamps, sort data!"

        # Initialize pyproj with appropriate ellipsoid
        geodesic = pyproj.Geod(ellps="WGS84")

        # Calculate forward azimuth and great circle distance between modelled coordinates
        self.data["platform_course"], backaz, self.data["platform_displacement"] = (
            geodesic.inv(
                self.data["longitude"].shift().tolist(),
                self.data["latitude"].shift().tolist(),
                self.data["longitude"].tolist(),
                self.data["latitude"].tolist(),
            )
        )

        # Convert azimuth from (-180° to 180°) to (0° to 360°)
        self.data["platform_course"] = (
            (self.data["platform_course"] + 360) % 360
        ).round(2)

        ## Note here that no displacement (same lat/lon repeated) yields platform_course 180
        ## and displacement 0 in the NH.  In SH it is course 0 and displacement 0 which might
        ## need to be considered if there is SH data.
        self.data.loc[self.data["platform_displacement"] == 0, "platform_course"] = (
            np.nan
        )

        # Calculate time delta between rows (in seconds)
        time_delta = self.data["timestamp"].diff().dt.total_seconds()

        # Calculate speed in m/s
        self.data["platform_speed_wrt_ground"] = (
            self.data["platform_displacement"] / time_delta
        )

        # Round columns
        # to assess whether there are consecutive duplicate positions, comment these lines

        self.data["platform_displacement"] = self.data["platform_displacement"].round(0)
        self.data["platform_course"] = self.data["platform_course"].round(0)
        self.data["platform_speed_wrt_ground"] = self.data[
            "platform_speed_wrt_ground"
        ].round(3)

        # set property
        self.speeded = True
        self.log.debug("Calculated displacement, direction and speed for track")

        # recalculate stats here since things may have changed
        self.refresh_stats(speed=False)

    def speed_limit(self, threshold=5):
        """
        Remove gross speeding violations from data.

        Note the intent here is to remove only the very worst rows from datasets.  It is
        a very crude way to cut down on _clearly wrong_ position data.  Note high speeds are
        often due to inaccurate positions, but also inprecise positions over short periods
        of time. (eg., an Argos position 1-2 min apart may exceed a threshold even if
        the precision is relatively good).

        It is important to be careful not to cut out good data.

        The default value here is 5 m/s or 18 kph or 432 km/d (this is very conservative
        to avoid throwing away data - especially for ARGOS beacons. It could easily be set
        lower for GNSS-based systems - likely 2 m/s would be fine)

        Note, if the speed limit is higher than allowed from point 1 to point 2 then it
        is implicitly assumed that the 2nd point is invalid, from the way the script works.
        This may not be true but finding out, is challenging. If this situation arises,
        a warning will be set.

        Parameters
        ----------
        threshold : float, optional
            A threshold, beyond which rows are removed (m/s). The default is 5.

        """
        # needs to be in a loop since if there is a fly-away point, you have going out and coming back
        before = len(self.data)
        while (self.data["platform_speed_wrt_ground"] > threshold).any():
            # if the first speed is over threshold, we need to investigate:
            if (
                self.data[self.data["platform_speed_wrt_ground"] > threshold].index[0]
                == 1
            ):
                self.log.warning(
                    "Track rows: Bad position might be the first row (rerun w/ debug log to see"
                )
            self.log.debug(
                f'Removing position at {self.data.loc[self.data["platform_speed_wrt_ground"] > threshold, "timestamp"].iloc[0]} due to speed limit violations'
            )
            self.data.drop(
                self.data[self.data["platform_speed_wrt_ground"] > threshold].index[0],
                inplace=True,
            )
            self.speed()
        self.log.info(
            f"Track rows: {before - len(self.data)} rows removed due to speeds > {threshold} m/s"
        )
        self.log.info(f"Track rows: {len(self.data)} - after speed filter")

        self.speedlimited = True

        # recalculate stats here since things may have changed
        self.refresh_stats()

    def trim(self):
        """
        Trim a track from data_start/end with trim_start/end to yield a track from track_start to _end.

        The trim_start and trim_end track properties are used to determine where to trim.
        These need to be provided intentionally (as an argument or read from track metadata)
        during the initialization of the track or they will be set to data_start and data_end

        Note that you will want to AFTER running the speed and speed limit, since the
        track_start and track_end will automatically be moved if bad data are found
        at the start and end of the track.

        Returns
        -------
        None.

        """
        if self.trim_start:
            self.data.drop(
                self.data[self.data["timestamp"] < self.trim_start].index,
                inplace=True,
            )
            self.log.info(
                f"Values at the start of the track from {self.data_start} up to {self.trim_start} were trimmed"
            )
            self.trimmed_start = True

        if self.trim_end:
            self.data.drop(
                self.data[self.data["timestamp"] > self.trim_end].index,
                inplace=True,
            )
            self.log.info(
                f"Values at the end of the track following {self.trim_end} to {self.data_end} were trimmed"
            )
            self.trimmed_end = True

        # Also flag that general trimming was done here.
        self.trimmed = True

        # recalculate stats here since things may have changed
        self.refresh_stats()
        self.log.info(f"Track rows: {len(self.data)} after trim")

    def geo(self):
        """
        Add a geodataframe of track points and a track line to the track object.

        This method converts track data into a geospatial points (self.trackpoints) and
        linestring (self.trackline).


        """
        # Convert to GeoPandas dataframe
        self.trackpoints = gpd.GeoDataFrame(
            self.data,
            geometry=gpd.points_from_xy(self.data["longitude"], self.data["latitude"]),
        )

        # Set CRS
        self.trackpoints.crs = "EPSG:4326"

        # Convert to line
        self.trackline = self.trackpoints.groupby(["platform_id"])["geometry"].apply(
            lambda x: LineString(x.tolist())
        )

        # Set CRS
        self.trackline.crs = "EPSG:4326"
        self.geoed = True
        self.log.info("Track geospatial data created")

    def output(self, types=["csv"], path_output=".", file_name=None):
        """
        Output the track to a file.

        Note the default is a csv (non-spatial) format.  Other options include track points
        (_pt) or track lines (_ln) in a gpkg or kml file [let's move on from shapefiles, eh!].
        See types option below.

        The script checks for an existing file. If one is there, that will be logged.
        The file will not be overwritten and data export will fail.

        Notes about data formats:
            - the csv file is the output format 'of record'
            - the gpkg _ln and _pt files are the recommended geospatial format. The _pt
              version contains and attribute table. Since the _pt data is a far larger
              file, it seemed like a good idea to keep _ln and _pt data separate.
            - the kml _ln and _pt files are meant for a quick look only (convienient to view):
                - there is no fancy symbology in the kml output.
                - the kml_pt output is restricted to platform_id and the timestamp.
                - sometimes the kml_pt file loads slowly.

        Parameters
        ----------
        types : list of output types to generate ['csv', 'pt_kml', 'ln_kml', 'pt_gpkg','ln_gpkg']. The default is 'csv'.
        path_output : str, optional
            Path to put the output. The default is the current directory
        file_name : str, optional
            filename of output. The default is None, which will autogenerate on the platform_id

        Returns
        -------
        None.

        """
        if not file_name:
            file_name = self.platform_id

        if types is None:
            return

        # test if the geo method was run or not.
        if not self.geoed:
            self.geo()

        # output part
        if "csv" in types:
            # Write CSV file without index column
            if not os.path.isfile(f"{os.path.join(path_output, file_name)}.csv"):
                self.data.to_csv(
                    f"{os.path.join(path_output, file_name)}.csv",
                    index=False,
                    # date_format='%Y-%m-%d %H:%M:%S', # easy to read natively with Excel/Libre
                    # date_format="%Y-%m-%dT%H:%M:%SZ", # one ISO8601 format
                    date_format="%Y-%m-%dT%H:%M:%S%:z",  # another ISO8601 format (python 3.12 and up)
                    na_rep="NA",  # Sets no data to NA
                )
                self.log.info("Track output as csv file")
            else:
                self.log.error("File already exists, writing as csv failed!")

        if "pt_gpkg" in types:
            if not os.path.isfile(f"{os.path.join(path_output, file_name)}_pt.gpkg"):
                self.trackpoints.to_file(
                    f"{os.path.join(path_output, file_name)}_pt.gpkg", driver="GPKG"
                )
                self.log.info("Track output as trackpoint gpkg file")
            else:
                self.log.error(
                    "File already exists, writing as trackpoint gpkg failed!"
                )

        if "ln_gpkg" in types:
            if not os.path.isfile(f"{os.path.join(path_output, file_name)}_ln.gpkg"):
                self.trackline.to_file(
                    f"{os.path.join(path_output, file_name)}_ln.gpkg", driver="GPKG"
                )
                self.log.info("Track output as trackline gpkg file")
            else:
                self.log.error("File already exists, writing as trackline gpkg failed!")

        if "pt_kml" in types:
            if not os.path.isfile(f"{os.path.join(path_output, file_name)}_pt.kml"):
                # note the name will be the beacon id and the description will be the timestamp.
                self.trackpoints[["platform_id", "timestamp", "geometry"]].to_file(
                    f"{os.path.join(path_output, file_name)}_pt.kml", driver="KML"
                )
                self.log.info("Track output as trackpoint kml file")
            else:
                self.log.error("File already exists, writing as trackpoint kml failed!")

        if "ln_kml" in types:
            if not os.path.isfile(f"{os.path.join(path_output, file_name)}_ln.kml"):
                self.trackline.to_file(
                    f"{os.path.join(path_output, file_name)}_ln.kml", driver="KML"
                )
                self.log.info("Track output as trackline kml file")
            else:
                self.log.error("File already exists, writing as trackline kml failed!")

    def resample(self, timestep="D", agg_function=None, first=True):
        """
        Resample track to a given time step.

        Timestep can be D for daily or h for hourly or multiples of D or h (eg '7D', '12h')
        After resampling other track properties will be refreshed.
        Other agg_fuctions might be wanted (max?, min?) but these are not implemented.
        One day maybe there will be interpolations?

        Since the track data and properies will be overwritten, it is a good idea to make
        a copy first:
            track_6h = copy.deepcopy(track)
            track_6h.resample(timestep="6h")

        Note this method has not been thoroughly tested!

        Parameters
        ----------
        timestep : str, optional
            Give the code for the timestep to sample to. The default is "D". See above.
        agg_function : str, optional
            Aggregation function: median, mean, None. The default is None.
        first : bool, optional
            If agg_fuction is none, or for columns that cannot be aggregated,
                take first (True) or last (False) value for the time period. The default
                is True.

        Returns
        -------
        None.

        """
        sdf = self.data
        # need to have a datetime index or use the 'on' keyword
        sdf = sdf.set_index("timestamp")
        sdf["u"] = np.sin(np.radians(sdf.platform_orientation))
        sdf["v"] = np.cos(np.radians(sdf.platform_orientation))

        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html
        # Note that you can control whether bin intervals are closed on left or right
        # and which label to use.  e.g., daily, closed left with left label:
        #   All data from 2024-08-02 00:00:00 to 2024-08-02 23:59:59 are in August 2.
        # Weekly and Montly have different behaviour by default so use use '7D' instead
        # of 'W' for weekly resampling for consistency
        # https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#api-groupby

        if agg_function == None:
            if first:
                number_f = "first"
            else:
                number_f = "last"
        elif agg_function == "mean":
            number_f = "mean"
        elif agg_function == "median":
            number_f = "median"
        else:
            self.log.error("Check resampling parameters")
            raise Exception("Check resampling parameters")
        if first:
            string_f = "first"
        else:
            string_f = "last"

        # this sort of thing is trivial
        # sdf.resample("D").last()
        # sdf.resample("D").first()

        # with mixed data types the way you aggregate needs to be controlled for each column
        sdf_ = sdf.resample(timestep).agg(
            platform_id=("platform_id", string_f),
            latitude=("latitude", number_f),
            longitude=("longitude", number_f),
            air_temperature=("air_temperature", number_f),
            internal_temperature=("internal_temperature", number_f),
            surface_temperature=("surface_temperature", number_f),
            air_pressure=("air_pressure", number_f),
            platform_pitch=("platform_pitch", number_f),
            platform_roll=("platform_roll", number_f),
            platform_orientation=("platform_orientation", string_f),
            voltage_battery_volts=("voltage_battery_volts", number_f),
            argo_position_accuracy=("argo_position_accuracy", number_f),
            u=("u", number_f),
            v=("v", number_f),
        )

        sdf_["platform_orientation"] = (
            360 + np.rad2deg(np.atan2(sdf_.u, sdf_.v))
        ) % 360

        self.data = sdf_.drop(["u", "v"], axis=1)

        # after doing the resampling it will be important to run:
        self.refresh_stats()

    def track_metadata(self, path_output=".", meta_export=None, verbose=False):
        """
        Make a dataframe and dictionary of the known track metadata for export.

        Put metadata into categories. The json is nested but the dataframe is not.

        Parameters
        ----------
        path_output : str, optional
            path where the output should be saved.  The default is the current directory.
        meta_export : str, optional
            Specify 'pandas' or 'json' format or 'both'. The default (None) does not export a file.
        verbose : Bool
            If true, return/export all the metadata, otherwise only the most useful.

        Returns
        -------
        track_meta : pandas dataframe
            available track metadata.

        """
        # get all the properties of the track
        t_meta = copy.deepcopy(self.__dict__)

        # get the extra metadata for the beacon here if available
        if hasattr(self, "meta_dict"):
            x_meta = copy.deepcopy(self.meta_dict)
            t_meta = t_meta | x_meta

        # don't want every entry: this is data or log
        remove_keys = [
            "log",
            "data",
            "meta_dict",
            "trackpoints",
            "trackline",
        ]
        for key in remove_keys:
            t_meta.pop(key, None)  # Use pop to avoid KeyError if key doesn't exist

        # now parse out the metadata fields based on what categories are defined:
        identifier_metadata = [
            "platform_id",
            "year",
            "id",
            "wmo",
            "iceberg_name",
            "iceberg_source",
        ]

        format_metadata = ["reader"]

        comments_metadata = ["comments"]

        track_metadata = [
            "track_start",
            "track_end",
            "data_start",
            "data_end",
            "trim_start",
            "trim_end",
            "trim_start_flag",
            "trim_end_flag",
            "latitude_start",
            "longitude_start",
            "latitude_end",
            "longitude_end",
            "observations",
            "duration",
            "distance_travelled",
        ]

        project_metadata = [
            "project",
            "data_contributor",
            "data_contact",
            "data_contact_email",
        ]

        deployment_metadata = [
            "deployed_by",
            "deployment_method",
            "deployment_platform",
            "photo_credit",
        ]

        morphology_metadata = [
            "shape",
            "size",
            "length",
            "length_flag",
            "width",
            "width_flag",
            "area",
            "area_flag",
            "height",
            "height_flag",
            "thickness",
            "thickness_flag",
            "draft",
            "draft_flag",
        ]

        beacon_metadata = [
            "model",
            "make",
            "transmitter",
            "air_deployable",
            "buoyant",
            "has_internal_temperature",
            "has_surface_temperature",
            "has_air_temperature",
            "has_voltage_battery_volts",
            "has_air_pressure",
            "has_platform_pitch",
            "has_platform_roll",
            "has_platform_orientation",
        ]

        valid_range_metadata = [
            "latitude_min",
            "latitude_max",
            "longitude_min",
            "longitude_max",
            "air_temperature_min",
            "air_temperature_max",
            "internal_temperature_min",
            "internal_temperature_max",
            "surface_temperature_min",
            "surface_temperature_max",
            "air_pressure_min",
            "air_pressure_max",
            "platform_pitch_min",
            "platform_pitch_max",
            "platform_roll_min",
            "platform_roll_max",
            "platform_orientation_min",
            "platform_orientation_max",
            "voltage_battery_volts_min",
            "voltage_battery_volts_max",
            "argo_position_accuracy_min",
            "argo_position_accuracy_max",
        ]

        process_metadata = [
            "datafile",
            "raw_data",
            "purged",
            "sorted",
            "speeded",
            "speedlimited",
            "trimmed",
            "trimmed_start",
            "trimmed_end",
            "geoed",
        ]

        # Create dictionaries based on each of the categories
        beacon_metadata_dict = {k: t_meta[k] for k in beacon_metadata if k in t_meta}
        comments_metadata_dict = {
            k: t_meta[k] for k in comments_metadata if k in t_meta
        }
        deployment_metadata_dict = {
            k: t_meta[k] for k in deployment_metadata if k in t_meta
        }
        format_metadata_dict = {k: t_meta[k] for k in format_metadata if k in t_meta}
        identifier_metadata_dict = {
            k: t_meta[k] for k in identifier_metadata if k in t_meta
        }
        morphology_metadata_dict = {
            k: t_meta[k] for k in morphology_metadata if k in t_meta
        }
        process_metadata_dict = {k: t_meta[k] for k in process_metadata if k in t_meta}
        project_metadata_dict = {k: t_meta[k] for k in project_metadata if k in t_meta}
        track_metadata_dict = {k: t_meta[k] for k in track_metadata if k in t_meta}
        valid_range_metadata_dict = {
            k: t_meta[k] for k in valid_range_metadata if k in t_meta
        }

        excluded_keys = set().union(
            beacon_metadata,
            comments_metadata,
            deployment_metadata,
            format_metadata,
            identifier_metadata,
            morphology_metadata,
            process_metadata,
            project_metadata,
            track_metadata,
            valid_range_metadata,
        )

        leftover_metadata_dict = {
            k: t_meta[k] for k in t_meta if k not in excluded_keys
        }

        if verbose:
            track_meta_dict = {
                "identifier_metadata": identifier_metadata_dict,
                "track_metadata": track_metadata_dict,
                "project_metadata": project_metadata_dict,
                "deployment_metadata": deployment_metadata_dict,
                "format_metadata": format_metadata_dict,
                "morphology_metadata": morphology_metadata_dict,
                "comments_metadata": comments_metadata_dict,
                "beacon_metadata": beacon_metadata_dict,
                "process_metadata": process_metadata_dict,
                "valid_range_metadata": valid_range_metadata_dict,
                "leftover_metadata": leftover_metadata_dict,
            }

        else:
            track_meta_dict = {
                "identifier_metadata": identifier_metadata_dict,
                "track_metadata": track_metadata_dict,
                "project_metadata": project_metadata_dict,
                "deployment_metadata": deployment_metadata_dict,
                "format_metadata": format_metadata_dict,
                "morphology_metadata": morphology_metadata_dict,
                "comments_metadata": comments_metadata_dict,
                "beacon_metadata": beacon_metadata_dict,
            }

        flattened = {}
        for meta_category, meta_category_dict in track_meta_dict.items():
            for key, value in meta_category_dict.items():
                flattened[f"{key}"] = value

        # Convert to DataFrame
        track_meta_df = pd.DataFrame([flattened])

        # order columns
        col_order = (
            identifier_metadata
            + track_metadata
            + project_metadata
            + deployment_metadata
            + format_metadata
            + morphology_metadata
            + comments_metadata
            + beacon_metadata
            + process_metadata
            + valid_range_metadata
            + list(leftover_metadata_dict.keys())
        )

        new_columns = [col for col in col_order if col in track_meta_df.columns]
        track_meta_df = track_meta_df[new_columns]

        if meta_export == "json" or meta_export == "both":
            track_meta_json = json.dumps(
                track_meta_dict, indent=4, default=json_serialize
            )
            with open(
                f"{Path(path_output)/self.platform_id}_meta.json", "w"
            ) as file_export:
                file_export.write(track_meta_json)

        if meta_export == "pandas" or meta_export == "both":
            track_meta_df.to_csv(
                f"{Path(path_output)/self.platform_id}_meta.csv",
                index=False,
                na_rep="NA",
            )

        return track_meta_df

    # The following graphing functions are in track_fig.py but listed here so they can
    # be a method of Track.

    def plot_map(self, path_output=".", interactive=False, dpi=300):
        """
        Plot a map of the track.

        See track_fig.py for more details.

        Parameters
        ----------
        path_output : str, optional
            Path to save output. The default is the current directory.
        dpi : int, optional
            Resolution of the graph in dots per inch. The default is 300.
        interactive: bool
            Create an interactive map that can be panned and zoomed.

        Returns
        -------
        None.

        """
        # call the function in track_fig.py
        plot_map(
            self,
            path_output=path_output,
            dpi=dpi,
            interactive=interactive,
            log=self.log,
        )

    def plot_trim(self, path_output=".", interactive=False, dpi=300):
        """
        Plot a trim diagnostic graph for the track.

        See track_fig.py for more details.

        Parameters
        ----------
        path_output : str, optional
            Path to save output. The default is the current directory.
        dpi : int, optional
            Resolution of the graph in dots per inch. The default is 300.
        interactive: bool
            Create an interactive map that can be panned and zoomed.

        Returns
        -------
        None.

        """
        # call the function in track_fig.py
        plot_trim(
            copy.deepcopy(
                self
            ),  # since the function modifies the track, work with copy
            path_output=path_output,
            dpi=dpi,
            interactive=interactive,
            log=self.log,
        )

    def plot_dist(self, path_output=".", interactive=False, dpi=300):
        """
        Plot distributions along the track.

        See track_fig.py for more details.

        Parameters
        ----------
        path_output : str, optional
            Path to save output. The default is the current directory.
        dpi : int, optional
            Resolution of the graph in dots per inch. The default is 300.
        interactive: bool
            Create an interactive map that can be panned and zoomed.

        Returns
        -------
        None.

        """
        # call the function in track_fig.py
        plot_dist(
            self,
            path_output=path_output,
            dpi=dpi,
            interactive=interactive,
            log=self.log,
        )

    def plot_time(self, path_output=".", interactive=False, dpi=300):
        """
        Plot a timeseries of the track.

        See track_fig.py for more details.

        Parameters
        ----------
        path_output : str, optional
            Path to save output. The default is the current directory.
        dpi : int, optional
            Resolution of the graph in dots per inch. The default is 300.
        interactive: bool
            Create an interactive map that can be panned and zoomed.

        Returns
        -------
        None.

        """
        # call the function in track_fig.py
        plot_time(
            copy.deepcopy(
                self
            ),  # since the function modifies the track, work with copy
            path_output=path_output,
            dpi=dpi,
            interactive=interactive,
            log=self.log,
        )
