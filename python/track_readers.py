"""
track_readers.py.

A collection of functions that convert from various beacon data formats to
a standardized format.

Note that the formats ingested by these functions (known as the raw data in the ITBD)
are determined by the data owner/provider and do not necessarily represent the format of
the original communication from the device itself.

Each function takes the raw data file path/name and puts it into the standardized format
At a minimum, the date/time, latitude and longitude are required.  Other columns are 
optional.


--
Created December 2022 by Adam Garbo based on R scripts from Derek Mueller, Cindy
Lopes, Anna Crawford and Jill Rajewicz
Rewritten June-July 2024 by Derek Mueller

"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
from collections import namedtuple
import re


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


def create_sdf(nrows):
    """
    Create an empty dataframe in the standardized format filled with NAs.

    Also the place to (re)define the standardized format if needed

    Parameters
    ----------
    nrows : int
        Number of rows to make.

    Returns
    -------
    sdf : Pandas Dataframe
        Empty dataframe in standard format

    """
    # define the column names
    columns = [
        "beacon_id",
        "datetime_data",
        "datetime_transmit",
        "latitude",
        "longitude",
        "temperature_air",
        "temperature_internal",
        "temperature_surface",
        "pressure",
        "pitch",
        "roll",
        "heading",
        "satellites",
        "voltage",
        "loc_accuracy",
        "distance",
        "speed",
        "direction",
    ]

    col_dtypes = {
        "beacon_id": str,
        "datetime_data": np.dtype("datetime64[ns]"),
        "datetime_transmit": np.dtype("datetime64[ns]"),
        "latitude": float,
        "longitude": float,
        "temperature_air": float,
        "temperature_internal": float,
        "temperature_surface": float,
        "pressure": float,
        "pitch": float,
        "roll": float,
        "heading": float,
        # "satellites" : int,
        "voltage": float,
        # "loc_accuracy": int,
        "distance": float,
        "speed": float,
        "direction": float,
    }

    # define sdf as dtype Int64 (which can contain NA)
    sdf = pd.DataFrame(
        np.nan, index=np.arange(nrows), columns=columns, dtype=pd.Int64Dtype()
    )

    # change dtype of all other columns as required
    sdf = sdf.astype(col_dtypes)

    # explicitly set these as UTC (tz aware)
    sdf.datetime_data = pd.to_datetime(sdf.datetime_data, utc=True)
    sdf.datetime_transmit = pd.to_datetime(sdf.datetime_transmit, utc=True)

    return sdf


#### Helper functions


def dms2dd(string):
    """
    Extract degrees, minutes and seconds from a string

    Assumes that the format is deg min sec or deg min, or deg.  Never deg sec, etc.
    Also assumes there will be no negative min or sec.

    Parameters
    ----------
    string : str
        A string containing deg, min, seconds like 71° 20' 23.20800"

    Returns
    -------
    dd : float
        decimal degree.

    """
    # Find all numbers in the string (all integers and floating point numbers in string)
    dms = re.findall(r"\d+\.\d+|-?\d+", string)
    # convert them all to floating point numbers
    dms = [float(i) for i in dms]
    # how many are there?
    match len(dms):
        case 3:
            pass
        case 2:
            dms.append(0)
        case 1:
            dms.append(0)
            dms.append(0)
        case _:
            dms = [0, 0, 0]
    degs, mins, secs = dms

    decSec = float(secs) / 3600
    decMin = float(mins) / 60
    dd = abs(degs) + decMin + decSec
    if degs < 0:
        dd = dd * -1
    return dd


# -----------------------------------------------------------------------------
# Standardization functions
# -----------------------------------------------------------------------------


def standard(raw_data_file, log=None):
    """
    Convert standardized data format to standardized dataframe.

    This script doesn't do anything to the data, but it allows users to use a
    standardized csv file to be read in and worked on within the same workflow
    as we have already.

    The script follows the same template, if there is a problem, like a raw file is used
    instead of a standardized one, you will hear about it.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file (which in this case is in the standard format)
    log : logger
        A logger instance

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized beacon track pandas dataframe.

    Data format
    -----------------------------------
    See create_sdf function

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:
        sdf["datetime_data"] = pd.to_datetime(rdf["datetime_data"], utc=True)

        sdf["latitude"] = rdf["latitude"]
        sdf["longitude"] = rdf["longitude"]

        sdf["datetime_transmit"] = pd.to_datetime(rdf["datetime_transmit"], utc=True)
        sdf["temperature_air"] = rdf["temperature_air"]
        sdf["temperature_internal"] = rdf["temperature_internal"]
        sdf["temperature_surface"] = rdf["temperature_surface"]
        sdf["pressure"] = rdf["pressure"]
        sdf["pitch"] = rdf["pitch"]
        sdf["roll"] = rdf["roll"]
        sdf["heading"] = rdf["heading"]
        sdf["satellites"] = rdf["satellites"]
        sdf["voltage"] = rdf["voltage"]
        sdf["loc_accuracy"] = rdf["loc_accuracy"]
        sdf["distance"] = rdf["distance"]
        sdf["speed"] = rdf["speed"]
        sdf["direction"] = rdf["direction"]

    except:
        print(f"Problem with standard data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def calib_argos(raw_data_file, log=None):
    """
    Convert raw data from CALIB ARGOS format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    A challenging format to read!

    There are four types of lines -
        blank line '\n'
        obs1 = begining of the obs - starts with id
        obs2 = end of the obs - starts with '('
        a comment  - none of the above

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    try:
        # Read the whole file:
        f = open(raw_data_file, mode="r", errors="replace")
        fp = f.readlines()
        f.close()

        # get the id of the beacon from the file name and the year
        year, ID = Path(raw_data_file).stem.split("_")

        # empty lists to fill with data
        latitude = []
        longitude = []
        loc_accuracy = []
        datetime_transmit = []
        datetime_data = []
        message_index = []
        pressure = []
        voltage = []
        temperature_internal = []
        comments = []

        # obs1 and 2 hold the index for each one of the 2 data lines
        obs1 = []
        obs2 = []
        for i, line in enumerate(fp):
            fields = line.split()
            if len(fields) == 0:  # nothing on the line
                continue
            elif ID in fields[0]:
                if (
                    fields[1] != "NO"
                ):  # sometimes NO LOCATION happens... then the record needs to be ignored.
                    obs1.append(i)
            elif "(" in fields[0]:  # hope that no comment lines start with '('
                obs2.append(i)
            # all other non-blank lines are comments
            else:
                comments.append(line)

        # need to keep track of the year (for both trans and data)
        trans_year = year
        data_year = year
        trans_doy = 0
        data_doy = 0

        # pass through again and load up the lists
        for o1 in obs1:
            if o1 + 1 in obs2:  # this makes sure that the obs1 is followed by obs2
                fields = fp[o1].split()

                # Lat and Lon:
                for j in range(1, 3):
                    strLoc = fields[j]
                    if strLoc[-1:] == "S" or strLoc[-1:] == "W":
                        numLoc = float(strLoc[:-1]) * -1
                    elif strLoc[-1:] == "N" or strLoc[-1:] == "E":
                        numLoc = float(strLoc[:-1])
                    else:
                        print(line)
                    if j == 1:
                        latitude.append(numLoc)
                    else:
                        longitude.append(numLoc)
                # location quality
                loc_accuracy.append(int(fields[3]))

                # dates/times
                trans_time, data_time = fields[4].split("-")

                # trans time is time of transmission
                if trans_time[0] == "?":
                    datetime_transmit.append(np.nan)
                else:
                    if (
                        trans_doy > 350 and int(trans_time.split("/")[0]) < 15
                    ):  # happy new year!
                        trans_year = str(int(trans_year) + 1)
                    trans_doy = int(trans_time.split("/")[0])
                    datetime_transmit.append(
                        dt.datetime.strptime(
                            f"{trans_year}-{trans_time}", "%Y-%j/%H%MZ"
                        )
                    )

                # data time is position time
                if data_time[0] == "?":
                    datetime_data.append(np.nan)
                else:

                    # There is no year recorded in the data, only DOY
                    # Detect and deal with year rollover in December/January
                    # this here looks to see if the year is about to roll over.
                    # the approach is not perfect since if the data is missing over a large
                    # part of Dec and January these conditions won't match
                    if data_doy > 350 and int(data_time.split("/")[0]) < 15:
                        data_year = str(int(data_year) + 1)
                    data_doy = int(data_time.split("/")[0])
                    datetime_data.append(
                        dt.datetime.strptime(f"{data_year}-{data_time}", "%Y-%j/%H%M")
                    )

        for o2 in obs2:
            if o2 - 1 in obs1:  # this makes sure that the obs2 is preceded by obs1

                fields = fp[o2].split()

                # sometimes there are "?" in the data but they are not meaningful
                fields = list(
                    filter(lambda item: item != "?", fields)
                )  # sometimes "?" are present
                fields = list(
                    filter(lambda item: item != "(", fields)
                )  # remove this from list

                message_index.append(
                    int(fields[0].replace("(", "")[:-1])
                )  # remove the "(" in 2 digit values

                # These sensors are either floating points (in scientific notation) or
                # They are hex values  or they are integers.
                # See manual for how to map values.  The problem is distinguishing int and hex
                # For temperatures > -20 there will be 3 digits for int and only 2 for hex
                #  this is the best guess, may not work always

                # pressure
                if "E+" in fields[1] or "E-" in fields[1]:
                    pressure.append(float(fields[1]))
                else:
                    if len(fields[3]) == 3:
                        pressure.append(int(fields[1]) * 0.1511 + 920)
                    else:
                        pressure.append(int(fields[1], 16) * 0.1511 + 920)

                # voltage
                if "E+" in fields[2] or "E-" in fields[2]:
                    voltage.append(float(fields[2]))
                else:
                    if len(fields[3]) == 3:
                        voltage.append(int(fields[2]) * 0.2 + 6)
                    else:
                        voltage.append(int(fields[2], 16) * 0.2 + 6)

                # temperature
                if "E+" in fields[3] or "E-" in fields[3]:
                    temperature_internal.append(float(fields[3]))
                else:
                    if len(fields[3]) == 3:
                        temperature_internal.append(int(fields[3]) * 0.3 - 50)
                    else:
                        temperature_internal.append(int(fields[3], 16) * 0.3 - 50)

        # create an empty standard data frame - sdf - filled with NAs
        sdf = create_sdf(len(latitude))

        sdf["beacon_id"] = Path(raw_data_file).stem
        sdf["datetime_data"] = pd.to_datetime(datetime_data, utc=True)
        sdf["datetime_transmit"] = pd.to_datetime(datetime_transmit, utc=True)
        sdf["latitude"] = latitude
        sdf["longitude"] = longitude
        sdf["temperature_internal"] = temperature_internal
        sdf["pressure"] = pressure
        sdf["voltage"] = voltage
        sdf["loc_accuracy"] = loc_accuracy

        if len(comments) > 0:
            log.info("Start of comments from raw ARGOS file:")
            for comment in comments:
                log.info(comment)
            log.info("End of comments from raw ARGOS file:")
    except:
        log.error(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def calib_iridium(raw_data_file, log=None):
    """
    Convert raw data from Metocean iCALIB format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Columns:
        Asset Name
        Asset Id
        Data Date (UTC)
        Received Date (UTC)
        LATITUDE
        LONGITUDE
        FMTID
        YEAR
        MONTH
        DAY
        HOUR
        MIN
        SST
        BP
        BPT
        VBAT
        GPSDELAY
        SNR
        TTFF
        SBDTIME
        Report Body

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:

        # Data timestamp
        if "DATA DATE (UTC)" in rdf:
            sdf["datetime_data"] = pd.to_datetime(rdf["DATA DATE (UTC)"], utc=True)
        elif "Data Date (UTC)" in rdf:
            sdf["datetime_data"] = pd.to_datetime(rdf["Data Date (UTC)"], utc=True)
        else:
            raise

        # Data transmission timestamp
        if "RECEIVED DATE (UTC)" in rdf:
            sdf["datetime_transmit"] = pd.to_datetime(
                rdf["RECEIVED DATE (UTC)"], utc=True
            )
        elif "Received Date (UTC)" in rdf:
            sdf["datetime_transmit"] = pd.to_datetime(
                rdf["Received Date (UTC)"], utc=True
            )

        sdf["latitude"] = rdf["LATITUDE"]
        sdf["longitude"] = rdf["LONGITUDE"]

        # Surface temperature
        if "SST" in rdf:
            sdf["temperature_surface"] = rdf["SST"]

        # Barometric pressure
        if "BP" in rdf:
            sdf["pressure"] = rdf["BP"]

        # Battery voltage
        if "VBAT" in rdf:
            sdf["voltage"] = rdf["VBAT"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def canatec(raw_data_file, log=None):
    """
    Convert raw data from Canatec format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Columns are:
        ReadingDate
        Latitude
        Longitude
        Elevation
        Heading
        Speed
        Fix
        Satellites
        HDOP
        VDOP
        VerticalVelocity
        Pressure
        TempExternal
        TempInternal
        BeaconAlarmState
        BatteryVoltage
        ModemVoltage
        WindSpeed
        WindDirection

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:
        sdf["datetime_data"] = pd.to_datetime(
            rdf["ReadingDate"], utc=True, yearfirst="True"
        )
        sdf["latitude"] = rdf["Latitude"]
        sdf["longitude"] = rdf["Longitude"]

        if "TempExternal" in rdf:
            sdf["temperature_air"] = rdf["TempExternal"]

        if "TempInternal" in rdf:
            sdf["temperature_internal"] = rdf["TempInternal"]

        if "Pressure" in rdf:
            sdf["pressure"] = rdf["Pressure"]

        if "BatteryVoltage" in rdf:
            sdf["voltage"] = rdf["BatteryVoltage"]

        if "Satellites" in rdf:
            sdf["satellites"] = rdf["Satellites"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def pathfinder_ccore(raw_data_file, log=None):
    """
    Convert raw data from Pathfinder C-CORE format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Columns (case sensitive):
        Buoy Name
        Time (2011 Jun 18 18:20:02 UTC)
        Latitude
        Longitude
        Temperature
        Drogue Depth (m)

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:
        sdf["datetime_data"] = pd.to_datetime(
            rdf["Time"], format="%Y %b %d %H:%M:%S UTC", utc=True
        )
        sdf["latitude"] = rdf["Latitude"]
        sdf["longitude"] = rdf["Longitude"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def cryologger(raw_data_file, log=None):
    """
    Convert raw data from Cryologger format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    See Cryologger_ITB in beacon model info for more details

    Data columns (case sensitive):
        imei
        momsn
        transmit_time
        iridium_latitude
        iridium_longitude
        iridium_cep
        data
        unixtime
        temperature_int
        humidity_int
        pressure_int
        pitch
        roll
        heading
        latitude
        longitude
        satellites
        hdop
        voltage
        transmitDuration
        messageCounter
    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:
        sdf["datetime_data"] = pd.to_datetime(rdf["unixtime"], unit="s", utc=True)
        sdf["latitude"] = rdf["latitude"]
        sdf["longitude"] = rdf["longitude"]

        #  Data transmission timestamp (UTC)
        if "transmit_time" in rdf:
            sdf["datetime_transmit"] = pd.to_datetime(rdf["transmit_time"])

        #  Battery voltage
        if "voltage" in rdf:
            sdf["voltage"] = rdf["voltage"]

        # Internal temperature
        if "temperature_int" in rdf:
            sdf["temperature_internal"] = rdf["temperature_int"]

        #  Barometric pressure
        if "pressure_int" in rdf:
            sdf["pressure"] = rdf["pressure_int"] * 10  # Convert to hPa

        # Pitch
        if "pitch" in rdf:
            sdf["pitch"] = rdf["pitch"]

        # Roll
        if "roll" in rdf:
            sdf["roll"] = rdf["roll"]

        # Tilt-compensated heading
        if "heading" in rdf:
            sdf["heading"] = rdf["heading"]

        # GNSS Satellites
        if "satellites" in rdf:
            sdf["satellites"] = rdf["satellites"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def iabp(raw_data_file, log=None):
    """
    Convert raw data from IABP website format to standardized dataframe.

    Note: Many different beacon types report to the IABP repository

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Data columns:
        BuoyID
        Date
        Year
        Hour
        Min
        DOY
        POS_DOY
        Lat
        Lon
        BP
        Ta
        Ts
    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:
        # use the position doy for the timestamp (along with year)
        sdf["datetime_data"] = pd.to_datetime(rdf["Year"], format="%Y", utc=True) + rdf[
            "POS_DOY"
        ].sub(1).apply(pd.Timedelta, unit="D")
        # round to the nearest second
        sdf["datetime_data"].dt.round(freq="s")

        sdf["latitude"] = rdf["Lat"]
        sdf["longitude"] = rdf["Lon"]

        # Air temperature
        if "Ta" in rdf:
            sdf["temperature_air"] = rdf["Ta"]

        # Surface temperature
        if "Ts" in rdf:
            sdf["temperature_surface"] = rdf["Ts"]

        # Pressure
        if "BP" in rdf:
            sdf["pressure"] = rdf["BP"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def iip(raw_data_file, log=None):
    """
    Convert raw data from Metocean IIP GlobalStar format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Columns:
        INDEX
        ID
        DATETIME (2019-04-29T23:03UTC)
        LATITUDE
        LONGITUDE
    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:

        sdf["datetime_data"] = pd.to_datetime(
            rdf["DATETIME"], format="%Y-%m-%dT%H:%MUTC", utc=True
        )

        sdf["latitude"] = rdf["LATITUDE"]

        sdf["longitude"] = rdf["LONGITUDE"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def oceanetic(raw_data_file, log=None):
    """
    Convert raw data from Oceanetic format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Columns:
        beacon id
        yr
        mm
        dd
        hr
        lat
        long
    """

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:
        rdf.rename(
            columns={"yr": "year", "mm": "month", "dd": "day", "hr": "hour"},
            inplace=True,
        )
        sdf["datetime_data"] = pd.to_datetime(
            rdf[["year", "month", "day", "hour"]], utc=True
        )

        # Latitude
        sdf["latitude"] = rdf["lat"]

        # Longitude
        sdf["longitude"] = rdf["long"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def ceos(raw_data_file, log=None):
    """
    Convert raw data from CEOS format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Columns:
        imei
        unknown
        unknown
        date
        time
        lat
        hemi
        long
        hemi
        unknown
        unknown
        unknown
        unknown
    """
    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:
        sdf["datetime_data"] = pd.to_datetime(
            rdf["date"].str.cat(rdf["time"], sep=" "), utc=True
        )

        # Latitude
        sdf["latitude"] = rdf["lat"]

        # Longitude
        sdf["longitude"] = rdf["long"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def wirl_sbd(raw_data_file, log=None):
    """
    Convert raw data from WIRL SBD decode format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Columns:
        IMEI
        Year
        Month
        Day
        Hour
        Minute
        Latitude
        Longitude
        Temperature
        Voltage Battery
        AtmPress
        FormatID

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:

        if "Year" in rdf:
            sdf["datetime_data"] = pd.to_datetime(
                rdf[["Year", "Month", "Day", "Hour", "Minute"]], utc=True
            )

        # Latitude
        if "Latitude" in rdf:
            sdf["latitude"] = rdf["Latitude"]

        # Longitude
        if "Longitude" in rdf:
            sdf["longitude"] = rdf["Longitude"]

        if "Temperature" in rdf:
            sdf["temperature_internal"] = rdf["Temperature"]
            sdf.loc[sdf["temperature_internal"] == -99, "temperature_internal"] = np.nan

        if "AtmPress" in rdf:
            sdf["pressure"] = rdf["AtmPress"]
            sdf.loc[sdf["pressure"] == -99, "pressure"] = np.nan

        if "Voltage Battery" in rdf:
            sdf["voltage"] = rdf["Voltage Battery"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def rockstar(raw_data_file, log=None):
    """
    Convert raw data from Rockstar/Yellowbrick format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Columns:
        ID
        GPS Time (UTC) (17-08-2016 20:00:06)
        Latitude
        Longitude
        Temperature (only sometimes)

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    with open(raw_data_file, "r") as fp:
        lines = fp.readlines()
        for row in lines:
            word = "ID,GPS"  # ------
            if row.find(word) != -1:
                skip = lines.index(row)

    rdf = pd.read_csv(raw_data_file, index_col=False, skiprows=skip)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:
        # Datetime format for beacon IDs: 2011_300034013463170 and 0082470
        # there are 2 columns with the same label.  they seem to be always the same...
        sdf["datetime_data"] = pd.to_datetime(
            rdf["GPS Time (UTC)"], dayfirst=True, utc=True
        )

        sdf["latitude"] = rdf["Latitude"]
        sdf["longitude"] = rdf["Longitude"]

        if "Temperature" in rdf:
            sdf["temperature_internal"] = rdf["Temperature"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def solara(raw_data_file, log=None):
    """
    Convert raw data from Solara format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Columns:
        serial
        alias
        lat
        long
        timestamp
            format_1 = "%Y-%m-%d %H:%M:%S"
            format_2 = "%d/%m/%Y %H:%M:%S"

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:

        sdf["datetime_data"] = pd.to_datetime(rdf["timestamp"], utc=True)
        sdf["latitude"] = rdf["lat"]
        sdf["longitude"] = rdf["long"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def svp_old_throw_away(raw_data_file, log=None):
    """
    Convert raw data from Metocean SVP format to standardized dataframe.

    - SVP-I-BXGS-LP beacons have BP, GPS, and SST
    - SVP-I-XXGS-LP beacons have GPS and SST, no BP
    - SVP-I-BXGSA-L-AD beacons have BP, GPS, SST, AT, lithium battery and are
    designed for air deployment in Arctic regions

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Columns:
        Asset Name / Asset.Name
        Asset Id  / Asset.Id /
        Data Date (UTC) / DataDate_UTC
        Received Date (UTC)
        LATITUDE   # this is in dd.dddd format
        LONGITUDE  # this is in dd.dddd format
        FMTID
        YEAR
        MONTH
        DAY
        HOUR
        MIN
        SST
        BP
        BPT
        AT
        VBAT
        GPSDELAY
        SNR
        TTFF
        SBDTIME
        Report Body / Report.Body

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:

        # Data timestamp
        if "DATA DATE (UTC)" in rdf:
            sdf["datetime_data"] = pd.to_datetime(rdf["DATA DATE (UTC)"], utc=True)
        elif "Data Date (UTC)" in rdf:
            sdf["datetime_data"] = pd.to_datetime(rdf["Data Date (UTC)"], utc=True)
        elif "DataDate_UTC" in rdf:
            sdf["datetime_data"] = pd.to_datetime(rdf["DataDate_UTC"], utc=True)

        # Data transmission timestamp
        if "RECEIVED DATE (UTC)" in rdf:
            sdf["datetime_transmit"] = pd.to_datetime(
                rdf["RECEIVED DATE (UTC)"], utc=True
            )
        elif "Received Date (UTC)" in rdf:
            sdf["datetime_transmit"] = pd.to_datetime(
                rdf["Received Date (UTC)"], utc=True
            )

        sdf["latitude"] = rdf["LATITUDE"]
        sdf["longitude"] = rdf["LONGITUDE"]

        # Air temperature
        if "AT" in rdf:
            sdf["temperature_air"] = rdf["AT"]

        # Surface temperature
        if "SST" in rdf:
            sdf["temperature_surface"] = rdf["SST"]

        # Barometric pressure
        if "BP" in rdf:
            sdf["pressure"] = rdf["BP"]

        # Battery voltage
        if "VBAT" in rdf:
            sdf["voltage"] = rdf["VBAT"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def svp(raw_data_file, log=None):
    """
    Convert raw data from Metocean SVP format to standardized dataframe.

    - SVP-I-BXGS-LP beacons have BP, GPS, and SST
    - SVP-I-XXGS-LP beacons have GPS and SST, no BP
    - SVP-I-BXGSA-L-AD beacons have BP, GPS, SST, AT, lithium battery and are
    designed for air deployment in Arctic regions

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    Columns:
        Asset Name / Asset.Name
        Asset Id  / Asset.Id / Modem ID
        Data Date (UTC) / DataDate_UTC  / Date(GMT)
        Received Date (UTC)
        LATITUDE   # this is in dd.dddd format / GPS LATITUDE (DEGREES) # dd mm.mmmmmm format
        LONGITUDE  # this is in dd.dddd format / GPS LONGITUDE (DEGREES) # dd mm.mmmmmm format
        FMTID / FORMAT ID
        YEAR
        MONTH
        DAY / DAY (day)
        HOUR / HOUR (hr)
        MIN  / MINUTE (min)
        SST  / Sea Surface Temperature (�C)
        BP  / Barometric Pressure (mbar)
        BPT / Barometric Pressure Tendency (mbar)
        AT
        VBAT / Battery Voltage (V)
        GPSDELAY / Time Since Last GPS fix (min)
        SNR  / GPS reported Signal to Noise ratio (dB)
        TTFF / Time to First fix (s)
        SBDTIME  / Iridium Transmission Duration (s)
        Report Body / Report.Body / Hex Data

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(
        raw_data_file, index_col=False, skipinitialspace=True, encoding_errors="replace"
    )

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:

        # Data timestamp
        if "DATA DATE (UTC)" in rdf:
            sdf["datetime_data"] = pd.to_datetime(rdf["DATA DATE (UTC)"], utc=True)
        elif "Data Date (UTC)" in rdf:
            sdf["datetime_data"] = pd.to_datetime(rdf["Data Date (UTC)"], utc=True)
        elif "DataDate_UTC" in rdf:
            sdf["datetime_data"] = pd.to_datetime(rdf["DataDate_UTC"], utc=True)
        elif "Date(GMT)" in rdf:
            sdf["datetime_data"] = pd.to_datetime(rdf["Date(GMT)"], utc=True)

        # Data transmission timestamp
        if "RECEIVED DATE (UTC)" in rdf:
            sdf["datetime_transmit"] = pd.to_datetime(
                rdf["RECEIVED DATE (UTC)"], utc=True
            )
        elif "Received Date (UTC)" in rdf:
            sdf["datetime_transmit"] = pd.to_datetime(
                rdf["Received Date (UTC)"], utc=True
            )

        if "GPS LATITUDE (DEGREES)" in rdf:
            sdf["latitude"] = rdf["GPS LATITUDE (DEGREES)"].apply(dms2dd)
        else:
            sdf["latitude"] = rdf["LATITUDE"]
        if "GPS LONGITUDE (DEGREES)" in rdf:
            sdf["longitude"] = rdf["GPS LONGITUDE (DEGREES)"].apply(dms2dd)
        else:
            sdf["longitude"] = rdf["LONGITUDE"]

        # Air temperature
        if "AT" in rdf:
            sdf["temperature_air"] = rdf["AT"]

        # Surface temperature
        if "SST" in rdf:
            sdf["temperature_surface"] = rdf["SST"]
        if "Sea Surface Temperature (�C)" in rdf:
            sdf["temperature_surface"] = rdf["Sea Surface Temperature (�C)"]

        # Barometric pressure
        if "BP" in rdf:
            sdf["pressure"] = rdf["BP"]
        if "Barometric Pressure (mbar)" in rdf:
            sdf["pressure"] = rdf["Barometric Pressure (mbar)"]

        # Battery voltage
        if "VBAT" in rdf:
            sdf["voltage"] = rdf["VBAT"]
        if "Battery Voltage (V)" in rdf:
            sdf["voltage"] = rdf["Battery Voltage (V)"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def bio(raw_data_file, log=None):
    """
    Convert raw data from BIO Icetracker2 format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    BID
    VOLTAGE
    GPS_DATE (08-09-15 13:00:00.000000000 or 2011-03-20 13:00:00.0)

    LATITUDE
    LONGITUDE
    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    # transfer the data from rdf to sdf
    try:
        sdf["datetime_data"] = pd.to_datetime(
            rdf["GPS_DATE"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce", utc=True
        ).fillna(
            pd.to_datetime(
                rdf["GPS_DATE"],
                format="%y-%m-%d %H:%M:%S.%f",
                errors="coerce",
                utc=True,
            )
        )
        sdf["latitude"] = rdf["LATITUDE"]
        sdf["longitude"] = rdf["LONGITUDE"]

        if "VOLTAGE" in rdf:
            sdf["voltage"] = rdf["VOLTAGE"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


def navidatum(raw_data_file, log=None):
    """
    Convert raw data from Skywave DMR800L format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    -----------------------------------
    Date (02/08/2012 21:01)
    Latitude
    Longitude

    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()

    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file, index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:
        sdf["datetime_data"] = pd.to_datetime(
            rdf["Date"], format="%d/%m/%Y %H:%M", utc=True
        )

        sdf["latitude"] = rdf["Latitude"]
        sdf["longitude"] = rdf["Longitude"]

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf


'''
def fn_template(raw_data_file, log=None):
    """
    Convert raw data from <PROVIDER/MODEL> format to standardized dataframe.

    Parameters
    ----------
    raw_data_file : string
        Path to raw data CSV file.

    Returns
    -------
    sdf : Pandas DataFrame
        Standardized Pandas dataframe ready for cleaning.

    Raw data format
    ----------------------------------
    <INSERT HERE>
    """
    # set up the logger to output nowhere if None
    if log == None:
        log = nolog()
        
    # read in the raw data frame - rdf
    rdf = pd.read_csv(raw_data_file,  index_col=False, skipinitialspace=True)

    # create an empty standard data frame - sdf - filled with NAs
    sdf = create_sdf(len(rdf))

    # Unique beacon identifier
    sdf["beacon_id"] = Path(raw_data_file).stem

    try:
        1 + 1  # placeholder - remove to use template
        # <INSERT HERE>

    except:
        print(f"Problem with raw data file {raw_data_file}, check formatting")
        sys.exit(1)

    return sdf
'''
