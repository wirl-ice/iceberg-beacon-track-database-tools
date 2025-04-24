# Iceberg Beacon Track Database (IBTD) Processing Tools

This repository provides the core Python scripts and modules for processing, standardizing, visualizing, and collating iceberg beacon track data into the Iceberg Beacon Track Database (IBTD). The processing workflow is designed to read-in various raw data formats from multiple beacon types/telemetry providers, standardize them, perform quality control, and output processed data files and visualizations.  

![Map](https://github.com/user-attachments/assets/ceb666c2-5f4f-4d98-aae3-6058d2131b4f)
A visual overview of the Iceberg Beacon Track Database (IBTD).  a) Map of all the tracks in the IBTD (version 1). b) Photo of 6 km-long ice island MIL20-A-1-A (begining of track 2021_300234011751690) off the coast of Axel Heiberg Island in August 2021 (photo: Mike Knox/Darryl Herman, Kenn Borek Air/PCSP); c) Photo of unnamed large tabular iceberg (begining of track 2013_300234011240410) in Nares Strait in August 2013 (photo: Meaghan Shields, CEOS, University of Manitoba); d) Photo of 'Station 3' a 4.5 km long ice island (beginning of tracks 2015_300134010505190, 2015_300234062791420, and 2015_300134010907780), which is about to be laser scanned from the CCGS Amundsen in April 2013 (photo: Anna Crawford, WIRL, Carleton University).  

## IBTD background
Researchers at Carleton University's Water and Ice Research Laboratory, in collaboration with the Canadian Ice Service (Meteorological Service of Canada, Environment and Climate Change Canada), have recently compiled one of the most comprehensive databases of in-situ iceberg tracking beacon drift trajectories in the Northern Hemisphere, with over 719,000 observations collected between 1997 and 2024. 

The database contains iceberg and ice island positions and meteorological data measured by tracking beacons as well as information about the icebergs at the time of deployment (e.g. shape, dimensions, source, thickness) when available. Data have been contributed by government, academic, and industry sources from tracking beacons deployed on icebergs in the western and eastern Canadian Arctic, and off the East Coast of Canada. 

Drift location, direction and speed along with other environmental data from tracking beacons deployed on icebergs and ice islands can be used to understand how icebergs drift, and to develop and validate models of iceberg drift, in order to improve predictions of ice hazard occurrence and behaviour. The software in this repository was used to compile the Database and might be helpful to Database users who may wish to analyze or visualize beacon track data. 

The Database itself can be accessed here: 

[Garbo A, Rajewicz, J, Mueller D, Tivy A, Copland L, Turnbull I, Desjardins L, Barber D, Peterson I, Prinsenberg S, Dalton A, Briggs R, Crawford A, Lamontagne P, Roth, JC, King T, Hicks M, Spring W, Conner S, Hill C. (2025) Iceberg Beacon Track Database [version 1.0]. Canadian Cryospheric Information Network (CCIN), Waterloo, Canada. doi:10.21963/13340.](https://doi.org/10.21963/13340)

More details on the IBTD can be found in the Database documentation: 

[Mueller D, Rajewicz, J, Garbo A, Tivy A (2025) Iceberg Beacon Track Database [version 1.0] Documentation. Water and Ice Research Lab, Department of Geography and Environmental Studies, Carleton University, Ottawa. 32 pp. doi:10.22215/wirl/2025.1](https://doi.org/10.22215/wirl/2025.1)

## Overview

IBTD Processing Tools contains a modular workflow that:

* Reads and standardizes raw beacon track data from various formats.
* Purges, filters, and trims track data according to command-line arguments or track metadata and beacon specifications.
* Generates standardized outputs in csv format, as well as point and line geospatial formats (GeoPackage, KML) and 3 types of overview plots (a map, a timeseries plot and a plot of velocity/speed distributions).
* Batch processes and collates tracks into the IBTD and outputs summary database statistics for documentation.


## Modules

| Module | Description |
| :-- | :-- |
| `track_readers.py` | Functions to ingest and standardize raw data from numerous beacon formats. |
| `ibtd.py` | Core classes: `Track`, `Meta`, and `Specs` for representing track data, track metadata, and beacon specifications. |
| `track_process.py` | Command-line/API interface for processing single tracks, including reading, standardizing, purging, sorting, deduplicating, calculating speed, trimming, filtering and plotting. |
| `track_fig.py` | Visualization scripts to produce maps, time series, distribution plots and and a preview of the trimming points. |
| `track_collate.py` | Batch processing, collation, and database/statistics generation for all available tracks. |


## Processing workflow

### 1. Prepare metadata and specifications

* Ensure you have a **track metadata file** (e.g., `track_metadata_raw.ods` or `track_metadata.ods`) and a **beacon model specification file** (e.g., `beacon_specs.ods`).
* These files are required for batch processing, and are handy to have for processing a single track.
* These files can be found in the Database or users can create their own if they wish to reprocess track data. 

### 2. Process Individual Tracks

* Use `track_process.py` to work on a single beacon track.
* This script contains functions to convert raw iceberg beacon track data into the Database standard format. It can also work on standard format (already procesed) beacon track files for plotting and other operations. 
* This script runs at the command line, but the workflow functions can be called within custom Python scripts. 
* Works on a single track to: 
    * Run the appropriate **reader** function to access raw data
    * **Standardize** the data format after rejecting invalid data
    * **Purge** data according to stated minimum and maximum values in the beacon specifications (Data values outside beacon model specifications are set to NaN)
    * **Sort** and deduplicate the observations
    * Calculate **displacement, speed and course bearing** between observations
    * **Trim** the track based on supplied `trim_start` and `trim_end` timestamps     
    * **Speed filter** observations iteratively to remove points where speed values are above a set threshold
    * **Calculate statistics** and properties of the track
    * **Create output** files (as required by the user):
        * A csv file of all track observations in the standard format (this is the output format 'of record')
        * A track point and track line GeoPackage file (*_pt.gpkg, *_ln.gpkg)
        * A track point and track line Google Earth file (*_pt.kml, *_ln.kml)
        * A metadata file in json or csv format
        * A map of the track (a map plot, *_map.png)
        * A graph of temperature, displacement and velocity through time (a time plot, *_time.png)
        * A graph of the speed distribution (as a histogram and cumulative histogram) and the distribution of velocities (a polar plot) for the track (a dist plot, *_dist.png)
        * A graph previewing when `trim_start` and `trim_end` timestamps are placed against a time series of raw data (a trim plot, *_trim.png)
        * A log file that details the processing of the raw track data

* Example (trimming a standard track from August 22 to 26, 2021 and outputting a map and a csv file of the data):

```sh
python track_process.py 2021_300434065868240.csv . -s '2021-08-22 13:00:00' -e '2021-08-26 21:00:00' -op map -ot csv
```

* See `track_process.py` for full CLI options and usage details.

### 3. Batch Collation and Database Creation

* Use `track_collate.py` to process all tracks in a directory tree and collate them into a Database.
* `track_collate.py`, in turn, calls `track_process.py` to process tracks with the workflow described above. 
* The script copies the original directory structure and outputs the csv, GeoPackage, KML, metadata, map plot, dist plot and time plot there. 
* The log file and trim plot are placed in the raw folder
* An overview map and some tables for reporting are also generated

* Note that the parameters in `track_collate.py` (input/output directories, run name, etc.) are hard-coded since Database collation is not done frequently.  Edit these parameters and run the script: 

```sh
python track_collate.py
```
* The script will prompt before overwriting output files.

## Standardized Data Format

All tracks are converted to a standard format with the following columns, where available.  Missing data are denoted by `NA`:

| Field                      | Informal description                                                                                          | Units/Format              | Vocabulary | URI                                                                                                                  | Data type |
|:---------------------------|:--------------------------------------------------------------------------------------------------------------|:--------------------------|------------|:---------------------------------------------------------------------------------------------------------------------|:-----------|
| `platform_id`                | Unique identifier from satellite data provider: e.g., IMEI (15 digits) or Argos transmitter number (5 digits) | year_id                   | CF         | https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#platform_id               | string    |
| `timestamp`                  | Date and time of position and other measurements in UTC                                                       | YYYY-MM-DDTHH:MM:SS+HH:MM | ISO8601    | https://www.iso.org/iso-8601-date-and-time-format.html                                                               | string    |
| `latitude`                   | Latitude (WGS84)                                                                                              | degree                    | CF         | https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#latitude                  | float     |
| `longitude`                  | Longitude (WGS84)                                                                                             | degree                    | CF         | https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#longitude                 | float     |
| `air_temperature`            | Air temperature                                                                                               | K                         | CF         | https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#air_temperature           | float     |
| `internal_temperature`       | Temperature measured within beacon enclosure                                                                  | K                         | None       | -                                                                                                                    | float     |
| `surface_temperature`        | Ice/water surface temperature                                                                                 | K                         | CF         | https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#surface_temperature       | float     |
| `air_pressure`               | Barometric pressure                                                                                           | Pa                        | CF         | https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#air_pressure              | float     |
| `platform_pitch`             | Pitch measurement as recorded by accelerometer                                                                | degree                    | CF         | https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#platform_pitch            | float     |
| `platform_roll`              | Roll measurement as recorded by accelerometer                                                                 | degree                    | CF         | https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#platform_roll             | float     |
| `platform_orientation`       | Tilt-compensated heading measurement (relative to magnetic north) of the beacon                               | degree                    | CF         | https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#platform_orientation      | float     |
| `voltage_battery_volts`      | Battery voltage                                                                                               | volts                     | R14        | http://vocab.nerc.ac.uk/collection/R14/current/T100529/                                                              | float     |
| `platform_displacement`      | Distance from previous to current track position                                                              | m                         | None       | -                                                                                                                    | float     |
| `platform_speed_wrt_ground`  | Speed from previous to current track position                                                                 | ms-1                      | CF         | https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#platform_speed_wrt_ground | float     |
| `platform_course`            | Azimuth relative to true north from previous to current track position                                        | degrees                   | CF         | https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html#platform_course           | float     |

## **Dependencies**

* python 3
* pandas, numpy, geopandas>=1.0, matplotlib, seaborn, cartopy, openpyxl, odfpy

## Installation
Dependencies can be installed using conda and the environment.yml file as follows: 

`conda env create -f environment.yml`

`conda activate ibtd`

## Citation and Contact

This software was developed at the Water and Ice Research Lab at Carleton University by Derek Mueller and Adam Garbo with contributions from Jill Rajewicz and Anna Crawford. For questions, contributions, or bug reports, please contact Derek Mueller (derek.mueller@carleton.ca).

Mueller D, Garbo A (2025) Iceberg Beacon Track Database (IBTD) Processing Tools.  (more soon)

## Licence

This software is licenced under GNU General Public License v3 See `LICENSE` file for details.

## References

- For detailed documentation and usage, see...(sphinx to come..)
