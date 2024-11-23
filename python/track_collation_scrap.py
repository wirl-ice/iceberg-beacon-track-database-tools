#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_collation.py

Batch process many tracks and/or collate many tracks into a larger database 


Created on Sun Jul 28 04:33:24 2024

@author: dmueller
"""

import os
import pandas as pd
import track_processing
from itbd import Meta, Models, Track

from pathlib import Path

meta_file = (
    "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/database/metadata.ods"
)
spec_file = "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/database/models.ods"
scandir = "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/raw_data/"
outdir = "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/test/"

# note this will remove all files and folders from ... press y to continue

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

log = track_processing.tracklog("Test", outdir, level="INFO")
log.info("Starting run.....")

metadata = Meta(meta_file, log)
modeldata = Models(spec_file, log)

alltrack_meta = pd.DataFrame()

for root, dirs, files in os.walk(scandir, topdown=True):
    dirs.sort()  # this will keep the processing more ordered
    for d in dirs:
        if not os.path.isdir(os.path.join(outdir, root[prefix:], d)):
            os.mkdir(os.path.join(outdir, root[prefix:], d))
    for f in files:
        if Path(f).stem in metadata.df.beacon_id.values:
            print(f"\n\n......{f}.......")

            # df = track_processing.track_process(
            #     os.path.join(root, f),
            #     os.path.join(outdir, root[prefix:]),
            #     metadata=metadata,
            #     reader=None,
            #     specs=modeldata,
            #     track_start=None,  # need to pull from Meta
            #     track_end=None,  # need to pull from Meta
            #     output_file=None,
            #     output_types=["csv"],
            # )

            trk = Track(os.path.join(root, f), metadata=metadata, logger=log)
            trk.load_model_specs(modeldata)
            trk.clean()
            trk.sort()
            trk.speed()
            # trk.trim()
            # trk.speed_limit()
            trk.output(
                ["csv", "pt_kml", "ln_kml"],
                path_output=os.path.join(outdir, root[prefix:]),
            )

            # do this after all the data have been cleaned.
            trk_meta = trk.track_metadata("pandas")

            trk.plot_map(path_output=os.path.join(outdir, root[prefix:]))
            trk.plot_temp(os.path.join(outdir, root[prefix:]))
            trk.plot_dist(os.path.join(outdir, root[prefix:]))
            trk.plot_time(os.path.join(outdir, root[prefix:]))

            log.info("Completed track processing... \n")

            alltrack_meta = pd.concat([alltrack_meta, trk_meta]).reset_index(drop=True)


alltrack_meta.to_csv(os.path.join(outdir, "test.csv"), index=False)
