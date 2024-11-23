#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 19:30:52 2024

@author: dmueller
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd

# custom modules
from itbd import Track, Meta, Models, nolog

meta = Meta(
    "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/database/metadata.ods"
)
models = Models(
    "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/database/models.ods"
)
data_file = "/home/dmueller/Desktop/cis_iceberg_beacon_database_0.3/raw_data/2000/11254/2000_11254.txt"
trk = Track(data_file, metadata=meta)
trk.load_model_specs(models)
trk.clean()
trk.sort()
trk.speed()
trk.speed_limit()
# trk.trim()
# trk.output(output_types, path_output=output_path, file_output=output_file)
# trk_meta = trk.track_metadata("pandas")
# trk.plot_map(interactive=True)
trk.plot_temp(interactive=True)

trk.load_model_specs(models)
x = trk.track_metadata("pandas", export=True)
