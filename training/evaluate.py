# Copyright (C) 2023 Daniel Enériz and Antonio Rodriguez-Almeida
# 
# This file is part of PCG Segmentation Model Optimization.
# 
# PCG Segmentation Model Optimization is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PCG Segmentation Model Optimization is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PCG Segmentation Model Optimization.  If not, see <http://www.gnu.org/licenses/>.

# Usage:
# evaluate.py [-h] [--datasets_list DATASETS_LIST [DATASETS_LIST ...]]
#             [--N_list N_LIST [N_LIST ...]] [--n0_list N0_LIST [N0_LIST ...]]
#             [--nenc_list NENC_LIST [NENC_LIST ...]] [--CV]
#
# Evaluate the model
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --datasets_list DATASETS_LIST [DATASETS_LIST ...]
#                         List of datasets to evaluate
#   --N_list N_LIST [N_LIST ...]
#                         List of N values to evaluate
#   --n0_list N0_LIST [N0_LIST ...]
#                         List of n0 values to evaluate
#   --nenc_list NENC_LIST [NENC_LIST ...]
#                         List of nenc values to evaluate
#   --CV                  Evaluate the cross-validation models

import argparse
from utils.posttraining import parse_metrics, parse_cv_metrics

parser = argparse.ArgumentParser(description='Evaluate the model')

parser.add_argument("--datasets_list", type=int, nargs='+', default=[2016, 2022], help="List of datasets to evaluate")

parser.add_argument("--N_list", type=int, nargs='+', default=[64, 128, 256, 512], help="List of N values to evaluate")
parser.add_argument("--n0_list", type=int, nargs='+', default=[8, 7, 6, 5, 4], help="List of n0 values to evaluate")
parser.add_argument("--nenc_list", type=int, nargs='+', default=[4, 3, 2, 1], help="List of nenc values to evaluate")

# Add an optional argument to parse the cross-validation metrics
parser.add_argument("--CV", action="store_true", help="Evaluate the cross-validation models")

args = parser.parse_args()

if args.CV:
    parse_cv_metrics(args.datasets_list, args.N_list, args.n0_list, args.nenc_list)
else:
    parse_metrics(args.datasets_list, args.N_list, args.n0_list, args.nenc_list)