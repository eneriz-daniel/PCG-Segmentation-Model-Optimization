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

# Usage preparedata.py [-h] [--datasets_list DATASETS_LIST [DATASETS_LIST ...]]
#                        [--N_list N_LIST [N_LIST ...]]
#                        [--n0_list N0_LIST [N0_LIST ...]]
#                        [--nenc_list NENC_LIST [NENC_LIST ...]]
#
# Prepare data for HLS
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --datasets_list DATASETS_LIST [DATASETS_LIST ...]
#                         List of datasets to prepare. Default: 2016, 2022
#   --N_list N_LIST [N_LIST ...]
#                         List of N values to prepare. Default: 64, 128, 256, 512
#   --n0_list N0_LIST [N0_LIST ...]
#                         List of n0 values to prepare. Default: 8, 7, 6, 5, 4
#   --nenc_list NENC_LIST [NENC_LIST ...]
#                         List of nenc values to prepare. Default: 4, 3, 2, 1

import argparse
import numpy as np
from hls_utils import save_model_paramters_as_npy
from models import get_model_parametrized
import os

parser = argparse.ArgumentParser(description='Prepare data for HLS')
parser.add_argument('--datasets_list', type=int, nargs='+', default=[2016, 2022],
                    help='List of datasets to prepare. Default: 2016, 2022')
parser.add_argument('--N_list', type=int, nargs='+', default=[64, 128, 256, 512],
                    help='List of N values to prepare. Default: 64, 128, 256, 512')
parser.add_argument('--n0_list', type=int, nargs='+', default=[8, 7, 6, 5, 4],
                    help='List of n0 values to prepare. Default: 8, 7, 6, 5, 4')
parser.add_argument('--nenc_list', type=int, nargs='+', default=[4, 3, 2, 1],
                    help='List of nenc values to prepare. Default: 4, 3, 2, 1')

args = parser.parse_args()

bs=250
for dataset in args.datasets_list:
    for N in args.N_list:

        X = np.load('{}-data/N{}-data/test.npz'.format(dataset, N))['X']

        for n0 in args.n0_list:
            for nenc in args.nenc_list:
                # Load the model
                model = get_model_parametrized(N, n0, nenc)
                model.load_weights('models/{}/N{}/n0{}/nenc{}/parameters.h5'.format(dataset, N, n0, nenc))

                # Save the model parameters as npy files
                save_model_paramters_as_npy(model, 'models/{}/N{}/n0{}/nenc{}/'.format(dataset, N, n0, nenc), nenc)
        
        #Create a inputs folder in the N{}-data folder
        if not os.path.exists('{}-data/N{}-data/inputs'.format(dataset, N)):
            os.makedirs('{}-data/N{}-data/inputs'.format(dataset, N))

        # Split the data into batchs of size 250 (as it is near the maximum allowe by
        # the C compiler) and same them in X_%d.npy files
        for i in range(X.shape[0]//bs):
            np.save('{}-data/N{}-data/inputs/X_{}.npy'.format(dataset, N, i), X[i*bs:(i+1)*bs])
        
        # Save the last batch
        np.save('{}-data/N{}-data/inputs/X_{}.npy'.format(dataset, N, X.shape[0]//bs), X[(X.shape[0]//bs)*bs:])

        # Save the number of test elements in test_data_elements.txt
        with open('{}-data/N{}-data/test_data_elements.txt'.format(dataset, N), 'w') as f:
            f.write(str(X.shape[0]))