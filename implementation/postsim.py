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

# Usage: postsim.py [-h] --implementation IMPLEMENTATION
#                     [--datasets_list DATASETS_LIST [DATASETS_LIST ...]]
#                     [--N_list N_LIST [N_LIST ...]]
#                     [--n0_list N0_LIST [N0_LIST ...]]
#                     [--nenc_list NENC_LIST [NENC_LIST ...]]
#                     [--W_list W_LIST [W_LIST ...]]
#                     [--I_list I_LIST [I_LIST ...]]
#
# Post-simulation analysis of CSim results.
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --implementation IMPLEMENTATION, -i IMPLEMENTATION
#                         The implementation to analyze. Can be "preliminary",
#                         "memory-sharing" or "stream".
#   --datasets_list DATASETS_LIST [DATASETS_LIST ...]
#                         The datasets to analyze. Defaults to 2016 and 2022.
#   --N_list N_LIST [N_LIST ...]
#                         The N values to analyze. Defaults to 64, 128, 256 and 512.
#   --n0_list N0_LIST [N0_LIST ...]
#                         The n0 values to analyze. Defaults to 8, 7, 6, 5 and 4.
#   --nenc_list NENC_LIST [NENC_LIST ...]
#                         The nenc values to analyze. Defaults to 4, 3, 2 and 1.
#
# WI:
#   The W and I values to analyze.
#
#   --W_list W_LIST [W_LIST ...]
#                         The W values to analyze. Defaults to 16.
#   --I_list I_LIST [I_LIST ...]
#                         The I values to analyze. Only combinations where W > I will be used. Defaults to 8.


import os
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.metrics import CategoricalAccuracy
from typing import Tuple
import pandas as pd
from models import get_model_parametrized
from tqdm import trange
import argparse

# Remove TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def evaluate_csim(project_directory: str, model: Model, X: np.ndarray, S: np.ndarray, outfile: str = None) -> float:
  """Evaluates the csim results computing its accuracy.

  Args:
    project_directory (str): Path to the CSim project directory.
    model (Model): The floating-point-based model to compare to.
    X (np.ndarray): The test data.
    S (np.ndarray): The test labels.
    outfile (str, optional): Path to the output file. Defaults to None.

  Return:
    (float, float): The accuracies of the fixed-point model and the floating-point model.
  """

  # If project_directory does not end with '/' add it
  if project_directory[-1] != '/':
    project_directory += '/'

  # Open the csim-results.txt file and check the number of lines is the same as the number of test samples, raise an error otherwise
  with open(project_directory + 'csim-results.txt', 'r') as f:
    lines = f.readlines()
    if len(lines) != S.shape[0]:
      raise ValueError('The number of lines in csim-results.txt is not the same as the number of test samples.')

  # Load the csim-results.txt file and unflatten the outputs
  S_fixed = np.loadtxt(project_directory + 'csim-results.txt')
  S_fixed = S_fixed.reshape(S.shape)
  
  # Compute the categorical accuracy
  cat_acc = CategoricalAccuracy()
  cat_acc.reset_states()
  cat_acc.update_state(S, S_fixed)

  fixed_point_acc = cat_acc.result().numpy()

  S_float = model.predict(X)

  cat_acc.reset_states()
  cat_acc.update_state(S, S_float)
  floating_point_acc = cat_acc.result().numpy()

  if outfile is not None:
    with open(outfile, 'w') as f:
        f.write('Fixed point accuracy: {} %\n'.format(round(100*fixed_point_acc, 3)))
        f.write('Floating point accuracy: {} %\n'.format(round(100*floating_point_acc, 3)))
        f.write('Difference: {} %\n'.format(round(100*(floating_point_acc - fixed_point_acc), 3)))

  return fixed_point_acc, floating_point_acc

# Parse the arguments
parser = argparse.ArgumentParser(description='Post-simulation analysis of CSim results.')
parser.add_argument('--implementation', '-i', type=str, help='The implementation to analyze. Can be "preliminary", "memory-sharing" or "stream".', required=True)
parser.add_argument('--datasets_list', type=int, nargs='+', default=[2016, 2022], help='The datasets to analyze. Defaults to 2016 and 2022.')
parser.add_argument('--N_list', type=int, nargs='+', default=[64, 128, 256, 512], help='The N values to analyze. Defaults to 64, 128, 256 and 512.')
parser.add_argument('--n0_list', type=int, nargs='+', default=[8,7,6,5,4], help='The n0 values to analyze. Defaults to 8, 7, 6, 5 and 4.')
parser.add_argument('--nenc_list', type=int, nargs='+', default=[4,3,2,1], help='The nenc values to analyze. Defaults to 4, 3, 2 and 1.')

# Create a group to parse the values of W and I
WI_group = parser.add_argument_group('WI', 'The W and I values to analyze.')
WI_group.add_argument('--W_list', type=int, nargs='+', default=[16], help='The W values to analyze. Defaults to 16.')
WI_group.add_argument('--I_list', type=int, nargs='+', default=[8], help='The I values to analyze. Only combinations where W > I will be used. Defaults to 8.')
                      
args = parser.parse_args()

# Check the implementation is valid
if args.implementation not in ['preliminary', 'memory-sharing', 'stream']:
  raise ValueError('Invalid implementation. Must be "preliminary", "memory-sharing" or "stream".')

WI_list = [[W, I] for W in args.W_list for I in args.I_list if W > I]

for dataset in args.datasets_list:
  for WI in WI_list:

    # Check if the directory exists
    if not os.path.isdir('csim-runs/{}/{}/W{}I{}/'.format(args.implementation, dataset, WI[0], WI[1])):
      raise ValueError('The directory csim-runs/{}/{}/W{}I{}/ does not exist.'.format(args.implementation, dataset, WI[0], WI[1]))

    fixedp_acc_table = np.zeros((len(args.N_list), len(args.n0_list), len(args.nenc_list)))
    acc_table = np.zeros((len(args.N_list), len(args.n0_list), len(args.nenc_list)))
    diff_table = np.zeros((len(args.N_list), len(args.n0_list), len(args.nenc_list)))

    # Create the excel file
    writer = pd.ExcelWriter('csim-runs/{}/{}/W{}I{}/csim-acc-{}-{}-W{}I{}.xlsx'.format(args.implementation, dataset, WI[0], WI[1], args.implementation, dataset, WI[0], WI[1]), engine='xlsxwriter')

    counter = 1
    max_count = len(args.N_list)*len(args.n0_list)*len(args.nenc_list)

    for i, N in enumerate(args.N_list):
        for j, n0 in enumerate(args.n0_list):
            for k, nenc in enumerate(args.nenc_list):
                print('Analyzing {} implementation dataset={}, N={}, n0={}, nenc={}. {}/{}'.format(args.implementation, dataset, N, n0, nenc, counter, max_count))

                project_dict = 'csim-runs/{}/{}/W{}I{}/N{}n0{}nenc{}/'.format(args.implementation, dataset, WI[0], WI[1], N, n0, nenc)

                with np.load('{}-data/N{}-data/test.npz'.format(dataset, N)) as data:
                    X = data['X']
                    S = data['S']
                
                model = get_model_parametrized(N, n0, nenc)
                model.load_weights('models/{}/N{}/n0{}/nenc{}/parameters.h5'.format(dataset, N, n0, nenc))

                print(' - Loaded floating point total rec accuracy: {} %'.format(round(100*np.load('models/{}/N{}/n0{}/nenc{}/test_metrics.npz'.format(dataset, N, n0, nenc))['total_rec_acc'], 3)))
                print(' - Loaded floating point global accuracy: {} %'.format(round(100*np.load('models/{}/N{}/n0{}/nenc{}/test_metrics.npz'.format(dataset, N, n0, nenc))['global_acc'], 3)))

                fixedp_acc_table[i,j,k], acc_table[i,j,k] = evaluate_csim(project_dict, model, X, S)
                diff_table[i,j,k] = acc_table[i,j,k] - fixedp_acc_table[i,j,k]
                print(' - Fixed point accuracy: {} %'.format(round(100*fixedp_acc_table[i,j,k], 3)))
                print(' - Floating point accuracy: {} %'.format(round(100*acc_table[i,j,k], 3)))
                
                counter += 1
        
        # Save data in a dataframe
        fixedp_acc_df = pd.DataFrame(fixedp_acc_table[i], index=args.n0_list, columns=args.nenc_list)
        fixedp_acc_df.index.name = 'Base filter \ Number of blocks'

        acc_df = pd.DataFrame(acc_table[i], index=args.n0_list, columns=args.nenc_list)
        acc_df.index.name = 'Base filter \ Number of blocks'

        diff_df = pd.DataFrame(diff_table[i], index=args.n0_list, columns=args.nenc_list)
        diff_df.index.name = 'Base filter \ Number of blocks'

        # Save dataframe as a excel file
        acc_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=0)
        fixedp_acc_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=6)
        diff_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=12)

        # Add the title to each table
        sheet = writer.sheets['N{}'.format(N)]
        sheet.write_string(0, 0, 'Floating point accuracy')
        sheet.write_string(0, 6, 'Fixed point accuracy')
        sheet.write_string(0, 12, 'Difference')
    
    # Save the numpy tables
    np.savez('csim-runs/{}/{}/W{}I{}/csim-acc-W{}I{}.npz'.format(args.implementation, dataset, WI[0], WI[1], WI[0], WI[1]), fixedp_acc_table=fixedp_acc_table, acc_table=acc_table, diff_table=diff_table)

    writer.save()

