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

# Usage: postsynth.py [-h] --implementation IMPLEMENTATION [--pragmas]
#                     [--N_list N_LIST [N_LIST ...]]
#                     [--n0_list N0_LIST [N0_LIST ...]]
#                     [--nenc_list NENC_LIST [NENC_LIST ...]]
#                     [--W_list W_LIST [W_LIST ...]]
#                     [--I_list I_LIST [I_LIST ...]]
#
# Post-synthesis analysis.
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --implementation IMPLEMENTATION, -i IMPLEMENTATION
#                         The implementation to analyze. Can be "preliminary",
#                         "memory-sharing" or "stream".
#   --pragmas, -p         Analyze the optimized implementation if passed. If
#                         not, it will use the default implementation results.
#   --N_list N_LIST [N_LIST ...]
#                         The N values to analyze. Defaults to 64, 128, 256 and
#                         512.
#   --n0_list N0_LIST [N0_LIST ...]
#                         The n0 values to analyze. Defaults to 8, 7, 6, 5 and
#                         4.
#   --nenc_list NENC_LIST [NENC_LIST ...]
#                         The nenc values to analyze. Defaults to 4, 3, 2 and 1.
#
# WI:
#   The W and I values to analyze.
#
#   --W_list W_LIST [W_LIST ...]
#                         The W values to analyze. Defaults to 16.
#   --I_list I_LIST [I_LIST ...]
#                         The I values to analyze. Only combinations where
#                         W > I will be used. Defaults to 8.


import os
import numpy as np
import pandas as pd
import argparse
import json

# Parse the arguments
parser = argparse.ArgumentParser(description='Post-synthesis analysis.')
parser.add_argument('--implementation', '-i', type=str, help='The implementation to analyze. Can be "preliminary", "memory-sharing" or "stream".', required=True)
parser.add_argument('--pragmas', '-p', action='store_true', help='Analyze the optimized implementation if passed. If not, it will use the default implementation results.', default=False)
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

optimization = 'optimized' if args.pragmas else 'default'

for WI in WI_list:

  bram_table = np.zeros((len(args.N_list), len(args.n0_list), len(args.nenc_list)))
  lut_table = np.zeros((len(args.N_list), len(args.n0_list), len(args.nenc_list)))
  ff_table = np.zeros((len(args.N_list), len(args.n0_list), len(args.nenc_list)))
  dsp_table = np.zeros((len(args.N_list), len(args.n0_list), len(args.nenc_list)))

  latency_best_table = np.zeros((len(args.N_list), len(args.n0_list), len(args.nenc_list)))
  latency_avg_table = np.zeros((len(args.N_list), len(args.n0_list), len(args.nenc_list)))
  latency_worst_table = np.zeros((len(args.N_list), len(args.n0_list), len(args.nenc_list)))

  # Create the excel file
  writer = pd.ExcelWriter('synth-runs/{}/{}/W{}I{}/synth-{}-{}-W{}I{}.xlsx'.format(args.implementation, optimization, WI[0], WI[1], args.implementation, optimization, WI[0], WI[1]), engine='xlsxwriter')

  for i, N in enumerate(args.N_list):
      for j, n0 in enumerate(args.n0_list):
          for k, nenc in enumerate(args.nenc_list):

              json_file = 'synth-runs/{}/{}/W{}I{}/N{}n0{}nenc{}/segmenter-synth/solution1/solution1_data.json'.format(args.implementation, optimization, WI[0], WI[1], N, n0, nenc)
              with open(json_file) as f:
                  data = json.load(f)
              
              bram_table[i,j,k] = data['ModuleInfo']['Metrics']['Segmenter']['Area']['BRAM_18K']
              lut_table[i,j,k] = data['ModuleInfo']['Metrics']['Segmenter']['Area']['LUT']
              ff_table[i,j,k] = data['ModuleInfo']['Metrics']['Segmenter']['Area']['FF']
              dsp_table[i,j,k] = data['ModuleInfo']['Metrics']['Segmenter']['Area']['DSP48E']

              # Check is there is latency data
              latency_data_available = data['ModuleInfo']['Metrics']['Segmenter']['Latency']['LatencyBest'] != ''

              if latency_data_available:
                  latency_best_table[i,j,k] = data['ModuleInfo']['Metrics']['Segmenter']['Latency']['LatencyBest']
                  latency_avg_table[i,j,k] = data['ModuleInfo']['Metrics']['Segmenter']['Latency']['LatencyAvg']
                  latency_worst_table[i,j,k] = data['ModuleInfo']['Metrics']['Segmenter']['Latency']['LatencyWorst'] 
      
      # Save data in a dataframe
      bram_df = pd.DataFrame(bram_table[i], index=args.n0_list, columns=args.nenc_list)
      bram_df.index.name = 'Base filter \ Number of blocks'

      lut_df = pd.DataFrame(lut_table[i], index=args.n0_list, columns=args.nenc_list)
      lut_df.index.name = 'Base filter \ Number of blocks'

      ff_df = pd.DataFrame(ff_table[i], index=args.n0_list, columns=args.nenc_list)
      ff_df.index.name = 'Base filter \ Number of blocks'

      dsp_df = pd.DataFrame(dsp_table[i], index=args.n0_list, columns=args.nenc_list)
      dsp_df.index.name = 'Base filter \ Number of blocks'

      latency_best_df = pd.DataFrame(latency_best_table[i], index=args.n0_list, columns=args.nenc_list)
      latency_best_df.index.name = 'Base filter \ Number of blocks'

      latency_avg_df = pd.DataFrame(latency_avg_table[i], index=args.n0_list, columns=args.nenc_list)
      latency_avg_df.index.name = 'Base filter \ Number of blocks'

      latency_worst_df = pd.DataFrame(latency_worst_table[i], index=args.n0_list, columns=args.nenc_list)
      latency_worst_df.index.name = 'Base filter \ Number of blocks'

      # Save dataframe as a excel file
      bram_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=0)
      dsp_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=6)
      ff_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=12)
      lut_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=18)
      
      latency_best_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=24)
      latency_avg_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=30)
      latency_worst_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=36)

      # Add the title to each table
      sheet = writer.sheets['N{}'.format(N)]
      sheet.write_string(0, 0, 'BRAM')
      sheet.write_string(0, 6, 'DSP')
      sheet.write_string(0, 12, 'FF')
      sheet.write_string(0, 18, 'LUT')
      sheet.write_string(0, 24, 'LatencyBest')
      sheet.write_string(0, 30, 'LatencyAvg')
      sheet.write_string(0, 36, 'LatencyWorst')

  #Save the numpy tables
  np.savez('synth-runs/{}/{}/W{}I{}/synth-{}-{}-W{}I{}.npz'.format(args.implementation, optimization, WI[0], WI[1], args.implementation, optimization, WI[0], WI[1]), bram_table=bram_table, lut_table=lut_table, ff_table=ff_table, dsp_table=dsp_table, latency_best_table=latency_best_table, latency_avg_table=latency_avg_table, latency_worst_table=latency_worst_table)

  writer.save()