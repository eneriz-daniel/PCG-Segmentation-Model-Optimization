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

# Usage: hls_launcher.py [-h] [-b] [-m | -s] [-p]
#                          ([--csim N n0 nenc dataset W I | --csimid file_path id |]
#                          [--synth N n0 nenc W I | --synthid file_path id)] 
#                          [--project_dir project_dir]
#
# Launch HLS processes
#
# optional arguments:
#   -h, --help            show this help message and exit
#   -b, --background      Run as subprocess
#   -m, --memory_sharing  Run memory_sharing implementation. If not passed, baseline version is used.
#   -s, --stream          Run stream implementation. If not passed, baseline version is used.
#   -p, --pragmas         Use pragmas to optimize HLS implementation. It only affects in synthesis.
#   --csim N n0 nenc dataset W I
#                         Launch csim using custom arguments
#   --csimid file_path id
#                         Launch csim using file driven arguments
#   --synth N n0 nenc W I
#                         Launch synth using custom arguments
#   --synthid file_path id
#                         Launch synth using file driven arguments
#   --project_dir project_dir
#                         Custom project directory

from hls_utils import launch_csim, launch_synthesis
import argparse
import os
import json

# Get absolute path to the current file
current_file_path = os.path.dirname(os.path.abspath(__file__))+'/'

print("Current file path: ", current_file_path)

parser = argparse.ArgumentParser(description='Launch HLS processes')

# Parse subprocess flag
parser.add_argument('-b', '--background', help='Run as subprocess', action='store_true')

# Create a non-required mutually exclusive group to parse the implementation type
implementation_type_group = parser.add_mutually_exclusive_group(required=False)

# Parse memory_sharing implementation flag
implementation_type_group.add_argument('-m', '--memory_sharing', help='Run memory_sharing implementation. If not passed, baseline version is used.', action='store_true')

# Parse stream implementation flag
implementation_type_group.add_argument('-s', '--stream', help='Run stream implementation. If not passed, baseline version is used.', action='store_true')

# Parse optimization flags. If used, pragmas are used to optimize the HLS implementation
parser.add_argument('-p', '--pragmas', help='Use pragmas to optimize HLS implementation. It only affects in synthesis.', action='store_true')

mode = parser.add_mutually_exclusive_group(required=True)

csim_group = mode.add_mutually_exclusive_group()
# Add the csim argument to enable access to the launch csim function with custom arguments [N, n0, nenc, dataset, W, I]
csim_group.add_argument('--csim', nargs=6, help='Launch csim using custom arguments', metavar=('N', 'n0', 'nenc', 'dataset', 'W', 'I'), type=int)
# Add the csimid argument to enable access to the launch csim function with file driven arguments
csim_group.add_argument('--csimid', nargs=2, help='Launch csim using file driven arguments', metavar=('file_path', 'id'), type=str)

synth_group = mode.add_mutually_exclusive_group()
# Add the synth argument to enable access to the launch synth function with custom arguments [N, n0, nenc, W, I]
synth_group.add_argument('--synth', nargs=5, help='Launch synth using custom arguments', metavar=('N', 'n0', 'nenc', 'W', 'I'), type=int)
# Add the synthid argument to enable access to the launch synth function with file driven arguments
synth_group.add_argument('--synthid', nargs=2, help='Launch synth using file driven arguments', metavar=('file_path', 'id'), type=str)

# Parse a custom project directory
parser.add_argument('--project_dir', help='Custom project directory', metavar='project_dir')

args = parser.parse_args()

print('Running in background: {}'.format(args.background))

if args.memory_sharing:
    print('Running memory_sharing implementation')
    implementation_type = 'memory-sharing'
elif args.stream:
    print('Running stream implementation')
    implementation_type = 'stream'
else:
    print('Running baseline implementation')
    implementation_type = 'baseline'

if args.pragmas:
    print('Using pragmas to optimize HLS implementation')
    optimization = 'optimized'
else:
    print('Using default HLS implementation')
    optimization = 'default'

if args.csim:
    if args.pragmas:
        print('Pragmas are ignored in csim')
    if args.project_dir:
        launch_csim(args.project_dir, current_file_path, args.csim[0], args.csim[1], args.csim[2], args.csim[3], args.csim[4], args.csim[5], args.background, implementation=implementation_type)
    else:
        project_dir = '{}csim-runs/{}/{}/W{}I{}/N{}n0{}nenc{}/'.format(current_file_path, implementation_type, args.csim[3], args.csim[4], args.csim[5], args.csim[0], args.csim[1], args.csim[2])
        launch_csim(project_dir, current_file_path, args.csim[0], args.csim[1], args.csim[2], args.csim[3], args.csim[4], args.csim[5], args.background, implementation=implementation_type)

if args.csimid:
    if args.pragmas:
        print('Pragmas are ignored in csim')
        
    # Open file_path as a json file
    with open(args.csimid[0]) as f:
        data = json.load(f)
    
    # Get the arguments from the corresponding id in the json file
    csim_args = data[args.csimid[1]]

    if args.project_dir:
        launch_csim(args.project_dir, current_file_path, csim_args['N'], csim_args['n0'], csim_args['nenc'], csim_args['dataset'], csim_args['W'], csim_args['I'], args.background, implementation=implementation_type)
    else:
        project_dir = '{}csim-runs/{}/{}/W{}I{}/N{}n0{}nenc{}/'.format(current_file_path, implementation_type, csim_args['dataset'], csim_args['W'], csim_args['I'], csim_args['N'], csim_args['n0'], csim_args['nenc'])
        launch_csim(project_dir, current_file_path, csim_args['N'], csim_args['n0'], csim_args['nenc'], csim_args['dataset'], csim_args['W'], csim_args['I'], args.background, implementation=implementation_type)

if args.synth:
    if args.project_dir:
        launch_synthesis(args.project_dir, args.synth[0], args.synth[1], args.synth[2], args.synth[3], args.synth[4], args.background, implementation=implementation_type, optimized=args.pragmas)
    else:
        project_dir = '{}synth-runs/{}/{}/W{}I{}/N{}n0{}nenc{}/'.format(current_file_path, implementation_type, optimization, args.synth[3], args.synth[4], args.synth[0], args.synth[1], args.synth[2])
        launch_synthesis(project_dir, args.synth[0], args.synth[1], args.synth[2], args.synth[3], args.synth[4], args.background, implementation=implementation_type, optimized=args.pragmas)

if args.synthid:
    # Open file_path as a json file
    with open(args.synthid[0]) as f:
        data = json.load(f)
    
    # Get the arguments from the corresponding id in the json file
    synth_args = data[args.synthid[1]]

    if args.project_dir:
        launch_synthesis(args.project_dir, synth_args['N'], synth_args['n0'], synth_args['nenc'], synth_args['W'], synth_args['I'], args.background, implementation=implementation_type, optimized=args.pragmas)
    else:
        project_dir = '{}synth-runs/{}/{}/W{}I{}/N{}n0{}nenc{}/'.format(current_file_path, implementation_type, optimization, synth_args['W'], synth_args['I'], synth_args['N'], synth_args['n0'], synth_args['nenc'])
        launch_synthesis(project_dir, synth_args['N'], synth_args['n0'], synth_args['nenc'], synth_args['W'], synth_args['I'], args.background, implementation=implementation_type, optimized=args.pragmas)