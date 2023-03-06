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
# train_launcher.py [-h]
#                   (--parameters_file file_path id | --parameters N n0 nenc)
#                   [--CV] [--fold FOLD]
#                   [--training_hyperparameters EPOCHS BATCH_SIZE LEARNING_RATE]
#                   [--model_path MODEL_PATH]
#                   {2016,2022}
#
# Train the segmentation model passing the model parameters through the command line
#
# positional arguments:
#   {2016,2022}           Dataset to use for training
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --parameters_file file_path id
#                         Use file driven model parameters
#   --parameters N n0 nenc
#                         Use custom model parameters. Input window size, number of
#                         filters in the first layer and number of encoder/decoder
#                         blocks.
#   --training_hyperparameters EPOCHS BATCH_SIZE LEARNING_RATE
#                         Use custom training hyperparameters
#   --model_path MODEL_PATH
#                         Path to store the trained model and training details
#
# Cross validation:
#   --CV                  Use 10-fold cross validation.
#   --fold FOLD           Fold number to use for training

import argparse
import json
from utils.training import train, train_fold
import numpy as np

epochs_default = 15
batch_size_default = 1
learning_rate_default = 1e-4

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train the segmentation model passing the model parameters through the command line")

# Add mandatory arguments, the model parameters or the file driven parameters
model_params_selection = parser.add_mutually_exclusive_group(required=True)

# Add the paramters_file argument
model_params_selection.add_argument("--parameters_file", nargs=2, help='Use file driven model parameters', metavar=('file_path', 'id'), type=str)

# Create a group for the model parameters. The first 3 are mandatory, the last one is optional
model_params_selection.add_argument("--parameters", nargs=3, help='Use custom model parameters. Input window size, number of filters in the first layer and number of encoder/decoder blocks.', metavar=('N', 'n0', 'nenc'), type=int)

# Add a CV argument to select if 10-fold CV is used. If it is used, the parameters file must contain the fold number or the fold number must be passed as an argument
# Add a group for the CV
CV_selection = parser.add_argument_group('Cross validation')
CV_selection.add_argument("--CV", help='Use 10-fold cross validation.', action='store_true')
CV_selection.add_argument("--fold", help='Fold number to use for training', type=int)

# Add the training hyperparameters as optional arguments
parser.add_argument("--training_hyperparameters", nargs=3, help='Use custom training hyperparameters', metavar=('EPOCHS', 'BATCH_SIZE', 'LEARNING_RATE'), default=[epochs_default, batch_size_default, learning_rate_default])

parser.add_argument("--model_path", help='Path to store the trained model and training details')

# Add the desired dataset as a mandatory argument. It must be 2016 or 2022
parser.add_argument("dataset", help='Dataset to use for training', choices=['2016', '2022'])

args = parser.parse_args()

print(args.CV)

# Catch the training hyperparameters
epochs = int(args.training_hyperparameters[0])
batch_size = int(args.training_hyperparameters[1])
learning_rate = float(args.training_hyperparameters[2])

# Catch the model hyperparameters
if args.parameters_file:
    # Open the file
    with open(args.parameters_file[0], 'r') as f:
        # Load the hyperparameters
        parameters_dict = json.load(f)[args.parameters_file[1]]
        N = int(parameters_dict['N'])
        n0 = int(parameters_dict['n0'])
        nenc = int(parameters_dict['nenc'])

        if args.CV:
            try:
                fold = int(parameters_dict['fold'])
            except KeyError:
                raise KeyError('Under CV mode, the hyperparameters file must contain the fold number')
else:
    N = int(args.parameters[0])
    n0 = int(args.parameters[1])
    nenc = int(args.parameters[2])

    fold = args.CV

# Check if the fold number is valid
if args.CV:
    if fold < 0 or fold > 9:
        raise ValueError('The fold number must be between 1 and 10')

# If 10-fold CV is used
if args.CV:

    tau = N//8

    X_sets = []
    S_sets = []
    dicts = []
    # Load the 10 folds X, S and dicts from {}-CV/N{}-data/fold_{}.npz.format(args.dataset, N, i)
    for i in range(10):
        X_sets.append(np.load('{}-CV/N{}-data/fold_{}.npz'.format(args.dataset, N, i))['X'])
        S_sets.append(np.load('{}-CV/N{}-data/fold_{}.npz'.format(args.dataset, N, i))['S'])
        with open('{}-CV/N{}-data/fold_{}.json'.format(args.dataset, N, i)) as f:
            dicts.append(json.load(f))

    if args.model_path is None:
        model_path = 'models-cv/{}/N{}/n0{}/nenc{}/'.format(args.dataset, N, n0, nenc)
    else:
        model_path = args.model_path
    
    # Call the training function
    train_fold(fold, X_sets, S_sets, dicts, model_path, N, tau, n0, nenc, epochs, batch_size, learning_rate, args.dataset)

# When 10-fold CV is not used
else:
    # Load the data
    X_train = np.load('{}-data/N{}-data/train.npz'.format(args.dataset, N))['X']
    S_train = np.load('{}-data/N{}-data/train.npz'.format(args.dataset, N))['S']
    X_val = np.load('{}-data/N{}-data/valid.npz'.format(args.dataset, N))['X']
    S_val = np.load('{}-data/N{}-data/valid.npz'.format(args.dataset, N))['S']
    X_test = np.load('{}-data/N{}-data/test.npz'.format(args.dataset, N))['X']
    S_test = np.load('{}-data/N{}-data/test.npz'.format(args.dataset, N))['S']

    # Load the test dict
    with open('{}-data/N{}-data/test_dict.json'.format(args.dataset, N), 'r') as f:
        test_dict = json.load(f)

    tau = N//8

    if args.model_path is None:
        model_path = 'models/{}/N{}/n0{}/nenc{}/'.format(args.dataset, N, n0, nenc)
    else:
        model_path = args.model_path

    train(X_train, S_train, X_val, S_val, X_test, S_test, test_dict, model_path, N, tau, n0, nenc, epochs, batch_size, learning_rate, args.dataset)
