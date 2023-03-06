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

# This files enables the user to prepare the data for the 2016 or 2022
# datasets. The user can choose to use 10-fold cross-validation or not. The
# user can choose to use all N values or a subset of them. The user can choose
# to use the default directories for the data or to use custom directories.
# The user can choose to use the default proportions for the data or to use custom
# proportions. The user can choose to compress the test data or not. The user can
# choose to force the data to be re-prepared or not.
#
# Usage:
# preparedata.py [-h] [--p2016 | --p2022] [--CV] [--N_list N_LIST [N_LIST ...]]
#                [--process_dir_2016 PROCESS_DIR_2016]
#                [--process_dir_2022 PROCESS_DIR_2022]
#                [--out_dir_2016 OUT_DIR_2016] [--out_dir_2022 OUT_DIR_2022]
#                [--proportions train valid test]
#                [--compress [file_name_fmt_{year}.tgz]] [--force]
#
# Prepare data for 2016 and/or 2022 dataset
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --p2016               Prepare 2016 dataset
#   --p2022               Prepare 2022 dataset
#   --CV                  Use 10-fold cross-validation
#   --N_list N_LIST [N_LIST ...], -N N_LIST [N_LIST ...]
#                         N value(s) to use
#   --process_dir_2016 PROCESS_DIR_2016
#                         Processing data directory for 2016 dataset
#   --process_dir_2022 PROCESS_DIR_2022
#                         Processing data directory for 2022 dataset
#   --out_dir_2016 OUT_DIR_2016
#                         Output data directory for 2016 dataset
#   --out_dir_2022 OUT_DIR_2022
#                         Output data directory for 2022 dataset
#   --proportions train valid test, -p train valid test
#                         Proportion of data to use for training, validation and
#                         testing
#   --compress [file_name_fmt_{year}.tgz], -c [file_name_fmt_{year}.tgz]
#                         Compress the test data into a TGZ file. You can give a
#                         file name format to use. Use "test_data_{}.tgz" as
#                         default
#   --force, -f           Force the data to be re-prepared

from utils.preprocessing import prepare_dataset_2022, find_valid_data_2022, \
    create_train_valid_test_datasets, generate_X_S_from_dict_2022, \
    prepare_dataset_2016, find_valid_data_2016, generate_X_S_from_dict_2016, \
    equally_sum_subset_partition
import os
import numpy as np
import json
import argparse

# Fix numpy random seed
np.random.seed(33)

# Catch arguments from CLI to launch 2016 ando/or 2022 dataset preparation
parser = argparse.ArgumentParser(description='Prepare data for 2016 and/or 2022 dataset')

# Add a group of arguments, one for each dataset. If none is given, prepare both datasets
group = parser.add_mutually_exclusive_group()
group.add_argument('--p2016', action='store_true', help='Prepare 2016 dataset')
group.add_argument('--p2022', action='store_true', help='Prepare 2022 dataset')

# Add a CV argument to catch if the user wants to use 10-fold cross-validation. If not given, do not use cross-validation
parser.add_argument('--CV', action='store_true', help='Use 10-fold cross-validation')

# Add optional arguments to parser to catch:
# N value. If not given, use all N values [64, 128, 256, 512]
parser.add_argument('--N_list', '-N', type=int, nargs='+', help='N value(s) to use')

# Data directory for each dataset. If not given, use default directories
parser.add_argument('--process_dir_2016', type=str, help='Processing data directory for 2016 dataset')
parser.add_argument('--process_dir_2022', type=str, help='Processing data directory for 2022 dataset')

# Output directory for each dataset. If not given, use default directories
parser.add_argument('--out_dir_2016', type=str, help='Output data directory for 2016 dataset')
parser.add_argument('--out_dir_2022', type=str, help='Output data directory for 2022 dataset')

# Proportion of data to use for training, validation and testing. Use a list of 3 floats. If not given, use [0.6, 0.2, 0.2], use metavar to show the format
parser.add_argument('--proportions', '-p', type=float, nargs=3, metavar=('train', 'valid', 'test'), help='Proportion of data to use for training, validation and testing')

# Compress the test data into a TGZ file. If not given, do not compress. Admit a str as argument to give the name of the TGZ file. Use 'test_data_{year}.tgz' as default
parser.add_argument('--compress', '-c', type=str, nargs='?', metavar='file_name_fmt_{year}.tgz', const='test_data_{}.tgz', help='Compress the test data into a TGZ file. You can give a file name format to use. Use "test_data_{}.tgz" as default')

# Force the data to be re-prepared. If not given, do not re-prepare
parser.add_argument('--force', '-f', action='store_true', help='Force the data to be re-prepared')

# Parse arguments
args = parser.parse_args()

# CHECK THE ARGUMENTS

# If no dataset is given, prepare both
if not args.p2016 and not args.p2022:
    args.p2016 = True
    args.p2022 = True

# If CV is not given, do not use cross-validation
if not args.CV:
    args.CV = False

# If no N value is given, use all N values
if not args.N_list:
    args.N_list = [64, 128, 256, 512]

# If no data directory is given, use default directories
if not args.process_dir_2016:
    args.process_dir_2016 = "2016-proc/"
# Else, check if the given directory exists
else:
    # If under Windows, check if the given directory has a trailing slash
    if os.name == 'nt':
        if args.process_dir_2016[-1] != "\\":
            args.process_dir_2016 += "\\"
    # Else, check if the given directory has slash
    else:
        if args.process_dir_2016[-1] != "/":
            args.process_dir_2016 += "/"

    if not os.path.exists(args.process_dir_2016):
        os.mkdir(args.process_dir_2016)

# If no data directory is given, use default directories
if not args.process_dir_2022:
    args.process_dir_2022 = "2022-proc/"
# Else, check if the given directory exists
else:
    # If under Windows, check if the given directory has a trailing slash
    if os.name == 'nt':
        if args.process_dir_2022[-1] != "\\":
            args.process_dir_2022 += "\\"
    # Else, check if the given directory has slash
    else:
        if args.process_dir_2022[-1] != "/":
            args.process_dir_2022 += "/"
    
    if not os.path.exists(args.process_dir_2022):
        os.mkdir(args.process_dir_2022)

# If no output directory is given, use default directories
if not args.out_dir_2016:
    # If cross-validation is used, use a different directory
    if args.CV:
        args.out_dir_2016 = "2016-CV/"
    # Else, use the default directory
    else:
        args.out_dir_2016 = "2016-data/"
# Else, check if the given directory exists
else:
    # If under Windows, check if the given directory has a trailing slash
    if os.name == 'nt':
        if args.out_dir_2016[-1] != "\\":
            args.out_dir_2016 += "\\"
    # Else, check if the given directory has slash
    else:
        if args.out_dir_2016[-1] != "/":
            args.out_dir_2016 += "/"

    if not os.path.exists(args.out_dir_2016):
        os.mkdir(args.out_dir_2016)

# If no output directory is given, use default directories
if not args.out_dir_2022:
    # If cross-validation is used, use a different directory
    if args.CV:
        args.out_dir_2022 = "2022-CV/"
    # Else, use the default directory
    else:
        args.out_dir_2022 = "2022-data/"
# Else, check if the given directory exists
else:
    # If under Windows, check if the given directory has a trailing slash
    if os.name == 'nt':
        if args.out_dir_2022[-1] != "\\":
            args.out_dir_2022 += "\\"
    # Else, check if the given directory has slash
    else:
        if args.out_dir_2022[-1] != "/":
            args.out_dir_2022 += "/"
    
    if not os.path.exists(args.out_dir_2022):
        os.mkdir(args.out_dir_2022)

# If no proportions are given, use [0.6, 0.2, 0.2]
if not args.proportions:
    args.proportions = [0.6, 0.2, 0.2]
# Else, check if the proportions are valid
else:
    if sum(args.proportions) != 1:
        raise ValueError("Proportions must sum to 1")

# If no compression is given, do not compress
if not args.compress:
    args.compress = False
# Else, check if the file name format is valid
else:
    try:
        args.compress.format(2022)
    except:
        raise ValueError("Invalid file name format for compression")

# If no force is given, do not force
if not args.force:
    args.force = False

# PREPARE THE DATASETS

if args.CV:
    print("Preparing data for 10-fold cross-validation.")

    # Prepare 2016 dataset
    if args.p2016:
            
        print("Preparing 2016 dataset...")

        # Check if 2016-segmentation/data_dict.json exists
        if os.path.exists("{}data_dict.json".format(args.process_dir_2016)) and not args.force:
            print("Found a data_dict.json file inside the '{}' directory. Loading it to save time preparing the data".format(args.process_dir_2016))

            # Load data_dict.json
            with open("{}data_dict.json".format(args.process_dir_2016), "r") as f:
                data_dict = json.load(f)
        else:
            data_dict = prepare_dataset_2016(preprocesed_path=args.process_dir_2016)
        
        fold_sizes_list = np.zeros((len(args.N_list), 10))

        for i, N in enumerate(args.N_list):
            print("Preparing data for N = {}...".format(N))

            tau = N//8

            # If N{} folder does not exist, create it
            if not os.path.exists("{}/N{}-data".format(args.out_dir_2016, N)):
                os.makedirs("{}/N{}-data".format(args.out_dir_2016, N))
            
            valid_data = find_valid_data_2016(data_dict, N, tau, preprocesed_path=args.process_dir_2016)

            # Save valid data in '2016-data/N{}-data/data_dict.json'.format(N)
            with open("{}N{}-data/data_dict.json".format(args.out_dir_2016, N), "w") as f:
                json.dump(valid_data, f)
            
            # Split data into 10 folds
            folds_dicts = equally_sum_subset_partition(valid_data, 10)

            # Save folds in '2016-data/N{}-data/fold_{}.json'.format(N, i)
            for j in range(10):
                with open("{}N{}-data/fold_{}.json".format(args.out_dir_2016, N, j), "w") as f:
                    json.dump(folds_dicts[j], f)
            
            # Generate the X and S ndarrays for each fold
            for j in range(10):
                X, S = generate_X_S_from_dict_2016(folds_dicts[j], N, tau, processed_path=args.process_dir_2016)
                np.savez("{}N{}-data/fold_{}.npz".format(args.out_dir_2016, N, j), X=X, S=S)

                fold_sizes_list[i, j] = X.shape[0]

        if args.compress:
            os.system('tar -czvf cv-{} {}N*-data/fold_*.npz'.format(args.compress.format(2016), args.out_dir_2016))

    # Prepare 2022 dataset
    if args.p2022:
        print("Preparing 2022 dataset...")

        # Check if 2022-segmentation/data_dict.json exists
        if os.path.exists("{}data_dict.json".format(args.process_dir_2022)) and not args.force:
            print("Found a data_dict.json file inside the '{}' directory. Loading it to save time preparing the data".format(args.process_dir_2022))

            # Load data_dict.json
            with open("{}data_dict.json".format(args.process_dir_2022), "r") as f:
                data_dict = json.load(f)
        else:
            data_dict = prepare_dataset_2022(preprocesed_path=args.process_dir_2022)
        
        fold_sizes_list = np.zeros((len(args.N_list), 10))

        for i, N in enumerate(args.N_list):
            print("Preparing data for N = {}...".format(N))

            tau = N//8

            # If N{} folder does not exist, create it
            if not os.path.exists("{}/N{}-data".format(args.out_dir_2022, N)):
                os.makedirs("{}/N{}-data".format(args.out_dir_2022, N))
            
            valid_data = find_valid_data_2022(data_dict, N, tau, preprocesed_path=args.process_dir_2022)

            # Save valid data in '2022-data/N{}-data/data_dict.json'.format(N)
            with open("{}N{}-data/data_dict.json".format(args.out_dir_2022, N), "w") as f:
                json.dump(valid_data, f)
            
            # Split data into 10 folds
            folds_dicts = equally_sum_subset_partition(valid_data, 10)

            # Save folds in '2022-data/N{}-data/fold_{}.json'.format(N, i)
            for j in range(10):
                with open("{}N{}-data/fold_{}.json".format(args.out_dir_2022, N, j), "w") as f:
                    json.dump(folds_dicts[j], f)
            
            # Generate the X and S ndarrays for each fold
            for j in range(10):
                X, S = generate_X_S_from_dict_2022(folds_dicts[j], N, tau, processed_path=args.process_dir_2022)
                np.savez("{}N{}-data/fold_{}.npz".format(args.out_dir_2022, N, j), X=X, S=S)

                fold_sizes_list[i, j] = X.shape[0]

        if args.compress:
            os.system('tar -czvf cv-{} {}N*-data/fold_*.npz'.format(args.compress.format(2022), args.out_dir_2022))

else:
    print("Preparing data for training and testing.")
    # Prepare 2016 dataset
    if args.p2016:

        print("Preparing 2016 dataset...")

        # Check if 2016-segmentation/data_dict.json exists
        if os.path.exists("{}data_dict.json".format(args.process_dir_2016)) and not args.force:
            print("Found a data_dict.json file inside the '{}' directory. Loading it to save time preparing the data".format(args.process_dir_2016))

            # Load data_dict.json
            with open("{}data_dict.json".format(args.process_dir_2016), "r") as f:
                data_dict = json.load(f)
        else:
            data_dict = prepare_dataset_2016(preprocesed_path=args.process_dir_2016)
        
        train_sizes_list = []
        valid_sizes_list = []
        test_sizes_list = []

        for N in args.N_list:
            tau = N//8

            # If N{} folder does not exist, create it
            if not os.path.exists("{}/N{}-data".format(args.out_dir_2016, N)):
                os.makedirs("{}/N{}-data".format(args.out_dir_2016, N))
            
            valid_data = find_valid_data_2016(data_dict, N, tau, preprocesed_path=args.process_dir_2016)

            # Save valid data in '2016-data/N{}-data/data_dict.json'.format(N)
            with open("{}N{}-data/data_dict.json".format(args.out_dir_2016, N), "w") as f:
                json.dump(valid_data, f)
            
            # Split it into training, valid and test set
            train_dict, valid_dict, test_dict = create_train_valid_test_datasets(valid_data, train_prop=args.proportions[0], valid_prop=args.proportions[1], test_prop=args.proportions[2], save_dicts_fmt='{}/N{}-data/'.format(args.out_dir_2016, N)+'{}_dict.json')

            # Generate the corresponding X S ndarrays
            X_train, S_train = generate_X_S_from_dict_2016(train_dict, N, tau, processed_path=args.process_dir_2016)
            X_valid, S_valid = generate_X_S_from_dict_2016(valid_dict, N, tau, processed_path=args.process_dir_2016)
            X_test, S_test = generate_X_S_from_dict_2016(test_dict, N, tau, processed_path=args.process_dir_2016)

            train_sizes_list.append(X_train.shape[0])
            valid_sizes_list.append(X_valid.shape[0])
            test_sizes_list.append(X_test.shape[0])

            # Save the ndarrays
            np.savez('{}N{}-data/train.npz'.format(args.out_dir_2016, N), X=X_train, S=S_train)
            np.savez('{}N{}-data/valid.npz'.format(args.out_dir_2016, N), X=X_valid, S=S_valid)
            np.savez('{}N{}-data/test.npz'.format(args.out_dir_2016, N), X=X_test, S=S_test)

        # Add all the 2016-data/N{}-data/test.npz files to a TGZ file
        if args.compress:
            os.system('tar -czvf {} {}N*-data/test.npz'.format(args.compress.format(2016), args.out_dir_2016))

        # Print the sizes of the datasets
        print("N          : {}".format(args.N_list))
        print("Train sizes: {}".format(train_sizes_list))
        print("Valid sizes: {}".format(valid_sizes_list))
        print("Test sizes : {}".format(test_sizes_list))




    # Prepare 2022 dataset	
    if args.p2022:  

        print("Preparing 2022 dataset...")

        # Check if circor-segmentation/data_dict.json exists
        if os.path.exists("{}data_dict.json".format(args.process_dir_2022)) and not args.force:

            print("Found a data_dict.json file inside the '{}' directory. Loading it to save time preparing the data".format(args.process_dir_2022))

            # Load data_dict.json
            with open("{}data_dict.json".format(args.process_dir_2022), "r") as f:
                data_dict = json.load(f)
        else:
            data_dict = prepare_dataset_2022(preprocesed_path=args.process_dir_2022)

        train_sizes_list = []
        valid_sizes_list = []
        test_sizes_list = []

        for N in args.N_list:
            tau = N//8

            # If N{} folder does not exist, create it
            if not os.path.exists("{}/N{}-data".format(args.out_dir_2022, N)):
                os.makedirs("{}/N{}-data".format(args.out_dir_2022, N))

            valid_data = find_valid_data_2022(data_dict, N, tau, preprocesed_path=args.process_dir_2022)

            # Save valid data in '2022-data/N{}-data/data_dict.json'.format(N)
            with open("{}N{}-data/data_dict.json".format(args.out_dir_2022, N), "w") as f:
                json.dump(valid_data, f)

            # Split it into training, valid and test set
            train_dict, valid_dict, test_dict = create_train_valid_test_datasets(valid_data, train_prop=args.proportions[0], valid_prop=args.proportions[1], test_prop=args.proportions[2], save_dicts_fmt='{}/N{}-data/'.format(args.out_dir_2022, N)+'{}_dict.json')

            # Generate the corresponding X S ndarrays
            X_train, S_train = generate_X_S_from_dict_2022(train_dict, N, tau, processed_path=args.process_dir_2022)
            X_valid, S_valid = generate_X_S_from_dict_2022(valid_dict, N, tau, processed_path=args.process_dir_2022)
            X_test, S_test = generate_X_S_from_dict_2022(test_dict, N, tau, processed_path=args.process_dir_2022)

            train_sizes_list.append(X_train.shape[0])
            valid_sizes_list.append(X_valid.shape[0])
            test_sizes_list.append(X_test.shape[0])

            # Save the ndarrays
            np.savez('{}N{}-data/train.npz'.format(args.out_dir_2022, N), X=X_train, S=S_train)
            np.savez('{}N{}-data/valid.npz'.format(args.out_dir_2022, N), X=X_valid, S=S_valid)
            np.savez('{}N{}-data/test.npz'.format(args.out_dir_2022, N), X=X_test, S=S_test)

        # Add all the 2022-data/N{}-data/test.npz files to a TGZ file
        if args.compress:
            os.system('tar -czvf {} {}N*-data/test.npz'.format(args.compress.format(2022), args.out_dir_2022))

        # Print the sizes of the datasets
        print("N          : {}".format(args.N_list))
        print("Train sizes: {}".format(train_sizes_list))
        print("Valid sizes: {}".format(valid_sizes_list))
        print("Test sizes : {}".format(test_sizes_list))