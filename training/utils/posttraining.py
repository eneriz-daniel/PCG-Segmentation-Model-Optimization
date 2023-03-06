import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from utils.models import get_model_parametrized
import matplotlib.pyplot as plt
import os

def get_maccs(model):
    """Get the number of MACCs of a model.

    Args:
    -----
        model (Model): Model instance.

    Returns:
    --------
        maccs (int): Number of MACCs.
    """

    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])]);

    graph_info = profile(forward_pass.get_concrete_function().graph,
                            options=ProfileOptionBuilder.float_operation());

    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    return graph_info.total_float_ops // 2;

def parse_metrics(dataset_list, N_list, n0_list, nenc_list,
                  excel_file_fmt = 'models/{}/test_metrics.xlsx'):
    """Parse the metrics of the models and save them in an excel file.

    Args:
    -----
        dataset_list (list): List of datasets.
        N_list (list): List of N values.
        n0_list (list): List of n0 values.
        nenc_list (list): List of nenc values.
        excel_file_fmt (str): Format of the excel file. Must contain a single
            placeholder for the dataset name. Default:
            'models/{}/test_metrics.xlsx'
    """
    global_acc_table = np.zeros((len(N_list), len(n0_list), len(nenc_list)))
    total_rec_acc_table = np.zeros((len(N_list), len(n0_list), len(nenc_list)))
    sens_table = np.zeros((len(N_list), len(n0_list), len(nenc_list)))
    ppv_table = np.zeros((len(N_list), len(n0_list), len(nenc_list)))

    for dataset in dataset_list:

        print('Parsing metrics for dataset {}...'.format(dataset))

        # Check if dataset is either '2016' or '2022'
        if str(dataset) not in ['2016', '2022']:
            raise ValueError('Dataset must be either 2016 or 2022.')

        # Create the excel file
        writer = pd.ExcelWriter(excel_file_fmt.format(dataset))

        # Create a table for each N
        for i, N in enumerate(N_list):
            for j, n0 in enumerate(n0_list):
                for k, nenc in enumerate(nenc_list):

                    # Read the file
                    f = np.load('models/{}/N{}/n0{}/nenc{}/test_metrics.npz'.format(dataset, N, n0, nenc))
                    
                    # Get the accuracy, sensitivity, and PPV
                    global_acc = f['global_acc']
                    total_rec_acc = f['total_rec_acc']
                    sens = f['sensitivity']
                    ppv = f['ppv']

                    global_acc_table[i,j,k] = global_acc
                    total_rec_acc_table[i,j,k] = total_rec_acc
                    sens_table[i,j,k] = sens
                    ppv_table[i,j,k] = ppv
            
            # Create a pd.DataFrame for each table
            global_acc_df = pd.DataFrame(global_acc_table[i], index=n0_list, columns=nenc_list)
            global_acc_df.index.name = 'Base filter \ Number of blocks'

            total_rec_acc_df = pd.DataFrame(total_rec_acc_table[i], index=n0_list, columns=nenc_list)
            total_rec_acc_df.index.name = 'Base filter \ Number of blocks'

            sens_df = pd.DataFrame(sens_table[i], index=n0_list, columns=nenc_list)
            sens_df.index.name = 'Base filter \ Number of blocks'

            ppv_df = pd.DataFrame(ppv_table[i], index=n0_list, columns=nenc_list)
            ppv_df.index.name = 'Base filter \ Number of blocks'

            global_acc_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=0)
            total_rec_acc_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=8, startcol=0)
            sens_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=15, startcol=0)
            ppv_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=22, startcol=0)

            sheet = writer.sheets['N{}'.format(N)]
            sheet.write_string(0, 0, 'Global accuracy')
            sheet.write_string(7, 0, 'Total recording accuracy')
            sheet.write_string(14, 0, 'Sensitivity')
            sheet.write_string(21, 0, 'PPV')

        writer.save()

def get_macc_params_tables(N_list, n0_list, nenc_list):
    """Get the number of MACCs and parameters of the models.

    Args:
    -----
        N_list (list): List of N values.
        n0_list (list): List of n0 values.
        nenc_list (list): List of nenc values.

    Returns:   
    --------
        maccs_table (np.ndarray): Table of the number of MACCs.
        params_table (np.ndarray): Table of the number of parameters.
    """

    maccs_table = np.zeros((len(N_list), len(n0_list), len(nenc_list)))
    params_table = np.zeros((len(N_list), len(n0_list), len(nenc_list)))

    for i, N in enumerate(N_list):
        for j, n0 in enumerate(n0_list):
            for k, nenc in enumerate(nenc_list):
                # Read the file
                model = get_model_parametrized(N, n0, nenc)
                maccs = get_maccs(model)
                params = model.count_params()

                maccs_table[i,j,k] = maccs
                params_table[i,j,k] = params

    return maccs_table, params_table

def parse_cv_metrics(dataset_list, N_list, n0_list, nenc_list):
    """Parse the cross-validation metrics of the models and save them in an
    excel file.

    Args:
    -----
        dataset_list (list): List of datasets.
        N_list (list): List of N values.
        n0_list (list): List of n0 values.
        nenc_list (list): List of nenc values.
    """
    for dataset in dataset_list:
        # Firstly check all the models are available
        error = False
        for N in N_list:
            for n0 in n0_list:
                for nenc in nenc_list:
                    for l in range(10):
                        if not os.path.exists('models-cv/{}/N{}/n0{}/nenc{}/fold_{}/test_metrics.npz'.format(dataset, N, n0, nenc, l)):
                            print('Missing test metrics from: dataset: {}, N: {}, n0: {}, nenc: {}, fold: {}'.format(dataset, N, n0, nenc, l))
                            error = True

        if error:
            print('Some models are missing. Please check the above messages.')
            exit()

        # Create the excel file
        writer = pd.ExcelWriter('models-cv/{}/test_metrics_.xlsx'.format(dataset))
        # Initialize the tables to store the '{mean}±{std}' strings
        global_acc_table = np.zeros((len(N_list), len(n0_list), len(nenc_list)), dtype=object)
        total_rec_acc_table = np.zeros((len(N_list), len(n0_list), len(nenc_list)), dtype=object)
        ppv_table = np.zeros((len(N_list), len(n0_list), len(nenc_list)), dtype=object)
        sens_table = np.zeros((len(N_list), len(n0_list), len(nenc_list)), dtype=object)

        for i, N in enumerate(N_list):
            for j, n0 in enumerate(n0_list):
                for k, nenc in enumerate(nenc_list):
                    
                    # Initialize the test metrics accumulator
                    test_metrics = np.zeros((10, 4))

                    for l in range(10):
                        with np.load('models-cv/{}/N{}/n0{}/nenc{}/fold_{}/test_metrics.npz'.format(dataset, N, n0, nenc, l)) as data:
                            test_metrics[l, 0] = data['global_acc']
                            test_metrics[l, 1] = data['total_rec_acc']
                            test_metrics[l, 2] = data['ppv']
                            test_metrics[l, 3] = data['sensitivity']
                    
                    # Save the mean and standard deviation of the test metrics
                    np.savez('models-cv/{}/N{}/n0{}/nenc{}/test_metrics.npz'.format(dataset, N, n0, nenc),
                        global_acc_mean=np.mean(test_metrics[:, 0]),
                        global_acc_std=np.std(test_metrics[:, 0]), 
                        global_accs=test_metrics[:, 0],
                        total_rec_acc_mean=np.mean(test_metrics[:, 1]),
                        total_rec_acc_std=np.std(test_metrics[:, 1]),
                        total_rec_accs=test_metrics[:, 1],
                        ppv_mean=np.mean(test_metrics[:, 2]),
                        ppv_std=np.std(test_metrics[:, 2]),
                        ppvs=test_metrics[:, 2],
                        sensitivity_mean=np.mean(test_metrics[:, 3]),
                        sensitivity_std=np.std(test_metrics[:, 3]),
                        sensitivities=test_metrics[:, 3])
                    
                    global_acc_table[i, j, k] = '{:.1f}±{:.1f}'.format(100*np.mean(test_metrics[:, 0]), 100*np.std(test_metrics[:, 0]))
                    total_rec_acc_table[i, j, k] = '{:.1f}±{:.1f}'.format(100*np.mean(test_metrics[:, 1]), 100*np.std(test_metrics[:, 1]))
                    ppv_table[i, j, k] = '{:.1f}±{:.1f}'.format(100*np.mean(test_metrics[:, 2]), 100*np.std(test_metrics[:, 2]))
                    sens_table[i, j, k] = '{:.1f}±{:.1f}'.format(100*np.mean(test_metrics[:, 3]), 100*np.std(test_metrics[:, 3]))

                    # Create a pd.DataFrame for each table
                    global_acc_df = pd.DataFrame(global_acc_table[i], index=n0_list, columns=nenc_list)
                    global_acc_df.index.name = 'Base filter \ Number of blocks'

                    total_rec_acc_df = pd.DataFrame(total_rec_acc_table[i], index=n0_list, columns=nenc_list)
                    total_rec_acc_df.index.name = 'Base filter \ Number of blocks'

                    ppv_df = pd.DataFrame(ppv_table[i], index=n0_list, columns=nenc_list)
                    ppv_df.index.name = 'Base filter \ Number of blocks'

                    sens_df = pd.DataFrame(sens_table[i], index=n0_list, columns=nenc_list)
                    sens_df.index.name = 'Base filter \ Number of blocks'

                    global_acc_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=1, startcol=0)
                    total_rec_acc_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=8, startcol=0)
                    sens_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=15, startcol=0)
                    ppv_df.to_excel(writer, sheet_name='N{}'.format(N), startrow=22, startcol=0)

                    sheet = writer.sheets['N{}'.format(N)]
                    sheet.write_string(0, 0, 'Global Accuracy')
                    sheet.write_string(7, 0, 'Total recording accuracy')
                    sheet.write_string(14, 0, 'Sensitivity')
                    sheet.write_string(21, 0, 'PPV')

        writer.save()