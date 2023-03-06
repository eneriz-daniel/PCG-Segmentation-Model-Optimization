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

import numpy as np
from typing import Tuple
from tqdm import tqdm
import tensorflow as tf
from utils.models import *
import os
import warnings
from tensorflow.keras.utils import plot_model
import time 
import pickle
import matplotlib.pyplot as plt
from typing import Dict
import json
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.preprocessing import rolling_strided_window, check_valid_sequence, unroll_strided_windows, find_valid_data_2022, generate_X_S_from_dict_2022 
from tensorflow.keras.utils import to_categorical

def compute_tp_fp_ttot(s_true_orig: np.ndarray, s_pred_orig: np.ndarray) -> Tuple[int, int, int]:
    """Compute the true positives, false positives and the total number of S1
    and S2 true sounds, using the method described in [1].

    Args:
    -----
        s_true (np.ndarray): True states.
        s_pred (np.ndarray): Predicted states.

    Returns:
    --------
        tp (int): True positives.
        fp (int): False positives.
        ttot (int): Total number of S1 and S2 true sounds.
    
    References:
    -----------
        [1] S. E. Schmidt et al., "Segmentation of heart sound recordings by a
        duration-dependent hidden Markov model," Physiol. Meas., vol. 31, no. 4,
        pp. 513-29, Apr. 2010
    """
    # Copy the original arrays
    s_true = s_true_orig.copy()
    s_pred = s_pred_orig.copy()

    # The inputs are supposed to be 1D arrays of integers between 1 and 4 with
    # the same length, so make sure to decategoricalize them.
    
    # Check if s_true is a numpy array of 1D
    if not isinstance(s_true, np.ndarray) or s_true.ndim != 1:
        raise TypeError('s_true must be a numpy array of 1D.')
    
    # Check if s_pred is a numpy array of 1D
    if not isinstance(s_pred, np.ndarray) or s_pred.ndim != 1:
        raise TypeError('s_pred must be a numpy array of 1D.')
    
    # Check if s_true and s_pred have the same length
    if s_true.shape[0] != s_pred.shape[0]:
        raise ValueError('s_true and s_pred must have the same length.')
 
    # Make systolic and diastolic states 0 (i.e. the 2 and 4 states are zero now)
    s_true[s_true == 2] = 0
    s_true[s_true == 4] = 0

    s_pred[s_pred == 2] = 0
    s_pred[s_pred == 4] = 0

    # Obtain the difference of each array
    true_diff = np.diff(s_true)
    pred_diff = np.diff(s_pred)

    # If the first non-zero element of [true|pred]_diff is negative, substitude it by 0
    # to remove a non-complete S1 or S2 sound
    # Obtain the first non-zero element of [true|pred]_diff
    first_non_zero = np.nonzero(true_diff)[0][0]
    if true_diff[first_non_zero] < 0:
        true_diff[first_non_zero] = 0
    
    if np.count_nonzero(pred_diff):
        first_non_zero = np.nonzero(pred_diff)[0][0]
        if pred_diff[first_non_zero] < 0:
            pred_diff[first_non_zero] = 0

    # Calculate the number of complete S1 states in each array. The start of
    # these states are marked with 1's in the arrays abd the end are marked with
    # -1's. The number of complete S1 states is the minimum.
    true_S1 = min(np.count_nonzero(true_diff == 1), np.count_nonzero(true_diff == -1))
    pred_S1 = min(np.count_nonzero(pred_diff == 1), np.count_nonzero(pred_diff == -1))

    # Find the centers of each S1 state in the true and predicted arrays
    true_S1_centers = np.zeros(true_S1)
    for i in range(true_S1):
        lower_bound = np.where(true_diff == 1)[0][i]
        upper_bound = np.where(true_diff[lower_bound:] == -1)[0][0] + lower_bound + 1
        
        true_S1_centers[i] = (lower_bound + upper_bound) / 2

    pred_S1_centers = np.zeros(pred_S1)
    for i in range(pred_S1):
        lower_bound = np.where(pred_diff == 1)[0][i]
        upper_bound = np.where(pred_diff[lower_bound:] == -1)[0][0] + lower_bound + 1
        
        pred_S1_centers[i] = (lower_bound + upper_bound) / 2
    
    # Calculate the difference between the centers of the S1 states in the true
    # and predicted arrays using the nearest element and mark them as true
    # positive if the difference is less than 60 ms
    true_positives = np.zeros(pred_S1, dtype=bool)
    if true_S1:
        for i in range(pred_S1):
            # Find the nearest true_S1_center
            nearest_true_S1_center = true_S1_centers[np.argmin(np.abs(true_S1_centers - pred_S1_centers[i]))]

            # Mark the true positive if the difference is less than 60 ms (3 samples)
            if np.abs(pred_S1_centers[i] - nearest_true_S1_center) < 3:
                true_positives[i] = True
    
    # Calculate the true positives and false positives
    tp = np.count_nonzero(true_positives)
    fp = np.count_nonzero(np.logical_not(true_positives))

    # Do the same process for the S2 states

    # Calculate the number of complete S2 states in each array. The start of
    # these states are marked with 3's in the arrays abd the end are marked with
    # -3's. The number of complete S2 states is the minimum.
    true_S2 = min(np.count_nonzero(true_diff == 3), np.count_nonzero(true_diff == -3))
    pred_S2 = min(np.count_nonzero(pred_diff == 3), np.count_nonzero(pred_diff == -3))

    # Find the centers of each S2 state in the true and predicted arrays
    true_S2_centers = np.zeros(true_S2)
    for i in range(true_S2):
        lower_bound = np.where(true_diff == 3)[0][i]
        upper_bound = np.where(true_diff[lower_bound:] == -3)[0][0] + lower_bound + 1
        
        true_S2_centers[i] = (lower_bound + upper_bound) / 2

    pred_S2_centers = np.zeros(pred_S2)
    for i in range(pred_S2):
        lower_bound = np.where(pred_diff == 3)[0][i]
        upper_bound = np.where(pred_diff[lower_bound:] == -3)[0][0] + lower_bound + 1
        
        pred_S2_centers[i] = (lower_bound + upper_bound) / 2
    
    # Calculate the difference between the centers of the S2 states in the true
    # and predicted arrays using the nearest element and mark them as true
    # positive if the difference is less than 60 ms
    true_positives = np.zeros(pred_S2, dtype=bool)
    if true_S2:
        for i in range(pred_S2):
            # Find the nearest true_S2_center
            nearest_true_S2_center = true_S2_centers[np.argmin(np.abs(true_S2_centers - pred_S2_centers[i]))]

            # Mark the true positive if the difference is less than 60 ms (3 samples)
            if np.abs(pred_S2_centers[i] - nearest_true_S2_center) < 3:
                true_positives[i] = True
    
    # Calculate the true positives and false positives
    tp += np.count_nonzero(true_positives)
    fp += np.count_nonzero(np.logical_not(true_positives))

    return tp, fp, true_S1 + true_S2

def seq_max_temporal_model(x: np.ndarray) -> np.ndarray:
    """Implementation of the sequential max temporal modeling. It forces the input
    states sequence to contain only admisible transitions among heart states
    (S1->systolic->S2->diastolic->S1).

    Args:
    -----
        x (np.ndarray): Input sequence of states. The elements must be integers
        between 1 and 4.

    Returns:
    --------
        y (np.ndarray): Output sequence of states, where only admisible
        transitions are present.
    """

    # Check if x is a numpy array of 1D
    if not isinstance(x, np.ndarray) or x.ndim != 1:
        raise TypeError('x must be a numpy array of 1D.')
    
    # Create y as an empty array of same size as x
    y = np.zeros(x.shape)

    # Set that the first element of y is the same as the first element of x
    y[0] = x[0]

    # Iterate over the rest of the elements of x
    for i in range(1, x.shape[0]):
        # If x[i] = (x[i-1] + 1) % 4, then y[i] = x[i]
        if y[i-1] % 4 + 1 == x[i]:
            y[i] = x[i]
        # Otherwise, y[i] = y[i-1]
        else:
            y[i] = y[i-1]
    
    return y

def ppv_sens_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """This computes the positive predicted value and the sensitivity as
    described in [1]. It expects the arguments to be generated during the Keras
    fit/evaluate runtime, so first it decategorizes the inputs to be a heart
    state sequence 1->2->3->4->1, then calls `compute_tp_fp_ttot` and finally
    computes the positive predicted value and the sensitivity for the entire
    batch size.

    Args:
    -----
        y_true: The true labels.
        y_pred: The predicted labels.
    
    Returns:
    --------
        ppv: The positive predicted value.
        sens: The sensitivity.

    References:
    -----------
        [1] S. E. Schmidt et al., "Segmentation of heart sound recordings by a
        duration-dependent hidden Markov model," Physiol. Meas., vol. 31, no. 4,
        pp. 513-29, Apr. 2010
    """
    # If the batch size is None, return 0 for both metrics
    if y_true.shape[0] == None:
        return 0, 0
    
    # Iterate over the batches
    tp = 0
    fp = 0
    ttot = 0
    for i in tqdm(range(y_true.shape[0]), desc="Computing PPV and Sens"):
        # Decategorize the arrays
        s_pred = np.squeeze(y_pred[i, :, :].argmax(axis=-1)+1)
        s_true = np.squeeze(y_true[i, :, :].argmax(axis=-1)+1)

        # Use que sequential max temporal modeling
        s_pred = seq_max_temporal_model(s_pred)

        # Compute the true positives, false positives and true total
        tp_i, fp_i, ttot_i = compute_tp_fp_ttot(s_true, s_pred)

        # Update the counters
        tp += tp_i
        fp += fp_i
        ttot += ttot_i

    # Compute the positive predicted value and the sensitivity
    ppv = tp / (tp + fp + tf.keras.backend.epsilon())
    sens = tp / ttot

    return ppv, sens

class CustomMetrics(tf.keras.callbacks.Callback):
    """Custom callback to calculate the positive predicted value and sensitivity
    metrics in the validation data after each epoch.

    Attributes:
    -----------
        data (tuple): A tuple containing the validation data (x, s).
        prefix (str): A prefix to add to the metrics names.
        ppv (list): A list containing the positive predicted value for each
            epoch.
        sensitivity (list): A list containing the sensitivity for each epoch. 

    Methods:
    --------
        on_epoch_end: Called at the end of each epoch.

    """

    def __init__(self, data, prefix=''):
        """Constructor method. It initializes the attributes.
        
        Args:
        -----
            data (tuple): A tuple containing the validation data (x, s).
            prefix (str): A prefix to add to the metrics names. Default is ''.
        
        Returns:
        --------
            None
        """
        super(CustomMetrics, self).__init__()
        self.data = data
        self.prefix = prefix
        self.ppv = []
        self.sensitivity = []

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch. It calculates the positive predicted
        value and sensitivity metrics in the validation data and appends them to
        the corresponding lists. It also prints the metrics.
        
        Args:
        -----
            epoch (int): The current epoch.
            logs

        Returns:
        --------
            None
        """

        x, s = self.data
        _ppv, _sensitivity = ppv_sens_metrics(s, self.model.predict(x))
        
        self.ppv.append(_ppv)
        self.sensitivity.append(_sensitivity)
        print('- {}ppv: {} - {}sensitivity: {}'.format(self.prefix, _ppv, self.prefix, _sensitivity), end=' ')

def test_model(model: tf.keras.Model, X_test, S_test, test_dict: Dict, N: int, tau: int,
               dataset : str, processed_path: str) -> Tuple[float, float, float, float]:
    """Test the model on the test dataset defined by `test_dict`, returning the
    accuracy, positive predicted value and sensitivity, as defined in [1, 2].
    This can take a while to be processed.

    Args:
    -----
        model (tf.keras.Model): The model to test.
        X_test (np.ndarray): The test dataset.
        S_test (np.ndarray): The test labels.
        test_dict (Dict): The test dataset defined in a Dict, using the same
        structure as the `data_dict` generated by `prepare_dataset_{dataset}`.
        N (int): The window length.
        tau (int): Stride used in the windowing.
        dataset (str): The dataset to use. Can be '2016' or '2022'.
        processed_path (str): The path to the processed data.
    
    Returns:
    --------
        global_acc: The global accuracy, i.e. the categorical accuracy computed
        over the entire test dataset window by window, being unaware of the belonging
        of each window to a specific recording and patient.
        total_rec_acc: The total recording accuracy, i.e. the accuracy computed
        over the entire test dataset recording by recording, being aware of the
        belonging of each window to a specific recording and patient.
        ppv: The positive predicted value.
        sens: The sensitivity.

    References:
    -----------
        [1] Renna, F., Oliveira, J., & Coimbra, M. T. (2019). Deep Convolutional
        Neural Networks for Heart Sound Segmentation. IEEE journal of biomedical
        and health informatics, 23(6), 2435-2445.
        https://doi.org/10.1109/JBHI.2019.2894222
        [2] S. E. Schmidt et al., "Segmentation of heart sound recordings by 
        a duration-dependent hidden Markov model," Physiol. Meas., vol. 31, no.
        4, pp. 513-29, Apr. 2010
    """

    # Check the model is a tf.keras.Model
    if not isinstance(model, tf.keras.Model):
        raise ValueError("The model must be a tf.keras.Model.")
    
    # Check the test_dict is a Dict
    if not isinstance(test_dict, Dict):
        raise ValueError("The test_dict must be a Dict.")

    # If running under Windows, replace '/' with '\\' in the processed_path
    if os.name == 'nt':
        processed_path = processed_path.replace('/', '\\')
    
    # Check if the dataset is valid
    if dataset not in ['2016', '2022']:
        raise ValueError("The dataset must be '2016' or '2022'.")
    
    # Compute the 'global' accuracy using the sliced dataset
    cat_acc = tf.keras.metrics.CategoricalAccuracy()
    cat_acc.update_state(S_test, model.predict(X_test))
    global_acc = cat_acc.result().numpy()
    
    # Initialize 0-sized arrays to store the entire predictions and the entire
    # ground truth to compute the accuracy
    s_pred_global = np.zeros(0)
    s_true_global = np.zeros(0)

    # Initialize the true positives, false positives and true total counters
    tp = 0
    fp = 0
    ttot = 0

    if dataset == '2022':
        # Iterate over the patients
        for patient_id in tqdm(test_dict, desc='Testing model'):
            for location in test_dict[patient_id]["Unique locations"]:       
                for j in [key for key in test_dict[patient_id][location].keys() if key != 'Count']:
                    for k in [key for key in test_dict[patient_id][location][j].keys() if key != 'Number of splits']:
                        # Load the .npz file
                        with np.load(processed_path + test_dict[patient_id][location][j][k]["File name"]) as f:
                            x = f['x']
                            s = f['s']

                            # Obtain the X and S
                            X = rolling_strided_window(x, N, tau)
                            S = rolling_strided_window(s, N, tau)

                            # Annotate the number of windows
                            n_windows = S.shape[0]

                            # Check if the sequence is valid
                            X, S = check_valid_sequence(X, S, 2)

                            # If the sequence is not valid, skip to the next sequence
                            if S.shape[0] != n_windows:
                                continue

                            # Reshape X and S to have the valid format for the model
                            # Create a new axis in S
                            S = S[:, np.newaxis, :]

                            # Swap axes to format the data as channels_last
                            X = np.swapaxes(X, 1, 2)
                            S = np.swapaxes(S, 1, 2)

                            # Transform S to categorical
                            S = to_categorical(S-1)

                            # Predict the labels
                            S_pred = model.predict(X)

                            # Unroll predictions
                            s_pred = unroll_strided_windows(S_pred, tau)

                            # Decategorize the predictions
                            s_pred = np.squeeze(s_pred.argmax(axis=-1)+1)

                            # Use que sequential max temporal modeling
                            s_pred = seq_max_temporal_model(s_pred)

                            # Store the predictions and the ground truth in the global arrays
                            s_pred_global = np.append(s_pred_global, s_pred)
                            s_true_global = np.append(s_true_global, s[:s_pred.size])

                            # Compute the true positives, false positives and true total
                            tp_i, fp_i, ttot_i = compute_tp_fp_ttot(s[:s_pred.size], s_pred)

                            # Update the counters
                            tp += tp_i
                            fp += fp_i
                            ttot += ttot_i
    
    else: # dataset == '2016'
        # Iterate over the patients
        for patient_id in tqdm(test_dict, desc='Testing model'):
            # Iterate over the recordings
            for recording in [recording_num for recording_num in test_dict[patient_id] if recording_num not in ['Count', 'Number of valid windows']]:
                # Load the .npz file
                with np.load(processed_path + test_dict[patient_id][recording]["File name"]) as f:
                    x = f['x']
                    s = f['s']

                    # Obtain the X and S
                    X = rolling_strided_window(x, N, tau)
                    S = rolling_strided_window(s, N, tau)

                    # Annotate the number of windows
                    n_windows = S.shape[0]

                    # Check if the sequence is valid
                    X, S = check_valid_sequence(X, S, 2)

                    # If the sequence is not valid, skip to the next sequence
                    if S.shape[0] != n_windows:
                        continue

                    # Reshape X and S to have the valid format for the model
                    # Create a new axis in S
                    S = S[:, np.newaxis, :]

                    # Swap axes to format the data as channels_last
                    X = np.swapaxes(X, 1, 2)
                    S = np.swapaxes(S, 1, 2)

                    # Transform S to categorical
                    S = to_categorical(S-1)

                    # Predict the labels
                    S_pred = model.predict(X)

                    # Unroll predictions
                    s_pred = unroll_strided_windows(S_pred, tau)

                    # Decategorize the predictions
                    s_pred = np.squeeze(s_pred.argmax(axis=-1)+1)

                    # Use que sequential max temporal modeling
                    s_pred = seq_max_temporal_model(s_pred)

                    # Store the predictions and the ground truth in the global arrays
                    s_pred_global = np.append(s_pred_global, s_pred)
                    s_true_global = np.append(s_true_global, s[:s_pred.size])

                    # Compute the true positives, false positives and true total
                    tp_i, fp_i, ttot_i = compute_tp_fp_ttot(s[:s_pred.size], s_pred)

                    # Update the counters
                    tp += tp_i
                    fp += fp_i
                    ttot += ttot_i
    
    # Compute the 'total recording' accuracy using the global arrays
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(s_true_global, s_pred_global)
    total_rec_acc = accuracy.result().numpy()

    # Compute the positive predicted value and the sensitivity
    ppv = tp / (tp + fp + tf.keras.backend.epsilon())
    sens = tp / ttot

    return global_acc, total_rec_acc, ppv, sens

def train(X_train: np.ndarray, S_train: np.ndarray, X_valid: np.ndarray,
          S_valid: np.ndarray, X_test: np.ndarray, S_test: np.ndarray,
          test_dict: Dict, model_path: str, N: int, tau: int, n0: int,
          nenc: int, epochs: int, batch_size: int, lr: float, dataset_name: str):
    """Trains a model with the given parameters with the given training
    hyperparamters using the given dataset.

    Args:
    -----
        X_train (np.ndarray): The input training data.
        S_train (np.ndarray): The output training data.
        X_valid (np.ndarray): The input validation data.
        S_valid (np.ndarray): The output validation data.
        X_test (np.ndarray): The input test data.
        S_test (np.ndarray): The output test data.
        test_dict (Dict): The test dataset defined in a Dict.
        model_path (str): Path to the directory where the model will be saved.
        N (int): Window length.
        tau (int): Stride of the windowing.
        n0 (int): Number of filters in the first layer.
        nenc (int): Number encoder/decoder blocks.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        dataset_name (str): Name of the dataset.
    
    Returns:
    --------
        If test_dict is not None, returns the global accuracy, the total
        recording accuracy, the positive predicted value and the sensitivity.
    """

    print('Start of the training...')

    # Create the model
    model = get_model_parametrized(N, n0, nenc)

    # Model compile 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()]) 

    # Create a folder to save the model and the history of the training
    # If running under Windows, replace the '/' in the path with '\\'
    if os.name == 'nt':
        model_path = model_path.replace('/', '\\')

    # Make the directory if it doesn't exist  
    try:
        os.makedirs(model_path)
    except(FileExistsError):
        warnings.warn('Folder already exists. Overwriting it.', UserWarning)
        pass
    
    callbacks = []
    # Create the model checkpoint if valid_dict is not None
    checkpoint = ModelCheckpoint(model_path+'parameters.h5'.format(N, tau), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks.append(checkpoint)

    # Create a callback to evaluate the PPV and the sensitivity after each epoch   
    custom_metrics_train = CustomMetrics([X_train, S_train])
    callbacks.append(custom_metrics_train)
    custom_metrics_valid = CustomMetrics([X_valid, S_valid], prefix='val')
    callbacks.append(custom_metrics_valid)

    # Save model summary in a txt file
    with open(model_path+'modelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Save training configuration in a txt file
    with open(model_path+'training.txt', 'w') as f:
        f.write('Model hyperparameters:\n')
        f.write('N = {}\n'.format(N))
        f.write('tau = {}\n'.format(tau))
        f.write('n0 = {}\n'.format(n0))
        f.write('nenc = {}\n'.format(nenc))
        f.write('\nTraining hyperparameters:\n')
        f.write('epochs = {}\n'.format(epochs))
        f.write('batch_size = {}\n'.format(batch_size))
        f.write('lr = {}\n'.format(lr))
        f.write('\nDataset:\n')
        f.write('dataset_name = {}\n'.format(dataset_name))


    # Save model plot in a png file
    try:
        plot_model(model, to_file=model_path+'model_plot.png')
    except(ImportError):
        warnings.warn('Could not plot the model. Install plot_model dependencies to that.', UserWarning)
        pass

    # Annote the initial time
    t0 = time.time()

    # Model training
    history = model.fit(
        x=X_train,
        y=S_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_valid, S_valid),
        callbacks=callbacks
    )

    # Calculate the time elapsed
    t1 = time.time() - t0

    # Add the time elapsed to the training txt
    with open(model_path+'training.txt', 'a') as f:
        f.write('time elapsed: {} s'.format(t1))

    # Save history object
    with open(model_path+'history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Save custom metrics too
    with open(model_path+'custom_metrics_train.pkl', 'wb') as f:
        pickle.dump({'ppv': custom_metrics_train.ppv, 'sensitivity': custom_metrics_train.sensitivity}, f)
    with open(model_path+'custom_metrics_valid.pkl', 'wb') as f:
        pickle.dump({'ppv': custom_metrics_valid.ppv, 'sensitivity': custom_metrics_valid.sensitivity}, f)

    # Plot training and validation loss
    fig, axs = plt.subplots(4,1, sharex=True, figsize=(10,5))
    axs[0].plot(history.history['loss'], label='Training')
    axs[0].plot(history.history['val_loss'], label='Validation')
    axs[0].set_ylabel('Loss, Cross Entropy')

    axs[0].legend()

    axs[1].plot(history.history['categorical_accuracy'], label='Training')
    axs[1].plot(history.history['val_categorical_accuracy'], label='Validation')
    axs[1].set_ylabel('Accuracy')

    # Plot custom metrics in the two last axes
    axs[2].plot(custom_metrics_train.ppv, label='Training')
    axs[2].plot(custom_metrics_valid.ppv, label='Validation')
    axs[2].set_ylabel('PPV')

    axs[3].plot(custom_metrics_train.sensitivity, label='Training')
    axs[3].plot(custom_metrics_valid.sensitivity, label='Validation')
    axs[3].set_ylabel('Sensitivity')

    axs[-1].set_xlabel('Epoch')

    # Reduce the space between the subplots
    fig.subplots_adjust(hspace=0)

    plt.suptitle('Training evolution: N = {}, tau = {}'.format(N, tau))

    # Save the figure
    plt.savefig(model_path+'training_evolution.png')

    if test_dict is not None:
        # Compute the metrics on the test set
        # First load the saved model
        model = tf.keras.models.load_model(model_path+'parameters.h5')

        # Compute the metrics
        global_acc, total_rec_acc, ppv, sensitivity = test_model(model, X_test, S_test, test_dict, N, tau, dataset_name, '{}-proc/'.format(dataset_name))

        # Save the metrics in a txt file
        with open(model_path+'test_metrics.txt', 'w') as f:
            f.write('Global accuracy: {}\n'.format(global_acc))
            f.write('Total recording accuracy: {}\n'.format(total_rec_acc))
            f.write('PPV: {}\n'.format(ppv))
            f.write('Sensitivity: {}\n'.format(sensitivity))

        # Save the metrics in a npz file
        np.savez(model_path+'test_metrics.npz', global_acc=global_acc, total_rec_acc=total_rec_acc, ppv=ppv, sensitivity=sensitivity)

        # Print the final test metrics
        print('------------------------------------------------------')
        print('FINAL TEST METRICS:')
        print('Global accuracy: {} %'.format(round(100*global_acc,3)))
        print('Total recording accuracy: {} %'.format(round(100*total_rec_acc,3)))
        print('PPV: {} %'.format(round(100*ppv,3)))
        print('Sensitivity: {} %'.format(round(100*sensitivity,3)))
        print('------------------------------------------------------')
        
        return global_acc, total_rec_acc, ppv, sensitivity

def train_fold(fold: int, X_sets : Tuple, S_sets : Tuple, dicts_sets : Tuple,
               model_path: str, N: int, tau: int, n0: int, nenc: int,
               epochs: int, batch_size: int, lr: float, dataset_name: str):
    """Train a model with 10-fold cross-validation. Each fold is used once as
    validation set and the other 9 folds are used as training set. The model is
    saved in the directory specified by model_path, each fold is saved in a
    different subdirectory with the format 'fold_X/' where X is the fold number.

    Args:
    -----
        fold (int): Fold to be used as test set.
        X_sets (Tuple): Tuple of 10 numpy arrays containing the input data.
        S_sets (Tuple): Tuple of 10 numpy arrays containing the output data.
        dicts_sets (Tuple): Tuple of 10 dictionaries containing the data
        reference of each set.
        model_path (str): Path to the directory where the model will be saved.
        N (int): Number of samples per window.
        tau (int): Padding of the recordings windows.
        n0 (int): Number of filters in the first convolutional layer.
        nenc (int): Number of encoder/decoder layers.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        dataset_name (str): Name of the dataset. Must be either '2016' or '2022'. 
    
    Returns:
    --------
        None
    """

    # Check fold is a integer between 0 and 9
    if not isinstance(fold, int) or fold < 0 or fold > 9:
        raise ValueError('fold must be an integer between 0 and 9.')

    # Validate the sets inputs, labels and dictionaries
    if len(X_sets) != 10 or len(S_sets) != 10 or len(dicts_sets) != 10:
        raise ValueError('X_sets, S_sets and dicts_sets must be tuples of 10 numpy arrays.')
    if not all([isinstance(X, np.ndarray) for X in X_sets]):
        raise ValueError('X_sets must be a tuple of 10 numpy arrays.')
    if not all([isinstance(S, np.ndarray) for S in S_sets]):
        raise ValueError('S_sets must be a tuple of 10 numpy arrays.')
    if not all([isinstance(dicts, dict) for dicts in dicts_sets]):
        raise ValueError('dicts_sets must be a tuple of 10 dictionaries.')
    
    # Validate the dataset name
    if dataset_name not in ['2016', '2022']:
        raise ValueError('dataset_name must be either 2016 or 2022.')
    
    # Create the main directory to save the models
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create the directory to save the models of each fold
    if not os.path.exists(model_path+'fold_{}/'.format(fold)):
        os.mkdir(model_path+'fold_{}/'.format(fold))
    
    print('------------------------------------------------------')
    print('TRAINING FOLD {}'.format(fold))
    print('------------------------------------------------------')

    # Create the training and validation sets
    X_train = np.concatenate([X_sets[j] for j in range(10) if j != fold])
    S_train = np.concatenate([S_sets[j] for j in range(10) if j != fold])
    
    X_val = X_sets[fold]
    S_val = S_sets[fold]
    val_dict = dicts_sets[fold]

    # Train the model
    train(X_train, S_train,
          X_val, S_val,
          X_val, S_val,
          val_dict,
          '{}fold_{}/'.format(model_path, fold),
          N, tau, n0, nenc,
          epochs, batch_size, lr,
          dataset_name
    )

def generate_train_config_file(N_list, n0_list, nenc_list,
                               file_name = 'train_config.json'):
    """Generate a json file with the training configurations. The number of
    training configurations is reported in the display. This allows to launch
    training jobs through the `train_launcher.py` `--hyperparameters_file`
    option.

    Args:
    -----
        N_list (list): List of the number of samples per window.
        n0_list (list): List of the number of filters in the first convolutional
        layer.
        nenc_list (list): List of the number of encoder/decoder layers.
        file (str): Name of the file to be saved. Default is
            'train_config.json'.
    """

    data = {}
    id = 0
    for N in N_list:
        for n0 in n0_list:
            for nenc in nenc_list:
                data[id] = {'N': N, 'n0': n0, 'nenc': nenc}
                id += 1

    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

    print('Number of training configurations: {}'.format(id))

def generate_cv_train_config_file(N_list, n0_list, nenc_list,
                                  fold_list = list(range(10)),
                                  file_name = 'train_config_cv.json'):
    """Generate a json file with the training configurations for the 10-fold
    cross-validation. The number of training configurations is reported in the
    display. This allows to launch training jobs through the `train_launcher.py`
    `--CV --hyperparameters_file` options.

    Args:
    -----
        N_list (list): List of the number of samples per window.
        n0_list (list): List of the number of filters in the first convolutional
        layer.
        nenc_list (list): List of the number of encoder/decoder layers.
        fold_list (list): List of the folds to be used as validation set.
            Default is all, i.e. [0,1,2,3,4,5,6,7,8,9].
        file (str): Name of the file to be saved. Default is
            'train_config_cv.json'.
    """

    data = {}
    id = 0
    for N in N_list:
        for n0 in n0_list:
            for nenc in nenc_list:
                for fold in fold_list:
                    data[id] = {}
                    data[id]['N'] = N
                    data[id]['n0'] = n0
                    data[id]['nenc'] = nenc
                    data[id]['fold'] = fold

                    id += 1

    with open(file_name, 'w') as f:
        json.dump(data, f)

    print('Number of training configurations: {}'.format(id))