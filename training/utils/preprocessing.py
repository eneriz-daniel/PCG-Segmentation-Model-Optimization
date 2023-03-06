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
from typing import Tuple, Dict
import scipy.signal
import pywt
import warnings
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import os
import pandas as pd
import copy
import json
import librosa
import matplotlib.pyplot as plt
from scipy import io as sio

DATASET2022_PATH = "../physionet.org/files/circor-heart-sound/1.0.3"
DATASET2016_PATH = "../physionet.org/files/hss/1.0/extracted_files"

# Load patient data as a string.
def load_patient_data(filename : str) -> str:
    """This function loads the patient data from a file. This function is taken
    from the [PhysioNet 2022 Challenge repository](https://github.com/physionetchallenges/python-classifier-2022).
    
    Args:
    -----
        filename (str): The path to the file containing the patient data.
    
    Returns:
    --------
        str: The patient data.
    """

    with open(filename, 'r') as f:
        data = f.read()
    return data

# Define the spike removal function
def schmidt_spike_removal(original_data: np.ndarray, fs: float) -> np.ndarray:
    """This function removes the spikes from the data as described in [1]. It is
    based on the [MATLAB implementation of David Springer](https://github.com/davidspringer/Schmidt-Segmentation-Code/blob/master/schmidt_spike_removal.m)
    and the [Python implementation of Mutasem Aldmour](https://github.com/mutdmour/mlnd_heartsound/blob/master/project/sampleModel/schmidt_spike_removal.py)
    
    Args:
    -----
        data (np.ndarray): The data to be preprocessed.
        fs (float): The sampling frequency of the data.
    
    Returns:
    --------
        np.ndarray: The preprocessed data.
    
    References:
    -----------
        [1] S. E. Schmidt et al., "Segmentation of heart sound recordings by a
        duration-dependent hidden Markov model," Physiol. Meas., vol. 31, no. 4,
        pp. 513-29, Apr. 2010
    """
    # Find the window size to be 500ms
    window_size = int(np.round(fs*0.5))

    # Find any samples outside of a integer number of windows
    trailing_samples = int(np.mod(original_data.size, window_size))

    # Reshape the singal into a number of windows
    #sample_frames = original_data[:original_data.size-trailing_samples].reshape(window_size, -1)
    sample_frames = np.array(np.split(original_data[:original_data.size-trailing_samples], round((original_data.size-trailing_samples)/window_size), axis=0))
    
    # Find the MAAs (Maximum Absolute Amplitude) of each window
    maa = np.max(np.abs(sample_frames), axis=1)

    # While there are still samples greater than 3* the median value of the MAAs,
    # then remove those spikes
    while np.any(maa > 3*np.median(maa)):
        
        # Find the window with the maximum MAAs
        window_num = np.argmax(maa)

        # Find the position of the spike within that window
        spike_idx = np.argmax(np.abs(sample_frames[window_num, :]))

        # Finding zero crossings (where there may not be actual 0 values, just
        # a change from positive to negative or vice versa)):
        zero_crossings = np.abs(np.diff(np.sign(sample_frames[window_num][:])))
        if (len(zero_crossings) == 0):
            zero_crossings = [0]
        zero_crossings = [1 if i > 1 else 0 for i in zero_crossings ] + [0]

        #Find the start of the spike, finding the last zero crossing before
        # spike position. If that is empty, take the start of the window:
        spike_start = np.nonzero(zero_crossings[0:spike_idx+1])[0]
        # print spike_start
        if (len(spike_start) > 0):
            spike_start = spike_start[-1]
        else:
            spike_start = 0

        # Find the end of the spike, finding the first zero crossing after spike
        # position. If that is empty, take the end of the window:
        zero_crossings[0:spike_idx+1] = [0]*(spike_idx+1)
        spike_end = np.nonzero(zero_crossings)[0]
        if (len(spike_end) > 0):
            spike_end = spike_end[0] + 1
        else:
            spike_end = window_size

        # Set to zero the samples within the spike
        sample_frames[window_num, spike_start:spike_end] = 0.0001

        # Recalculate the MAAs
        maa = np.max(np.abs(sample_frames), axis=1)
    
    # Reshape the signal back into a single vector
    data = sample_frames.reshape(-1)

    # Add the trailing samples back to the signal
    if trailing_samples:
        data = np.append(data, original_data[-trailing_samples:])

    return data

# Get number of recording locations from patient data.
def get_num_locations(data) -> int:
    """This function gets the number of recording locations from the patient
    data. This function is taken from the [PhysioNet 2022 Challenge repository](https://github.com/physionetchallenges/python-classifier-2022).

    Args:
    -----
        data (str): The patient data.

    Returns:    
    --------
        int: The number of recording locations.
    """
    num_locations = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                num_locations = int(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_locations

def load_recordings(data_folder: str, data: str, get_frequencies: bool = False,
                    get_filenames: bool = False) -> Tuple:
    """This function loads the recordings from the patient included in `data`.
    It is based on (a function of the PhysioNet Challenge 2022 example
    code)[https://github.com/physionetchallenges/python-classifier-2022/blob/edf2c64cdbc8d467ac69253d31ff76243c96e041/helper_code.py#L66]
    but adapted to use librosa to read the wavs and to optionally output the
    filenames.

    Args:
    -----
        data_folder (str): The folder where the data is stored.
        data (str): The patient data. This must be loaded using `helper_code.load_patient_data`.
        get_frequencies (bool): Whether to output the frequencies of the recordings.
        get_filenames (bool): Whether to output the filenames of the recordings.
    
    Returns:
    --------
        A tuple containing these elements zipped:
            Tuple of np.ndarray: The recordings.
            Tuple of sampling frequencies. (Optional)
            Tuple of filenames. (Optional)
    """
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations+1]

    recordings = list()
    frequencies = list()
    filenames = list()
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = librosa.load(filename, sr=None)

        # If the recording lentgh is odd, remove the last sample
        if len(recording) % 2 == 1:
            recording = recording[:-1]

        recordings.append(recording)
        frequencies.append(frequency)
        filenames.append(filename)

    if get_frequencies:
        if get_filenames:
            return zip(recordings, frequencies, filenames)
        else:
            return zip(recordings, frequencies)
    else:
        if get_filenames:
            return zip(recordings, filenames)
        else:
            return recordings

# Filter the BadCoefficients warning
warnings.filterwarnings('ignore', category=scipy.signal.BadCoefficients)

# Define the function to compute the Homomorphic Envelope and the Hilbert envelope:
def homomorphic_envelogram_with_hilbert(input_signal: np.ndarray, fs: float, lpf_frequency=8) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the homomorphic envelope of a signal based on the Hilbert's
    envelope, which is also returned. This implementation is based on Mutasem
    Aldmour's work https://github.com/mutdmour/mlnd_heartsound which is a
    reimplementation of the MATLAB version published by David Springer
    https://github.com/davidspringer/Schmidt-Segmentation-Code.

    In [1-3], the researchers found the homomorphic envelope of Shannon energy.
    However, in [4], the authors state that the singularity at 0 when using the
    natural logarithm (resulting in values of -inf) can be fixed by using a
    complex valued signal. They motivate the use of the Hilbert transform to
    find the analytic signal, which is a converstion of a real-valued signal to
    a complex-valued signal, which is unaffected by the singularity.

    The usage of the Hilbert transform is explained in [5].

    Args:
    -----
        input_signal: 1D numpy array containing the input signal.
        fs: Sampling frequency of the input signal.
        lpf_frequency: Cutoff frequency of the low-pass filter. Default is 8 Hz.

    Returns:
    --------
        hilbert_envelope: 1D numpy array containing the Hilbert envelope.
        homomorphic_envelope: 1D numpy array containing the homomorphic envelope
        of the input signal.

    References:
    -----------
        [1] S. E. Schmidt et al., Segmentation of heart sound recordings by a 
        duration-dependent hidden Markov model., Physiol. Meas., vol. 31, no. 4,
        pp. 513?29, Apr. 2010.
        
        [2] C. Gupta et al., Neural network classification of homomorphic segmented
        heart sounds, Appl. Soft Comput., vol. 7, no. 1, pp. 286-297, Jan. 2007.
        
        [3] D. Gill et al., Detection and identification of heart sounds using 
        homomorphic envelogram and self-organizing probabilistic model, in 
        Computers in Cardiology, 2005, pp. 957-960.

        [4] I. Rezek and S. Roberts, Envelope Extraction via Complex Homomorphic
        Filtering. Technical Report TR-98-9, London, 1998

        [5] Choi et al, Comparison of envelope extraction algorithms for cardiac
        sound signal segmentation, Expert Systems with Applications, 2008.
    """
    # Check if the input signal is a 1D numpy array:
    if not isinstance(input_signal, np.ndarray) or len(input_signal.shape) != 1:
        raise ValueError('The input signal must be a 1D numpy array.')

	#8Hz, 1st order, Butterworth LPF
    B_low, A_low  = scipy.signal.butter(1,2*lpf_frequency/fs,'low')

    # Obtain the Hilbert transform:
    hilbert_envelope = np.abs(scipy.signal.hilbert(input_signal))

    # Calculate the homomorphic envelope of the signal using the Hilbert transform to obtain the analytic signal.
    homomorphic_envelope = np.exp(scipy.signal.filtfilt(B_low, A_low, np.log(hilbert_envelope)))

    return hilbert_envelope, homomorphic_envelope

# Define the function to compute the PSD envelope:
def psd(input_signal: np.ndarray, fs: float, fl_low: float = 40, fl_high: float = 60, resample: bool = True) -> np.ndarray:
    """Calculate the PSD envelope of a signal between fl_low and fl_high. This
    implementation is based on the description available at [1].
    
    Args:
    -----
        input_signal: 1D numpy array containing the input signal.
        fs: Sampling frequency of the input signal
        fl_low: Lower frequency limit of the PSD envelope. Defaults to 40 Hz.
        fl_high: Higher frequency limit of the PSD envelope. Defaults to 60 Hz.
        resample: If True, the signal is resampled to fs. Defaults to True.

    Returns:
    --------
        psd_envelope: 1D numpy array containing the PSD envelope of the input
        signal.

    References:
    -----------
        [1] D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
        Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
    """

    # * It reduces significantly the equivalent sampling frecuency 

    # Check if the input is a 1D numpy array
    if not isinstance(input_signal, np.ndarray) or len(input_signal.shape) != 1:
        raise ValueError('The input signal must be a 1D numpy array.')

    # Check if frecuecy limits are valid
    if fl_low < 0 or fl_low > fs/2:
        raise ValueError('The lower frequency limit must be between 0 and fs/2.')
    if fl_high < 0 or fl_high > fs/2:
        raise ValueError('The higher frequency limit must be between 0 and fs/2.')
    if fl_low > fl_high:
        raise ValueError('The lower frequency limit must be smaller than the higher frequency limit.')
    
    # Find the spectrogram of the signal. The window width is set to 0.05
    # seconds, with 50% overlap.
    f, t, Sxx = scipy.signal.spectrogram(input_signal, fs, nperseg=int(fs*0.5*0.05), noverlap=int(fs*0.05*0.5*0.5), nfft=int(fs))

    # Find the lower and higher frequency limits of the PSD envelope
    fl_low_index = np.argmin(np.abs(f-fl_low))
    fl_high_index = np.argmin(np.abs(f-fl_high))

    # Find the mean PSD over the frequency range
    psd_envelope = np.mean(Sxx[fl_low_index:fl_high_index,:], axis=0)

    if resample:
        # Resample the PSD envelope to the input signal's sampling frequency
        psd_envelope = scipy.signal.resample(psd_envelope, len(input_signal))

    return psd_envelope

# Partially extracted from: https://github.com/cmescobar/Heart_sound_prediction/blob/master/hsp_utils/envelope_functions.py
def wavelet(signal_in, wavelet='db1', levels=[4],
            start_level=1, end_level=5, erase_pad=True)-> np.ndarray:
    """Calculates and returns the Wavelet envelope of a signal based on Shannon's 
    energy [1,2]. This implementation is based on Christian Escobar Arcer's work 
    https://github.com/cmescobar/Heart_sound_prediction. He used the Stationary Wavelet
    Decomposition, that corresponds to the Discrete Wavelet Transform without 
    decimating the signal. 

    In [3], the researches used a similar method to produce one of the four different 
    envelopes that would be the input of a segmentation Convolutional Neural Network 
    (CNN).

    Args:
    -----
    signal_in: 1D numpy array containing the input signal.
    wavelet: used wavelet (https://www.pybytes.com/pywavelets/ref/wavelets.html)
    levels: selected levels for 
    start_level: Wavelet decomposition start level
    end_level: Wavelet decomposition end level
    erase_pad: boolean that indicates is padding to compute the 
                Stationary Wavelet Transform is erased or not 

    Returns:
    --------
    DWT_envs: n-D numpy array containing the Shannon envelope for the selected
    Wavelet decomposition levels (i.e., "levels" values).

    References:
    -----------

    [1] L. Huiying et al., A Heart Sound Segmentation Algorithm using Wavelet 
    Decomposition and Reconstruction, 19th International Conference - 
    IEEE/EMBS, 1997. 

    [2] Choi et al, Comparison of envelope extraction algorithms for cardiac
    sound signal segmentation, Expert Systems with Applications, 2008.

    [3] Renna et al., Deep Convolutional Neural Network for Heart Sound Segmentation,
    IEEE Journal of Biomedical and Health Informatics, 2019.
    """  
    # Original data points number
    N = signal_in.shape[0]

    # If N is odd, delete the last point
    if N % 2 == 1:
        signal_in = signal_in[:-1]
        N = N - 1
        add_a_zero = True
    else:
        add_a_zero = False

    # Amount of desired points 
    points_desired = 2 ** int(np.ceil(np.log2(N)))

    # Padding points number definition 
    pad_points = (points_desired-N) // 2

    # Padding to reach the needed power of two Paddeando 
    audio_pad = np.pad(signal_in, pad_width=pad_points, 
                        constant_values=0)

    # Stationary Wavelet Decomposition
    coeffs = pywt.swt(audio_pad, wavelet=wavelet, level=end_level, 
                      start_level=start_level)

    # Array to store decomposition levels 
    wav_coeffs = np.zeros((len(coeffs[0][1]), 0))

    for level in levels:
        # Indexes according to pywt.swt(.) output
        coef_i =  np.expand_dims(coeffs[-level + start_level][1], -1)
        
        # Coefficients concatenation
        wav_coeffs = np.concatenate((wav_coeffs, coef_i), axis=1)
        
    # Erase padding points if needed.
    # The pad_points is added to skip pad_points = 0 cases
    if erase_pad and pad_points:
        wav_coeffs_out = wav_coeffs[pad_points:-pad_points]
    else:
        wav_coeffs_out = wav_coeffs
    
    # Normalization of decomposition coefficients
    coeffs_norm = wav_coeffs_out/np.max(wav_coeffs_out)

    # Substitute zero values with an small number to avoid log(0)
    coeffs_norm[coeffs_norm == 0] = 1e-10
    
    # Computation of Shannon Energy 
    DWT_envs = -(np.square(coeffs_norm)*np.log(np.square(coeffs_norm)))

    # Supress second dimension of array 
    DWT_envs = DWT_envs[:, 0]

    # Add a zero if needed
    if add_a_zero:
        DWT_envs = np.concatenate((DWT_envs, np.zeros(1)))

    return np.abs(DWT_envs)

# Define a dictionary with the default parameters for the envelope/envelogram
# extraction functions
default_envelope_config = {
    'homomorphic_envelogram_with_hilbert': {'lpf_frequency': 8},
    'psd': {'fl_low': 40, 'fl_high': 60, 'resample': True},
    'wavelet': {'wavelet': 'db1',
                'levels': [4],
                'start_level': 1,
                'end_level': 6,
                'erase_pad': True}
}

def get_envelopes(input_signal: np.ndarray, fs: float,
    config_dict: Dict = default_envelope_config) -> np.ndarray:
    """Gets the envelopes and evelograms from the input signal, using the
    parameters defined in config_dict.
    
    Args:
    -----
        input_signal (np.ndarray): 1D numpy array containing the input signal.
        fs (float): Sampling frequency of the input signal.
        config_dict (Dict): Dictionary with the configuration of the
        envelopes.
    
    Returns:
    --------
        x (np.ndarray): Stack of the selected the envelopes/envelograms. The
        default order is homomorphic, hilbert, psd, and wavelet.
    """

    # Check if the input is a 1D numpy array
    if not isinstance(input_signal, np.ndarray) or len(input_signal.shape) != 1:
        raise ValueError('The input signal must be a 1D numpy array.')

    # Check if the dictionary is valid
    if not isinstance(config_dict, dict):
        raise ValueError('The config_dict must be a dictionary.')

    # Get the envelopes
    envelopes = []
    for key, value in config_dict.items():
        if key == 'homomorphic_envelogram_with_hilbert':
            hilbert_envelope, homomorphic_envelope = homomorphic_envelogram_with_hilbert(input_signal, fs, config_dict[key]['lpf_frequency'])
            envelopes.append(homomorphic_envelope)
            envelopes.append(hilbert_envelope)
        elif key == 'psd':
            psd_envelope = psd(input_signal, fs, config_dict[key]['fl_low'], config_dict[key]['fl_high'], config_dict[key]['resample'])
            envelopes.append(psd_envelope)
        elif key == 'wavelet':
            wavelet_envelope = wavelet(input_signal, config_dict[key]['wavelet'], config_dict[key]['levels'], config_dict[key]['start_level'],config_dict[key]['end_level'],config_dict[key]['erase_pad'])
            envelopes.append(wavelet_envelope)

    # Stack the envelopes
    x = np.stack(envelopes, axis=0)

    return x

def renna_preprocess_wave(input_signal: np.ndarray, fs: float,
                     config_dict: Dict = default_envelope_config) -> np.ndarray:
    """This function preprocess the data as described in [1]. First a bandpass
    filter is applied to the signal between 25 and 400Hz. Then the spike removal
    method described in [2] is used, which is followed by the envelogram
    generation used in [3]. After this, the envelograms are downsampled to 50 Hz.
    Finally, a standardization (normalization with mean 0 and deviation 1) is used.

    Args:
    -----
        input_signal (np.ndarray): The data to be preprocessed.
        fs (float): Sampling frecuency of the signal.
        config_dict (Dict): Configuration dictionary for the envelopes
        generation.
    
    Returns:
    --------
        x, an np.ndarray containing the envelograms, with shape (4, 50*T), where
        T is the duration of input_signal.

    References:
    -----------
        [1] Renna, F., Oliveira, J., & Coimbra, M. T. (2019). Deep Convolutional
        Neural Networks for Heart Sound Segmentation. IEEE journal of biomedical
        and health informatics, 23(6), 2435-2445.
        https://doi.org/10.1109/JBHI.2019.2894222

        [2] S. E. Schmidt et al., "Segmentation of heart sound recordings by 
        a duration-dependent hidden Markov model," Physiol. Meas., vol. 31, no.
        4, pp. 513-29, Apr. 2010

        [3] D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
        Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
    """

    # Check if the input is a 1D numpy array
    if not isinstance(input_signal, np.ndarray) or len(input_signal.shape) != 1:
        raise ValueError('The input signal must be a 1D numpy array.')
    
    # Desing a BP filter between 25 and 400 Hz.
    filter = scipy.signal.butter(4, [25, 400], btype='bandpass', fs=fs, output='sos')
    # Apply the filter
    data = scipy.signal.sosfilt(filter, input_signal)

    # Spike removal from Schmidt et al. 2010 https://pubmed.ncbi.nlm.nih.gov/20208091/
    data = schmidt_spike_removal(data, fs)

    # Generate the envelograms
    x = get_envelopes(data, fs, config_dict)

    # Downsample the envelograms to 50 Hz. 
    x = scipy.signal.decimate(x, int(fs/50))

    # Standardize the envelograms
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)

    return x

def renna_preprocess_circor_annotations(input_data: np.ndarray) -> np.ndarray:
    """Takes the raw read annotations from the .tsv file of the CirCor dataset
    [1] and generates the s array, containing the labels of the heart states at
    50 Hz. Note that here s(t) is in the range (0 - 4), where 0 is the
    unclassified label and the values 1 to 4 represent each heart state.

    Args:
    -----
        input_data (np.ndarray): The raw annotations from the .tsv file.
    
    Returns:
    --------
        s (np.ndarray): The heart states at 50 Hz.
    
    References:
    -----------
        [1] J. H. Oliveira et al., "The CirCor DigiScope Dataset: From Murmur
        Detection to Murmur Classification," in IEEE Journal of Biomedical and
        Health Informatics, doi: 10.1109/JBHI.2021.3137048.
    """
    # Check if the input is a 2D numpy array with the second dimension of size 3
    if not isinstance(input_data, np.ndarray) or len(input_data.shape) != 2 or input_data.shape[1] != 3:
        raise ValueError('The input data must be a 2D numpy array with the second dimension of size 3.')

    # Find the duration in seconds of the recording (last value of the second column)
    T = input_data[-1,1]
    
    # Generate a time array with 50 Hz sampling rate
    t = np.arange(0, T, 1.0/50.0)

    # Generate the s array
    s = np.zeros_like(t)

    # For each time instant, find its heart state and assign it to the s array
    for i in range(len(t)):
        # Find the first value greater than t(i)
        j = np.argmax(input_data[:,1] > t[i])
        # Assign the heart state to s(i)
        s[i] = input_data[j,2]

    return s

def rolling_strided_window(x: np.ndarray, N: int, tau: int) -> np.ndarray:
    """Takes the input 2D array and extracts windows of size N and stride tau
    along de second dimension. If the input array is 1D another dimesion is
    expanded to perform de operation.

    Args:
    -----
        x (np.ndarray): Input 2D array.
        N (int): Size of the windows.
        tau (int): Stride of the windows.

    Returns:
    --------
        x_windows (np.ndarray): 2D array with the windows.
    """

    orig_input_1D = False

    # Check if the input is 1D
    if x.ndim == 1:
        x = x[np.newaxis, :]
        orig_input_1D = True
    
    # Check if the input is 2D
    if x.ndim != 2:
        raise ValueError("The input array must be 2D or 1D to be expanded.")

    # Check if the stride is a positive integer
    if not isinstance(tau, int) or tau <= 0:
        raise ValueError("The stride must be a positive integer.")
    
    # Check if the window size is a positive integer
    if not isinstance(N, int) or N <= 0:
        raise ValueError("The window size must be a positive integer.")
    
    x_windows = np.lib.stride_tricks.sliding_window_view(x, (x.shape[0],N))[:, 0::tau]

    # Remove the first dimension since it always outputs a 4D if the input is 2D
    x_windows = x_windows[0,:,:,:]

    # If the input was 1D, remove the extra dimension added at the beginning
    if orig_input_1D:
        x_windows = x_windows[:,0,:]

    return x_windows
   
def check_valid_sequence(x: np.ndarray, s: np.ndarray, verbose: int) -> Tuple[np.ndarray, np.ndarray]:
    """As there are annotations that have illegal heart state transitions, as
    the first 3 -> 1 in the 500086_MV.tsv file, this function checks if the
    heart state sequence of each window in s is valid, removing the windows
    that are not. Also it removes the windows with less than 3 heart
    transitions.

    Args:
    -----
        x (np.ndarray): The signal of the recording.
        s (np.ndarray): The heart state sequence of the recording.
        verbose (int): Verbosity level.

    Returns:
    --------
        x_valid (np.ndarray): The signal of the recording with only the valid
        windows.
        s_valid (np.ndarray): The heart state sequence of the recording with
        only the valid windows.
    """

    # Check if the first dimension is the same in x and s
    if x.shape[0] != s.shape[0]:
        raise ValueError('The first dimension of x and s must be the same.')
    
    # Iterate over the windows
    invalid_idxs = []
    for i in range(s.shape[0]):
        # Check if the heart state sequence is valid
        if np.unique(s[i, :]).shape[0] < 4:
            invalid_idxs.append(i)
            if verbose >= 2:
                warnings.warn('Window {} has less than 3 heart transitions'.format(i), RuntimeWarning)
        else:
            for j in range(1, s.shape[1]):
                if not (s[i, j] == s[i, j-1] or s[i, j] == s[i, j-1] % 4 + 1):
                    invalid_idxs.append(i)
                    if verbose >= 2:
                        warnings.warn('Invalid transition from {} to {} in window {}.'.format(s[i, j-1], s[i, j], i), RuntimeWarning)
                    break
        
    
    # Remove the invalid windows
    x_valid = np.delete(x, invalid_idxs, axis=0)
    s_valid = np.delete(s, invalid_idxs, axis=0)

    return x_valid, s_valid

def generate_X_S_2022(filenames: list, data_folder: str, N: int, tau: int,
                      verbose: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates the X and S arrays for the CirCor dataset [1]. The X array
    contains windows of size N and stride tau of the envelograms of the
    recordings sampled at 50 Hz, as done in Renna et al. [2]. The S array
    contains the heart state sequence of the recordings.

    Args:
    -----
        filenames (list): List of strings with the filenames of patient data
        (`.txt` files).
        data_folder (str): Path to the folder containing the data.
        N (int): Size of the windows.
        tau (int): Stride of the windows.
        verbose (int): Verbosity level.
    
    Returns:
    --------
        X (np.ndarray): 3D array with the windows of the recordings. Its shape
        is (number of windows, N, number of envelograms).
        S (np.ndarray): 3D array with the heart state sequence of the
        recordings. Its shape is (number of windows, N, number of heart states).
    """

    # Generate an XS vector from x and s with length N and stride tau
    # Initialize the X and S arrays
    X = np.zeros((0, 4, N))
    S = np.zeros((0, N))

    # Extract all recording for each patient
    for name in tqdm(filenames, desc='Iterating over patients') if verbose >=1 else filenames:
        
        data = load_patient_data(name)

        for recording, filename in load_recordings(data_folder, data, get_filenames=True):
            
            # Get envelopes
            x_global = renna_preprocess_wave(recording, fs = 4000)

            annotations = np.loadtxt(filename[:-4] + '.tsv')

            try:
                # Extract segmentation anotation from .tsv
                s_global = renna_preprocess_circor_annotations(annotations)
            except ValueError:
                # If the annotations are not valid, skip this recording
                continue

            # Extract the indexes of the s elements with heart state information
            labeled_idxs_global = np.where(s_global!=0)[0]

            # Find 0 intervals between heart state changes
            zero_intervals = np.diff(labeled_idxs_global)-1 != 0

            # Split x and s between those intervals
            x_split = np.split(x_global, labeled_idxs_global[1:][zero_intervals], axis=1)
            s_split = np.split(s_global, labeled_idxs_global[1:][zero_intervals])

            for k in range(len(x_split)):

                # Extract the indexes of the s elements with heart state information
                labeled_idxs = np.where(s_split[k]!=0)[0]

                # Use only data with heart state information
                x = x_split[k][:, labeled_idxs]
                s = s_split[k][labeled_idxs]
                
                if x.shape[1] < N:
                    # If the window is smaller than N, discard it
                    continue

                x = rolling_strided_window(x, N, tau)
                s = rolling_strided_window(s, N, tau)

                x, s = check_valid_sequence(x, s, verbose)

                # Stack the windows
                X = np.vstack((X, x))
                S = np.vstack((S, s))
        
    # Create a new axis in S and concatenate X and S
    S = S[:, np.newaxis, :]
    XS = np.concatenate((X, S), axis=1)

    # Shuffle the samples
    np.random.shuffle(XS)

    # Return the X and S arrays
    X = XS[:, :x.shape[1], :]
    S = XS[:, x.shape[1]:, :]

    # Swap axes to format the data as channels_last
    X = np.swapaxes(X, 1, 2)
    S = np.swapaxes(S, 1, 2)

    # Transform S to categorical
    S = to_categorical(S-1)

    return X, S

def read_single_wav_2022(patient : int, location : str,
                         multirecording_id : int = 0) -> Tuple[np.ndarray, float, str]:
    """Reads a single wav file.
    
    Args:
    -----
        patient: The patient ID.
        location: The location of the recording.
        multirecording_id: The ID of the recording if there are more than one
        recording in the same location per subject. Defaults to 0, that implies
        just one recording exists.
    Returns:
    --------
        The data, its sampling frequency and the path to the file accesed.
    """
    if multirecording_id:
        file_subpath = '/training_data/{}_{}_{}.wav'.format(patient, location, multirecording_id)

        # Check the OS
        if os.name == 'nt':
            file_path = DATASET2022_PATH.replace('/', '\\') + file_subpath.replace('/', '\\')
            wav_data, fs = librosa.load(file_path, sr=None)
        else:
            file_path = DATASET2022_PATH + file_subpath
            wav_data, fs = librosa.load(file_path, sr=None)
       
    else:
        file_subpath = '/training_data/{}_{}.wav'.format(patient, location)
        
        # Check the OS
        if os.name == 'nt':
            file_path = DATASET2022_PATH.replace('/', '\\') + file_subpath.replace('/', '\\')
            wav_data, fs = librosa.load(file_path, sr=None)
        else:
            file_path = DATASET2022_PATH + file_subpath
            wav_data, fs = librosa.load(file_path, sr=None)
    
    # Check if wav_data has even length and remove last element if it is odd.
    if len(wav_data) % 2 == 1:
        wav_data = wav_data[:-1]
    
    return wav_data, fs, file_path

def read_single_tsv_2022(patient : int, location : str,
                         multirecording_id : int = 0) -> Tuple[np.ndarray, str]:
    """This function reads a single tsv file.
    Args:
    -----
        patient: The patient ID.
        location: The location of the recording.
        multirecording_id: The ID of the recording if there are more than one
        recording in the same location per subject. Defaults to 0, that implies
        just one recording exists.
        
    Returns:
    --------
        The data and the path to the file accesed.
    """

    if multirecording_id:
        # Read the tsv file.
        
        file_subpath = '/training_data/{}_{}_{}.tsv'.format(patient, location, multirecording_id)
        # Check the OS
        if os.name == 'nt':
            file_path = DATASET2022_PATH.replace('/', '\\') + file_subpath.replace('/', '\\')
            tsv_data = np.loadtxt(file_path)
        else:
            file_path = DATASET2022_PATH + file_subpath
            tsv_data = np.loadtxt(file_path)
       
    else:
        file_subpath = '/training_data/{}_{}.tsv'.format(patient, location)
        # Check the OS
        if os.name == 'nt':
            file_path = DATASET2022_PATH.replace('/', '\\') + file_subpath.replace('/', '\\')
            tsv_data = np.loadtxt(file_path)
        else:
            file_path = DATASET2022_PATH + file_subpath
            tsv_data = np.loadtxt(file_path)
    
    return tsv_data, file_path

def prepare_dataset_2022(dataset_path: str = DATASET2022_PATH,
                         preprocesed_path: str = './circor-segmentation',
                         save_dict_name : str = '/data_dict.json',
                         verbose: bool = True) -> Dict:
    """This function prepares the data of the dataset [1] for the segmentation
    model. The data of each recording is read from the dataset path,
    preprocessed and saved in a npz file containing the envelograms (np.ndarray
    x) and the fundamental heart states (np.ndarray s) at a frecuency of 50 Hz.

    Args:
    -----
        dataset_path (str): Path to the dataset.
        preprocesed_path (str): Path to the directory where the preprocessed
        data is saved. Default is './circor-segmentation'.
        save_dict_name (str): Filename to save the data dictionary. Default is
        data_dict.json.
        verbose (bool): If True, prints the progress of the function. Default is
        True.
    
    Returns:
    --------
        data_dict (Dict): Dictionary containing the .npz file names in function
        of the patient ID, the location, the multirecording id and the split id,
        and othe useful data.
    
    References:
    -----------
        [1] J. H. Oliveira et al., "The CirCor DigiScope Dataset: From Murmur
        Detection to Murmur Classification," in IEEE Journal of Biomedical and
        Health Informatics, doi: 10.1109/JBHI.2021.3137048.
    """
    # Set the subpath of the .csv file containing a summary of the recordings
    csv_subpath = '/training_data.csv'

    # Check if the OS is Windows and reformat the paths if so
    if os.name == 'nt':
        dataset_path = dataset_path.replace('/', '\\')
        preprocesed_path = preprocesed_path.replace('/', '\\')
        csv_subpath = csv_subpath.replace('/', '\\')
        save_dict_name = save_dict_name.replace('/', '\\')

        no_labeled_recording = no_labeled_recording.replace('/', '\\')
    
    # Create the new folder
    if not os.path.exists(preprocesed_path):
        os.mkdir(preprocesed_path)

    # Read the .csv file containing a summary of the recordings
    df = pd.read_csv(dataset_path + csv_subpath)
    
    # Initialize the data dictionary
    data_dict = {}

    posible_locations = ['AV', 'TV', 'PV', 'MV', 'Phc']

    # Iterate over the rows in the dataframe
    for index, row in tqdm(df.iterrows(), total=df.shape[0]) if verbose else df.iterrows():

        # Save the patient id
        patient_id = str(row['Patient ID'])

        # Obtain existing locations of patient data
        existing_locations = row['Recording locations:'].split('+')

        # Count the number of recordings for each location
        locations_count = np.zeros(len(posible_locations), dtype=int)
        for i, location in enumerate(posible_locations):
            locations_count[i] = existing_locations.count(location)
        
        # Save the patient general data
        data_dict[patient_id] = {"Existing locations": existing_locations,
                                 "Location count": locations_count.tolist(),
                                 "Unique locations": np.unique(existing_locations).tolist(),}

        # Iterate over posible locations
        for i, location in enumerate(posible_locations):

            data_dict[patient_id][location] = {"Count": int(locations_count[i])}

            # Iterate over the number of recordings for each location
            for j in range(locations_count[i]):

                # Read the recording data
                # If more than one recording for the same location use this line
                if locations_count[i] > 1:
                    data, fs, audio_file = read_single_wav_2022(patient_id, location, j+1)
                else:
                    data, fs, audio_file = read_single_wav_2022(patient_id, location)

                # Read the annotations data
                if locations_count[i] > 1:
                    annotations, _ = read_single_tsv_2022(patient_id, location, j+1)
                else:
                    annotations, _ = read_single_tsv_2022(patient_id, location)

                try: # Try de feature and label generation
                    # Preprocess the data
                    x_global = renna_preprocess_wave(data, fs)

                    # Preprocess the annotations
                    s_global = renna_preprocess_circor_annotations(annotations)
                    
                except ValueError:
                    # Go to the next recording
                    continue

                # Extract the indexes of the s elements with heart state information
                labeled_idxs_global = np.where(s_global!=0)[0]

                # Find 0 intervals between heart state changes
                zero_intervals = np.diff(labeled_idxs_global)-1 != 0

                # Split x and s between those intervals
                x_split = np.split(x_global, labeled_idxs_global[1:][zero_intervals], axis=1)
                s_split = np.split(s_global, labeled_idxs_global[1:][zero_intervals])

                data_dict[patient_id][location][str(j)] = {"Number of splits": len(x_split)}

                for k in range(len(x_split)):

                    # Extract the indexes of the s elements with heart state information
                    labeled_idxs = np.where(s_split[k]!=0)[0]

                    # Use only data with heart state information
                    x = x_split[k][:, labeled_idxs]
                    s = s_split[k][labeled_idxs]

                    # Save the data
                    # If more than one recording for the same location use this format:
                    if locations_count[i] > 1:
                        file_name = '/{}_{}_{}-split{}.npz'.format(patient_id, location, j, k)
                    else:
                        file_name = '/{}_{}-split{}.npz'.format(patient_id, location, k)
                    
                    # If working on Windows, use the following format:
                    if os.name == 'nt':
                        file_name = file_name.replace('/', '\\')
                        
                    np.savez(preprocesed_path + file_name, x=x, s=s)

                    data_dict[patient_id][location][str(j)][str(k)] = {"File name": file_name,
                                                                       "Number of frames": x.shape[1]}
    
    if save_dict_name is not None:
        # Save data dict under the json file in preprocesed_path/save_dict_name
        with open(preprocesed_path + save_dict_name, 'w') as fp:
            json.dump(data_dict, fp)

    return data_dict

def find_valid_data_2022(data_dict: Dict, N: int, tau: int,
                         preprocesed_path: str = './circor-segmentation',
                         save_dict_name: str = None) -> Dict:
    """Takes the dictionary generated by `prepare_dataset_2022` and creates
    another one with only the data that has at least N frames.

    Args:
    -----
        data_dict (dict): Dictionary generated by `prepare_dataset_2022`.
        N (int): Minimum number of frames.
        tau (int): Stride for rolling window.
        preprocesed_path (str): Path to the preprocessed data.
        save_dict_name (str): Name of the json file to save the data dict.

    Returns:
    --------
        data_dict_valid (dict): Dictionary with only the data that has at least
        N frames. It contains also the number of valid windows of size N in each
        patient and in each sample.
    """
    # Check if N is a positive integer
    if not isinstance(N, int) or N <= 0:
        raise ValueError('N must be a positive integer.')
    
    # Check if tau is a positive integer
    if not isinstance(tau, int) or tau <= 0:
        raise ValueError('tau must be a positive integer.')
    
    # If running under Windows, replace '/' with '\\'
    if os.name == 'nt':
        preprocesed_path = preprocesed_path.replace('/', '\\')

    # Check if the preprocesed_path exists
    if not os.path.exists(preprocesed_path):
        raise FileNotFoundError('The preprocesed_path does not exist.')
    
    posible_locations = ['AV', 'TV', 'PV', 'MV', 'Phc']

    data_dict_valid = copy.deepcopy(data_dict)

    for patient_id in data_dict:
        for location in data_dict[patient_id]["Unique locations"]:       
            for j in [key for key in data_dict[patient_id][location].keys() if key != 'Count']:
                for k in [key for key in data_dict[patient_id][location][j].keys() if key != 'Number of splits']:
                    # If the split is not valid, remove it
                    if data_dict[patient_id][location][j][k]["Number of frames"] < N:
                        data_dict_valid[patient_id][location][j].pop(k)

                        # Update the number of splits
                        data_dict_valid[patient_id][location][j]["Number of splits"] -= 1
                
                # If there are no splits left, remove the multi recording
                if data_dict_valid[patient_id][location][j]["Number of splits"] == 0:
                    data_dict_valid[patient_id][location].pop(j)

                    # Update the number of recordings
                    data_dict_valid[patient_id][location]["Count"] -= 1

                    # Update the location count list
                    data_dict_valid[patient_id]["Location count"][posible_locations.index(location)] -= 1

                    # Update the existing locations list
                    data_dict_valid[patient_id]["Existing locations"].remove(location)
            
            # If there are no recordings left, remove the location
            if data_dict_valid[patient_id][location]["Count"] == 0:
                data_dict_valid[patient_id].pop(location)

                # Update the list of unique locations
                data_dict_valid[patient_id]["Unique locations"].remove(location)

        # If there are no locations left, remove the patient
        if np.sum(data_dict_valid[patient_id]["Location count"]) == 0:
            data_dict_valid.pop(patient_id)
        
    # Compute the number of valid windows of size N in each patient
    for patient_id in data_dict_valid:
        n_samples_patient = 0
        data_dict_valid[patient_id]["Number of valid windows"] = 0
        for location in data_dict_valid[patient_id]["Unique locations"]:
            for j in [key for key in data_dict_valid[patient_id][location].keys() if key != 'Count']:
                for k in [key for key in data_dict_valid[patient_id][location][j].keys() if key != 'Number of splits']:
                    x = np.load(preprocesed_path + data_dict_valid[patient_id][location][j][k]["File name"])['x']
                    n_samples = rolling_strided_window(x, N, tau).shape[0]
                    data_dict_valid[patient_id][location][j][k]['Number of valid windows'] = n_samples
                    n_samples_patient += n_samples
        data_dict_valid[patient_id]["Number of valid windows"] = n_samples_patient
    
    if save_dict_name is not None:
        # If the first char in save_dict_name is not '/', add it
        if save_dict_name[0] != '/':
            save_dict_name = '/' + save_dict_name
        
        # If running under Windows, replace '/' with '\\'
        if os.name == 'nt':
            save_dict_name = save_dict_name.replace('/', '\\')

        # Save data dict under the json file in preprocesed_path/save_dict_name
        with open(preprocesed_path + save_dict_name, 'w') as fp:
            json.dump(data_dict_valid, fp)

    return data_dict_valid

def equally_sum_subset_partition(data_dict: Dict, n_splits: int,
                                 verbose: bool = True) -> Tuple[Dict]:
    """Takes data_dict which is supposed to be the output of find_valid_data and
    creates n_splits dictionaries with cuasiequaly distributed patients, trying
    to keep the same number of samples in each split. It uses the Greedy number
    partiton algorithm.

    Args:
    -----
        data_dict (Dict): Dictionary generated by `find_valid_data`, that has
        the number of samples per patient.
        n_splits (int): Number of splits to create.

    Returns:
    --------
        data_dict_splits (Tuple[Dict]): Tuple of `n_splits` dictionaries with
        the same structure as data_dict, with the patients and samples info per
        each split.
    """
    # Check if n_splits is a positive integer
    if not isinstance(n_splits, int) or n_splits <= 0:
        raise ValueError('n_splits must be a positive integer.')
    
    # Check if data_dict is a dictionary
    if not isinstance(data_dict, dict):
        raise TypeError('data_dict must be a dictionary.')
    
    # Check if data_dict has the number of samples per patient
    for patient_id in data_dict:
        if not 'Number of valid windows' in data_dict[patient_id]:
            raise KeyError('The data_dict does not have the number of samples in patient {}.'.format(patient_id))
    
    # Create a dataframe with the number of samples per patient
    df_n_samples_per_patient = pd.DataFrame({'Patient ID': list(data_dict.keys()), 'Number of valid windows': [data_dict[patient_id]['Number of valid windows'] for patient_id in list(data_dict.keys())]})
    
    # Sort it by number of samples
    df_n_samples_per_patient = df_n_samples_per_patient.sort_values(by='Number of valid windows', ascending=False)

    # Create a list of n_splits datasets
    df_splits = [pd.DataFrame({'Patient ID': [], 'Number of valid windows': []})]*n_splits

    # Iterate over the sorted dataframe
    for index, row in df_n_samples_per_patient.iterrows():
        # Find the split with the smallest number of samples
        min_split = np.argmin([df_splits[i]['Number of valid windows'].sum() for i in range(n_splits)])
        
        # Add the patient to the split using pd.concat
        df_splits[min_split] = pd.concat([df_splits[min_split], df_n_samples_per_patient.iloc[[index]]])
    
    # If verbose, print the number of samples per split
    if verbose:
        print('Samples have been distributed using equally sum partion in {} splits:'.format(n_splits))
        for i in range(n_splits):
            print('\tSplit {}: {}'.format(i, int(df_splits[i]['Number of valid windows'].sum())))

    # Create a list of dictionaries with the same structure as data_dict from df_splits
    data_dict_splits = []
    for i in range(n_splits):
        data_dict_splits.append({})
        for patient_id in df_splits[i]['Patient ID']:
            data_dict_splits[i][patient_id] = data_dict[patient_id]
    
    return data_dict_splits

def create_train_valid_test_datasets(data_dict: Dict, train_prop: float = 0.6,
                                     valid_prop: float = 0.2,
                                     test_prop: float = 0.2,
                                     verbose: bool = True,
                                     save_dicts_fmt = None) -> Tuple[Dict, Dict, Dict]:
    """Takes data_dict, which is supposed to include the number of valid samples
    per patient, i.e. its the output of `find_valid_data`, and creates three
    dictionaries with the samples per patient for the train, validation and
    test sets using the proprtions defined in the arguments.

    Args:
    -----
        data_dict (Dict): Dictionary generated by `find_valid_data`, that has
        the number of samples per patient.
        train_prop (float): Proportion of the samples to use for the train set.
        valid_prop (float): Proportion of the samples to use for the validation set.
        test_prop (float): Proportion of the samples to use for the test set.
        verbose (bool): If True, prints the number of samples per set.
        save_dicts_fmt (str): If not None, its the format to save the train,
        valid and test

    Returns:
    --------
        data_dict_train (Dict): Dictionary with the samples per patient for the
        train set.
        data_dict_valid (Dict): Dictionary with the samples per patient for the
        validation set.
        data_dict_test (Dict): Dictionary with the samples per patient for the
        test set.
    """
    # Check if train_prop, valid_prop and test_prop are floats between 0 and 1
    if not (isinstance(train_prop, float) and isinstance(valid_prop, float) and isinstance(test_prop, float)):
        raise TypeError('train_prop, valid_prop and test_prop must be floats.')
    if not (0 <= train_prop <= 1 and 0 <= valid_prop <= 1 and 0 <= test_prop <= 1):
        raise ValueError('train_prop, valid_prop and test_prop must be floats between 0 and 1.')
    
    # Check if train_prop, valid_prop and test_prop sum 1
    if not (train_prop + valid_prop + test_prop == 1):
        raise ValueError('train_prop, valid_prop and test_prop must sum 1.')
    
    # Check if train_prop, valid_prop and test_prop are multiples of 0.1
    if not (train_prop*10 % 1 == 0 and valid_prop*10 % 1 == 0 and test_prop*10 % 1 == 0):
        raise ValueError('train_prop, valid_prop and test_prop must be multiples of 0.1.')
    
    # Check if data_dict is a dictionary
    if not isinstance(data_dict, dict):
        raise TypeError('data_dict must be a dictionary.')
    
    # Check if data_dict has the number of samples per patient
    for patient_id in data_dict:
        if not 'Number of valid windows' in data_dict[patient_id]:
            raise KeyError('The data_dict does not have the number of samples in patient {}.'.format(patient_id))
    
    # Check if save_dicts_fmt is a string
    if save_dicts_fmt is not None and not isinstance(save_dicts_fmt, str):
        raise TypeError('save_dicts_fmt must be a string.')
    
    # Check if save_dicts_fmt has just one {} in it
    if save_dicts_fmt is not None and not save_dicts_fmt.count('{}') == 1:
        raise ValueError('save_dicts_fmt must contain one {}.')
    
    # Check if save_dicts_fmt ends with .json
    if save_dicts_fmt is not None and not save_dicts_fmt.endswith('.json'):
        raise ValueError('save_dicts_fmt must end with .json.')


    # Equally distribute the samples per patient in 10 splits
    data_dict_splits = equally_sum_subset_partition(data_dict, 10, verbose=False)

    # Merge the first train_prop*10 splits to form the train set
    data_dict_train = {}
    for i in range(int(train_prop*10)):
        for patient_id in data_dict_splits[i]:
            data_dict_train[patient_id] = data_dict_splits[i][patient_id]
    
    # Do the same for the validation set
    data_dict_valid = {}
    for i in range(int(train_prop*10), int(train_prop*10 + valid_prop*10)):
        for patient_id in data_dict_splits[i]:
            data_dict_valid[patient_id] = data_dict_splits[i][patient_id]
    
    # And for the test set
    data_dict_test = {}
    for i in range(int(train_prop*10 + valid_prop*10), len(data_dict_splits)):
        for patient_id in data_dict_splits[i]:
            data_dict_test[patient_id] = data_dict_splits[i][patient_id]
    
    if verbose:
        print('Train set samples: {}'.format(sum([data_dict_train[patient_id]['Number of valid windows'] for patient_id in data_dict_train])))
        print('Validation set samples: {}'.format(sum([data_dict_valid[patient_id]['Number of valid windows'] for patient_id in data_dict_valid])))
        print('Test set samples: {}'.format(sum([data_dict_test[patient_id]['Number of valid windows'] for patient_id in data_dict_test])))

    if save_dicts_fmt is not None:
        # Save the dictionaries in json format
        with open(save_dicts_fmt.format('train'), 'w') as f:
            json.dump(data_dict_train, f)
        with open(save_dicts_fmt.format('valid'), 'w') as f:
            json.dump(data_dict_valid, f)
        with open(save_dicts_fmt.format('test'), 'w') as f:
            json.dump(data_dict_test, f)
    
    return data_dict_train, data_dict_valid, data_dict_test

def generate_X_S_from_dict_2022(data_dict: Dict, N: int, tau: int,
                      processed_path: str = './circor-segmentation') -> Tuple[np.ndarray, np.ndarray]:
    """For each .npz included in `data_dict` inside the folder `processed_path`,
    it extracts windows of size `N` with stride `tau` from `x` and `s` and
    stackes them in `X` and `S`, shuffling the samples.

    Args:
    -----
        data_dict (dict): Dictionary that has been filtered by `find_valid`.
        N (int): Window size.
        tau (int): Stride.
        processed_path (str): Path to the preprocessed data. Defaults to
        './circor-segmentation'.

    Returns:
    --------
        X (np.ndarray): Numpy array with shape (n_samples, x.shape[0], N).
        S (np.ndarray): Numpy array with shape (n_samples, N).
    """
    # Check if data_dict is a dictionary
    if not isinstance(data_dict, dict):
        raise TypeError('data_dict must be a dictionary.')
    
    # Check if N is a positive integer
    if not isinstance(N, int) or N <= 0:
        raise ValueError('N must be a positive integer.')
    
    # Check if tau is a positive integer
    if not isinstance(tau, int) or tau <= 0:
        raise ValueError('tau must be a positive integer.')
    
    # If running under Windows, replace '/' with '\\'
    if os.name == 'nt':
        processed_path = processed_path.replace('/', '\\')

    # Check if the processed_path exists
    if not os.path.exists(processed_path):
        raise FileNotFoundError('The processed_path does not exist.')

    # Extract the first patient
    patient_id = list(data_dict.keys())[0]
    # Extract its first location
    location = data_dict[patient_id]["Unique locations"][0]
    # Extract its first multi recording
    recording = [key for key in data_dict[patient_id][location].keys() if key != 'Count'][0]
    # Extract its first split
    split = [key for key in data_dict[patient_id][location][recording].keys() if key != 'Number of splits'][0]

    with np.load(processed_path + data_dict[patient_id][location][recording][split]["File name"]) as f:
        x = f['x']

    # Initialize the X and S arrays
    X = np.zeros((0, x.shape[0], N))
    S = np.zeros((0, N))

    # Iterate over the patients
    for patient_id in tqdm(data_dict):
        for location in data_dict[patient_id]["Unique locations"]:       
            for j in [key for key in data_dict[patient_id][location].keys() if key != 'Count']:
                for k in [key for key in data_dict[patient_id][location][j].keys() if key != 'Number of splits']:
                    # Load the .npz file
                    with np.load(processed_path + data_dict[patient_id][location][j][k]["File name"]) as f:
                        x = f['x']
                        s = f['s']
                    
                    x = rolling_strided_window(x, N, tau)
                    s = rolling_strided_window(s, N, tau)

                    x, s = check_valid_sequence(x, s, 2)

                    # Stack the windows
                    X = np.vstack((X, x))
                    S = np.vstack((S, s))

    # Create a new axis in S and concatenate X and S
    S = S[:, np.newaxis, :]
    XS = np.concatenate((X, S), axis=1)

    # Shuffle the samples
    np.random.shuffle(XS)

    # Return the X and S arrays
    X = XS[:, :x.shape[1], :]
    S = XS[:, x.shape[1]:, :]

    # Swap axes to format the data as channels_last
    X = np.swapaxes(X, 1, 2)
    S = np.swapaxes(S, 1, 2)

    # Transform S to categorical
    S = to_categorical(S-1)

    return X, S

def unroll_strided_windows(S: np.ndarray, tau: int) -> np.ndarray:
    """Unrolls the input `S` 3D array with shape (n_windows, N, 4), which is
    supposed to be generated with stride `tau`, outputing a 2D vector. The
    elements in the overlapping positions are averaged.

    Args:
    -----
        S (np.ndarray): Input 3D array.
        tau (int): Stride of the input array.

    Returns:
    --------
        s (np.ndarray): 2D array with shape (tau*(n_windows-1) + N, 4).
    """
    
    # Check that the input array is a np.ndarray
    if not isinstance(S, np.ndarray):
        raise TypeError('Input array must be a np.ndarray')
    
    # Check that the input array is 2D
    if S.ndim != 3:
        raise ValueError('Input array must be 3D.')

    # Check that the stride is positive integer
    if not isinstance(tau, int):
        raise TypeError('Stride must be an integer.')

    # Obtain the window size and the number of windows
    N = S.shape[1]
    n_windows = S.shape[0]

    # Calculate the length of the output array
    s_len = tau*(n_windows-1) + N

    # Create a 2D array of NaNs of size (n_windows, s_len)
    s_expanded = np.full((n_windows, s_len, 4), np.nan)

    # Allocate each window to the corresponding position in the expanded array
    for i in range(n_windows):
        s_expanded[i, tau*i:tau*i+N, :] = S[i, :, :]
    
    # Calculate the mean of the expanded array in the first axis
    s = np.nanmean(s_expanded, axis=0)

    return np.squeeze(s)

def extract_2016_data_from_mat(mat_file: str = '../physionet.org/files/hss/1.0/example_data.mat',
                               save_path: str = DATASET2016_PATH):
    """Extracts the data from the 2016 dataset saved in the .mat file [1] and
    saves it to a folder in npy files. The data saved has the recording
    identifier in the first dimension.

    Args:
    -----
        mat_file (str): The path to the .mat file. Defaults to '../physionet.org/physionet-2016.mat'.
        save_path (str): The path to save the data. Defaults to DATASET2016_PATH.
    
    References:
    -----------
        [1] https://physionet.org/content/hss/1.0/
    """

    file_name_format = '/{}.npy'

    # If running under Windows, change the paths format
    if os.name == 'nt':
        mat_file = mat_file.replace('/', '\\')
        save_path = save_path.replace('/', '\\')
        file_name_format = file_name_format.replace('/', '\\')

    # Load the data
    data = sio.loadmat(mat_file)['example_data'][0,0]

    # If save_path does not exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Data names
    data_names = ['audio', 'ecg_annotations', 'patient_number', 'binary_diagnosis']

    for i, data_name in enumerate(data_names):
        
        # If the data[i] hasn't the long dimension first, transpose it and squeeze it
        if data[i].shape[0] != data[0].size:
            data[i] = data[i].T.squeeze()

        # Save the data
        np.save(save_path + file_name_format.format(data_name), data[i])

def read_single_2016_audio(record_id: int) -> Tuple[np.ndarray, int]:
    """Reads a single recording from the Physionet 2016 dataset. The data must
    be extrated first from the .mat file using the `extrac_2016_data_from_mat`
    function. The data is finally normalized and returned with the sampling
    frecuency.
    
    Args:
    -----
        record_id (int): The recording identifier. Must be an integer below 792.
    
    Returns:
    --------
        data (np.ndarray): The audio data.
        fs (int): The sampling frequency. It is always 1000 Hz [1].
    
    References:
    -----------
        [1] Renna, F., Oliveira, J., & Coimbra, M. T. (2019). Deep Convolutional
        Neural Networks for Heart Sound Segmentation. IEEE journal of biomedical
        and health informatics, 23(6), 2435-2445.
        https://doi.org/10.1109/JBHI.2019.2894222
    """
    # Check if the record_id is valid
    if not record_id < 792:
        raise ValueError('record_id must be an integer below 792')
    
    # Audio file
    audio_file = DATASET2016_PATH + '/audio.npy'

    # If running under Windows, change the paths format
    if os.name == 'nt':
        audio_file = audio_file.replace('/', '\\')
        
    # Load the data. Allow pickle is needed to load objects (np.arrays)
    data = np.load(audio_file, allow_pickle=True)

    # Get the selected data
    data = data[record_id].squeeze()

    # Normalize data to be between -1 and 1
    data = data/np.max(np.abs(data))

    # Return the squeezed data
    return data, 1000

def read_single_2016_annotations(record_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reads a single annotation array from the Physionet 2016 dataset. The data
    must be extrated first from the .mat file using the
    `extrac_2016_data_from_mat` function.
    
    Args:
    -----
        record_id (int): The recording identifier. Must be an integer below 792.
    
    Returns:
    --------
       Tuple of np.ndarray: The ECG annotations data, the first column is the
       R-peak position and the second the end-T-wave position.
    """
    
    # Check if the record_id is valid
    if not record_id < 792:
        raise ValueError('record_id must be an integer below 792')
    
    # Audio file
    audio_file = DATASET2016_PATH + '/ecg_annotations.npy'

    # If running under Windows, change the paths format
    if os.name == 'nt':
        audio_file = audio_file.replace('/', '\\')
        
    # Load the data. Allow pickle is needed to load objects (np.ndarray)
    annotations = np.load(audio_file, allow_pickle=True)

    # Return the squeezed data
    return annotations[record_id, 0].squeeze(), annotations[record_id, 1].squeeze()

def visualize_annotated_2016_pcg(record_id: int, t_off: float = 0, T: float = 4,
                                 fig_path: str = './annotated-2016-pgc.png'):
    """Visualizes the PCG and its R-peaks and end-T-waves annotations of a single
    recording from the Physionet 2016 dataset. The data must be extrated first
    from the .mat file using the `extrac_2016_data_from_mat` function.
    
    Args:
    -----
        record_id (int): The recording identifier. Must be an integer below 792.
        t_off (float): The time offset. Defaults to 0.
        T (float): The duration of the plot. Defaults to 4.
        fig_path (str): The path to save the figure.
    """
    # Check if the record_id is valid
    if not record_id < 792:
        raise ValueError('record_id must be an integer below 792')

    # If running under Windows, change the paths format
    if os.name == 'nt':
        fig_path = fig_path.replace('/', '\\')
    
    # Read the data
    data, fs = read_single_2016_audio(record_id)
    annotations = read_single_2016_annotations(record_id)

    plt.figure(figsize=(15, 4))

    plt.plot(np.linspace(0, data.size/fs, data.size), data, label='PCG')
    plt.vlines(annotations[0]/50, -1, 1, label='R-peak', colors='C1')
    plt.vlines(annotations[1]/50, -1, 1, label='end-T-wave', colors='C2')

    plt.legend()

    # Create axis labels
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Zoom to the selected time interval
    plt.xlim(t_off, T)

    # Use record id as title
    plt.title('Record #{}'.format(record_id))

    plt.savefig(fig_path)

def label_2016_pcg_positions(envelopes: np.ndarray, annotations: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Generates an array with the PCG states annotations at 50 Hz sampling
    rate. It uses the R-peak and end-T-wave annotations available in the
    Physionet 2016 dataset and the gaussian model for the time duration of the
    PCG states. This implementation is based on the the code developed Springer
    et al. 2016 [1] publicly availble on
    https://physionet.org/content/hss/1.0/labelPCGStates.m
    
    Args:
    -----
        envelopes (np.ndarray): PCG-related envelopes. This envelopes must be
        generated at 50 Hz.
        annotations (Tuple[np.ndarray, np.ndarray]): The ECG annotations data,
        the first column is the R-peak position and the second the end-T-wave
        position.
    
    Returns:
    --------
        s (np.ndarray): The PCG states annotations at 50 Hz sampling rate.
    
    References:
    -----------
        [1] Springer, M., & Coimbra, M. T. (2016). A novel approach to
        delineate the PCG states in ECG recordings. IEEE journal of biomedical
        and health informatics, 25(3), 716-724.
        https://doi.org/10.1109/JBHI.2016.2517098
        
        [2] Renna, F., Oliveira, J., & Coimbra, M. T. (2019). Deep Convolutional
        Neural Networks for Heart Sound Segmentation. IEEE journal of biomedical
        and health informatics, 23(6), 2435-2445.
        https://doi.org/10.1109/JBHI.2019.2894222

        [3] S. E. Schmidt et al., "Segmentation of heart sound recordings by a
        duration-dependent hidden Markov model," Physiol. Meas., vol. 31, no. 4,
        pp. 513-29, Apr. 2010

        [4] A. G. Tilkian and M. B. Conover, Understanding heart sounds and
        murmurs: with an introduction to lung sounds, 4th ed. Saunders, 2001.
    """   
    # Check if annotations is a list of 2 np.ndarray
    if not isinstance(annotations, Tuple) or len(annotations) != 2 or not isinstance(annotations[0], np.ndarray) or not isinstance(annotations[1], np.ndarray):
        raise ValueError('annotations must be a list of 2 np.ndarray')
    
    # Check if the envelopes is a np.ndarray with 2 dimensions and the first one has size 4
    if not isinstance(envelopes, np.ndarray) or envelopes.ndim != 2 or envelopes.shape[0] != 4:
        raise ValueError('envelopes must be a np.ndarray with 2 dimensions and the first one has size 4')
        
    # Get the R-peak and end-T-wave positions
    r_peak = annotations[0]
    end_t_wave = annotations[1]

    # Create the PCG states array
    s = np.zeros(envelopes[0].size)

    # Timing durations from Schmidt
    fs = 50
    mean_S1 = 0.122*fs
    std_S1 = 0.022*fs
    mean_S2 = 0.092*fs
    std_S2 = 0.022*fs

    mean_sys = 0.208*fs
    mean_dias = 0.523*fs

    # Setting the duration from each R-peak to R-peak + mean_S1 as the first state
    # The R-peak in the ECG coincides with the start of the S1 sound [4]
    # Therefore, the duration from each R-peak to the mean_S1 sound duration
    # later were labelled as the "true" positions of the S1 sounds:
    for i in range(r_peak.size):

        # If the annotation is out of the PCG range, skip it
        if r_peak[i] > envelopes.shape[1]:
            continue
        
        s[int(r_peak[i]):int(r_peak[i] + mean_S1)] = 1

        # Set the spaces between this S1 and the next end-T-wave as the second
        # state to preliminary label them
        # To find the next end-T-wave, compute the differences between this
        # R-peak and the end-T-wave positions
        diffs = end_t_wave - r_peak[i]

        # Set the negative values to infinity (a big number near 2^16) to
        # exclude them
        diffs[diffs < 0] = 60000

        # If the array does not have any non infinity value, the R-peak is the
        # last ECG label, so fill the states for the mean systolic duration
        if diffs[diffs < 60000].size == 0:
            s[int(r_peak[i] + mean_S1):int(r_peak[i] + mean_S1 + mean_sys)] = 2
        else:
            # Find the index of the minimum difference
            next_end_T_wave = np.argmin(diffs)
            s[int(r_peak[i] + mean_S1):end_t_wave[next_end_T_wave]+int(2*mean_S2)] = 2

    # Set S2 as state 3 depending on position of end T-wave peak in ECG:
    # The second heart sound occurs at approximately the same time as the
    # end-T-wave [4]. Therefore, for each end-T-wave, find the peak in the
    # envelope around the end-T-wave, setting a window centered on this peak as
    # the second heart sound state:
    for i in range(end_t_wave.size):

        # If the annotation is out of the PCG range, skip it
        if end_t_wave[i] > envelopes.shape[1]:
            continue
        
        # find search window of envelope:
        # T-end +- mean+1sd
        # Set upper and lower bounds, to avoid errors of searching outside size
        # of the signal
        lower_bound = max(0, np.floor(end_t_wave[i] - mean_S2 - std_S2))
        upper_bound = min(envelopes[0].size, np.ceil(end_t_wave[i] + mean_S2 + std_S2))

        # Find the maximum value index if the first envelope in the search window
        max_index = np.argmax(envelopes[0][int(lower_bound):int(upper_bound)])

        # Find the actual index in the envelpes array
        # Make sure this has a max value of the length of the signal
        max_index = min(envelopes[0].size, lower_bound + max_index)
        
        # Set the states to state 3, centered on the S2 peak, +- 1/2 of the
        # expected S2 sound duration. Again, making sure it does not try to set
        # a value outside of the length of the signal:
        upper_bound = min(envelopes[0].size, np.ceil(max_index + mean_S2/2))
        lower_bound = max(0, np.ceil(max_index - mean_S2/2))
        s[int(lower_bound):int(upper_bound)] = 3
        
        # Set the spaces between state 3 and the next R peak as state 4
        # We need to find the next R peal after this S2 sound, so substract the
        # position of this ent-T-wave from the R-peaks positions
        diffs = r_peak - end_t_wave[i]
        
        # Exclude those that are ngative (meaning that the R-peak is before this
        # end-T-wave occured), setting them to infinity (a big number near 2^16)
        diffs[diffs < 0] = 60000

        # If the array is empty, then there are no more R-peaks after this, so
        # set the state 4 for a mean diastolyc duration
        if diffs[diffs < 60000].size == 0:
            s[int(upper_bound):int(upper_bound + mean_dias)] = 4
        else:
            # Find the first R-peak after this end-T-wave
            next_r_peak = diffs.argmin()
            
            # Set the state 4, centered on the next R-peak
            s[int(upper_bound):r_peak[next_r_peak]] = 4
            
    return s

def prepare_dataset_2016(dataset_path: str = DATASET2016_PATH, 
                         preprocesed_path: str = './2016-segmentation',
                         save_dict_name : str = '/data_dict.json',
                         verbose: bool = True) -> Dict:
    """Reads the raw data from the 2016 Physionet dataset generated by
    `extract_2016_data_from_mat` and creates .npz for each sample containing the
    envelograms and the heart states labels sampled at 50Hz.
    
    Args:
    -----
        dataset_path (str): Path to the containing containing the raw data
        extracted by `extrac_2016_data_from_mat`.
        preprocessed_path (str): Path to the directory where the .npz files will be saved
        save_dict_name (str): Name of the .json file where the data dictionary will be saved
        verbose (bool): If True, prints the progress of the function
    
    Returns:
    --------
        data_dict (dict): Dictionary containing the data and the labels
    """
    patient_data_filename = '/patient_number.npy'

    # If running under Windows, replace the backslashes with forward slashes
    if os.name == 'nt':
        dataset_path = dataset_path.replace('\\', '/')
        preprocesed_path = preprocesed_path.replace('\\', '/')
        save_dict_name = save_dict_name.replace('\\', '/')
        patient_data_filename = patient_data_filename.replace('\\', '/')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        
        # If it does not exist, check if the physionet.org/example_data.mat exists
        mat_file = '/'.join(dataset_path.split('/')[:2]) + '/files/hss/1.0/example_data.mat'
        if not os.path.exists(mat_file):
            raise ValueError('The dataset path {} does not exist. And there is not {} availabe to extract the information from. Please download it from https://physionet.org/content/hss/1.0/example_data.mat'.format(dataset_path, mat_file))
        else:
            # If the .mat file exists, extract the data from it
            extract_2016_data_from_mat(mat_file, dataset_path)

    # Check if the preprocessed path exists
    if not os.path.exists(preprocesed_path):
        # Create the preprocessed path
        os.makedirs(preprocesed_path)

    # Check if the save_dict_name is a valid .json filename
    if not save_dict_name.endswith('.json'):
        raise ValueError('The save_dict_name {} is not a valid .json filename.'.format(save_dict_name))
    
    patient_data = np.load(DATASET2016_PATH + patient_data_filename, allow_pickle=True)

    data_dict = {}

    # Create the data_dictionary base, knowing there are 135 patients
    for i in range(1, 136):
        data_dict[str(i)] = {"Count" : 0}
    
    # Iterate over the recordings (792)
    for i in tqdm(range(len(patient_data))) if verbose else range(len(patient_data)):

        wave_data, fs = read_single_2016_audio(i)
        annotations = read_single_2016_annotations(i)
        patient = str(patient_data[i][0][0]) #It is awfully saved

        # The recording identifier is the data_dict[patient]['Count'], if there
        # are no recordings for the patient, is 0; if there is 1, is 1, etc.
        recording_id = str(data_dict[patient]['Count'])

        # Preprocess the data
        x = renna_preprocess_wave(wave_data, fs)

        # Label the heart states
        s = label_2016_pcg_positions(x, annotations)

        # Since there are not unlabeled data in the middle of the recordings,
        # understand each sample as a single split.
        # Remove the unlabeled (s=0) data from the beginning and the end of the
        # recordings
        labeled_idx = np.where(s != 0)[0]

        # Use the labeled data
        x = x[:, labeled_idx]
        s = s[labeled_idx]

        # Save the data
        file_name = '/{}_{}.npz'.format(patient, recording_id)

        # If working under Windows, replace the backslashes with forward slashes
        if os.name == 'nt':
            file_name = file_name.replace('\\', '/')

        np.savez(preprocesed_path + file_name, x=x, s=s)

        # Update the data_dictionary
        data_dict[patient][recording_id] = {}
        data_dict[patient][recording_id]["File name"] = file_name
        data_dict[patient][recording_id]["Number of frames"] = x.shape[1]
        data_dict[patient]['Count'] += 1
    
    # Save the data_dictionary
    with open(preprocesed_path + save_dict_name, 'w') as f:
        json.dump(data_dict, f)
    
    return data_dict

def find_valid_data_2016(data_dict: Dict, N: int, tau: int,
                         preprocesed_path: str = './2016-segmentation',
                         save_dict_name: str = None) -> Dict:
    """Takes the dictionary generated by `prepare_dataset_2016` and creates
    another one with only the data that has at least N frames.

    Args:
    -----
        data_dict (dict): Dictionary generated by `prepare_dataset_2016`.
        N (int): Minimum number of frames.
        tau (int): Stride for rolling window.
        preprocesed_path (str): Path to the preprocessed data.
        save_dict_name (str): Name of the json file to save the data dict.

    Returns:
    --------
        data_dict_valid (dict): Dictionary with only the data that has at least
        N frames. It contains also the number of valid windows of size N in each
        patient and in each sample.
    """
    # Check if N is a positive integer
    if not isinstance(N, int) or N <= 0:
        raise ValueError('N must be a positive integer.')
    
    # Check if tau is a positive integer
    if not isinstance(tau, int) or tau <= 0:
        raise ValueError('tau must be a positive integer.')

    # If save_dict_name is not None, check if it starts with a '/' and ends with a '.json'
    if save_dict_name is not None:
        if not save_dict_name.startswith('/'):
            save_dict_name = '/' + save_dict_name
        if not save_dict_name.endswith('.json'):
            raise(ValueError('The save_dict_name {} is not a valid .json filename.'.format(save_dict_name)))
    
    # If running under Windows, replace '/' with '\\'
    if os.name == 'nt':
        preprocesed_path = preprocesed_path.replace('/', '\\')
        if save_dict_name is not None:
            save_dict_name = save_dict_name.replace('/', '\\')

    # Check if the preprocesed_path exists
    if not os.path.exists(preprocesed_path):
        raise FileNotFoundError('The preprocesed_path does not exist.')

    data_dict_valid = copy.deepcopy(data_dict)

    # Iterate over the patients
    for patient in data_dict:
        # Iterate over the recordings
        for recording in [recording_num for recording_num in data_dict[patient] if recording_num != 'Count']:
            # If the number of frames is less than N, remove the recording
            if data_dict[patient][recording]['Number of frames'] < N:
                data_dict_valid[patient].pop(recording)
                # Update the count of recordings
                data_dict_valid[patient]['Count'] -= 1
        # If there are no recordings for the patient, remove the patient
        if data_dict_valid[patient]['Count'] == 0:
            data_dict_valid.pop(patient)
    
    # Iterate over the patients
    for patient in data_dict_valid:
        n_samples_patient = 0
        # Iterate over the recordings
        for recording in [recording_num for recording_num in data_dict_valid[patient] if recording_num != 'Count']:
            # Add the number of valid windows
            x = np.load(preprocesed_path + data_dict_valid[patient][recording]["File name"])['x']
            n_samples = rolling_strided_window(x, N, tau).shape[0]
            data_dict_valid[patient][recording]['Number of valid windows'] = n_samples
            n_samples_patient += n_samples
            

            # Update the data_dict_valid
        data_dict_valid[patient]["Number of valid windows"] = n_samples_patient

    # If save_dict_name is not None, save the data_dict_valid
    if save_dict_name is not None:
        with open(preprocesed_path + save_dict_name, 'w') as f:
            json.dump(data_dict_valid, f)

    return data_dict_valid

def generate_X_S_from_dict_2016(data_dict: Dict, N: int, tau: int,
                      processed_path: str = './2016-segmentation') -> Tuple[np.ndarray, np.ndarray]:
    """For each .npz included in `data_dict` inside the folder `processed_path`,
    it extracts windows of size `N` with stride `tau` from `x` and `s` and
    stack them in `X` and `S`, shuffling the samples.

    Args:
    -----
        data_dict (dict): Dictionary that has been filtered by
        `find_valid_2016`.
        N (int): Window size.
        tau (int): Stride.
        processed_path (str): Path to the preprocessed data. Defaults to
        './circor-segmentation'.

    Returns:
    --------
        X (np.ndarray): Numpy array with shape (n_samples, x.shape[0], N).
        S (np.ndarray): Numpy array with shape (n_samples, N).
    """
    # Check if data_dict is a dictionary
    if not isinstance(data_dict, dict):
        raise TypeError('data_dict must be a dictionary.')
    
    # Check if N is a positive integer
    if not isinstance(N, int) or N <= 0:
        raise ValueError('N must be a positive integer.')
    
    # Check if tau is a positive integer
    if not isinstance(tau, int) or tau <= 0:
        raise ValueError('tau must be a positive integer.')
    
    # If running under Windows, replace '/' with '\\'
    if os.name == 'nt':
        processed_path = processed_path.replace('/', '\\')

    # Check if the processed_path exists
    if not os.path.exists(processed_path):
        raise FileNotFoundError('The processed_path does not exist.')

    # Extract the first patient
    patient_id = list(data_dict.keys())[0]
    # Extract its recording
    recording = [recording_num for recording_num in data_dict[patient_id] if recording_num not in ['Count', 'Number of valid windows']][0]

    # Read one .npz file to get the number of envelopes included in x
    with np.load(processed_path + data_dict[patient_id][recording]["File name"]) as f:
        x = f['x']

    # Initialize the X and S arrays
    X = np.zeros((0, x.shape[0], N))
    S = np.zeros((0, N))

    # Iterate over the patients
    for patient_id in tqdm(data_dict):
        # Iterate over the recordings
        for recording in [recording_num for recording_num in data_dict[patient_id] if recording_num not in ['Count', 'Number of valid windows']]:
            # Load the .npz file
            with np.load(processed_path + data_dict[patient_id][recording]["File name"]) as f:
                x = f['x']
                s = f['s']

            x = rolling_strided_window(x, N, tau)
            s = rolling_strided_window(s, N, tau)

            x, s = check_valid_sequence(x, s, 2)

            # Stack the windows
            X = np.vstack((X, x))
            S = np.vstack((S, s))
    
    # Create a new axis in S and concatenate X and S
    S = S[:, np.newaxis, :]
    XS = np.concatenate((X, S), axis=1)

    # Shuffle the samples
    np.random.shuffle(XS)

    # Return the X and S arrays
    X = XS[:, :x.shape[1], :]
    S = XS[:, x.shape[1]:, :]

    # Swap axes to format the data as channels_last
    X = np.swapaxes(X, 1, 2)
    S = np.swapaxes(S, 1, 2)

    # Transform S to categorical
    S = to_categorical(S-1)

    return X, S