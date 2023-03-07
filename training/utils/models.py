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

from tensorflow.keras import Model, layers
from typing import Tuple
import tensorflow as tf

def encoding_block(x: tf.Tensor, filters: int, kernel_size: int, stride: int,
                   activation: str, padding: str, name_prefix: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Encoding block of the network, based on the description of [1].

    Args:
    -----
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters in the convolutional layers.
        kernel_size (int): Kernel size of the convolutional layers.
        stride (int): Stride of the convolutional layers.
        activation (str): Activation function of the convolutional layers.
        padding (str): Padding of the convolutional layers.
        name_prefix (str): Name prefix of the layers.
    
    Returns:
    --------
        x (tf.Tensor): Output tensor.
        enc_residual (tf.Tensor): Residual tensor to use in the decoder phase.
    
    References:
    -----------
        [1] F. Renna, J. Oliveira and M. T. Coimbra, "Deep Convolutional Neural
        Networks for Heart Sound Segmentation," in IEEE Journal of Biomedical
        and Health Informatics, vol. 23, no. 6, pp. 2435-2445, Nov. 2019, doi:
        10.1109/JBHI.2019.2894222.
    """
    # 2 x ( Conv + ReLU )
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                      activation=activation, padding=padding, use_bias=False,
                      name=name_prefix+'_conv_relu_0')(x)

    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                      activation=activation, padding=padding, use_bias=False,
                      name=name_prefix+'_conv_relu_1')(x)
    
    # Store tensor to be the input od the decoding phase
    enc_residual = x 
    
    # Max pooling to downsample in time
    x = layers.MaxPooling1D(pool_size=2, strides=2, padding="same",
                            name=name_prefix+'_maxpool')(x) 

    return x, enc_residual

def decoding_block(x: tf.Tensor, residual: tf.Tensor, filters: int,
                   kernel_size: int, stride: int, activation: str, padding: str,
                   name_prefix: str) -> tf.Tensor:
    """Decoding block of the network, based on the description of [1].

    Args:
    -----
        x (tf.Tensor): Input tensor.
        residual (tf.Tensor): Residual tensor from the encoder phase.
        filters (int): Number of filters in the convolutional layers.
        kernel_size (int): Kernel size of the convolutional layers.
        stride (int): Stride of the convolutional layers.
        activation (str): Activation function of the convolutional layers.
        padding (str): Padding of the convolutional layers.
        name_prefix (str): Name prefix of the layers.
    
    Returns:
    --------
        x (tf.Tensor): Output tensor.

    References:
    -----------
        [1] F. Renna, J. Oliveira and M. T. Coimbra, "Deep Convolutional Neural
        Networks for Heart Sound Segmentation," in IEEE Journal of Biomedical
        and Health Informatics, vol. 23, no. 6, pp. 2435-2445, Nov. 2019, doi:
        10.1109/JBHI.2019.2894222.
    """

    # Upsampling + 1 x ( Conv + ReLU )
    x = layers.UpSampling1D(size=2, name=name_prefix+'_upsample')(x)
    x = layers.Conv1D(filters=filters, kernel_size=3, strides=1,
                      activation=activation, padding=padding, use_bias=False,
                      name=name_prefix+'_up_conv_relu')(x)


    # Add residual + 2 x ( Conv + ReLU )
    x = layers.Concatenate(name=name_prefix+'_concatenate')([residual, x]) 
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                      activation=activation, padding=padding, use_bias=False,
                      name=name_prefix+'_conv_relu_0')(x)

    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                      activation=activation, padding=padding, use_bias=False,
                      name=name_prefix+'_conv_relu_1')(x)
      
    return x

def get_model_parametrized(N: int, base_filter: int, n_blocks) -> Model:
    """Returns the model described in [1] with the desired model parameters.

    Args:
    -----
        N (int): Number of samples in the input tensor.
        n_features (int): Number of features in the input tensor.
        num_classes (int): Number of classes in the output tensor.
    
    Returns:
    --------
        model (Model): CNN-model instance.
    
    References:
    -----------
        [1] F. Renna, J. Oliveira and M. T. Coimbra, "Deep Convolutional Neural
        Networks for Heart Sound Segmentation," in IEEE Journal of Biomedical
        and Health Informatics, vol. 23, no. 6, pp. 2435-2445, Nov. 2019, doi:
        10.1109/JBHI.2019.2894222.
    """
    # Input tensor
    input = layers.Input(shape=(N, 4))

    x = input

    res = []

    # Encoding phase of the network: downsampling inputs
    for i in range(n_blocks):
        x, res_tmp = encoding_block(x, filters=base_filter*(2**i), kernel_size=3, stride = 1, activation = "relu", padding = "same", name_prefix = "enc_"+str(i))
        res.append(res_tmp)

    # Intermediate layer 
    # 2 x ( Conv + ReLU )
    x = layers.Conv1D(filters=base_filter*(2**n_blocks), kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False, name='central_conv_relu_0')(x)
    x = layers.Conv1D(filters=base_filter*(2**n_blocks), kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False, name='central_conv_relu_1')(x)
    

    # Decoding phase of the network: upsampling inputs
    for i in range(n_blocks):
        x = decoding_block(x, res[n_blocks-i-1], filters=base_filter*(2**(n_blocks-i-1)), kernel_size=3, stride = 1, activation = "relu", padding = "same", name_prefix = "dec_"+str(i))

    # Output of the model 
    # 1 x ( Conv + Softmax )
    x = layers.Conv1D(filters=4, kernel_size=3, strides=1, padding="same", use_bias=False, name='final_conv')(x)
    output = layers.Softmax()(x)
    
    # Define the model
    model = Model(input, output)

    return model