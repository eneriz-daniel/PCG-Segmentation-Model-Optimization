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

from string import Template
from tensorflow.keras import Model
import os
import numpy as np
from models import get_model_parametrized
import pandas as pd
import json
import shutil
import time
from typing import Tuple

bs = 250 # Batch size for the csim, limited by the maximum size of an array in C

################################################################################
#                       TEMPLATE BLOCKS FOR THE HLS CODE                       #
################################################################################

baseline_basic_encoder_params_def_block = """
#define ENC_{i}_CONV_RELU_0_K K //Kernel size of the enc_{i}_conv_relu_0 layer
#define ENC_{i}_CONV_RELU_0_INPUT_FEATURES {input_features} //Number of input features of the enc_{i}_conv_relu_0 layer
#define ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES {output_features} //Number of output features of the enc_{i}_conv_relu_0 layer
#define ENC_{i}_CONV_RELU_0_N {N_dim} //Number of frames in the time dimension of the enc_{i}_conv_relu_0 layer
#define ENC_{i}_CONV_RELU_1_K K //Kernel size of the enc_{i}_conv_relu_1 layer
#define ENC_{i}_CONV_RELU_1_INPUT_FEATURES {output_features} //Number of input features of the enc_{i}_conv_relu_1 layer
#define ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES {output_features} //Number of output features of the enc_{i}_conv_relu_1 layer
#define ENC_{i}_CONV_RELU_1_N {N_dim} //Number of frames in the time dimension of the enc_{i}_conv_relu_1 layer
"""
baseline_basic_central_params_def_block = """
#define CENTRAL_CONV_RELU_0_K K //Kernel size of the central_conv_relu_0 layer
#define CENTRAL_CONV_RELU_0_INPUT_FEATURES {input_features} //Number of input features of the central_conv_relu_0 layer
#define CENTRAL_CONV_RELU_0_OUTPUT_FEATURES {output_features} //Number of output features of the central_conv_relu_0 layer
#define CENTRAL_CONV_RELU_0_N {N_dim} //Number of frames in the time dimension of the central_conv_relu_0 layer
#define CENTRAL_CONV_RELU_1_K K //Kernel size of the central_conv_relu_1 layer
#define CENTRAL_CONV_RELU_1_INPUT_FEATURES {output_features} //Number of input features of the central_conv_relu_1 layer
#define CENTRAL_CONV_RELU_1_OUTPUT_FEATURES {output_features} //Number of output features of the central_conv_relu_1 layer
#define CENTRAL_CONV_RELU_1_N {N_dim} //Number of frames in the time dimension of the central_conv_relu_1 layer
"""

baseline_basic_decoder_params_def_block = """
#define DEC_{i}_UP_CONV_RELU_K K //Kernel size of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_UP_CONV_RELU_INPUT_FEATURES {input_features} //Number of input features of the dec_{i}_up_conv_relu layer
#define DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES {output_features} //Number of output features of the dec_{i}_up_conv_relu layer
#define DEC_{i}_UP_CONV_RELU_N {N_dim} //Number of frames in the time dimension of the dec_{i}_up_conv_relu layer
#define DEC_{i}_CONV_RELU_0_K K //Kernel size of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_CONV_RELU_0_INPUT_FEATURES {input_features} //Number of input features of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES {output_features} //Number of output features of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_CONV_RELU_0_N {N_dim} //Number of frames in the time dimension of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_CONV_RELU_1_K K //Kernel size of the dec_{i}_conv_relu_1 layer
#define DEC_{i}_CONV_RELU_1_INPUT_FEATURES {output_features} //Number of input features of the dec_{i}_conv_relu_1 layer
#define DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES {output_features} //Number of output features of the dec_{i}_conv_relu_1 layer
#define DEC_{i}_CONV_RELU_1_N {N_dim} //Number of frames in the time dimension of the dec_{i}_conv_relu_1 layer
"""

baseline_basic_enc_paramters_input_func=\
"""               apfixed enc_{i}_conv_relu_0_w[ENC_{i}_CONV_RELU_0_K][ENC_{i}_CONV_RELU_0_INPUT_FEATURES][ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES],
               apfixed enc_{i}_conv_relu_1_w[ENC_{i}_CONV_RELU_1_K][ENC_{i}_CONV_RELU_1_INPUT_FEATURES][ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES],
"""

baseline_basic_dec_paramters_input_func=\
"""               apfixed dec_{i}_up_conv_relu_w[DEC_{i}_UP_CONV_RELU_K][DEC_{i}_UP_CONV_RELU_INPUT_FEATURES][DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES],
               apfixed dec_{i}_conv_relu_0_w[DEC_{i}_CONV_RELU_0_K][DEC_{i}_CONV_RELU_0_INPUT_FEATURES][DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES],
               apfixed dec_{i}_conv_relu_1_w[DEC_{i}_CONV_RELU_1_K][DEC_{i}_CONV_RELU_1_INPUT_FEATURES][DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES],
"""

baseline_basic_enc_pragmas = \
"""  #pragma HLS INTERFACE s_axilite port=enc_{i}_conv_relu_0_w
  #pragma HLS INTERFACE s_axilite port=enc_{i}_conv_relu_1_w
"""

baseline_basic_dec_pragmas = \
"""  #pragma HLS INTERFACE s_axilite port=dec_{i}_up_conv_relu_w
  #pragma HLS INTERFACE s_axilite port=dec_{i}_conv_relu_0_w
  #pragma HLS INTERFACE s_axilite port=dec_{i}_conv_relu_1_w
"""

baseline_enc_basic_featuremap_init = \
"""
  apfixed enc_{i}_conv_relu_0[ENC_{i}_CONV_RELU_0_N][ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES];
  apfixed enc_{i}_conv_relu_1[ENC_{i}_CONV_RELU_1_N][ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
  apfixed enc_{i}_maxpool[ENC_{i}_CONV_RELU_1_N/2][ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
"""

baseline_dec_basic_featuremap_init = \
"""
  apfixed dec_{i}_upsample[DEC_{i}_UP_CONV_RELU_N][DEC_{i}_UP_CONV_RELU_INPUT_FEATURES];
  apfixed dec_{i}_up_conv_relu[DEC_{i}_UP_CONV_RELU_N][DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES];
  apfixed dec_{i}_concatenate[DEC_{i}_UP_CONV_RELU_N][DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES*2];
  apfixed dec_{i}_conv_relu_0[DEC_{i}_CONV_RELU_0_N][DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES];
  apfixed dec_{i}_conv_relu_1[DEC_{i}_CONV_RELU_1_N][DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
"""

baseline_basic_encoder_block = Template(
"""
  //-----------------------------ENCODER ${i}--------------------------------------
  //-------------------------enc_${i}_conv_relu_0----------------------------------
  // Iterate over the number of filters
  enc_${i}_conv_relu_0_k: for(apint k=0; k<ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    enc_${i}_conv_relu_0_i: for(apint i=0; i<ENC_${i}_CONV_RELU_0_N; i++){
      // Calculate the auxiliary positions respect to the input
      l_min = max(0, i - ENC_${i}_CONV_RELU_0_K/2);
      l_max = min(ENC_${i}_CONV_RELU_0_N, i + ENC_${i}_CONV_RELU_0_K/2 + 1);
      acc = 0; // Reset the accumulator
      enc_${i}_conv_relu_0_l: for(apint l=l_min; l<l_max; l++){
        #pragma HLS loop_tripcount min=2 max=3 avg=3
        enc_${i}_conv_relu_0_j: for(apint j=0; j<ENC_${i}_CONV_RELU_0_INPUT_FEATURES; j++){
          // Multiply the input and the weight
          acc += ${input_feature_map}[l][j]*enc_${i}_conv_relu_0_w[l-i+ENC_${i}_CONV_RELU_0_K/2][j][k];
        }
      }
    enc_${i}_conv_relu_0[i][k] = ReLU(acc); // Save the accumulator value
    }  
  }
  //----------------------------------------------------------------------------
  //-------------------------enc_${i}_conv_relu_1----------------------------------
  // Iterate over the number of filters
  enc_${i}_conv_relu_1_k: for(apint k=0; k<ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    enc_${i}_conv_relu_1_i: for(apint i=0; i<ENC_${i}_CONV_RELU_1_N; i++){
      // Calculate the auxiliary positions respect to the input
      l_min = max(0, i - ENC_${i}_CONV_RELU_1_K/2);
      l_max = min(ENC_${i}_CONV_RELU_1_N, i + ENC_${i}_CONV_RELU_1_K/2 + 1);
      acc = 0; // Reset the accumulator
      enc_${i}_conv_relu_1_l: for(apint l=l_min; l<l_max; l++){
        #pragma HLS loop_tripcount min=2 max=3 avg=3
        enc_${i}_conv_relu_1_j: for(apint j=0; j<ENC_${i}_CONV_RELU_1_INPUT_FEATURES; j++){
          // Multiply the input and the weight
          acc += enc_${i}_conv_relu_0[l][j]*enc_${i}_conv_relu_1_w[l-i+ENC_${i}_CONV_RELU_1_K/2][j][k];
        }
      }
      enc_${i}_conv_relu_1[i][k] = ReLU(acc); // Save the accumulator value
    }  
  }
  //----------------------------------------------------------------------------
  //-----------------------------enc_${i}_maxpool----------------------------------
  // Iterate over the number of filters
  enc_${i}_maxpool_k: for(apint k=0; k<ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    enc_${i}_maxpool_i: for(apint i=0; i<ENC_${i}_CONV_RELU_1_N/2; i++){
      enc_${i}_maxpool[i][k] = max(enc_${i}_conv_relu_1[2*i][k], enc_${i}_conv_relu_1[2*i+1][k]);
    }
  }
  //----------------------------------------------------------------------------
""")

baseline_basic_decoder_block = Template(
"""
  //-----------------------------DECODER ${i}--------------------------------------
  //-----------------------------dec_${i}_upsample---------------------------------
  // Iterate over the number of filters
  dec_${i}_upsample_k: for(apint k=0; k<DEC_${i}_UP_CONV_RELU_INPUT_FEATURES; k++){
    // Iterate over the input matrix
    dec_${i}_upsample_i: for(apint i=0; i<DEC_${i}_UP_CONV_RELU_N/2; i++){
      dec_${i}_upsample[2*i][k] = ${input_feature_map}[i][k];
      dec_${i}_upsample[2*i+1][k] = ${input_feature_map}[i][k];
    }
  }
  //----------------------------------------------------------------------------
  //-------------------------dec_${i}_up_conv_relu----------------------------------
  // Iterate over the number of filters
  dec_${i}_up_conv_relu_k: for(apint k=0; k<DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    dec_${i}_up_conv_relu_i: for(apint i=0; i<DEC_${i}_UP_CONV_RELU_N; i++){
      // Calculate the auxiliary positions respect to the input
      l_min = max(0, i - DEC_${i}_UP_CONV_RELU_K/2);
      l_max = min(DEC_${i}_UP_CONV_RELU_N, i + DEC_${i}_UP_CONV_RELU_K/2 + 1);
      acc = 0; // Reset the accumulator
      dec_${i}_up_conv_relu_l: for(apint l=l_min; l<l_max; l++){
        #pragma HLS loop_tripcount min=2 max=3 avg=3
        dec_${i}_up_conv_relu_j: for(apint j=0; j<DEC_${i}_UP_CONV_RELU_INPUT_FEATURES; j++){
          // Multiply the input and the weight
          acc += dec_${i}_upsample[l][j]*dec_${i}_up_conv_relu_w[l-i+DEC_${i}_UP_CONV_RELU_K/2][j][k];
        }
      }
      dec_${i}_up_conv_relu[i][k] = ReLU(acc); // Save the accumulator value
    }  
  }
  //----------------------------------------------------------------------------
  //--------------------------dec_${i}_concatenate---------------------------------
  // Iterate over the number of filters
  dec_${i}_concatenate_k: for(apint k=0; k<DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    dec_${i}_concatenate_i: for(apint i=0; i<DEC_${i}_UP_CONV_RELU_N; i++){
      dec_${i}_concatenate[i][k] = ${res_feature_map}[i][k];
      dec_${i}_concatenate[i][k+DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES] = dec_${i}_up_conv_relu[i][k];
    }
  }
  //----------------------------------------------------------------------------
  //-------------------------dec_${i}_conv_relu_0----------------------------------
  // Iterate over the number of filters
  dec_${i}_conv_relu_0_k: for(apint k=0; k<DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    dec_${i}_conv_relu_0_i: for(apint i=0; i<DEC_${i}_CONV_RELU_0_N; i++){
      // Calculate the auxiliary positions respect to the input
      l_min = max(0, i - DEC_${i}_CONV_RELU_0_K/2);
      l_max = min(DEC_${i}_CONV_RELU_0_N, i + DEC_${i}_CONV_RELU_0_K/2 + 1);
      acc = 0; // Reset the accumulator
      dec_${i}_conv_relu_0_l: for(apint l=l_min; l<l_max; l++){
        #pragma HLS loop_tripcount min=2 max=3 avg=3
        dec_${i}_conv_relu_0_j: for(apint j=0; j<DEC_${i}_CONV_RELU_0_INPUT_FEATURES; j++){
          // Multiply the input and the weight
          acc += dec_${i}_concatenate[l][j]*dec_${i}_conv_relu_0_w[l-i+DEC_${i}_CONV_RELU_0_K/2][j][k];
        }
      }
      dec_${i}_conv_relu_0[i][k] = ReLU(acc); // Save the accumulator value
    }  
  }
  //----------------------------------------------------------------------------
  //-------------------------dec_${i}_conv_relu_1----------------------------------
  // Iterate over the number of filters
  dec_${i}_conv_relu_1_k: for(apint k=0; k<DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    dec_${i}_conv_relu_1_i: for(apint i=0; i<DEC_${i}_CONV_RELU_1_N; i++){
      // Calculate the auxiliary positions respect to the input
      l_min = max(0, i - DEC_${i}_CONV_RELU_1_K/2);
      l_max = min(DEC_${i}_CONV_RELU_1_N, i + DEC_${i}_CONV_RELU_1_K/2 + 1);
      acc = 0; // Reset the accumulator
      dec_${i}_conv_relu_1_l: for(apint l=l_min; l<l_max; l++){
        #pragma HLS loop_tripcount min=2 max=3 avg=3
        dec_${i}_conv_relu_1_j: for(apint j=0; j<DEC_${i}_CONV_RELU_1_INPUT_FEATURES; j++){
          // Multiply the input and the weight
          acc += dec_${i}_conv_relu_0[l][j]*dec_${i}_conv_relu_1_w[l-i+DEC_${i}_CONV_RELU_1_K/2][j][k];
        }
      }
      dec_${i}_conv_relu_1[i][k] = ReLU(acc); // Save the accumulator value
    }  
  }
  //----------------------------------------------------------------------------
""")

baseline_basic_enc_parameters_array_init = \
"""
    float enc_{i}_conv_relu_0_tmp[ENC_{i}_CONV_RELU_0_K*ENC_{i}_CONV_RELU_0_INPUT_FEATURES*ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES];
    int enc_{i}_conv_relu_0_shape[4];
    apfixed enc_{i}_conv_relu_0_w[ENC_{i}_CONV_RELU_0_K][ENC_{i}_CONV_RELU_0_INPUT_FEATURES][ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES];
    float enc_{i}_conv_relu_1_tmp[ENC_{i}_CONV_RELU_1_K*ENC_{i}_CONV_RELU_1_INPUT_FEATURES*ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
    int enc_{i}_conv_relu_1_shape[4];
    apfixed enc_{i}_conv_relu_1_w[ENC_{i}_CONV_RELU_1_K][ENC_{i}_CONV_RELU_1_INPUT_FEATURES][ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
"""

baseline_basic_dec_parameters_array_init = \
"""
    float dec_{i}_up_conv_relu_tmp[DEC_{i}_UP_CONV_RELU_K*DEC_{i}_UP_CONV_RELU_INPUT_FEATURES*DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES];
    int dec_{i}_up_conv_relu_shape[4];
    apfixed dec_{i}_up_conv_relu_w[DEC_{i}_UP_CONV_RELU_K][DEC_{i}_UP_CONV_RELU_INPUT_FEATURES][DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES];
    float dec_{i}_conv_relu_0_tmp[DEC_{i}_CONV_RELU_0_K*DEC_{i}_CONV_RELU_0_INPUT_FEATURES*DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES];
    int dec_{i}_conv_relu_0_shape[4];
    apfixed dec_{i}_conv_relu_0_w[DEC_{i}_CONV_RELU_0_K][DEC_{i}_CONV_RELU_0_INPUT_FEATURES][DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES];
    float dec_{i}_conv_relu_1_tmp[DEC_{i}_CONV_RELU_1_K*DEC_{i}_CONV_RELU_1_INPUT_FEATURES*DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
    int dec_{i}_conv_relu_1_shape[4];
    apfixed dec_{i}_conv_relu_1_w[DEC_{i}_CONV_RELU_1_K][DEC_{i}_CONV_RELU_1_INPUT_FEATURES][DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
"""

baseline_basic_enc_parameters_reading_block = Template(
"""
    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/enc_${i}_conv_relu_0.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, enc_${i}_conv_relu_0_tmp, enc_${i}_conv_relu_0_shape);

    // Reshape it
    for(int k=0; k<ENC_${i}_CONV_RELU_0_K; k++){
        for(int j=0; j<ENC_${i}_CONV_RELU_0_INPUT_FEATURES; j++){
            for(int i=0; i<ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES; i++){
                enc_${i}_conv_relu_0_w[k][j][i] = enc_${i}_conv_relu_0_tmp[i+ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES*j+ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES*ENC_${i}_CONV_RELU_0_INPUT_FEATURES*k];
            }
        }
    }

    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/enc_${i}_conv_relu_1.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, enc_${i}_conv_relu_1_tmp, enc_${i}_conv_relu_1_shape);

    // Reshape it
    for(int k=0; k<ENC_${i}_CONV_RELU_1_K; k++){
        for(int j=0; j<ENC_${i}_CONV_RELU_1_INPUT_FEATURES; j++){
            for(int i=0; i<ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES; i++){
                enc_${i}_conv_relu_1_w[k][j][i] = enc_${i}_conv_relu_1_tmp[i+ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES*j+ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES*ENC_${i}_CONV_RELU_1_INPUT_FEATURES*k];
            }
        }
    }
""")

baseline_basic_dec_parameters_reading_block = Template(
"""
    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/dec_${i}_up_conv_relu.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, dec_${i}_up_conv_relu_tmp, dec_${i}_up_conv_relu_shape);

    // Reshape it
    for(int k=0; k<DEC_${i}_UP_CONV_RELU_K; k++){
      for(int j=0; j<DEC_${i}_UP_CONV_RELU_INPUT_FEATURES; j++){
        for(int i=0; i<DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES; i++){
          dec_${i}_up_conv_relu_w[k][j][i] = dec_${i}_up_conv_relu_tmp[i+DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES*j+DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES*DEC_${i}_UP_CONV_RELU_INPUT_FEATURES*k];
        }
      }
    }

    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/dec_${i}_conv_relu_0.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, dec_${i}_conv_relu_0_tmp, dec_${i}_conv_relu_0_shape);

    // Reshape it
    for(int k=0; k<DEC_${i}_CONV_RELU_0_K; k++){
      for(int j=0; j<DEC_${i}_CONV_RELU_0_INPUT_FEATURES; j++){
        for(int i=0; i<DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES; i++){
          dec_${i}_conv_relu_0_w[k][j][i] = dec_${i}_conv_relu_0_tmp[i+DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES*j+DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES*DEC_${i}_CONV_RELU_0_INPUT_FEATURES*k];
        }
      }
    }

    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/dec_${i}_conv_relu_1.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, dec_${i}_conv_relu_1_tmp, dec_${i}_conv_relu_1_shape);

    // Reshape it
    for(int k=0; k<DEC_${i}_CONV_RELU_1_K; k++){
      for(int j=0; j<DEC_${i}_CONV_RELU_1_INPUT_FEATURES; j++){
        for(int i=0; i<DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES; i++){
          dec_${i}_conv_relu_1_w[k][j][i] = dec_${i}_conv_relu_1_tmp[i+DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES*j+DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES*DEC_${i}_CONV_RELU_1_INPUT_FEATURES*k];
        }
      }
    }
""")

baseline_basic_enc_parameters_to_func_block = \
"""          enc_{i}_conv_relu_0_w,
          enc_{i}_conv_relu_1_w,
"""

baseline_basic_dec_parameters_to_func_block = \
"""          dec_{i}_up_conv_relu_w,
          dec_{i}_conv_relu_0_w,
          dec_{i}_conv_relu_1_w,
"""

memory_sharing_basic_encoder_params_def_block = """
#define ENC_{i}_CONV_RELU_0_K K //Kernel size of the enc_{i}_conv_relu_0 layer
#define ENC_{i}_CONV_RELU_0_INPUT_FEATURES {input_features} //Number of input features of the enc_{i}_conv_relu_0 layer
#define ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES {output_features} //Number of output features of the enc_{i}_conv_relu_0 layer
#define ENC_{i}_CONV_RELU_0_N {N_dim} //Number of frames in the time dimension of the enc_{i}_conv_relu_0 layer

#define ENC_{i}_CONV_RELU_1_K K //Kernel size of the enc_{i}_conv_relu_1 layer
#define ENC_{i}_CONV_RELU_1_INPUT_FEATURES {output_features} //Number of input features of the enc_{i}_conv_relu_1 layer
#define ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES {output_features} //Number of output features of the enc_{i}_conv_relu_1 layer
#define ENC_{i}_CONV_RELU_1_N {N_dim} //Number of frames in the time dimension of the enc_{i}_conv_relu_1 layer
"""
memory_sharing_basic_central_params_def_block = """
#define CENTRAL_CONV_RELU_0_K K //Kernel size of the central_conv_relu_0 layer
#define CENTRAL_CONV_RELU_0_INPUT_FEATURES {input_features} //Number of input features of the central_conv_relu_0 layer
#define CENTRAL_CONV_RELU_0_OUTPUT_FEATURES {output_features} //Number of output features of the central_conv_relu_0 layer
#define CENTRAL_CONV_RELU_0_N {N_dim} //Number of frames in the time dimension of the central_conv_relu_0 layer

#define CENTRAL_CONV_RELU_1_K K //Kernel size of the central_conv_relu_1 layer
#define CENTRAL_CONV_RELU_1_INPUT_FEATURES {output_features} //Number of input features of the central_conv_relu_1 layer
#define CENTRAL_CONV_RELU_1_OUTPUT_FEATURES {output_features} //Number of output features of the central_conv_relu_1 layer
#define CENTRAL_CONV_RELU_1_N {N_dim} //Number of frames in the time dimension of the central_conv_relu_1 layer
"""

memory_sharing_basic_decoder_params_def_block = """
#define DEC_{i}_UP_CONV_RELU_K K //Kernel size of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_UP_CONV_RELU_INPUT_FEATURES {input_features} //Number of input features of the dec_{i}_up_conv_relu layer
#define DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES {output_features} //Number of output features of the dec_{i}_up_conv_relu layer
#define DEC_{i}_UP_CONV_RELU_N {N_dim} //Number of frames in the time dimension of the dec_{i}_up_conv_relu layer

#define DEC_{i}_CONV_RELU_0_K K //Kernel size of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_CONV_RELU_0_INPUT_FEATURES {input_features} //Number of input features of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES {output_features} //Number of output features of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_CONV_RELU_0_N {N_dim} //Number of frames in the time dimension of the dec_{i}_conv_relu_0 layer

#define DEC_{i}_CONV_RELU_1_K K //Kernel size of the dec_{i}_conv_relu_1 layer
#define DEC_{i}_CONV_RELU_1_INPUT_FEATURES {output_features} //Number of input features of the dec_{i}_conv_relu_1 layer
#define DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES {output_features} //Number of output features of the dec_{i}_conv_relu_1 layer
#define DEC_{i}_CONV_RELU_1_N {N_dim} //Number of frames in the time dimension of the dec_{i}_conv_relu_1 layer
"""

memory_sharing_basic_enc_paramters_input_func=\
"""               apfixed enc_{i}_conv_relu_0_w[ENC_{i}_CONV_RELU_0_K][ENC_{i}_CONV_RELU_0_INPUT_FEATURES][ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES],
               apfixed enc_{i}_conv_relu_1_w[ENC_{i}_CONV_RELU_1_K][ENC_{i}_CONV_RELU_1_INPUT_FEATURES][ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES],
"""

memory_sharing_basic_dec_paramters_input_func=\
"""               apfixed dec_{i}_up_conv_relu_w[DEC_{i}_UP_CONV_RELU_K][DEC_{i}_UP_CONV_RELU_INPUT_FEATURES][DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES],
               apfixed dec_{i}_conv_relu_0_w[DEC_{i}_CONV_RELU_0_K][DEC_{i}_CONV_RELU_0_INPUT_FEATURES][DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES],
               apfixed dec_{i}_conv_relu_1_w[DEC_{i}_CONV_RELU_1_K][DEC_{i}_CONV_RELU_1_INPUT_FEATURES][DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES],
"""

memory_sharing_basic_enc_pragmas = \
"""  #pragma HLS INTERFACE s_axilite port=enc_{i}_conv_relu_0_w
  #pragma HLS INTERFACE s_axilite port=enc_{i}_conv_relu_1_w
"""

memory_sharing_basic_dec_pragmas = \
"""  #pragma HLS INTERFACE s_axilite port=dec_{i}_up_conv_relu_w
  #pragma HLS INTERFACE s_axilite port=dec_{i}_conv_relu_0_w
  #pragma HLS INTERFACE s_axilite port=dec_{i}_conv_relu_1_w
"""

memory_sharing_basic_skip_conn_init = \
"""
  apfixed enc_{i}_conv_relu_1[ENC_{i}_CONV_RELU_1_N][ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES];"""

memory_sharing_basic_encoder_block = Template(
"""
  //-----------------------------ENCODER ${i}--------------------------------------

  //-------------------------enc_${i}_conv_relu_0----------------------------------
  // Iterate over the number of filters
  enc_${i}_conv_relu_0_k: for(apint k=0; k<ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    enc_${i}_conv_relu_0_i: for(apint i=0; i<ENC_${i}_CONV_RELU_0_N; i++){
      ${pragmas}
      // Calculate the auxiliary positions respect to the input
      l_min = max(0, i - ENC_${i}_CONV_RELU_0_K/2);
      l_max = min(ENC_${i}_CONV_RELU_0_N, i + ENC_${i}_CONV_RELU_0_K/2 + 1);
      acc = 0; // Reset the accumulator
      enc_${i}_conv_relu_0_l: for(apint l=l_min; l<l_max; l++){
        #pragma HLS loop_tripcount min=2 max=3 avg=3
        enc_${i}_conv_relu_0_j: for(apint j=0; j<ENC_${i}_CONV_RELU_0_INPUT_FEATURES; j++){
          // Multiply the input and the weight
          acc += ${input_feature_map}[l][j]*enc_${i}_conv_relu_0_w[l-i+ENC_${i}_CONV_RELU_0_K/2][j][k];
        }
      }
    feature_maps_0[i][k] = ReLU(acc); // Save the accumulator value
    }  
  }
  //----------------------------------------------------------------------------

  //-------------------------enc_${i}_conv_relu_1----------------------------------
  // Iterate over the number of filters
  enc_${i}_conv_relu_1_k: for(apint k=0; k<ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    enc_${i}_conv_relu_1_i: for(apint i=0; i<ENC_${i}_CONV_RELU_1_N; i++){
      ${pragmas}
      // Calculate the auxiliary positions respect to the input
      l_min = max(0, i - ENC_${i}_CONV_RELU_1_K/2);
      l_max = min(ENC_${i}_CONV_RELU_1_N, i + ENC_${i}_CONV_RELU_1_K/2 + 1);
      acc = 0; // Reset the accumulator
      enc_${i}_conv_relu_1_l: for(apint l=l_min; l<l_max; l++){
        #pragma HLS loop_tripcount min=2 max=3 avg=3
        enc_${i}_conv_relu_1_j: for(apint j=0; j<ENC_${i}_CONV_RELU_1_INPUT_FEATURES; j++){
          // Multiply the input and the weight
          acc += feature_maps_0[l][j]*enc_${i}_conv_relu_1_w[l-i+ENC_${i}_CONV_RELU_1_K/2][j][k];
        }
      }
      enc_${i}_conv_relu_1[i][k] = ReLU(acc); // Save the accumulator value
    }  
  }
  //----------------------------------------------------------------------------

  //-----------------------------enc_${i}_maxpool----------------------------------
  // Iterate over the number of filters
  enc_${i}_maxpool_k: for(apint k=0; k<ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    enc_${i}_maxpool_i: for(apint i=0; i<ENC_${i}_CONV_RELU_1_N/2; i++){
      feature_maps_1[i][k] = max(enc_${i}_conv_relu_1[2*i][k], enc_${i}_conv_relu_1[2*i+1][k]);
    }
  }
  //----------------------------------------------------------------------------
""")

memory_sharing_basic_decoder_block = Template(
"""
  //-----------------------------DECODER ${i}--------------------------------------
  //-----------------------------dec_${i}_upsample---------------------------------
  // Iterate over the number of filters
  dec_${i}_upsample_k: for(apint k=0; k<DEC_${i}_UP_CONV_RELU_INPUT_FEATURES; k++){
    // Iterate over the input matrix
    dec_${i}_upsample_i: for(apint i=0; i<DEC_${i}_UP_CONV_RELU_N/2; i++){
      feature_maps_${k}[2*i][k] = feature_maps_${j}[i][k];
      feature_maps_${k}[2*i+1][k] = feature_maps_${j}[i][k];
    }
  }
  //----------------------------------------------------------------------------

  //-------------------------dec_${i}_up_conv_relu----------------------------------
  // Iterate over the number of filters
  dec_${i}_up_conv_relu_k: for(apint k=0; k<DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    dec_${i}_up_conv_relu_i: for(apint i=0; i<DEC_${i}_UP_CONV_RELU_N; i++){
      ${pragmas}
      // Calculate the auxiliary positions respect to the input
      l_min = max(0, i - DEC_${i}_UP_CONV_RELU_K/2);
      l_max = min(DEC_${i}_UP_CONV_RELU_N, i + DEC_${i}_UP_CONV_RELU_K/2 + 1);
      acc = 0; // Reset the accumulator
      dec_${i}_up_conv_relu_l: for(apint l=l_min; l<l_max; l++){
        #pragma HLS loop_tripcount min=2 max=3 avg=3
        dec_${i}_up_conv_relu_j: for(apint j=0; j<DEC_${i}_UP_CONV_RELU_INPUT_FEATURES; j++){
          // Multiply the input and the weight
          acc += feature_maps_${k}[l][j]*dec_${i}_up_conv_relu_w[l-i+DEC_${i}_UP_CONV_RELU_K/2][j][k];
        }
      }
      feature_maps_${j}[i][k] = ReLU(acc); // Save the accumulator value
    }  
  }
  //----------------------------------------------------------------------------

  //--------------------------dec_${i}_concatenate---------------------------------
  // Iterate over the number of filters
  dec_${i}_concatenate_k: for(apint k=0; k<DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    dec_${i}_concatenate_i: for(apint i=0; i<DEC_${i}_UP_CONV_RELU_N; i++){
      feature_maps_${k}[i][k] = ${res_feature_map}[i][k];
      feature_maps_${k}[i][k+DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES] = feature_maps_${j}[i][k];
    }
  }
  //----------------------------------------------------------------------------

  //-------------------------dec_${i}_conv_relu_0----------------------------------
  // Iterate over the number of filters
  dec_${i}_conv_relu_0_k: for(apint k=0; k<DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    dec_${i}_conv_relu_0_i: for(apint i=0; i<DEC_${i}_CONV_RELU_0_N; i++){
      ${pragmas}
      // Calculate the auxiliary positions respect to the input
      l_min = max(0, i - DEC_${i}_CONV_RELU_0_K/2);
      l_max = min(DEC_${i}_CONV_RELU_0_N, i + DEC_${i}_CONV_RELU_0_K/2 + 1);
      acc = 0; // Reset the accumulator
      dec_${i}_conv_relu_0_l: for(apint l=l_min; l<l_max; l++){
        #pragma HLS loop_tripcount min=2 max=3 avg=3
        dec_${i}_conv_relu_0_j: for(apint j=0; j<DEC_${i}_CONV_RELU_0_INPUT_FEATURES; j++){
          // Multiply the input and the weight
          acc += feature_maps_${k}[l][j]*dec_${i}_conv_relu_0_w[l-i+DEC_${i}_CONV_RELU_0_K/2][j][k];
        }
      }
      feature_maps_${j}[i][k] = ReLU(acc); // Save the accumulator value
    }  
  }
  //----------------------------------------------------------------------------

  //-------------------------dec_${i}_conv_relu_1----------------------------------
  // Iterate over the number of filters
  dec_${i}_conv_relu_1_k: for(apint k=0; k<DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES; k++){
    // Iterate over the input matrix
    dec_${i}_conv_relu_1_i: for(apint i=0; i<DEC_${i}_CONV_RELU_1_N; i++){
      ${pragmas}
      // Calculate the auxiliary positions respect to the input
      l_min = max(0, i - DEC_${i}_CONV_RELU_1_K/2);
      l_max = min(DEC_${i}_CONV_RELU_1_N, i + DEC_${i}_CONV_RELU_1_K/2 + 1);
      acc = 0; // Reset the accumulator
      dec_${i}_conv_relu_1_l: for(apint l=l_min; l<l_max; l++){
        #pragma HLS loop_tripcount min=2 max=3 avg=3
        dec_${i}_conv_relu_1_j: for(apint j=0; j<DEC_${i}_CONV_RELU_1_INPUT_FEATURES; j++){
          // Multiply the input and the weight
          acc += feature_maps_${j}[l][j]*dec_${i}_conv_relu_1_w[l-i+DEC_${i}_CONV_RELU_1_K/2][j][k];
        }
      }
      feature_maps_${k}[i][k] = ReLU(acc); // Save the accumulator value
    }  
  }
  //----------------------------------------------------------------------------
""")

memory_sharing_basic_enc_parameters_array_init = \
"""
    float enc_{i}_conv_relu_0_tmp[ENC_{i}_CONV_RELU_0_K*ENC_{i}_CONV_RELU_0_INPUT_FEATURES*ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES];
    int enc_{i}_conv_relu_0_shape[4];
    apfixed enc_{i}_conv_relu_0_w[ENC_{i}_CONV_RELU_0_K][ENC_{i}_CONV_RELU_0_INPUT_FEATURES][ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES];

    float enc_{i}_conv_relu_1_tmp[ENC_{i}_CONV_RELU_1_K*ENC_{i}_CONV_RELU_1_INPUT_FEATURES*ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
    int enc_{i}_conv_relu_1_shape[4];
    apfixed enc_{i}_conv_relu_1_w[ENC_{i}_CONV_RELU_1_K][ENC_{i}_CONV_RELU_1_INPUT_FEATURES][ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
"""

memory_sharing_basic_dec_parameters_array_init = \
"""
    float dec_{i}_up_conv_relu_tmp[DEC_{i}_UP_CONV_RELU_K*DEC_{i}_UP_CONV_RELU_INPUT_FEATURES*DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES];
    int dec_{i}_up_conv_relu_shape[4];
    apfixed dec_{i}_up_conv_relu_w[DEC_{i}_UP_CONV_RELU_K][DEC_{i}_UP_CONV_RELU_INPUT_FEATURES][DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES];

    float dec_{i}_conv_relu_0_tmp[DEC_{i}_CONV_RELU_0_K*DEC_{i}_CONV_RELU_0_INPUT_FEATURES*DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES];
    int dec_{i}_conv_relu_0_shape[4];
    apfixed dec_{i}_conv_relu_0_w[DEC_{i}_CONV_RELU_0_K][DEC_{i}_CONV_RELU_0_INPUT_FEATURES][DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES];

    float dec_{i}_conv_relu_1_tmp[DEC_{i}_CONV_RELU_1_K*DEC_{i}_CONV_RELU_1_INPUT_FEATURES*DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
    int dec_{i}_conv_relu_1_shape[4];
    apfixed dec_{i}_conv_relu_1_w[DEC_{i}_CONV_RELU_1_K][DEC_{i}_CONV_RELU_1_INPUT_FEATURES][DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
"""

memory_sharing_basic_enc_parameters_reading_block = Template(
"""
    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/enc_${i}_conv_relu_0.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, enc_${i}_conv_relu_0_tmp, enc_${i}_conv_relu_0_shape);

    // Reshape it
    for(int k=0; k<ENC_${i}_CONV_RELU_0_K; k++){
        for(int j=0; j<ENC_${i}_CONV_RELU_0_INPUT_FEATURES; j++){
            for(int i=0; i<ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES; i++){
                enc_${i}_conv_relu_0_w[k][j][i] = enc_${i}_conv_relu_0_tmp[i+ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES*j+ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES*ENC_${i}_CONV_RELU_0_INPUT_FEATURES*k];
            }
        }
    }

    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/enc_${i}_conv_relu_1.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, enc_${i}_conv_relu_1_tmp, enc_${i}_conv_relu_1_shape);

    // Reshape it
    for(int k=0; k<ENC_${i}_CONV_RELU_1_K; k++){
        for(int j=0; j<ENC_${i}_CONV_RELU_1_INPUT_FEATURES; j++){
            for(int i=0; i<ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES; i++){
                enc_${i}_conv_relu_1_w[k][j][i] = enc_${i}_conv_relu_1_tmp[i+ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES*j+ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES*ENC_${i}_CONV_RELU_1_INPUT_FEATURES*k];
            }
        }
    }
""")

memory_sharing_basic_dec_parameters_reading_block = Template(
"""
    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/dec_${i}_up_conv_relu.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, dec_${i}_up_conv_relu_tmp, dec_${i}_up_conv_relu_shape);

    // Reshape it
    for(int k=0; k<DEC_${i}_UP_CONV_RELU_K; k++){
      for(int j=0; j<DEC_${i}_UP_CONV_RELU_INPUT_FEATURES; j++){
        for(int i=0; i<DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES; i++){
          dec_${i}_up_conv_relu_w[k][j][i] = dec_${i}_up_conv_relu_tmp[i+DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES*j+DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES*DEC_${i}_UP_CONV_RELU_INPUT_FEATURES*k];
        }
      }
    }

    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/dec_${i}_conv_relu_0.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, dec_${i}_conv_relu_0_tmp, dec_${i}_conv_relu_0_shape);

    // Reshape it
    for(int k=0; k<DEC_${i}_CONV_RELU_0_K; k++){
      for(int j=0; j<DEC_${i}_CONV_RELU_0_INPUT_FEATURES; j++){
        for(int i=0; i<DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES; i++){
          dec_${i}_conv_relu_0_w[k][j][i] = dec_${i}_conv_relu_0_tmp[i+DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES*j+DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES*DEC_${i}_CONV_RELU_0_INPUT_FEATURES*k];
        }
      }
    }

    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/dec_${i}_conv_relu_1.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, dec_${i}_conv_relu_1_tmp, dec_${i}_conv_relu_1_shape);

    // Reshape it
    for(int k=0; k<DEC_${i}_CONV_RELU_1_K; k++){
      for(int j=0; j<DEC_${i}_CONV_RELU_1_INPUT_FEATURES; j++){
        for(int i=0; i<DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES; i++){
          dec_${i}_conv_relu_1_w[k][j][i] = dec_${i}_conv_relu_1_tmp[i+DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES*j+DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES*DEC_${i}_CONV_RELU_1_INPUT_FEATURES*k];
        }
      }
    }
""")

memory_sharing_basic_enc_parameters_to_func_block = \
"""          enc_{i}_conv_relu_0_w,
          enc_{i}_conv_relu_1_w,
"""

memory_sharing_basic_dec_parameters_to_func_block = \
"""          dec_{i}_up_conv_relu_w,
          dec_{i}_conv_relu_0_w,
          dec_{i}_conv_relu_1_w,
"""

stream_basic_encoder_params_def_block = """
#define ENC_{i}_CONV_RELU_0_K K //Kernel size of the enc_{i}_conv_relu_0 layer
#define ENC_{i}_CONV_RELU_0_INPUT_FEATURES {input_features} //Number of input features of the enc_{i}_conv_relu_0 layer
#define ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES {output_features} //Number of output features of the enc_{i}_conv_relu_0 layer
#define ENC_{i}_CONV_RELU_0_N {N_dim} //Number of frames in the time dimension of the enc_{i}_conv_relu_0 layer

#define ENC_{i}_CONV_RELU_1_K K //Kernel size of the enc_{i}_conv_relu_1 layer
#define ENC_{i}_CONV_RELU_1_INPUT_FEATURES {output_features} //Number of input features of the enc_{i}_conv_relu_1 layer
#define ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES {output_features} //Number of output features of the enc_{i}_conv_relu_1 layer
#define ENC_{i}_CONV_RELU_1_N {N_dim} //Number of frames in the time dimension of the enc_{i}_conv_relu_1 layer
"""
stream_basic_central_params_def_block = """
#define CENTRAL_CONV_RELU_0_K K //Kernel size of the central_conv_relu_0 layer
#define CENTRAL_CONV_RELU_0_INPUT_FEATURES {input_features} //Number of input features of the central_conv_relu_0 layer
#define CENTRAL_CONV_RELU_0_OUTPUT_FEATURES {output_features} //Number of output features of the central_conv_relu_0 layer
#define CENTRAL_CONV_RELU_0_N {N_dim} //Number of frames in the time dimension of the central_conv_relu_0 layer

#define CENTRAL_CONV_RELU_1_K K //Kernel size of the central_conv_relu_1 layer
#define CENTRAL_CONV_RELU_1_INPUT_FEATURES {output_features} //Number of input features of the central_conv_relu_1 layer
#define CENTRAL_CONV_RELU_1_OUTPUT_FEATURES {output_features} //Number of output features of the central_conv_relu_1 layer
#define CENTRAL_CONV_RELU_1_N {N_dim} //Number of frames in the time dimension of the central_conv_relu_1 layer
"""

stream_basic_decoder_params_def_block = """
#define DEC_{i}_UP_CONV_RELU_K K //Kernel size of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_UP_CONV_RELU_INPUT_FEATURES {input_features} //Number of input features of the dec_{i}_up_conv_relu layer
#define DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES {output_features} //Number of output features of the dec_{i}_up_conv_relu layer
#define DEC_{i}_UP_CONV_RELU_N {N_dim} //Number of frames in the time dimension of the dec_{i}_up_conv_relu layer

#define DEC_{i}_CONV_RELU_0_K K //Kernel size of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_CONV_RELU_0_INPUT_FEATURES {input_features} //Number of input features of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES {output_features} //Number of output features of the dec_{i}_conv_relu_0 layer
#define DEC_{i}_CONV_RELU_0_N {N_dim} //Number of frames in the time dimension of the dec_{i}_conv_relu_0 layer

#define DEC_{i}_CONV_RELU_1_K K //Kernel size of the dec_{i}_conv_relu_1 layer
#define DEC_{i}_CONV_RELU_1_INPUT_FEATURES {output_features} //Number of input features of the dec_{i}_conv_relu_1 layer
#define DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES {output_features} //Number of output features of the dec_{i}_conv_relu_1 layer
#define DEC_{i}_CONV_RELU_1_N {N_dim} //Number of frames in the time dimension of the dec_{i}_conv_relu_1 layer
"""

stream_basic_enc_paramters_input_func=\
"""               apfixed enc_{i}_conv_relu_0_w[ENC_{i}_CONV_RELU_0_K][ENC_{i}_CONV_RELU_0_INPUT_FEATURES][ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES],
               apfixed enc_{i}_conv_relu_1_w[ENC_{i}_CONV_RELU_1_K][ENC_{i}_CONV_RELU_1_INPUT_FEATURES][ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES],
"""

stream_basic_dec_paramters_input_func=\
"""               apfixed dec_{i}_up_conv_relu_w[DEC_{i}_UP_CONV_RELU_K][DEC_{i}_UP_CONV_RELU_INPUT_FEATURES][DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES],
               apfixed dec_{i}_conv_relu_0_w[DEC_{i}_CONV_RELU_0_K][DEC_{i}_CONV_RELU_0_INPUT_FEATURES][DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES],
               apfixed dec_{i}_conv_relu_1_w[DEC_{i}_CONV_RELU_1_K][DEC_{i}_CONV_RELU_1_INPUT_FEATURES][DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES],
"""

stream_basic_enc_pragmas = \
"""  #pragma HLS INTERFACE s_axilite port=enc_{i}_conv_relu_0_w
  #pragma HLS INTERFACE s_axilite port=enc_{i}_conv_relu_1_w
"""

stream_basic_dec_pragmas = \
"""  #pragma HLS INTERFACE s_axilite port=dec_{i}_up_conv_relu_w
  #pragma HLS INTERFACE s_axilite port=dec_{i}_conv_relu_0_w
  #pragma HLS INTERFACE s_axilite port=dec_{i}_conv_relu_1_w
"""

stream_enc_basic_featuremap_stream_init = \
"""
  apfixed_stream enc_{i}_outstream("enc_{i}_outstream");"""

stream_dec_basic_featuremap_stream_init = \
"""
  apfixed_stream dec_{i}_outstream("dec_{i}_outstream");"""

stream_basic_skip_conn_init = \
"""
  apfixed enc_{i}_conv_relu_1_out[ENC_{i}_CONV_RELU_1_N][ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES];"""

stream_basic_encoder_block = Template(
"""
  Encoder${pipelined}<ENC_${i}_CONV_RELU_0_K, ENC_${i}_CONV_RELU_0_INPUT_FEATURES, ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES, ENC_${i}_CONV_RELU_0_N,
          ENC_${i}_CONV_RELU_1_K, ENC_${i}_CONV_RELU_1_INPUT_FEATURES, ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES, ENC_${i}_CONV_RELU_1_N>(${input_feature_map}, enc_${i}_conv_relu_0_w, enc_${i}_conv_relu_1_w, enc_${i}_conv_relu_1_out, enc_${i}_outstream);""")

stream_basic_decoder_block = Template(
"""
  Decoder${pipelined}<DEC_${i}_UP_CONV_RELU_K, DEC_${i}_UP_CONV_RELU_INPUT_FEATURES, DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES, DEC_${i}_UP_CONV_RELU_N,
          DEC_${i}_CONV_RELU_0_K, DEC_${i}_CONV_RELU_0_INPUT_FEATURES, DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES, DEC_${i}_CONV_RELU_0_N,
          DEC_${i}_CONV_RELU_1_K, DEC_${i}_CONV_RELU_1_INPUT_FEATURES, DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES, DEC_${i}_CONV_RELU_1_N>(${input_feature_map}, dec_${i}_up_conv_relu_w, ${res_feature_map}, dec_${i}_conv_relu_0_w, dec_${i}_conv_relu_1_w, dec_${i}_outstream); """)

stream_basic_enc_parameters_array_init = \
"""
    float enc_{i}_conv_relu_0_tmp[ENC_{i}_CONV_RELU_0_K*ENC_{i}_CONV_RELU_0_INPUT_FEATURES*ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES];
    int enc_{i}_conv_relu_0_shape[4];
    apfixed enc_{i}_conv_relu_0_w[ENC_{i}_CONV_RELU_0_K][ENC_{i}_CONV_RELU_0_INPUT_FEATURES][ENC_{i}_CONV_RELU_0_OUTPUT_FEATURES];

    float enc_{i}_conv_relu_1_tmp[ENC_{i}_CONV_RELU_1_K*ENC_{i}_CONV_RELU_1_INPUT_FEATURES*ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
    int enc_{i}_conv_relu_1_shape[4];
    apfixed enc_{i}_conv_relu_1_w[ENC_{i}_CONV_RELU_1_K][ENC_{i}_CONV_RELU_1_INPUT_FEATURES][ENC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
"""

stream_basic_dec_parameters_array_init = \
"""
    float dec_{i}_up_conv_relu_tmp[DEC_{i}_UP_CONV_RELU_K*DEC_{i}_UP_CONV_RELU_INPUT_FEATURES*DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES];
    int dec_{i}_up_conv_relu_shape[4];
    apfixed dec_{i}_up_conv_relu_w[DEC_{i}_UP_CONV_RELU_K][DEC_{i}_UP_CONV_RELU_INPUT_FEATURES][DEC_{i}_UP_CONV_RELU_OUTPUT_FEATURES];

    float dec_{i}_conv_relu_0_tmp[DEC_{i}_CONV_RELU_0_K*DEC_{i}_CONV_RELU_0_INPUT_FEATURES*DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES];
    int dec_{i}_conv_relu_0_shape[4];
    apfixed dec_{i}_conv_relu_0_w[DEC_{i}_CONV_RELU_0_K][DEC_{i}_CONV_RELU_0_INPUT_FEATURES][DEC_{i}_CONV_RELU_0_OUTPUT_FEATURES];

    float dec_{i}_conv_relu_1_tmp[DEC_{i}_CONV_RELU_1_K*DEC_{i}_CONV_RELU_1_INPUT_FEATURES*DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
    int dec_{i}_conv_relu_1_shape[4];
    apfixed dec_{i}_conv_relu_1_w[DEC_{i}_CONV_RELU_1_K][DEC_{i}_CONV_RELU_1_INPUT_FEATURES][DEC_{i}_CONV_RELU_1_OUTPUT_FEATURES];
"""

stream_basic_enc_parameters_reading_block = Template(
"""
    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/enc_${i}_conv_relu_0.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, enc_${i}_conv_relu_0_tmp, enc_${i}_conv_relu_0_shape);

    // Reshape it
    for(int k=0; k<ENC_${i}_CONV_RELU_0_K; k++){
        for(int j=0; j<ENC_${i}_CONV_RELU_0_INPUT_FEATURES; j++){
            for(int i=0; i<ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES; i++){
                enc_${i}_conv_relu_0_w[k][j][i] = enc_${i}_conv_relu_0_tmp[i+ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES*j+ENC_${i}_CONV_RELU_0_OUTPUT_FEATURES*ENC_${i}_CONV_RELU_0_INPUT_FEATURES*k];
            }
        }
    }

    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/enc_${i}_conv_relu_1.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, enc_${i}_conv_relu_1_tmp, enc_${i}_conv_relu_1_shape);

    // Reshape it
    for(int k=0; k<ENC_${i}_CONV_RELU_1_K; k++){
        for(int j=0; j<ENC_${i}_CONV_RELU_1_INPUT_FEATURES; j++){
            for(int i=0; i<ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES; i++){
                enc_${i}_conv_relu_1_w[k][j][i] = enc_${i}_conv_relu_1_tmp[i+ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES*j+ENC_${i}_CONV_RELU_1_OUTPUT_FEATURES*ENC_${i}_CONV_RELU_1_INPUT_FEATURES*k];
            }
        }
    }
""")

stream_basic_dec_parameters_reading_block = Template(
"""
    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/dec_${i}_up_conv_relu.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, dec_${i}_up_conv_relu_tmp, dec_${i}_up_conv_relu_shape);

    // Reshape it
    for(int k=0; k<DEC_${i}_UP_CONV_RELU_K; k++){
      for(int j=0; j<DEC_${i}_UP_CONV_RELU_INPUT_FEATURES; j++){
        for(int i=0; i<DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES; i++){
          dec_${i}_up_conv_relu_w[k][j][i] = dec_${i}_up_conv_relu_tmp[i+DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES*j+DEC_${i}_UP_CONV_RELU_OUTPUT_FEATURES*DEC_${i}_UP_CONV_RELU_INPUT_FEATURES*k];
        }
      }
    }

    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/dec_${i}_conv_relu_0.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, dec_${i}_conv_relu_0_tmp, dec_${i}_conv_relu_0_shape);

    // Reshape it
    for(int k=0; k<DEC_${i}_CONV_RELU_0_K; k++){
      for(int j=0; j<DEC_${i}_CONV_RELU_0_INPUT_FEATURES; j++){
        for(int i=0; i<DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES; i++){
          dec_${i}_conv_relu_0_w[k][j][i] = dec_${i}_conv_relu_0_tmp[i+DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES*j+DEC_${i}_CONV_RELU_0_OUTPUT_FEATURES*DEC_${i}_CONV_RELU_0_INPUT_FEATURES*k];
        }
      }
    }

    sprintf(subdirectory,"models/%d/N%d/n0%d/nenc%d/parameters/dec_${i}_conv_relu_1.npy", DATASET, N, BASE_FILTER_SIZE, NENC);
    GetFlatArrFromNpy(parent_path+subdirectory, dec_${i}_conv_relu_1_tmp, dec_${i}_conv_relu_1_shape);

    // Reshape it
    for(int k=0; k<DEC_${i}_CONV_RELU_1_K; k++){
      for(int j=0; j<DEC_${i}_CONV_RELU_1_INPUT_FEATURES; j++){
        for(int i=0; i<DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES; i++){
          dec_${i}_conv_relu_1_w[k][j][i] = dec_${i}_conv_relu_1_tmp[i+DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES*j+DEC_${i}_CONV_RELU_1_OUTPUT_FEATURES*DEC_${i}_CONV_RELU_1_INPUT_FEATURES*k];
        }
      }
    }
""")

stream_basic_enc_parameters_to_func_block = \
"""          enc_{i}_conv_relu_0_w,
          enc_{i}_conv_relu_1_w,
"""

stream_basic_dec_parameters_to_func_block = \
"""          dec_{i}_up_conv_relu_w,
          dec_{i}_conv_relu_0_w,
          dec_{i}_conv_relu_1_w,
"""

################################################################################
#                        FUNCTIONS DEFINITION                                  #
################################################################################

def generate_c_sources_stream(N: int, N_features: int, N_states: int,
  base_filter_size: int, n_blocks: int, dataset : str, test_files: int, total_samples: int,
  test_samples: int, W: int, I: int, project_directory: str, data_directory: str,
  optimized : bool = True) -> None:
    """Generates the C sources for HLS using the stream templates.

    Args:
    -----
        N (int): Size of the input time window in samples.
        N_features (int): Number of features in the input tensor.
        N_states (int): Number of heart states.
        base_filter_size (int): Size of the base filter.
        n_blocks (int): Number of blocks in the encoder and in the decoder.
        dataset (str): Name of the dataset to use. Must be either '2016' or '2022'
        test_files (int): Number of test files to evaluate.
        total_samples (int): Total number of samples to test.
        test_samples (int): Number of samples in each test file.
        W (int): Number of total bits in the to use in the fixed-point datatype.
        If it is None, float is used. 
        I (int): Number of integer bits in the to use in the fixed-point
        datatype. If it is None, float is used. 
        project_directory (str): Path to the directory to store the results.
        data_directory (str): Path to the directory where the test data and the
        model paramters are stored.
        optimized (bool): If True, optimization is used in the C sources. Default: True.
    """

    enc_pipeline = ['']*n_blocks
    dec_pipeline = ['']*n_blocks
    central_pipeline = ''

    if optimized:
      enc_pipeline[-1] = '_pipelined'
      dec_pipeline[0] = '_pipelined'
      if n_blocks > 1:
        dec_pipeline[1] = '_pipelined'
      central_pipeline = '_pipelined'

    # If project_directory does not end with a "/", add it
    if project_directory[-1] != "/":
        project_directory += "/"

    # If project_directory does not exist, create it
    if not os.path.exists(project_directory):
        os.makedirs(project_directory)

    if W is None or I is None:
        datatype = 'FLOAT'
        # Define the fixed-point parameters to write something in the C sources
        # and not raise an error when the user does not specify the W and I.
        W = 16
        I = 8
    else:
        datatype = 'FIXED'

    parameters_definition_block = ""
    enc_parameters_input_to_func = ""
    for i in range(n_blocks):
        if i==0:
            parameters_definition_block += stream_basic_encoder_params_def_block.format(i=i, input_features='N_FEATURES', output_features='BASE_FILTER_SIZE', N_dim='N')
        else:
            parameters_definition_block += stream_basic_encoder_params_def_block.format(i=i, input_features='BASE_FILTER_SIZE*{}'.format(2**(i-1)), output_features='BASE_FILTER_SIZE*{}'.format(2**i), N_dim='N/{}'.format(2**i))

        enc_parameters_input_to_func += stream_basic_enc_paramters_input_func.format(i=i)

    parameters_definition_block +=stream_basic_central_params_def_block.format(input_features = 'BASE_FILTER_SIZE*{}'.format(2**(n_blocks-1)), output_features = 'BASE_FILTER_SIZE*{}'.format(2**n_blocks), N_dim = 'N/{}'.format(2**n_blocks))

    dec_parameters_input_to_func = ""
    for i in range(n_blocks):
        parameters_definition_block += stream_basic_decoder_params_def_block.format(i=i, input_features='BASE_FILTER_SIZE*{}'.format(2**(n_blocks-i)), output_features='BASE_FILTER_SIZE*{}'.format(2**(n_blocks-i-1)), N_dim='N/{}'.format(2**(n_blocks-i-1)))
        dec_parameters_input_to_func += stream_basic_dec_paramters_input_func.format(i=i)

    with open('stream/segmenter_h_template.txt') as template:
        h_template = template.read()
    with open('{}/segmenter.h'.format(project_directory), 'w') as h_file:
        h_file.write(h_template.format(
            datatype=datatype,
            N=N,
            N_FEATURES=N_features,
            N_STATES=N_states,
            BASE_FILTER_SIZE=base_filter_size,
            NENC=n_blocks,
            layers_parameter_definition=parameters_definition_block,
            enc_parameters_input_to_func=enc_parameters_input_to_func,
            dec_parameters_input_to_func=dec_parameters_input_to_func,
            W=W,
            I=I,
            dataset=dataset,
            test_samples=test_samples,
            test_files=test_files,
            total_samples=total_samples
        ))


    enc_feature_maps_streams_initialization = ""
    dec_feature_maps_streams_initialization = ""
    skip_conn_init = ""
    encoder_block = ""
    decoder_block = ""
    enc_pragmas = ""
    dec_pragmas = ""
    for i in range(n_blocks):
        enc_pragmas += stream_basic_enc_pragmas.format(i=i)
        dec_pragmas += stream_basic_dec_pragmas.format(i=i)
        enc_feature_maps_streams_initialization += stream_enc_basic_featuremap_stream_init.format(i=i)
        dec_feature_maps_streams_initialization += stream_dec_basic_featuremap_stream_init.format(i=i)
        skip_conn_init += stream_basic_skip_conn_init.format(i=i)
        if i == 0:
            encoder_block += stream_basic_encoder_block.substitute(pipelined=enc_pipeline[i], i=i, input_feature_map='x')
            decoder_block += stream_basic_decoder_block.substitute(pipelined=dec_pipeline[i], i=i, input_feature_map='central_conv_relu_1_outstream', res_feature_map='enc_{}_conv_relu_1_out'.format(n_blocks-i-1))
        else:
            encoder_block += stream_basic_encoder_block.substitute(pipelined=enc_pipeline[i], i=i, input_feature_map='enc_{}_outstream'.format(i-1))
            decoder_block += stream_basic_decoder_block.substitute(pipelined=dec_pipeline[i], i=i, input_feature_map='dec_{}_outstream'.format(i-1), res_feature_map='enc_{}_conv_relu_1_out'.format(n_blocks-i-1))     

    with open('stream/segmenter_cpp_template.txt') as template:
        s_template = Template(template.read())
    with open('{}/segmenter.cpp'.format(project_directory), 'w') as s:
        s.write(s_template.substitute(
            enc_parameters_input_to_func=enc_parameters_input_to_func,
            dec_parameters_input_to_func=dec_parameters_input_to_func,
            enc_pragmas=enc_pragmas,
            dec_pragmas=dec_pragmas,
            skip_conn_init=skip_conn_init,
            enc_feature_maps_streams_initialization=enc_feature_maps_streams_initialization,
            dec_feature_maps_streams_initialization=dec_feature_maps_streams_initialization,
            encoder_block=encoder_block,
            central_pipeline=central_pipeline,
            decoder_block=decoder_block,
            max_enc_i = n_blocks-1
        ))

    enc_parameters_init_block = ""
    dec_parameters_init_block = ""
    enc_parameters_reading_block = ""
    dec_parameters_reading_block = ""
    enc_parameters_to_func_block = ""
    dec_parameters_to_func_block = ""
    for i in range(n_blocks):
        enc_parameters_init_block += stream_basic_enc_parameters_array_init.format(i=i)
        dec_parameters_init_block += stream_basic_dec_parameters_array_init.format(i=i)
        enc_parameters_reading_block += stream_basic_enc_parameters_reading_block.substitute(i=i)
        dec_parameters_reading_block += stream_basic_dec_parameters_reading_block.substitute(i=i)
        enc_parameters_to_func_block += stream_basic_enc_parameters_to_func_block.format(i=i)
        dec_parameters_to_func_block += stream_basic_dec_parameters_to_func_block.format(i=i)

    with open('stream/segmenter_tb_cpp_template.txt') as template:
        s_template = Template(template.read())
    with open('{}/segmenter_tb.cpp'.format(project_directory), 'w') as s:
        s.write(s_template.substitute(
            parent_directory = data_directory,
            project_directory = project_directory,
            enc_parameters_array_init_block=enc_parameters_init_block,
            dec_parameters_array_init_block=dec_parameters_init_block,
            enc_parameters_reading_block=enc_parameters_reading_block,
            dec_parameters_reading_block=dec_parameters_reading_block,
            enc_parameters_to_func_block=enc_parameters_to_func_block,
            dec_parameters_to_func_block=dec_parameters_to_func_block,
        ))

def generate_c_sources_memory_sharing(N: int, N_features: int, N_states: int,
  base_filter_size: int, n_blocks: int, dataset : str, test_files: int, total_samples: int,
  test_samples: int, W: int, I: int, project_directory: str, data_directory: str, optimized : bool = True) -> None:
    """Generates the C sources for HLS using the memory sharing approach.

    Args:
    -----
        N (int): Size of the input time window in samples.
        N_features (int): Number of features in the input tensor.
        N_states (int): Number of heart states.
        base_filter_size (int): Size of the base filter.
        n_blocks (int): Number of blocks in the encoder and in the decoder.
        dataset (str): Name of the dataset to use. Must be either '2016' or '2022'
        test_files (int): Number of test files to evaluate.
        total_samples (int): Total number of samples to test.
        test_samples (int): Number of samples in each test file.
        W (int): Number of total bits in the to use in the fixed-point datatype.
        If it is None, float is used. 
        I (int): Number of integer bits in the to use in the fixed-point
        datatype. If it is None, float is used. 
        project_directory (str): Path to the directory to store the results.
        data_directory (str): Path to the directory where the test data and the
        model paramters are stored.
        optimized (bool): If True, the generated code includes the optimization
        pragmas. Defaults to True.
    """

    pragmas = {'enc': ['']*n_blocks, 'central': '', 'dec': ['']*n_blocks}

    if optimized:
      for i in range(n_blocks):
        # Place pragmas in the last two encoder blocks, the central part and the first two decoders
        if i == n_blocks-1 or i == n_blocks-2:
          pragmas['enc'][i] = '#pragma HLS unroll\n      #pragma HLS PIPELINE'
        if i == 0 or i == 1:
          pragmas['dec'][i] = '#pragma HLS unroll\n      #pragma HLS PIPELINE'
      pragmas['central'] = '#pragma HLS unroll\n      #pragma HLS PIPELINE'


    # If project_directory does not end with a "/", add it
    if project_directory[-1] != "/":
        project_directory += "/"

    # If project_directory does not exist, create it
    if not os.path.exists(project_directory):
        os.makedirs(project_directory)

    if W is None or I is None:
        datatype = 'FLOAT'
        # Define the fixed-point parameters to write something in the C sources
        # and not raise an error when the user does not specify the W and I.
        W = 16
        I = 8
    else:
        datatype = 'FIXED'

    parameters_definition_block = ""
    enc_parameters_input_to_func = ""
    for i in range(n_blocks):
        if i==0:
            parameters_definition_block += memory_sharing_basic_encoder_params_def_block.format(i=i, input_features='N_FEATURES', output_features='BASE_FILTER_SIZE', N_dim='N')
        else:
            parameters_definition_block += memory_sharing_basic_encoder_params_def_block.format(i=i, input_features='BASE_FILTER_SIZE*{}'.format(2**(i-1)), output_features='BASE_FILTER_SIZE*{}'.format(2**i), N_dim='N/{}'.format(2**i))

        enc_parameters_input_to_func += memory_sharing_basic_enc_paramters_input_func.format(i=i)

    parameters_definition_block +=memory_sharing_basic_central_params_def_block.format(input_features = 'BASE_FILTER_SIZE*{}'.format(2**(n_blocks-1)), output_features = 'BASE_FILTER_SIZE*{}'.format(2**n_blocks), N_dim = 'N/{}'.format(2**n_blocks))

    dec_parameters_input_to_func = ""
    for i in range(n_blocks):
        parameters_definition_block += memory_sharing_basic_decoder_params_def_block.format(i=i, input_features='BASE_FILTER_SIZE*{}'.format(2**(n_blocks-i)), output_features='BASE_FILTER_SIZE*{}'.format(2**(n_blocks-i-1)), N_dim='N/{}'.format(2**(n_blocks-i-1)))
        dec_parameters_input_to_func += memory_sharing_basic_dec_paramters_input_func.format(i=i)

    with open('memory-sharing/segmenter_h_template.txt') as template:
        h_template = template.read()
    with open('{}/segmenter.h'.format(project_directory), 'w') as h_file:
        h_file.write(h_template.format(
            datatype=datatype,
            N=N,
            N_FEATURES=N_features,
            N_STATES=N_states,
            BASE_FILTER_SIZE=base_filter_size,
            NENC=n_blocks,
            layers_parameter_definition=parameters_definition_block,
            enc_parameters_input_to_func=enc_parameters_input_to_func,
            dec_parameters_input_to_func=dec_parameters_input_to_func,
            W=W,
            I=I,
            dataset=dataset,
            test_samples=test_samples,
            test_files=test_files,
            total_samples=total_samples
        ))


    skip_conn_initialization = ""
    encoder_block = ""
    decoder_block = ""
    enc_pragmas = ""
    dec_pragmas = ""
    for i in range(n_blocks):
        enc_pragmas += memory_sharing_basic_enc_pragmas.format(i=i)
        dec_pragmas += memory_sharing_basic_dec_pragmas.format(i=i)
        skip_conn_initialization += memory_sharing_basic_skip_conn_init.format(i=i)
        if i == 0:
          j=1
          k=0
          encoder_block += memory_sharing_basic_encoder_block.substitute(i=i, input_feature_map='x', pragmas=pragmas['enc'][i])
          decoder_block += memory_sharing_basic_decoder_block.substitute(i=i, j=j, k=k, res_feature_map='enc_{}_conv_relu_1'.format(n_blocks-i-1), pragmas=pragmas['dec'][i])
        else:
          # Exchange j and k
          l = j
          j = k
          k = l
          encoder_block += memory_sharing_basic_encoder_block.substitute(i=i, input_feature_map='feature_maps_1', pragmas=pragmas['enc'][i])
          decoder_block += memory_sharing_basic_decoder_block.substitute(i=i, j=j, k=k, res_feature_map='enc_{}_conv_relu_1'.format(n_blocks-i-1), pragmas=pragmas['dec'][i])     

    with open('memory-sharing/segmenter_cpp_template.txt') as template:
        s_template = Template(template.read())
    with open('{}/segmenter.cpp'.format(project_directory), 'w') as s:
        s.write(s_template.substitute(
            enc_parameters_input_to_func=enc_parameters_input_to_func,
            dec_parameters_input_to_func=dec_parameters_input_to_func,
            enc_pragmas=enc_pragmas,
            dec_pragmas=dec_pragmas,
            skip_conn_init=skip_conn_initialization,
            encoder_block=encoder_block,
            central_pragmas = pragmas['central'],
            decoder_block=decoder_block,
            max_enc_i = n_blocks-1,
            j=k,
            k=j
        ))

    enc_parameters_init_block = ""
    dec_parameters_init_block = ""
    enc_parameters_reading_block = ""
    dec_parameters_reading_block = ""
    enc_parameters_to_func_block = ""
    dec_parameters_to_func_block = ""
    for i in range(n_blocks):
        enc_parameters_init_block += memory_sharing_basic_enc_parameters_array_init.format(i=i)
        dec_parameters_init_block += memory_sharing_basic_dec_parameters_array_init.format(i=i)
        enc_parameters_reading_block += memory_sharing_basic_enc_parameters_reading_block.substitute(i=i)
        dec_parameters_reading_block += memory_sharing_basic_dec_parameters_reading_block.substitute(i=i)
        enc_parameters_to_func_block += memory_sharing_basic_enc_parameters_to_func_block.format(i=i)
        dec_parameters_to_func_block += memory_sharing_basic_dec_parameters_to_func_block.format(i=i)

    with open('memory-sharing/segmenter_tb_cpp_template.txt') as template:
        s_template = Template(template.read())
    with open('{}/segmenter_tb.cpp'.format(project_directory), 'w') as s:
        s.write(s_template.substitute(
            parent_directory = data_directory,
            project_directory = project_directory,
            enc_parameters_array_init_block=enc_parameters_init_block,
            dec_parameters_array_init_block=dec_parameters_init_block,
            enc_parameters_reading_block=enc_parameters_reading_block,
            dec_parameters_reading_block=dec_parameters_reading_block,
            enc_parameters_to_func_block=enc_parameters_to_func_block,
            dec_parameters_to_func_block=dec_parameters_to_func_block,
        ))

def generate_c_sources_baseline(N: int, N_features: int, N_states: int,
  base_filter_size: int, n_blocks: int, dataset : str, test_files: int, total_samples: int,
  test_samples: int, W: int, I: int, project_directory: str, data_directory: str) -> None:
    """Generates the C sources for HLS using the baseline architecture.

    Args:
    -----
        N (int): Size of the input time window in samples.
        N_features (int): Number of features in the input tensor.
        N_states (int): Number of heart states.
        base_filter_size (int): Size of the base filter.
        n_blocks (int): Number of blocks in the encoder and in the decoder.
        dataset (str): Name of the dataset to use. Must be either '2016' or '2022'
        test_files (int): Number of test files to evaluate.
        total_samples (int): Total number of samples to test.
        test_samples (int): Number of samples in each test file.
        W (int): Number of total bits in the to use in the fixed-point datatype.
        If it is None, float is used. 
        I (int): Number of integer bits in the to use in the fixed-point
        datatype. If it is None, float is used. 
        project_directory (str): Path to the directory to store the results.
        data_directory (str): Path to the directory where the test data and the
        model paramters are stored.
    """

    # If project_directory does not end with a "/", add it
    if project_directory[-1] != "/":
        project_directory += "/"

    # If project_directory does not exist, create it
    if not os.path.exists(project_directory):
        os.makedirs(project_directory)

    if W is None or I is None:
        datatype = 'FLOAT'
        # Define the fixed-point parameters to write something in the C sources
        # and not raise an error when the user does not specify the W and I.
        W = 16
        I = 8
    else:
        datatype = 'FIXED'

    parameters_definition_block = ""
    enc_parameters_input_to_func = ""
    for i in range(n_blocks):
        if i==0:
            parameters_definition_block += baseline_basic_encoder_params_def_block.format(i=i, input_features='N_FEATURES', output_features='BASE_FILTER_SIZE', N_dim='N')
        else:
            parameters_definition_block += baseline_basic_encoder_params_def_block.format(i=i, input_features='BASE_FILTER_SIZE*{}'.format(2**(i-1)), output_features='BASE_FILTER_SIZE*{}'.format(2**i), N_dim='N/{}'.format(2**i))

        enc_parameters_input_to_func += baseline_basic_enc_paramters_input_func.format(i=i)

    parameters_definition_block +=baseline_basic_central_params_def_block.format(input_features = 'BASE_FILTER_SIZE*{}'.format(2**(n_blocks-1)), output_features = 'BASE_FILTER_SIZE*{}'.format(2**n_blocks), N_dim = 'N/{}'.format(2**n_blocks))

    dec_parameters_input_to_func = ""
    for i in range(n_blocks):
        parameters_definition_block += baseline_basic_decoder_params_def_block.format(i=i, input_features='BASE_FILTER_SIZE*{}'.format(2**(n_blocks-i)), output_features='BASE_FILTER_SIZE*{}'.format(2**(n_blocks-i-1)), N_dim='N/{}'.format(2**(n_blocks-i-1)))
        dec_parameters_input_to_func += baseline_basic_dec_paramters_input_func.format(i=i)

    with open('baseline/segmenter_h_template.txt') as template:
        h_template = template.read()
    with open('{}/segmenter.h'.format(project_directory), 'w') as h_file:
        h_file.write(h_template.format(
            datatype=datatype,
            N=N,
            N_FEATURES=N_features,
            N_STATES=N_states,
            BASE_FILTER_SIZE=base_filter_size,
            NENC=n_blocks,
            layers_parameter_definition=parameters_definition_block,
            enc_parameters_input_to_func=enc_parameters_input_to_func,
            dec_parameters_input_to_func=dec_parameters_input_to_func,
            W=W,
            I=I,
            dataset=dataset,
            test_samples=test_samples,
            test_files=test_files,
            total_samples=total_samples
        ))


    enc_feature_maps_initialization = ""
    dec_feature_maps_initialization = ""
    encoder_block = ""
    decoder_block = ""
    enc_pragmas = ""
    dec_pragmas = ""
    for i in range(n_blocks):
        enc_pragmas += baseline_basic_enc_pragmas.format(i=i)
        dec_pragmas += baseline_basic_dec_pragmas.format(i=i)
        enc_feature_maps_initialization += baseline_enc_basic_featuremap_init.format(i=i)
        dec_feature_maps_initialization += baseline_dec_basic_featuremap_init.format(i=i)
        if i == 0:
            encoder_block += baseline_basic_encoder_block.substitute(i=i, input_feature_map='x')
            decoder_block += baseline_basic_decoder_block.substitute(i=i, input_feature_map='central_conv_relu_1', res_feature_map='enc_{}_conv_relu_1'.format(n_blocks-i-1))
        else:
            encoder_block += baseline_basic_encoder_block.substitute(i=i, input_feature_map='enc_{}_maxpool'.format(i-1))
            decoder_block += baseline_basic_decoder_block.substitute(i=i, input_feature_map='dec_{}_conv_relu_1'.format(i-1), res_feature_map='enc_{}_conv_relu_1'.format(n_blocks-i-1))     

    with open('baseline/segmenter_cpp_template.txt') as template:
        s_template = Template(template.read())
    with open('{}/segmenter.cpp'.format(project_directory), 'w') as s:
        s.write(s_template.substitute(
            enc_parameters_input_to_func=enc_parameters_input_to_func,
            dec_parameters_input_to_func=dec_parameters_input_to_func,
            enc_pragmas=enc_pragmas,
            dec_pragmas=dec_pragmas,
            enc_feature_maps_initialization=enc_feature_maps_initialization,
            dec_feature_maps_initialization=dec_feature_maps_initialization,
            encoder_block=encoder_block,
            decoder_block=decoder_block,
            max_enc_i = n_blocks-1
        ))

    enc_parameters_init_block = ""
    dec_parameters_init_block = ""
    enc_parameters_reading_block = ""
    dec_parameters_reading_block = ""
    enc_parameters_to_func_block = ""
    dec_parameters_to_func_block = ""
    for i in range(n_blocks):
        enc_parameters_init_block += baseline_basic_enc_parameters_array_init.format(i=i)
        dec_parameters_init_block += baseline_basic_dec_parameters_array_init.format(i=i)
        enc_parameters_reading_block += baseline_basic_enc_parameters_reading_block.substitute(i=i)
        dec_parameters_reading_block += baseline_basic_dec_parameters_reading_block.substitute(i=i)
        enc_parameters_to_func_block += baseline_basic_enc_parameters_to_func_block.format(i=i)
        dec_parameters_to_func_block += baseline_basic_dec_parameters_to_func_block.format(i=i)

    with open('baseline/segmenter_tb_cpp_template.txt') as template:
        s_template = Template(template.read())
    with open('{}/segmenter_tb.cpp'.format(project_directory), 'w') as s:
        s.write(s_template.substitute(
            parent_directory = data_directory,
            project_directory = project_directory,
            enc_parameters_array_init_block=enc_parameters_init_block,
            dec_parameters_array_init_block=dec_parameters_init_block,
            enc_parameters_reading_block=enc_parameters_reading_block,
            dec_parameters_reading_block=dec_parameters_reading_block,
            enc_parameters_to_func_block=enc_parameters_to_func_block,
            dec_parameters_to_func_block=dec_parameters_to_func_block,
        ))

def save_model_paramters_as_npy(model: Model, directory: str, n_blocks: int):
    """Saves the model parameters as npy files to be read from the cpp code. It
    assumes that the model layers are named with the format `enc_i_conv_relu_j`
    and `dec_i_conv_relu_j` where i is the block number and j is the layer
    number.

    Args:
    -----
        model (Model): The model to save its parameters.
        directory (str): The directory to save the model parameters.
        n_blocks (int): The number of blocks in the model.
    """

    # If directory does not end with '/' add it
    if directory[-1] != '/':
        directory += '/'
    
    directory += 'parameters/'
    
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(n_blocks):   
        w = model.get_layer(name='enc_{}_conv_relu_0'.format(i)).get_weights()[0]
        np.save('{}enc_{}_conv_relu_0.npy'.format(directory, i), w)

        w = model.get_layer(name='enc_{}_conv_relu_1'.format(i)).get_weights()[0]
        np.save('{}enc_{}_conv_relu_1.npy'.format(directory, i), w)
    
    w =  model.get_layer(name='central_conv_relu_0').get_weights()[0]
    np.save('{}central_conv_relu_0.npy'.format(directory), w)

    w =  model.get_layer(name='central_conv_relu_1').get_weights()[0]
    np.save('{}central_conv_relu_1.npy'.format(directory), w)

    for i in range(n_blocks):
        w = model.get_layer(name='dec_{}_up_conv_relu'.format(i)).get_weights()[0]
        np.save('{}dec_{}_up_conv_relu'.format(directory, i), w)

        w = model.get_layer(name='dec_{}_conv_relu_0'.format(i)).get_weights()[0]
        np.save('{}dec_{}_conv_relu_0.npy'.format(directory, i), w)

        w = model.get_layer(name='dec_{}_conv_relu_1'.format(i)).get_weights()[0]
        np.save('{}dec_{}_conv_relu_1.npy'.format(directory, i), w)
    
    w = model.get_layer(name='final_conv').get_weights()[0]
    np.save('{}final_conv.npy'.format(directory), w)


def launch_csim(project_directory: str, data_directory: str, N: int,
  base_filter: int, n_blocks: int, dataset : str, W: int = None,
  I: int = None, subprocess: bool = False, timing: bool = False,
  implementation: str = 'baseline'):
  """Launchs a vivado_hls csim.
  
    Args:
    -----
      project_directory (str): Path to save the results and the auxiliary files.
      data_directory (str): Path to the directory containing the test data and
      the model parameters.
      N (int): Size in samples of the time window.
      base_filter (int): The base filter size of the convolutional layers.
      n_blocks (int): The number of blocks in the model.
      dataset (str): The dataset name. Must be either '2016' or '2022'.
      W (int): Number of total bits to use in the fixed-point data. If None,
      floating point data is used. Default: None.
      I (int): Number of integer bits to use in the fixed-point data. If None,
      floating point data is used. Default: None.
      subprocess (bool): If True, the csim is launched as a subprocess. If False,
      the csim is launched using os.system. Default: False.
      timing (bool): If True, the csim is launched with the timing option,
      annotating the starting time and the ending time of the csim. It only
      works if subprocess is False. Default: False.
      implementation (str): The implementation to use. Must be either 'baseline',
      'memory-sharing' or 'stream'. Default: 'baseline'.
  """

  # If implementation is not baseline, memory_sharing or stream raise an error
  if implementation not in ['baseline', 'memory-sharing', 'stream']:
    raise ValueError('implementation must be either baseline, memory_sharing or stream.')

  # If directory does not end with '/' add it
  if project_directory[-1] != '/':
    project_directory += '/'
  
  # Create the directory if it does not exist
  if not os.path.exists(project_directory):
    os.makedirs(project_directory)

  # Check if dataset is a string
  if not isinstance(dataset, str):
    dataset = str(dataset)

  # Check the dataset
  if dataset not in ['2016', '2022']:
    raise ValueError('dataset must be either 2016 or 2022.')
  
  # Read test_data_elements
  with open(data_directory + '{}-data/N{}-data/test_data_elements.txt'.format(dataset, N), 'r') as f:
    test_data_elements = int(f.read())

  # Generate the C source files using the templates
  if implementation == 'stream':
    generate_c_sources_stream(N, 4, 4, base_filter, n_blocks, dataset, test_data_elements//bs+1,
      test_data_elements, bs, W, I, project_directory, data_directory, optimized = False)
  elif implementation == 'memory-sharing':
    generate_c_sources_memory_sharing(N, 4, 4, base_filter, n_blocks, dataset, test_data_elements//bs+1,
      test_data_elements, bs, W, I, project_directory, data_directory, optimized = False)
  elif implementation == 'baseline':
    generate_c_sources_baseline(N, 4, 4, base_filter, n_blocks, dataset, test_data_elements//bs+1,
      test_data_elements, bs, W, I, project_directory, data_directory)
  
  # Copy npy_reading.h and csim-launcher.tcl to the project directory
  shutil.copyfile('npy_reading.h', project_directory + 'npy_reading.h')
  shutil.copyfile('csim-launcher.tcl', project_directory + 'csim-launcher.tcl')
  
  print('Launching Vivado HLS CSim...')
  if subprocess:
    # Launch the csim as a screen subprocess
    os.system('cd {} && screen -d -m vivado_hls csim-launcher.tcl'.format(project_directory))
  
  else:
    if timing:
      start = time.time()
      with open(project_directory + 'timing.txt', 'w') as f:
        f.write('Start time: {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))))

    # Launch the csim using os.system
    os.system('cd {} && vivado_hls csim-launcher.tcl'.format(project_directory))

    if timing:
      end = time.time()
      with open(project_directory + 'timing.txt', 'a') as f:
        f.write('End time: {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))))
        f.write('Total time: {} hours'.format((end-start)/3600))
    
def launch_synthesis(project_directory: str, N: int, base_filter: int,
  n_blocks: int,  W: int = None, I: int = None, subprocess: bool = False,
  implementation : str = 'baseline', optimized: bool = True):
  """Launches the synthesis of the described model and saves the results.
  
  Args:
  -----
    project_directory (str): Path to save the results and the auxiliary files.
    N (int): Size in samples of the time window.
    base_filter (int): The base filter size of the convolutional layers.
    n_blocks (int): The number of blocks in the model.
    dataset (str): The dataset name. Must be either '2016' or '2022'.
    W (int): Number of total bits to use in the fixed-point data. If None,
    floating point data is used. Default: None.
    I (int): Number of integer bits to use in the fixed-point data. If None,
    floating point data is used. Default: None.
    subprocess (bool): If True, the csim is launched as a subprocess. If False,
    the csim is launched using os.system. Default: False.
    implementation (str): The implementation to use. Must be either 'baseline',
    'memory-sharing' or 'stream'. Default: 'baseline'.
    optimized (bool): If True, the optimized version of the model is used. If False,
    the non-optimized version of the model is used. Default: True.
  """

  # If implementation is not baseline, memory_sharing or stream raise an error
  if implementation not in ['baseline', 'memory-sharing', 'stream']:
    raise ValueError('implementation must be either baseline, memory_sharing or stream.')

  #Fix dataset since it is not used
  dataset = '2016'

  # Check if project_directory ends with '/' if not add it
  if project_directory[-1] != '/':
    project_directory += '/'
  
  # Create the directory if it does not exist
  if not os.path.exists(project_directory):
    os.makedirs(project_directory)

  # Generate the C source files using the templates
  if implementation == 'stream':
    generate_c_sources_stream(N, 4, 4, base_filter, n_blocks, dataset, 1, 1, 1, W, I, project_directory, project_directory, optimized) # Arbitrary values can be used for non-structural arguments for synthesis
  elif implementation == 'memory-sharing':
    generate_c_sources_memory_sharing(N, 4, 4, base_filter, n_blocks, dataset, 1, 1, 1, W, I, project_directory, project_directory, optimized) # Arbitrary values can be used for non-structural arguments for synthesis
  elif implementation == 'baseline':
    generate_c_sources_baseline(N, 4, 4, base_filter, n_blocks, dataset, 1, 1, 1, W, I, project_directory, project_directory)
  
  # Copy synth-launcher.tcl to the project directory
  shutil.copyfile('synth-launcher.tcl', project_directory + 'synth-launcher.tcl')

  print('Launching Vivado HLS Synthesis...')
  if subprocess:
    os.system('cd {} && screen -d -m vivado_hls synth-launcher.tcl'.format(project_directory))
  else:
    os.system('cd {} && vivado_hls synth-launcher.tcl'.format(project_directory))

def generate_config_json(dataset_list: Tuple, N_list: Tuple, n0_list: Tuple, nenc_list: Tuple, WI_list: Tuple, json_file_path: str = 'csims-config.json'):
  """Generates the csim config json file, which contains each csim configuration
  to run. It can be used to launch csims and synthesis from the *id command line
  tool of hls_launcher.py.

  Args:
    dataset_list (Tuple): List of datasets to test. The datasets must be either
    '2016' or '2022'.
    N_list (Tuple): List of N values to test.
    n0_list (Tuple): List of n0 values to test.
    nenc_list (Tuple): List of nenc values to test.
    W_list (Tuple): List of W values to test.
    I_list (Tuple): List of I values to test.
    json_file_path (str): Path to the json file to save. Default: 'csims-config.json'.
  """

  # Create the dictionary
  csims_config = {}

  id = 0

  for dataset in dataset_list:
    for N in N_list:
      for n0 in n0_list:
        for nenc in nenc_list:
          for W, I in WI_list:
            csims_config[id] = {
                'N': N,
                'n0': n0,
                'nenc': nenc,
                'dataset': dataset,
                'W': W,
                'I': I
            }

            id += 1
  print(id)
  
  # Save the dictionary as a json file
  with open(json_file_path, 'w') as f:
    json.dump(csims_config, f)