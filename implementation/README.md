# PCG Segmentation Model Optimization: Implementation

This repository is the code for the paper [Optimizing Heart Sound Segmentation Deep Model for Low-Cost FPGA Implementation](). If you use this code, please cite the following paper.

```bibtext
```
## Basic files description

This folder contains the three different C++ implementations templates tested for the PCG segmentation model, the Python scripts used to generate the input data and the models C++ implementations, and to launch the HLS C Simulations and Synthesis. As described in the paper, the three different implementations are:
- A simple C++ implementation of the model, whose templates are in the [`baseline`](baseline) folder.
- A memory optimized C++ implementation of the model, where the feature maps share the same memory space. The templates are in the [`memory-sharing`](memory-sharing) folder.
- A streaming dataflow C++ implementation of the model, where the feature maps are streamed through the network. The templates are in the [`stream`](stream) folder.

In each template folder, you will find three files:
- `segmenter_cpp_template.txt`: the C++ template of the source code, containing the top function to be implemented, i.e. the segmentation model, and other auxiliary functions.
- `segmenter_h_template.txt`: the C++ template of the header file, containing the declarations of the model structure, and other auxiliary declarations as the fixed-point datatypes aliases.
- `segmenter_tb_template.txt`: the C++ template of the testbench file, containing the main function to launch the HLS C Simulation.

Additionally, in the main `implementation` folder, there is an auxiliary header file called [`npy_reading.h`](npy_reading.h), which contains the functions to read the input data from the `.npy` files. This file is used in the testbench files.

As commented in the [main README](../README.md), the HLS C Simulations and Synthesis are launched from the CLI. Therefore two `.tcl` files are provided in the main `implementation` folder:
- [`csim-launcher.tcl`](csim-launcher.tcl): that creates a Vivado HLS project, adds the source and testbench files, sets the target FPGA part, sets the clock and finally launches the HLS C Simulation.
- [`synth-launcher.tcl`](synth-launcher.tcl): that creates a Vivado HLS project, adds the source and testbench files, sets the target FPGA part, sets the clock and finally launches the HLS Synthesis.

> The target FPGA part of this project is the `xc7z020-clg400-1`, which is the programming logic of the Xilinx Zynq 7020 SoC. You can easily change the target part by modifying the `.tcl` files.

> The clock frequency of this project is 100 MHz (corresponding to a 10 ns clock period). You can easily change the clock frequency by modifying the `.tcl` files.

## Data preparation

The first step is to transfer the test dataset and the model parameters from the [`training`](training) folder to the `implementation` folder. To do so, you can manually copy them or use the following commands:
```console
foo@bar:~/PCG-Segmentation-Model-Optimization$ cd training
foo@bar:~/···/training$ tar -cf models.tar models/*/*/*/*/parameters.h5 models/*/*/*/*/test_metrics.npz
foo@bar:~/···/training$ tar -cf test_data.tar *-data/N*-data/test.npz
foo@bar:~/···/training$ cp models.tar test_data.tar ../implementation/
foo@bar:~/···/training$ cd ../implementation
foo@bar:~/···/implementation$ tar -xf models.tar
foo@bar:~/···/implementation$ tar -xf test_data.tar
```

After this, you should find the following directory structure in the `implementation` folder:
```console
foo@bar:~/···/implementation$ tree
.
├── 2016-data
│   ├── N128-data
│   │   └── test.npz
│   ├── N256-data
│   │   └── ···
|   └── ···
├── 2022-data
│   ├── N128-data
│   │   └── ···
|   └── ···
├── models
│   ├── 2016
│   │   ├── N128
│   │   │   ├── n04
│   │   │   │   ├── nenc1
│   │   │   │   │   ├── parameters.h5
│   │   │   │   │   └── test_metrics.npz
│   │   │   │   └── ···
│   │   │   └── ···
│   │   └── ···
│   └── 2022
│       ├── N128
│       │   ├── n04
│       │   │   ├── nenc1
│       │   │   │   ├── parameters.h5
│       │   │   │   └── test_metrics.npz
│       │   │   └── ···
│       │   └── ···
│       └── ···
├── npy_reading.h
├── csim-launcher.tcl
└── ···
```

For the correct functioning of the HLS C Simulations, the input data and the corresponding model parameters must be provided in the `.npy` format. To do so, you can use the Python script [`preparedata.py`](preparedata.py), which will generate the `.npy` files from the `.npz` files. To launch the script, you can use the following command:
```console
foo@bar:~/···/implementation$ python preparedata.py
```
This will create a `parameters` folder inside each `models/*/N*/n*/nenc*/` folder for `N in [64, 128, 256, 512]`, `n0 in [8, 7, 6, 5, 4]` and `nenc in [4, 3, 2, 1]`. This folder will contain `.npy` files each of them corresponding to an individual layer parameters.  Additionally, it will create an `inputs` folder inside each `*-data/N*-data/` folder, containing `.npy` files each of them corresponding to a batch of input data. Also, you will find a `test_data_elements.txt` file in each `*-data/N*-data/` folder, which contains the number of elements of the test data.

> The batch size of input data of the test `npy` files is defined by the `bs` variable in the [`preparedata.py`](preparedata.py) script. The default value is 250. You can easily change this value by modifying the script.

Additionally, the [`preparedata.py`](preparedata.py) script allows the preparation of data for selected models. To do so, you can use the following command to specify the model parameters and datasets to be prepared:
```console
foo@bar:~/···/implementation$ python preparedata.py --datasets_list 2016 --N_list 128 --n0_list 8 5 4 --nenc_list 4 2
```

## HLS C Simulation

Each model parameters combination results in a unique implementation. The [`hls_utils.py`](hls_utils.py) script contains the functions to generate the C++ source, header and testbench files from the templates, and to launch the HLS C Simulation and Synthesis.

The [`hls_launcher.py`](hls_launcher.py) enables the interaction with this functions through the CLI. To launch the HLS C Simulation, you can use the following command:
```console
foo@bar:~/···/implementation$ python hls_launcher.py --csim 64 8 4 2016 16 8
```

This will launch the HLS C Simulation for the model parameters combination `N=64`, `n0=8`, `nenc=4` over the 2016 dataset using a (16, 8) fixed-point representation, which has 16 bits in total of which 8 are dedicated to the integer part. A `csim-runs` folder will be created in the `implementation` folder, containing the `baseline/2016/W16I8/N64n08nenc04` folder, which will contain the source, header and testbench files as well as HLS project and the C Simulation results, which will be written on the `csim-results.txt` file. This will contain the flattened output of the model, i.e., there will be one line for each test data sample with 4×`N` elements.

The default implementation used by `hls_launcher.py` is the `baseline` implementation. You can launch the HLS C Simulation for the `stream` and `memory-sharing` implementations by using the `-s` and `-m` flags respectively. For example, to launch the HLS C Simulation for the `stream` implementation, you can use the following command:
```console
foo@bar:~/···/implementation$ python hls_launcher.py -s --csim 64 8 4 2016 16 8
```
And to launch the HLS C Simulation for the `memory-sharing` implementation, you can use the following command:
```console
foo@bar:~/···/implementation$ python hls_launcher.py -m --csim 64 8 4 2016 16 8
```
There is also a flag dedicated to launch the HLS C Simulation in the background, which is the `-b` flag. For example, to launch the HLS C Simulation for the `stream` implementation in the background, you can use the following command:
```console
foo@bar:~/···/implementation$ python hls_launcher.py -s -b --csim 64 8 4 2016 16 8
```
The script also supports file driven launch of the HLS C Simulation. This is done by using the `--csimid` flag that requires the path to a `.json` and the corresponding `id` of the model parameters combination to be launched. An example of the `.json` file is the following:
```json
{
    "0" : 
    {
        "N" : 64,
        "n0" : 8,
        "nenc" : 4,
        "dataset" : "2016",
        "W" : 16,
        "I" : 8
    },
    "1" : ...
}
```
Let this to be the content of the `csim_ids.json` inside the `implementation` folder. To launch the HLS C Simulation for first element, `0`, using the streamed implementation, you can use the following command:
```console
foo@bar:~/···/implementation$ python hls_launcher.py -s --csimid csim_ids.json 0
```
>This HLS C Simulations take a long time to be completed, although not too much computing resources are needed. As reference, the HLS C Simulation for the `N=64`, `n0=8`, `nenc=4` model parameters combination over the 2022 dataset using a fixed-point representation with 16 bits in total of which 8 are dedicated to the integer part takes around 1 day to be completed.

### HLS C Simulation evaluation

To automate the evaluation of the different HLS C Simulations, the [`postsim.py`](postsim.py) script can be used. This script will read the `csim-results.txt` files and will calculate the metrics of the models. To launch the script, you must specify the implementaton type results you want to evaluate, that must be either `prelimimnary`, `memory.sharing` or `stream`. For the first one, you can use the following command:
```console
foo@bar:~/···/implementation$ python postsim.py -i baseline
```
This will analyze, the model resulting of all parameters combinations on both datasets with the (16, 8) datatypes. To specify the datasets, model parameters and datatypes combinations to be analyzed, you can use the `--datasets_list`, `--N_list`, `--n0_list`, `--nenc_list` `--W_list` and `--I_list` flags. For example, to analyze the models resulting of the `N=[64, 128]`, `n0=[8, 6, 4]`, `nenc=4` model parameters combination over the 2016 dataset using the (16, 8), (16, 4) and (16, 2) datatypes, you can use the following command:
```console
foo@bar:~/···/implementation$ python postsim.py -i baseline --datasets_list 2016 --N_list 64 128 --n0_list 8 6 4 --nenc_list 4 --W_list 16 --I_list 8 4 2
```
> Note that the datatype selection is formed from the combination of the `W_list` and `I_list` flags. Thus, only combinations where `W` is greater than `I` are considered.

The script will create a `csim-acc-{implementation}-{dataset}-W{W}I{I}.xlsx` file inside each `csim-runs/{implementation}/{dataset}/{datatype}/` folder, containing the global accuracies of the floating-point model (resulting from the Keras training), the fixed-point model (resulting from the HLS C Simulation) and their difference.

## HLS Synthesis

The [`hls_launcher.py`](hls_launcher.py) script also enables the interaction with the HLS Synthesis. To launch the HLS Synthesis, you can use the following command:

```console
foo@bar:~/···/implementation$ python hls_launcher.py --synth 64 8 4 16 8
```
This will launch the HLS Synthesis for the model parameters combination `N=64`, `n0=8`, `nenc=4` using a (16, 8) fixed-point representation, which has 16 bits in total of which 8 are dedicated to the integer part. A `synth-runs` folder will be created in the `implementation` folder, containing the `baseline/default/W16I8/N64n08nenc04` folder, which will contain the source, header and testbench files as well as HLS project, where the synthesis report will be written on the `segmenter-synth/solution1/syn/report/Segmenter_csynth.rpt` file.

> Note that the dataset is not required to launch the HLS Synthesis, since the HLS Synthesis is not dependent on the dataset.

As in the simulation mode, the default implementation used by `hls_launcher.py` is the `baseline` implementation. You can launch the HLS Synthesis for the `stream` and `memory-sharing` implementations by using the `-s` and `-m` flags respectively. Also, the file-driven mode is supported by the `--synthid` flag, where the expected `.json` file is:
```json
{
    "0" : 
    {
        "N" : 64,
        "n0" : 8,
        "nenc" : 4,
        "W" : 16,
        "I" : 8
    },
    "1" : ...
}

```

Additionally, the `hls_launcher.py` supports the inclusion of optimization `pragma` directives in the source files through the `--pragmas` or `-p` argument. This will use the optimization directives found during the implementation optimization stage for the `N=64`, `n0=8`, `nenc=4` model. For other models, the `pragma` directives are extrapolated from the `N=64`, `n0=8`, `nenc=4` model. For example, the following command will launch the HLS Synthesis for the `N=64`, `n0=8`, `nenc=4` model using the `stream` optimized implementation:
```console
foo@bar:~/···/implementation$ python hls_launcher.py -s -p --synth 64 8 4 16 8
```

### HLS Synthesis evaluation

The `postsynth.py` is the CLI-enabled script to evaluate the HLS Synthesis results. This script will read the model synthesis results inside the `synth-runs` folder and will summarize the results in a `.xlsx` file per datatype. For example, the following command will evaluate the HLS Synthesis results for the `baseline` implementation across all the `N`, `n0`, `nenc` using the (16, 8) datatype:
```console
foo@bar:~/···/implementation$ python postsynth.py -i baseline
```
This will write the results in the `synth-baseline-default-W16I8.xlsx` file inside the `synth-runs/baseline/default/W16I8/` folder.

As in the `postsim.py` script, a selection of the synthesized models can be done by using the `--N_list`, `--n0_list`, `--nenc_list` `--W_list` and `--I_list` flags.

In this case, the analysis of the optimized implementation is also supported by using the `--pragmas` or `-p` flag.