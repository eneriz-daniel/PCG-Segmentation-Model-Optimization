# PCG Segmentation Model Optimization

This repository is the code for the paper [Low-Cost FPGA Implementation of Deep Learning-based Heart Sound Segmentation for Real-Time CVDs Screening](). If you use this code, please cite the following paper.

```bibtext
```

This project aims to optimize the heart sound segmentation deep model presented in [1] for low-cost FPGA implementation, tested over the [2016 PhysioNet/Computing in Cardiology Challenge](https://moody-challenge.physionet.org/2016/) dataset [2] and the CirCor Digiscope dataset [3] (which was used for the George B. Moody/Physionet dataset). Firstly, two model reduction parameters are identified: the number of filters and the number of encoding/decoding layers. Then, the model is optimized for FPGA implementation using two different strategies: memory sharing for feature maps allocation and streaming for dataflow implementation. Additionally, fixed-point representation is used to reduce the memory footprint of the model.

Therefore the repository two main folders: `training` and `implementation`. The first one contains the code for the training of the model with the selected model parameters, and the second one contains the code for the optimization of the model for FPGA implementation using the Vivado High-Level Synthesis (HLS) tool. Additionally, in each folder there is auxiliary code to preprocess the data, evaluate the model and generate summary reports. In each folder there is a `README.md` file with more details about the code and the usage of the scripts.

Also, the complete results report of the experiments is available in the [`complete-results.xlsx`](complete-results.xlsx) file. The results are organized in the different sheets of the file.

## Requirements

### Python dependencies

This code was tested with Python 3.9.10 under Ubuntu 20.04.2 LTS. The requirements are listed in the `requirements.txt` file. To install the requirements, you can use any Python package manager, such as `pip` or `conda`. For example, using `pip`:

```console
foo@bar:~/PCG-Segmentation-Model-Optimization$ pip install -r requirements.txt
```

or using `conda`:

```console
foo@bar:~/PCG-Segmentation-Model-Optimization$ conda install --file requirements.txt
```

### Vivado HLS

The Vivado HLS tool is required to run the code for the FPGA implementation. The version used in this project is 2019.2. The tool can be downloaded from the [Xilinx website](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/archive.html).

The HLS C Simulations and Synthesis processes are launched from Python code, therefore the `vivado_hls` executable must be accessible. Under Linux, you can enable this command by sourcing the Vivado `settings64.sh` command:

```console
foo@bar:~/PCG-Segmentation-Model-Optimization$ source /path/to/vivado_hls/settings64.sh
```

### Datasets

The datasets used in this project are not included in this repository. The datasets can be downloaded from the following links:
- [2016 PhysioNet/Computing in Cardiology Challenge dataset](https://physionet.org/content/hss/1.0/example_data.mat) It is a `.mat`file with the example data provided by the paper [4]. Must be saved in the root folder of the repository under the `physionet.org` folder.
- [CirCor DigiScope dataset](https://physionet.org/content/circor-heart-sound/1.0.3/). Must be saved in the root folder of the repository under the `physionet.org` folder.

Both datasets can be downloaded using `wget`:

```console
foo@bar:~/PCG-Segmentation-Model-Optimization$ wget -r -N -c -np wget https://physionet.org/files/hss/1.0/example_data.mat
```

```console
foo@bar:~/PCG-Segmentation-Model-Optimization$ wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.3/
```
After downloading the datasets, the folder structure should be as follows:
```console
foo@bar:~/PCG-Segmentation-Model-Optimization$ tree
.
├── implementation
│   └── ···
├── physionet.org/
│   └── files
│       ├── circor-heart-sound
│       |   └── 1.0.3
│       |       ├── ···
│       |       ├── training_data.csv
│       |       └── training_data
│       |           └── ···
│       └── hss
│           └── 1.0
│               └── example_data.mat
├── training
│   └── ···
├── README.md
└── requirements.txt
└── ···
```

> If you want to use the datasets in a different folder, you must change the paths in the [`training/utils/preprocessing.py`](training/utils/preprocessing.py) script.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

The authors of the project are Daniel Enériz and Antonio J. Rodríguez-Almeida.

## References

[1] F. Renna, J. Oliveira and M. T. Coimbra, "Deep Convolutional Neural Networks for Heart Sound Segmentation," *IEEE Journal of Biomedical and Health Informatics*, vol. 23, no. 6, pp. 2435-2445, Nov. 2019, doi: [10.1109/JBHI.2019.2894222](https://doi.org/10.1109/JBHI.2019.2894222).

[2] C. Liu et al., "An open access database for the evaluation of heart sound algorithms", *Physiol. Meas.*, vol. 37, no. 12, pp. 2181–2213, Dec. 2016, doi: [10.1088/0967-3334/37/12/2181](https://doi.org/10.1088/0967-3334/37/12/2181).

[3] J. Oliveira et al., "The CirCor DigiScope Dataset: From Murmur Detection to Murmur Classification," *IEEE Journal of Biomedical and Health Informatics*, vol. 26, no. 6, pp. 2524-2535, June 2022, doi: [10.1109/JBHI.2021.3137048](https://doi.org/10.1109/JBHI.2021.3137048).
