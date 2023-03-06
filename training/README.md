# PCG Segmentation Model Optimization: Training

This repository is the code for the paper [Optimizing Heart Sound Segmentation Deep Model for Low-Cost FPGA Implementation](). If you use this code, please cite the following paper.

```bibtext
```

This folder contains the code to preprocess the datasets, train the model and evaluate it. The [`utils`](utils/) folder contains different submodules with the needed auxiliary functions. The scripts in the main folder enable the access to these functions. In order to run them you must first match the requirements described in the [main README](../README.md).

## Data preparation

### Training, validation and test datasets

The data preparation is done using the [`preparedata.py`](preparedata.py) script. This script takes the raw data from the `physionet.org` folder and preprocesses it to generate the different subdatasets used in the project. This is a CLI-enabled script, so it can be run from the command line. In order to run them, you must first access the `training` folder:

```console
foo@bar:~/PCG-Segmentation-Model-Optimization/$ cd training
```

To launch the script, you must specify the dataset to preprocess. The available options are `--p2016` or `--p2022` for the 2016 PhysioNet/Computing in Cardiology Challenge dataset and the 2022 PhysioNet/Computing in Cardiology Challenge dataset, respectively. For example, to preprocess the 2016 dataset, you can run:

```console
foo@bar:~/···/training/$ python preparedata.py --p2016
```

> This process will generate some warnings. This is because some recordings in the dataset are not valid. These warnings can be ignored.

> Some time is needed to preprocess the data. Don`t be alarmed if the script takes a while to finish.

The script will generate two directories in the `training` folder:
- `2016-proc`, which contains the preprocessed data of each valid recording available in the dataset in `.npz` format and a `data_dict.json` file with the information corresponding to each recording. Each `.npz` file has two numpy arrays, `X` and `S`, which contain the input and output data, respectively.
- `2016-data`, which contains the subfolders `N64-data`, `N128-data`, `N256-data` and `N512-data`, each of them corresponding a different number of samples per window, `N`. Each of these subfolders contains:
    - `data_dict.json`, which contains the information of the recordings in the subdataset corresponding to the selected `N` value.
    - Three `.npz` files, `train.npz`, `val.npz` and `test.npz`, which contain the preprocessed data of the recordings in the subdataset corresponding to the selected `N` value. Each file has two numpy arrays, `X` and `S`, which contain the input and output data, respectively.
    - Three `.json` files, `train_dict.json`, `val_dict.json` and `test_dict.json`, which contain the information of the recordings in the subdataset corresponding to the selected `N` value.

The resulting training, validation and test datasets are subject-exclusively disjoint. This means that the recordings of each subject are either in the training, validation or test set, but not in more than one of them. The proportion of recordings in each set is 60%, 20% and 20%, respectively. This can be changed by using the `--proportions` or `-p` argument, which takes a list of three values that sum up to 1. For example, to generate the subdatasets with a 80%, 10% and 10% proportion, you can run:

```console
foo@bar:~/···/training/$ python preparedata.py --p2016 -p 0.8 0.1 0.1
```

Also, if not all the `N` values are needed, the `--N_list` or `-N` argument can be used to specify the list of `N` values to generate. For example, to generate the subdatasets with `N` values of 64, 128 and 256, you can run:

```console
foo@bar:~/···/training/$ python preparedata.py --p2016 -N 64 128 256
```

The test subdatasets can be compressed by adding the `--compress` or `-c` argument. This will generate a `.tgz` file joining the `N*-data/test.npz` files. This is useful to easily transfer the test subdatasets.

Finally, the force subcommand can be used to force the script to overwrite the preprocessed data. Prior to preprocess the data, the script checks if the `*-proc` has been created. If it has, the script will not overwrite the data and it will use it to create the subdatasets of `*-data`. But this can be avoided by using the `--force` or `-f` argument.

All this functionality is also available for the 2022 dataset, whose corresponding main command is:

```console
foo@bar:~/···/training/$ python preparedata.py --p2022
```

### 10-fold cross-validation

This script also supports the generation of the 10-fold cross-validation subdatasets. This is done by using the `--CV` argument. For example, to generate the 10-fold cross-validation subdatasets for the 2016 dataset, you can run:

```console
foo@bar:~/···/training/$ python preparedata.py --p2016 --CV
```

This will generate the `2016-CV`folder, which contains the subfolders `N64-data`, `N128-data`, `N256-data` and `N512-data`, each of them corresponding a different number of samples per window, `N`. Each of these subfolders contains:	
- `data_dict.json`, which contains the information of the recordings in the subdataset corresponding to the selected `N` value.
- Ten `.npz` files, `fold_0.npz`, `fold_1.npz`, ..., `fold_9.npz`, which contain the preprocessed data of the recordings in the subdataset corresponding to the selected `N` value. Each file has two numpy arrays, `X` and `S`, which contain the input and output data, respectively.
- Ten `.json` files, `fold_0.json`, `fold_1.json`, ..., `fold_9.json`, which contain the information of the recordings in the subdataset corresponding to the selected `N` value.

Of course, the 2022 dataset can also be used with the `--CV` argument.

## Training

### Normal training

The training is done using the [`train_launcher.py`](train_launcher.py) script. This script takes the preprocessed data from the [`preparedata.py`](preparedata.py) and trains the model. This is a CLI-enabled script, so it can be run from the command line.

A basic training can be launched by running:

```console
foo@bar:~/···/training/$ python train_launcher.py --parameters 64 8 4 2016
```

> For the bigger models, the training can take a while. As reference, the training launched with the parameters `64 8 4` takes around 1 hour in a NVIDIA GeForce RTX 3090 GPU.

This will train the model with the parameters specified in the `parameters` argument: `N=64`, `n0=8`, `nenc=4`; over the 2016 dataset and using the default training hyperparameters: 15 epochs, 1 batch size and 1e-4 learning rate. The training will be done using the GPU if available, otherwise it will use the CPU. This will create a folder inside the `training` folder with the name `models`, with the subfolders `2016/N64/n08/nenc4/`. Inside this folder, the trained model will be saved in the `parameters.h5` file. Other auxiliary files will also be saved in this folder. After the training, the test set will be evaluated and the results will be saved in the `test_metrics.npz` file.

As you can see, the model architecture can be easy changed by the `--parameters` command, that takes a list of three values: `N`, `n0` and `nenc`. The `N` value is the number of samples per window, `n0` is the number of filters in the first convolutional layer and `nenc` is the number of encoders/decoders in the model. The explored values for these parameters are:
- `N`: 64, 128, 256 and 512
- `n0`: 8, 7, 6, 5 and 4
- `nenc`: 4, 3, 2 and 1

In order to enable the automated launch of the training, a file-driven mode is available. This mode is enabled by using the `--parameters_file` and can be used as follows:

```console
foo@bar:~/···/training/$ python train_launcher.py --parameters_file config.json 0 2016
```
Where `config.json` is a `.json` file containing the parameters to be used in the training. This file must have the following structure:

```json
{
    "0" : {
        "N" : 64,
        "n0" : 8,
        "nenc" : 4
    },

    "1" : ···
}
```
In this case, as the `0` identifier is used, the training will be done with the parameters specified in the `0` key: `N=64`, `n0=8`, `nenc=4`.

The training process described in [1] has the default hyperparameters. Anyways, this can be customized by using the `--training_hyperparamters` command takes a list of three values: `epochs`, `batch_size` and `learning_rate`. The default values are 15 epochs, 1 batch size and 1e-4 learning rate.

Of course, the same functionality is available for the 2022 dataset, whose corresponding main command is:

```console
foo@bar:~/···/training/$ python train_launcher.py --parameters 64 8 4 2022
```

### 10-fold cross-validation training

The 10-folf cross-validation training can be also launched with this script. This is done by using the `--CV` argument, as in the following example:

```console
foo@bar:~/···/training/$ python train_launcher.py --parameters 64 8 4 --CV 0 2016 
```

This will create the `models-cv` folder, and inside it, the subfolders `2016/N64/n08/nenc4/fold_0/`. Inside this folder, the trained models will be saved in the `parameters.h5` file. Other auxiliary files will also be saved in this folder. After the training, the test set will be evaluated and the results will be saved in the `test_metrics.npz` file. As you can see a single fold training is launched. This is for the sake of parallelizability if the training is done in a cluster. The training of the other folds can be launched by changing the `0` identifier in the `--CV` argument. Also, the file driven mode can be used with the `--CV` argument, but the fold identifier must be also included in the configuration file structure.

```json
{
    "0" : {
        "N" : 64,
        "n0" : 8,
        "nenc" : 4,
        "fold" : 0
    },

    "1" : ···
}
```

## Evaluation

The evaluation is done using the [`evaluate.py`](evaluate.py) script. This script takes the model test metrics, saved in the `test_metrics.npz` files, and parses them into a single excel file. This is a CLI-enabled script, so it can be run from the command line.

A basic evaluation can be launched by running:

```console
foo@bar:~/···/training/$ python evaluate.py
```

This will parse the test metrics of the models trained in both datasets for all the explored parameter combinations: `N in [64, 128, 256, 512]`, `n0 in [8, 7, 6, 5, 4]` and `nenc in [4, 3, 2, 1]`. The results will be saved in the `models/{dataset}/test_metrics.xlsx` file.

If not all the models have been trained, the evaluation can be done for a specific parameter combinations by using the `--datasets_list`, `--N_list`, `--n0_list` and `--nenc_list` arguments. For example, if we want to evaluate the models trained with `N=64`, `n0 in [8, 4]` and `nenc in [4, 3, 2]`, we can run:

```console
foo@bar:~/···/training/$ python evaluate.py --N_list 64 --n0_list 8 4 --nenc_list 4 3 2
```
And it will only evaluate the models trained with the specified parameters in both datasets, since the `--datasets_list` argument is not specified.

The same functionality is available for the cross-validation results by using the `--CV` argument. The main command is:

```console
foo@bar:~/···/training/$ python evaluate.py --CV
```

And the results are available in the `models-cv/{dataset}/test_metrics.xlsx` file. Note that the results of the 10-fold cross-validation are averaged, and the standard deviation is also computed.

## References

[1] F. Renna, J. Oliveira and M. T. Coimbra, "Deep Convolutional Neural Networks for Heart Sound Segmentation," in *IEEE Journal of Biomedical and Health Informatics*, vol. 23, no. 6, pp. 2435-2445, Nov. 2019, doi: [10.1109/JBHI.2019.2894222](https://doi.org/10.1109/JBHI.2019.2894222).