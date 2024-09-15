# SHARP
SHARP: Overcoming artificial structures in resolution-enhanced Hi-C data by signal decomposition and multi-scale attention.



## Overview
SHARP is a Python-based tool designed to enhance the resolution of Hi-C data. By leveraging signal decomposition and multi-scale attention, this tool enables researchers to achieve smooth high-resolution Hi-C data, facilitating more detailed genomic interaction analyses.


## Dependencies
The following versions are recommended for optimal performance when using SHARP:
- einops=0.4.1
- numpy=1.20.3
- python=3.7.11
- pytorch=1.6.0
- scikit-image=0.18.1
- scipy=1.6.2


## Data Preprocessing
SHARP accepts 2D matrix in the npz format with the keyword 'matrix' as input. If you have a 2D matrix, you can transform it to the desired format with the code: 
```
$ import numpy as np
$ np.savez_compressed(Output_path, matrix=Your_matrix)
```
### Data structure:

    │
    ├── Data/
    │   │── 1_150/                            #Enhance rate (Downsample rate)
    │   │   ├── 20_Type3_patches.npz          #The third type of the signals decomposited (contacts due to other fine structures)
    │   │   ├── chr20_low_matrix.npz          #The low-resolution maps (downsampled at the specific downsample rate)
    │   │   └── chr20_record.txt              #The record to remove the second type of signals
    │   │── Output/
    │   │   ├── SHARP_20_reconstructed.npz    #The final resolution-enhanced Hi-C contact matrix
    │   └── chr20_high_matrix.npz             #Actual high-resolution maps

If you want to train the model use the paired actual high-resolution contact matrices and the down-sampled low-resolution contact matrices, please put your actual high-resolution contact matrices in the `/Data` folder, and use the scripts in the Preprocessing folder to generate the contents under the `/1_150` (default downsample rate, please change it to the desired downsample rate) folder.

If you only want to enhancer your low-resolution Hi-C contact matrix, please create a folder indicating the enhance rate you want (The enhancement rate refers to the downsampling rate on which the chosen model is trained, e.g. `/1_150`) and put your low_resolution contact matrix in it.

### Preprocessing: 
```
$ cd Preprocessing
$ python Preprocessing.py --path2matrix '../Data/chr20_high_matrix.npz' 
```
`Preprocessing.py` accepts one parameter, which is `--path2matrix`, indicating path to the matrix to be preprocessed.

### (Optional) Downsample the high-resolution maps:
```
$ cd Preprocessing
$ python Downsample.py --chr_list 20 21 --rate_list 150 100 
```
`Downsample.py` accepts two parameters:
```
--chr_list          #List of chromosome numbers
--rate_list         #List of downsample rate
```

***Note***: If you only want to enhancer your low-resolution Hi-C contact matrix, we strongly recommend scaling it to match the data range of the model’s training data. To do this, calculate the median contact frequency along the main diagonal of both your low-resolution Hi-C contact matrix (chr20) and the provided GM12878 chr20 contact matrix. The ratio between these medians should then be used to divide by all values in your low-resolution Hi-C contact matrix.



## Signal Decomposition
```
$ cd Preprocessing
$ python Decomposition.py --chr 20 --k_para 3500000 --matrix "../Data/1_150/chr20_low_matrix.npz" --output_path "../Data/1_150/" --use_blacklist True --black_list1 0 5273 5788 6199 9579 --black_list2 14 5783 5853 6249 9580
$ python Decomposition_save.py  --chr 20 --k_para 3500000 --matrix "../Data/1_150/chr20_low_matrix.npz" --output_path "../Data/1_150/" --record_file "../Data/1_150/chr20_record.txt" --use_blacklist True --black_list1 0 5273 5788 6199 9579 --black_list2 14 5783 5853 6249 9580
$ python Decomposition_save.py  --chr 20 --k_para 160000000 --matrix "../Data/chr20_high_matrix.npz" --output_path "../Data/" --record_file "../Data/1_150/chr20_record.txt" --use_blacklist True --black_list1 0 5273 5788 6199 9579 --black_list2 14 5783 5853 6249 9580
```
`Decomposition.py` accepts the following parameters:
```
--chr               #Chromosome
--k_para            #K paratemer used for signal decomposition
--matrix            #Path to the matrix to be decomposited
--output_path       #Output path for the record of the decomposition
--use_blacklist     #Whether to use the blacklist
--black_list1       #Blacklist array1
--black_list2       #Blacklist array2
```
`Decomposition_save.py` accepts the following parameters:
```
--chr               #Chromosome
--k_para            #K paratemer used for signal decomposition
--matrix            #Path to the matrix to be decomposited
--output_path       #Output path for the record of the decomposition
--record_file       #Path to the record file
--use_blacklist     #Whether to use the blacklist
--black_list1       #Blacklist array1
--black_list2       #Blacklist array2
```
`Decomposition.py` outputs a txt file contains the record for the second type of signals during the signal decomposition process
`Decomposition_save.py` outputs a npz file named as *.Type3_patches.npz that contains the patches for the third type of signals that can be used for training and test.


## (Optional) Training:
```
$ cd Training
$ python Train.py
```
`Train.py` trains the MAXIM backbone with the given low-resolution signals and the high-resolution target and saves the model to the `/Model` folder. The configures can be found in the `Configure.py`.
```
--gpu                 #Whether to use GPUs
--gpu_device          #The gpu device to use
--train_batch_size    #Batch size for training
--test_batch_size     #Batch size for validation and test
--num_workers         #Number of workers used in Dataloader when training
--test_num_workers    #Number of workers used in Dataloader when testing
--learning_rate       #Learning rate
--weight_decay        #Weight decay
--seed                #Seed
--epochs              #Number of epochs
--log_dir             #Path to store the training log
--model_dir           #Path to store the trained model
--low_data_dir        #Path to the low-resolution patches to be trained
--high_data_dir       #Path to the actual high-resolution targets
```

## Evaluation:
### Integrate the three types of signals: 
```
$ cd Evaluation
$ python Addback.py -chr 20 --k_para 160000000 --matrix "../Data/1_150/chr20_low_matrix.npz" --model ../Model/SHARP_best_model.pt --multi 150 --output_path "../Data/Output/" --patches "../Data/1_150/20_Type3_patches.npz" --record_file "../Data/1_150/chr20_record.txt" --use_blacklist True --black_list1 0 5273 5788 6199 9579 --black_list2 14 5783 5853 6249 9580
```
`Addback.py` accepts the patches contains the third type of signals, enhance it with the trained SHARP model, and integrate the enhanced third type of signals with the enhanced first, second type of signals. The configures are as follows:
```
--chr               #Chromosome
--k_para            #K paratemer used for signal decomposition
--matrix            #Path to the matrix to be decomposited
--model             #The trained SHARP model
--multi             #The enhance rate
--output_path       #Output path for the enhanced Hi-C contact matrix
--patches           #Path to the npz file contains the third type signals
--record_file       #Path to the record file
--use_blacklist     #Whether to use the blacklist
--black_list1       #Blacklist array1
--black_list2       #Blacklist array2
```

### NBD
```
$ cd Evaluation
$ python NBD.py --matrix "../Data/Output/SHARP_20_reconstructed.npz"
```
`NBD.py` accepts one parameter, which is `--matrix`, indicating path to the matrix to be calculated the NBD score.


### TRS
```
$ cd Evaluation
$ python TRS.py --refer_TAD_list "PATH_to_the_refer_TAD_list" --detect_TAD_list "PATH_to_the_enhanced_TAD_list" --thres 2
```
`TRS.py` accepts the TAD list called by TAD calling method as input, the default TAD list should follow the following format:
```
Chromosome start end
chr10	5000	10000
chr10	10000	20000
chr10	20000	40000
```
The bin size is set to 5kb as default. The configures included in the TRS.py are as follows:
```
--refer_TAD_list      #TAD detected on the actual high-resolution maps    
--detect_TAD_list     #TAD detected on the reconstructed maps
--thres               #Threshold to judge whether a TAD close to a patch boundary
```
