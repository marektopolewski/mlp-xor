# Modelling the XOR problem using an MLP
Neural networks are currently one of the fastest growing areas of computer intelligence. They are also arguably our best attempt to model human brain thus far. This project attempts to provide a theoretical and empirical analysis of a problem that early neural networks were incapable of overcoming - simulation of the XOR logic gate. A common solution to this issue is the multilayer perceptron. This project implements and evaluates this neural network under a variety of parameters to determine their influence over the model’s performance. 

## Download
To import the source files an appropriate git command can be used:
```shell
git clone https://github.com/marektopolewski/mlp-xor.git
```
Alternatively, the green `Download` button can be utilised.
\
\
(more at: [https://help.github.com/en/articles/cloning-a-repository](https://help.github.com/en/articles/cloning-a-repository))

## Directory Structure

The following files are contained within the directory:
- CSV files
	- `clean_data.csv`  - 128 noisless samples
	- `nosiy_data.csv`  - 128 precomputed noisy samples (used in report)
	- `new_noisy_data.csv`  - stores 128 noisy samples generated at runtime
- Python scripts
	- `xor_mlp.py` - contains the MLP model as well as all methods necessary for training and testing
	- `ideal_output_map.py` - script used to generated the ideal target output map for the XOR problem
- Other
	- `CS310__Coursework_Report.pdf` - coursework report (submitted via Tabula)
	- `readme.me` - this file, contains instructions


## Usage instructions
The only part of the `xor_mlp.py` script that ought to be modified are the following lines of the main method (at the bottom of the script):
```
#====# Use these paraeters to adjust testing configs #====#
verbose = 2
batch = False 
seeded = True 
dataMode = 2
neuron_nums = [2, 4, 8]
data_sizes = [16, 32, 64]
#====#====#====#====#====#====#====#====#====#====#====#====#
```
Above values are the defaults, however, they can be assigned other values:
- `verbose` - a value in range `[0,3]` that adjusts the amount output to the user. `0` for no information, `3` for printing all the data.
- `batch` - if set to `False` then the online learning is utilised, otherwise, batch mode.
- `seeded` - if set to `True` then the random generator seeded to yield the same results as in the report, otherwise, random initilaisation.
- `dataMode` - on what data is the model trained? `0` for noiseless data, `1` for new noisy data and `2` for precomputed noisy data (like in report).
- `neuron_nums` - a list of hidden neuron numbers to be tested.
- `data_size` - a list of dataset sizes for the model to be trained on.

Python **3** is required with the following libraries installed:
```
numpy, pandas, matplotlib, math, time
```
To run the script execute the command below:
```
python xor_mlp.py
```

## Help and Acknowledgements
For help contact: M.Topolewski@warwick.ac.uk

Implementation based on: [https://github.com/miloharper/multi-layer-neural-network](https://github.com/miloharper/multi-layer-neural-network)