# Group Perception Project Readme 
# Copy your perception homework to here.

# Perception Project Readme
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/mReuPAWT)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=21902286&assignment_repo_type=AssignmentRepo)
# Perception Project
This project will introduce you to the basic concepts of perception through a practical
example for lane detection in traffic. In this exercise, similar to the Control Algorithms
project, we have prepared some parts of the algorithm ourselves and asked you to
provide some of the critical sections of the code to offer you a learning experience that
combines inspection, analysis and synthesis.
You will specifically be using the U-net segmentation to model the lane. You will be
developing a neural network (CNN) based model for the process and obtain the weights
by training. The trained model will then be validated and finally evaluated/tested using
lane images not previously used during training. Your grade will be based on the
accurate determination of weights and the performance of your
[evaluate](#Evaluate/Test U-net CNN model) function, which will be evaluated using our
lane images. We expect to receive the weight file of your model from you with the
evaluate function.

## Hints 
You must make sure not to overfit the model, which occurs when the same data set is
used repeatedly for training. To avoid overfitting, we provided different datasets that you
can feed to the model as input during the training process. These datasets are given
in [Data](#Data) section.


## Files
- Perception-Project
	- data
		- custom
                     	- inputs
		- Dataset
			- Sim
			- ...
			- Modified Carla
			- data.json
	- model
		- unet.py
	- utils
		- dataloader.py
		- loses.py
		- utils.py
	- dataset_creator.py
	- train.py
	- evaluate.py
## Data
The first step is downloading the data to be used for training and validation
from [this](https://drive.google.com/file/d/1OC0FQfcSh1VaF8Yualx_AVG4MhxCyUNP/view?usp=sharing) link.

## Preparation of Data
The second step involves dividing the data to form a data set for training and another for
validation. The relevant code is provided to you, and you are expected to use this code
to save the related training and validation images to the json file. The data in the Sim file
is obtained from Gazebo. However, for a successful training process without overfitting,
we have also provided different data sets obtained from other simulation platforms; e.g.
Modified Carla.

Below are the specific instructions:

Put your data dirs(e.g. Sim, Modified Carla) into the **Dataset** directory. Each data dir must
consist of two directories; i.e. for inputs and labels. The labels in the **labels** directory are the
labels of the images in the **inputs** directory, and these labels must match the labels of the
images in the input file, with only the extension **_Label** added to the end of the image
label. Once the above steps are completed, you are ready to create the **data.json** file
using the command below. **data.json** file consists of image paths and label paths separated for the
training and validation processes.

```python
python3 dataset_creator.py
```
This command will start data seperation.

## Unet model
The segmentation process in this project is based on the U-Net model presented in the
paper below. We recommend that all students review this paper:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (2015)

![network architecture](https://i.imgur.com/jeDVpqF.png)

## Training and Validation Step
The data prepared in the previous step will now be used as input to train and validate
the U-Net segmentation model. This section will return the model weights.
Once **data.json** file is created, the U-Net model is ready for training.

```python
python3 train.py
```
This command will start the training.

## Evaluate/Test U-net CNN model
At this step, you will test the developed CNN model with data that was not used in the
training and validation step. Here are the instructions:
1) Write an evaluation function that takes the trained model weights and test
images; then feed the network with the test images(You can find different test
images in **data/custom/input**). Apply these test images to the CNN model.
Analyze and comment on the performance.

2) Show the segmented output of the model on the test images.  

You can refer to the validation section while developing the evaluation function,
specifically the **train.py** file and data loader for reading, and **utils/dataloader.py**
for image processing.


