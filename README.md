# CollaborativeFiltering and Neural Networks
A simple codebase on a implicit recommendation system on Netflix data and experimentation on Digit Classification task using kNN, SVM and MLP classifiers.

# Setup Instructions
### 1. Data Preparation 
Pertains only to question 1. Place the Netflix.zip file downloaded from eLearning into the data directory. Extract the contents using the command below -

```
unzip -v Netflix.zip # You can ommit the -v if you do not want verbose printed
```

### 2. Environment setup
Common packages like numpy, pandas and scikit-learn are required for this codebase to run. If you are using anaconda please use the following command to create the environment.

```
conda env create -f environment.yml
```

### 3. Question 1 - Collaborative Filtering
To execute the collaborative filtering algorithm run the below command

```
python main.py --question 1 --dataset_name Netflix
```

### 4. Question 2 - Neural Networks

a. To execute model tuning on SVM Classifier run the below command.

```
python main.py --question 2 --dataset_name mnist_784 --model_name SVC
```

b. To execute model tuning on KNN Classifier run the below command

```
python main.py --question 2 --dataset_name mnist_784 --model_name KNN
```
    
c. To execute model tuning on MLP Classifier run the below command

```
python main.py --question 2 --dataset_name mnist_784 --model_name MLP
```
    
d. Model tuning will be automatically followed by execution of the model on the best performing model. If you want to explicitly run a model with the best set of parameters execute the below command

```
python main.py --question 2 --dataset_name mnist_784 --model_name MLP --no_tuning
```

# References 
After the assignment is graded I plan on releasing the codebase to public git under this repository - https://github.com/amajee11us/CollaborativeFiltering

## Author - Anay Majee (anay.majee@utdallas.edu)

