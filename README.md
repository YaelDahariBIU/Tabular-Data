# Tabular Data Science

## Overview
This repository contains the our Tabular Data Science course exercises and project. In each exercise we will gradually build our project.

We used the Kaggle competition [dataset](https://www.kaggle.com/competitions/playground-series-s4e9/data) of used cars.

## Part 1

[Part 1](./Part_1) - The main focus of this exercise is to explore and analyze tabular data using Python.

In this part, we first decided to consider the competition's training dataset as our whole dataset, since we have no use for the unlabeled test dataset.
<br>
* Hence, we start by splitting the dataset to train and test sets.

* Then, we created a pipeline to transform the dataset, fill missing values, create new features. <br>
The pipeline also converts certain predetermined features to one-hot-type features while limiting their values to the most frequent ones.

* Later, we trained a simple Linear Regression model on our transformed train dataset and tested its accuracy on the transforrmed test dataset. We followed by analyzing the model's errors.

* Finally, we saved our model and our transforrmed dataset so that they can be further used in the next parts.

*Note: The SHAP-library generated graphs in the graphs directory are not exported well. Please refer to the norebook to see these plots properly.*

## How to run the code

To run the code, follow these steps:

1. Clone the repository:
	```bash
	git clone https://github.com/YaelDahariBIU/Tabular-Data.git
	cd Tabular-Data
	```

2. Optional - Create and activate a virtual environment:
	```bash
	python3 -m venv venv
	source venv/bin/activate
	```

3. Install the required dependencies:
	```bash
	pip install -r requirements.txt
	```
4. Enter a part's directory: 
	```bash
	cd Part_n # Replace n
	```

5. Run `part_n.ipynb` in your favorite code editor.
