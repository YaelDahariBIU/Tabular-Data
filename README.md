# Tabular Data Science - Exercise 1

## Overview

This repository contains the first exercise for the Tabular Data Science course. The main focus of this exercise is to explore and analyze tabular data using Python.

We used the Kaggle competition [dataset](https://www.kaggle.com/competitions/playground-series-s4e9/data) of used cars.

We created a pipeline to transform the dataset, filling missing values, creating new features and converting certain attributes to one-hot type features after heaving limited their values to the most frequent ones.

Later, we trained a simple Linear Regression model on the transformed train and test datasets and analyzed the model's errors.

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

4. Run `part1.ipynb` in your favorite code editor.
