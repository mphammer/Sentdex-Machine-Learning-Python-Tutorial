# Machine Learning Tutorial Python Introduction
## By sentdex

https://pythonprogramming.net/machine-learning-tutorial-python-introduction/

## Introduction

Machine Learning Definition: "field of study that gives computers the ability to learn without explicitly being programmed." - Arthur Samuel 

numpy - python computational library for n-dimensional arrays. 
scipy - a scientific computation library that uses NumPy underneath. 
scikit-learn - popular machine learning library
matplotlib - popular plotting/graphing/visualizing library
pandas - popular data analysis/manipulation library

## Regression - Intro and Data

[quandl](https://data.nasdaq.com/) - library for financial, economic and alternative datasets. 

Each column of the DataFrame (table) is a feature. You want to make sure your dataset has a lot of meaningful features and not a lot of useless features - simplify the dataset as much as you can. 
Useless features can cause a lot of problems.

## Regression - Features and Labels 

The label is what you're trying to predict. 
 
It's not always a good idea to delete data. If there are missing values in a table you might want to fill the empty values with something else rather than drop the entire row (df.dropna()).

## Regression - Training and Testing

