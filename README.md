# Intro to AI – Stock Price Prediction Project

**Team Members**: Christopher Brice, Gabe Sanders, Evan Archer, Marvin Wocheslander

## Project Overview

This project applies machine learning techniques to predict the stock prices of JPMorgan Chase & Co. (JPM) using historical stock data. Our primary goal was to evaluate multiple models and determine which machine learning approach yields the most accurate predictions for JPM stock prices.

We explored both traditional and deep learning methods and evaluated their performance. Below are the models we tested:

* **Linear Regression** – Implemented in `Model4.py`
  A simple baseline model used to understand how well a linear approach performs on stock price prediction.

* **Decision Tree Regressor** – Implemented in `Model2.py`
  A tree-based model that attempts to learn non-linear patterns in the stock data.

* **LSTM (Long Short-Term Memory)** – Implemented in `Model1.ipynb`
  A type of recurrent neural network (RNN) capable of learning from sequences, particularly suited for time series data.

* **GRU (Gated Recurrent Unit)** – Implemented in `Model3.ipynb`
  A more lightweight version of LSTM, also designed for sequence learning, potentially faster with similar performance.

Each model was trained and evaluated to compare predictive accuracy and computational efficiency.


## Project Structure

* **`DATA/`**
  Contains the raw and preprocessed datasets used for training and testing the models.

* **`models/`**
  Includes individual scripts and notebooks for each machine learning model.

* **`compare-models/`**
  Scripts used to compare model performance using metrics like RMSE, MAE, and visual plots.

* **`assets/`**
  Visualizations and graphs, such as model comparisons and predictions vs. actual prices.

* **`lab3/`**
  Code and files used in Lab 3 for experimenting with machine learning concepts related to this project.


## Requirements

* Python 3.10 or later
* Jupyter Notebook
