# Sally Storm Prediction Project 

## About Project
This project contains a python package for utility functions and notebooks that contains the code for deep learning training process.

We divided the project into 4 tasks for model training (Link to each notebooks provided):
- [Task 1A](https://github.com/esemsc-mc824/Storm-Forecasting/blob/main/Task1A_final.ipynb): This task is to predict 12 future 'vil' images from 12 sequential 'vil' images.
- [Task 1B](https://github.com/esemsc-mc824/Storm-Forecasting/blob/main/Task1B_final.ipynb): This task is to predict 12 future 'vil' images from 12 sequential 'vil', 'vis', 'ir069', and 'ir107' images.
- [Task 2](https://github.com/esemsc-mc824/Storm-Forecasting/blob/main/Task2_final.ipynb): This task is to predict 'vil' images from 'vis', 'ir069', and 'ir107' images.
- [Task 3](https://github.com/esemsc-mc824/Storm-Forecasting/blob/main/Task3_final.ipynb): This task is to predict number and location of lightning flashes from 'vil', 'vis', 'ir069', and 'ir107' images.

## Introduction

This project aims to tackle the challenge of improving real-time lightning storm forecasting using deep learning techniques. With the increasing unpredictability of weather patterns due to climate change, accurate forecasting of lightning storms is critical to protect lives, infrastructure, and the environment.

Each year, lightning storms result in significant damage, including wildfires, power outages, and disruptions to air travel. Traditional forecasting methods struggle to provide timely and precise predictions, often leaving little time for emergency response. By leveraging advanced ML models, we seek to enhance the ability to predict the evolution of storms, enabling better preparedness and more effective responses in real-time.

Through this project, our goal is to design a predictive system that can accurately forecast lightning storm behavior, offering valuable insights for decision-makers across various sectors, including emergency management, transportation, and energy. This solution could ultimately play a role in reducing the devastating impacts of these storms in our increasingly volatile climate.


## Additional Information

The dataset used consists of 800 example storm events, each accompanied by four satellite imagery types and a corresponding time series of lightning flashes. The available satellite images for each event include:

- Visible ('vis') – Standard optical imagery of the storm system.
- Water Vapor Infrared ('ir069') – Captures moisture content in the atmosphere.
- Cloud/Surface Temperature Infrared ('ir107') – Measures thermal radiation to estimate cloud-top and surface temperatures.
- Vertically Integrated Liquid ('vil') – Estimates the total liquid water content in a storm system.

Each storm event is also associated with a time series of time and location of lightning flashes, providing insights into storm development.


## DataProcessig Package

**DataProcessig** is a utility Python package designed for processing and predicting lightning storm data. It provides a suite of functions and classes to handle data normalization, loading, visualization, and model definition, facilitating efficient development and analysis of lightning storm prediction models.

### Features

- **Data Processing**
  - Compute global mean and standard deviation.
  - Compute global minimum and maximum values for image data.

- **Normalization**
  - Normalize image data using min-max or standard normalization.

- **Data Loading**
  - Custom PyTorch Dataset for loading and preprocessing data from HDF5 files.
  - Functions to load single or multiple events' data.

- **Visualization**
  - Create animated GIFs from image frames.
  - Plot events, correlations, and lightning distributions.
  - Generate geographic distribution plots of lightning events.

- **Models**
  - Sample neural network model for predicting future VIL images.