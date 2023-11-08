# MLZC_Midterm_Project

# Engine Health & Performance
## Problem description
The dataset allows the user to do an engine health status check and build a predictive maintenance model for automobiles. This could help vehicle owners and mechanics address potential issues before they become more severe. Further down the road, this dataset could include parameters like make, model, year and existing mileage on the car could paint a broader picture about the automobile market.

## About the project


After analyzing different algorithms, fine tuning them and validating them, final model was selected to be XG Boost and has been trained since XG Boost gave the best validation accuracy and AUC for the predictions.    

The predictions can be made by running engine_predict_test.py file.    

The model has been deployed using flask during production and waitress for deployment.    

The same has been containarized using docker.

## Installing dependencies and activating the environment

* Install pipenv : **pip install pipenv**
* Install python version 3.8 or greater
* Install other required libraries: **pipenv install numpy pandas sklearn====1.1.3 flask waitress====2.1.2**
* activating the shell : **pipenv shell**
* Install xgboost : **pipenv install xgboost**
* Install any dependency you need to as shown in the error message, if it throws an error as some module is missing while running the program


## How to run the engine health prediction service on your system


* build the image using  **docker build -t zoomcamp-test .** on the terminal.
* Open a new terminal and run **docker run -it -p 9696:9696 zoomcamp-test:latest**
* open a new terminal in your IDE and type : **python engine_predict_test.py**

Alternatively, you can directly run it from the terminal :

* Download all the files/clone the repo to your system
* Navigate to the project directory in terminal
* run **python predict.py**
* run **python engine_predict_test.py**


|File name|Description|
|---------|-------------------------------------------------------|
|Dockerfile | file to build and run the docker image in your system|
|engine_data.csv|data set used to train the models |
|Midterm Project - Automotive Vehicles Engine Health.ipynb|For all the preprocessing, EDA, model                                        training,validation,correlation,feature importance analysis and tuning.
|predict.py|script to run the flask app|
|train.py|script to train and save the final model|
|engine_predict_test.py| file to print output response of the model.|
|Pipfile & Pipfile.lock|pipenv files to prepare virtual enviroment| 
|engine_xgb_model.bin| The file in which trained model/ weights are pickled and saved|

