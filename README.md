# AnticontagionStrategies  
This repository contains the underlying code and data used in the paper "Effect of Non-Pharmacological Interventions on COVID-19 in China: Based on Parallel Evolution and Control Method"
The model code is written in Python, and has been tested on Python3.8. No non-standard hardware is required to run the model code.

In the code, we use a number of open source third-party packages, the names and versions of which are as follows:  

  matplotlib	3.3.0  
  numpy		    1.18.1+mkl  
  pandas		  1.1.0  
  sklearn		  0.0  

All scripts and functions require only a standard computer with enough RAM to support the operations defined by a user. For minimal performance, this will be a computer with about 2 GB of RAM. For optimal performance, we recommend a computer with the following specs:

  RAM: 16+ GB  
  CPU: 4+ cores, 3.3+ GHz/core  

Relevant scripts and functions are contained in "NPIs-SEIR.py" and "PECR.py", and the data required for models is contained in the subfolder 'data'. By downloading these files, a user will be able to run the relevant model functions. This download should be completed in a few minutes. 

The epidemiological key parameter models "alpha" and "beta" can be obtained by running the script "PECR.py", the output of which is stored under the path "data\model_judge". Simulation and prediction of epidemic transmission can be obtained by running the script "NPIs-SEIR.py". Different epidemic transmission results can be obtained by adjusting the strategy structure. PECR.py should take ~3 hours to run.


# Model functions
The function 'NPIs-SEIR.py' is the main function of the model, simulating the spread of infection in Wuhan from January 11, 2020 to April 9, 2020. This function is able to simulate epidemic transmission trends in different scenarios by controlling Non-Pharmacological Interventions strategies.

The function "PECR.py" is the main function for training the "alpha" and "beta" models, which uses a data-driven approach to optimize the models through cross-validation and regularization. The "alpha" and "beta" models are the key parameters of the epidemiological model and need to be trained first.
