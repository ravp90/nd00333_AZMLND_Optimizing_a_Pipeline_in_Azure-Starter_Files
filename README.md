# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains bank marketing data pertaining to individuals alongside the binary classification of whether they would subscribe to the bank. The aim of this study is to identify a model that can accurately use this information to correctly classify these individuals, and to then more broadly use the model to predict outcomes for future individuals captured in future marketing campaigns.

The best perfoming model was an AutoML Voting Ensemble with an accuracy os 0.9172, but there were a few other models trained with the AutoML that had a very similar accuracy.

## Scikit-learn Pipeline
The Scikit-learn pipeline downloads the data from the provided url and loads it into a pandas dataframe. Then the data is cleaned and pre-processed in place. 
The cleaning procedure drops the nulls, creates dummies from the categorical variables, converts strings to booleans where appropriate, and extracts the output category into a second dataframe.

For Hyperdrive, the regularization parameter and maximum iterations were the two hyperparameters selected for tuning and the classification model is a logistic regression model, and the goal of the optimization is to maximise the primary metric; accuracy.

### Parameter Sampler
The parameter sampler pertaining to the Hyperdrive optimisation is sampling the regularization paramter, C, as a continuous uniform value between 0.001 and 10.0. The maximum iterations on the other hand is chosen as a choice between 50, 100 and 200. The reason for setting up in this way is because the regularization parameter can take a continuous value given the type that this parameter accepts is a float, and the absolute value or range for this parameter could be very small or very large, so a reasonably large range of values is chosen for the optimisation to search within. For the maximum number of iterations, that is more about the number of solution calculations required to converge a solution, so ideally we would like to have as few as possible that we can get away with to reduce computational cost while also achieving an accurate model. For this reason I created a choice sampler between 50, 100 and 200.

### Early Stopping Policy
The BanditPolicy was chosen for the Hyperdrive optimisation because computation resource should be used as effectively as possible, so if a new set of parameters are tested and start with an accuracy significantly lower than the current best, it would make little sense to iteratively seek further gains in accuracy with those parameter values, rather it is more convenient to switch to a new set of values.  

## AutoML
For AutoML, the configuration for the pipeline is set to have an experiment timeout at 30 minutes, the task is again a classification with the primary metric being accuracy, but this pipeline also includes a n-fold cross validation, for which, 5 folds were generated. 

## Pipeline comparison
The Hyperdrive model was a logistic regression and the accuracy was 0.9111. The AutoML model was a voting ensemble with an accuracy of 0.9172. The AutoML generated a marginally more accurate model. The two types of model are fundamentally difference because a logistic regression model aims to form a separation between two classes using a logistic function. Whereas, the voting ensemble uses multiple previous iterations of AutoML to construct a voting classifier which determines the best model to use for a given scenario, thus if multiple models are created and perform well in different regions of the optimization space, the model with the strongest performance is called as appropriate from the voting classifier. 

## Future work
For hyperdrive I would look to use a broader range for the regularization parameter and I would consider more choices for the maximum iterations, or even the use of the quniform sampler to create a broader range for maximum iterations. I would also increase the maximum number of runs from 32 to some value much greater, as it appeared that most of the runs did not yield an increase in accuracy from the model.

For the AutoML, there could be a benefit to increasing the experiment time from 30 minutes to more, I would also look at adding more folds in the cross-validation given this dataset is very large. 

