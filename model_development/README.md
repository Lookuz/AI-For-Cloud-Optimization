# Model Development

This directory contains the python modules and scripts used in the development, testing and tuning of models for prediction algorithms used in resource recommendation of jobs. Some of the dependencies and modules required by the model scripts are in the `resource_recommendation/` folder and may need to be transferred over.

The scripts used for developing, training and tuning each model are:

- `xgb_tuning.py`: Script for developing, training and tuning XGBRegressor model
- `cb_tuning.py`: Script for developing, training and tuning CatBoostRegressor model
- `rf_tuning.py`: Script for developing, training and tuning RandfomForestRegressor model
- `svr_tuning.py`: Script for developing, training and tuning SVR(RBF Kernel) model
- `gbr_tuning.py`: Script for developing, training and tuning GradientBoostingRegressor model

Each script first reads in the data which is saved as a serialized pickle(.pkl) file, and processes the data by applying the appropriate data transformation, feature selection and engineering using operations provided in the `ai_cloud_etl` module. The model then is fitted on the dataset, and the performance of each model is evaluated using the following metrics:

- **R2 Score**: Score that explains how well fitted a model is using the ratio of the variance of the fit model's predictions over the total variance that is observed in the dataset. 
- **Mean Squared Error(MSE)**: Loss function that is minimized during the training phase when models are fit on a dataset. Main metric used to compare between models.

After fitting the model on the training dataset, hyperparameter tuning is performed to search for the best set of hyperparameters for each model. The set of hyperparameters to search for each model is given by `param_grid`.<br>
For hyperparameter search and tuning, we have opted to use Bayesian Optimization due to the observation that hyperparameter combinations can be difficult to express as a closed form function. This is done using the SciKit-Optimize(`skopt`) library.

Each model script has a objective function implemented by default to be used with each specific model. When using the model development scripts, it is also possible to provided a custom objective function to be provided for the particular model for use with `skopt`. However, it is important to take note that the objective function is one that should be *minimized* rather than maximizing the results for that combination of hyperparameters.<br>
By default, the `bayes_opt` function provided in `ai_cloud_model` uses the Expected Improvement(EI) acquisition function and Gaussian Processes. However, it is also possible to use a different acquisition function and other methods of bayesian Optimization(Decision Trees, etc). For more information, please see https://scikit-optimize.github.io/.

---
### Note
When developing and tuning the SVR(RBF) model, take note that the time taken is drastically longer than the other models as the size of the training data increases, the time taken for training the SVR model increases quadratically(O(*n<sup>2<sup>*)) due to the fact that SVR needs to compute distance between each data point to identify support vectors. It is recommended to downsize the dataset if the dataset is too large to reduce the training time significantly.

---

#### Stacking

Model script for developing, training and tuning the L2 model is also provided as `model_stacking.py`. `model_stacking.py` requires the base models to already been trained and tuned, as well as saved to a serialized pickle file(e.g `xgb.pkl` for XGBRegressor). These files will be used to load the base models using the `load_models` function in `ai_cloud_model`. Alternatively, it also possible to provided a custom list of L1 models.

Mathematically, if the base models are represent as functions where _f<sub>i<sub>_ is the _i<sup>th<sup>_ model as such:

<img float="left" height="40" src="https://github.com/Lookuz/AI-For-Cloud-Optimization/blob/master/model_development/images/l1_models.png">

Then, for each training example *x*, a new feature vector *x<sub>s<sub>* is produced as follows by obtaining the prediction of each base model _f<sub>i<sub>_ on the training example *x*:
  
<img float="left" height="50" src="https://github.com/Lookuz/AI-For-Cloud-Optimization/blob/master/model_development/images/l2_features.png">

Then, this new feature vector *x<sub>s<sub>* is used as a training example for the L2 meta learner which produces a prediction value *y<sub>pred<sub>* based on *x<sub>s<sub>* as input features.
  
<img float="left" height="50" src="https://github.com/Lookuz/AI-For-Cloud-Optimization/blob/master/model_development/images/l2_prediction.png">

where the L2 meta learner can be viewed as a representation of vector of weights *w*.

Overview of prediction pipeline for prediction using model stacking:

<img align="center" src="https://github.com/Lookuz/AI-For-Cloud-Optimization/blob/master/model_development/images/stacking_overview.png">


The base models are stacked to produce a new set of features using the predictions of the base models. This set of new features are then used as training data for the L2 meta learner model in the script to fit and train on. Similar to the base model scripts, the L2 model is then tuned to find the best hyperparameters using Bayesian Optimization via the SciKit-Optimize library.

Similar to training the SVR(RBF) model, model stacking can take a long time to complete. If the dataset gets large, it is recommended to either downsize the dataset to a reasonable size, or be prepared to wait a long time before results are returned.<br>

---
### Note
Additionally, it is possible to also provide custom base L1 models to be used in place of the default models provided in `resource_recommendation/` directory to train the L2 model. However, it is important that the L2 model trained from the L1 models must be used with the same set of base L1 models in the resource recommendation tool as the shape of the stacked features from L1 prediction must match the shape of the fit L2 model.<br>
Also, It should be noted that if custom L1 models are provided, the models should be fit and trained on the same shape as the dataset of the default base models. 

---