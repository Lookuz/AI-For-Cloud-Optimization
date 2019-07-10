# Resource Recommendation

 This directory contains the modules and files that are required for resource prediction, recommendation and processing for a given job script.

 To run the resource recommendation tool, use the following format to run the `resource_recommendation.py` script:<br>

 `python3 resource_recommendation.py <job_script> [-t] [-v]`<br>
 where `<job_script>` is the job script to produce the recommendation of resources for.

 The resource recommendation tool first processes the job script provided into key-value pairs. This is done in the `job_script_process` module. This is followed by conversion of the job information into a `DataFrame` object, and the values are transformed and normalized. Features are also engineered as necessary. This step is handled by operations provided in the `ai_cloud_etl` module.

 During the prediction phase, the processed job information is used as features for prediction using the 5 layer-1(L1) models as the default. The models are loaded from a serialized file:

 - `cb.pkl`: CatBoostRegressor 
 - `xgb.pkl`: XGBRegressor 
 - `rf.pkl`: RandomForestRegressor
 - `svr.pkl`: SVR(RBF Kernel)
 - `gbr.pkl`: GradientBoostingRegressor

 The predictions from these base models are then stacked and used as a feature vector for the L2 model. The following models are available to choose from as L2 models for second level prediction:

 - `xgb_l2.pkl`: L2 XGBRegressor(Default)
 - `cb_l2.pkl`: L2 CatBoostRegressor
 - `lr_l2.pkl`: L2 LinearRegressor

 By default, the L2 XGBRegressor is used as the L2 model for stacked prediction of the base models. To change the L2 model to be used for stacking, simply change the `l2_model=` argument in the `l2_predict` function from `ai-cloud_model`. For more information on model stacking, see http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/, as well as the description on model stacking under https://github.com/Lookuz/AI-For-Cloud-Optimization/edit/master/model_development/README.md.

 The prediction algorithms provided by the `ai_cloud_model` module also allows for custom L1 and L2 models to be used in place of the defaults provided. However, do note that the L1 models needs to be trained and fitted on the same structure and shape of data as the L1 models provided, and the L2 model provided must be fitted on the shape equal to the number of L1 models used in base prediction. 

 Upon obtaining the recommended resources from the prediction algorithms, it is then processed using operations from `queue_recommendation` module to provide the recommended number of cores and nodes for the queue that is specified in the job script. Additionally, `queue_recommendation` also provides the capability to recommend the queue based on the predicted resources. This is done using a metric that places penalty on both the wastage of resources of bottlenecking nodes, as well as the number of cores.


 #### Options:
 - `-t`: Also outputs the performance of the script in terms of runtime. Lists the time taken for each segment for the recommendation process(e.g job information extraction, prediction, etc). By default, time is not shown to the user
 - `-v`: Controls the verbosity of the script. If this option is specified, then all warning/error/messages will be shown when running the resource recommendation tool. By default, verbosity is set to false untless this flag is specified

 ### Modules
 - `ai_cloud_etl`: Module that provides operations for data extraction, feature engineering and selection
- `ai_cloud_model`: Module that contains parameters for loading the overall prediction algorithm, as well as model-related supporting operations
- `job_script_process`: Subroutine script to process the details of a given PBS job script and return the required information on the job
- `queue_extract`/`mem_extract`: Routine script to obtain the default parameters(CPU, memory) of each queue in the cluster. Using to handle missing values and allocate defaults, or provide upper/lower bounding should values be too high/low
- `queue_recommendation`: Provides the resource recommendation given the predicted resources, as well as queue recommendation using a self defined metric to select the most optimal queue for the user
- `recommendation_global`: Module that contains all the global variables and parameters used across model development and resource recommendation tools
