### Define overall data set parameters ###

# Set the Client
client = 'Client_Name'
# Datamodel ID, can be found in the URL
datamodel_id = 'Datamodel_ID'
# Case Key - concatenated primary keys of case table 
case_key = {'case_key': 'PQL_Statement'}
# Activity Table Name
activity_table = 'Name_of_Activity_Table'
# Activity Column Name
activity_name = 'Name_of_Activity_Column'
# Target KPI
target = {'Target_Name': f'''PQL_Statement'''}
# Filter for Data Set
data_filter = f"""FILTER_PQL_Statement"""
# Number of causes that the model should select per case
cause_number_to_identify = 4 #Number


### Define parameters for input data set creation ###

## Define parameters for Context Table
# Tables to be included in the model (names must match names in data model)
tables_to_include = [
     {'name': 'Table_Name', 'exclude': ['Column_Name','Column_Name']}
    ,{'name': 'Table_Name', 'exclude': []}   
]
# Additional columns that should be included, e.g., aggregated features from tables with multiple entries per case, KPIs, Splitted Features, Binned Features, etc.
additional_columns = {
     'Feature_Name': f'''PQL_Statement'''
    ,'Feature_Name': f'''PQL_Statement'''
}

## Define parameters for Days Between Table
# Events whose workdays in between should to be included in the model
events_days_between = {
     'Event_Name': f'''PQL_Statement'''
    ,'Event_Name': f'''PQL_Statement'''
}

## Define parameters for Multiple Value Feature Table
# Feature that should be "one-hot" encoded (with counts)
multiple_value_features = {
     'Feature_Name': f'''PQL_Statement'''
}


### Define ML parameters ###

# Define proportion of test size in train/test split
test_size = 0.1

## Feature Selection (Random Forest Feature Importance)
# Parameters for RandomForestClassifier
params_RandomForestClassifier = {
     'n_estimators': 100
    ,'max_depth': 20
    ,'max_features': 'auto'
    ,'n_jobs': -2
    ,'class_weight': 'balanced' 
}
# Number of features to be selected in 'Feature Selection'
feature_number_to_select = 100
# Method of Feature Importance Calculation ['impurity','permutation']
feature_importance_method = 'impurity'


## Prediction Model (CatBoost)
# Turn grid search on/off
grid_search = False
# Grid search parameters to be searched if grid_search = True
params_grid_search = {
    'iterations': [600],
    'learning_rate': [0.05, 0.1,0.15],
    'depth': [4,7,10,15,20],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}
# Parameters for CatBoostClassifier if grid_search = False
params_CatBoostClassifier = {
     'iterations': 700
    ,'depth': 10
}
# Parameters for CatBoost fitting
params_fit_CatBoost = {
     'use_best_model': True
    ,'early_stopping_rounds': 50
    ,'plot': True    
}

