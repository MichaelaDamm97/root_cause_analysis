# root_cause_analysis

Problem Statement:
Process Mining enables the visualization of business process flows with the goal of process optimization. Root Cause Analysis is essential for process improvement in order to take the right actions but usually time consuming and complex. These difficulties show demand for improvement and support in Root Cause Analysis in Process Mining.

Solution Approach: 
Leverage Machine Learning to support Root Cause Analysis in Process Mining.  
The approach  focuses on efficiently analyzing large process datasets and detecting patterns that have been disregarded by process experts. These two aspects are achieved by using performant Machine Learning algorithms capturing the interaction of numerous features. In addition, the model allows to incorporate expert knowledge to improve accuracy. 

Included files:
README.md - Introduction to the Python project and explanation of the files contained in the repository.
requirements.txt - Requirements file that contains a list of Python packages with the respective version number that needs to be installed before running the script.
login_template.py - Template that can be used for connecting to the Celonis IBC. It includes a placeholder for the login credentials, which are the Celonis Cloud-URL of the team and the corresponding API key.
config_template.py - Template that includes all the parameters that must be set before running the main code. This includes the parameters important for data extraction (e.g., data model id, case key, target, data filter), parameters that define the specific columns to be included (e.g., tables to include, additional columns, events to be considered for days between table) and the Machine Learning parameters for Random Forest and CatBoost.
utils.py - Methods that do not directly refer to data engineering or Machine Learning packages.
data_engineering.py - Methods used for dataset creation.
machine_learning.py	- Methods used for feature selection, creation of the prediction model, and interpretation of the model.
RCA.sql - SQL Script to transform the output tables in Celonis.
main.pybn - Jupyter notebook that includes the code to be executed.
