#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from pycelonis.celonis_api.pql.pql import PQL, PQLColumn, PQLFilter
from pycelonis.celonis_api.event_collection.data_model import Datamodel
from utils import *
from sklearn.preprocessing import LabelEncoder


def add_selected_table_columns_to_pql(datamodel: Datamodel, tables_to_include: list):
    '''
        Returns a PQL query that includes the columns of the given tables. Datetime columns are added as 3 features (Weekday, Day, Month)
        input:
            datamodel(Datamodel) - the datamodel of the process
            tables_to_include(list) - list with tables that should be cosidered, and columns to be excluded
        output:
            query with PQLColumns that includes the columns of the given tables
    '''
    query = PQL()
    for table_name in [x['name'] for x in tables_to_include]:
        table = datamodel.tables.find(table_name)
        for column in table.columns:
            c_name = column.get("name")
            c_type = column.get("type")
            # define column name and cell name
            name = f"{table.name}.{c_name}"
            cel_name = f'"{table.name}"."{c_name}"'
            # skip excluded columns
            exclude_cols = [x['exclude'] for x in tables_to_include if x['name']==table_name][0]
            if c_name in exclude_cols:
                continue
            # transform date columns to categorical values
            if c_type == "DATE":
                query += [PQLColumn(f"'Weekday '||DAY_OF_WEEK({cel_name})", name+'_WEEKDAY')]
                query += [PQLColumn(f"'Day '||DAY({cel_name})", name+'_DAY')]               
                query += [PQLColumn(f"'Month '||MONTH({cel_name})", name+'_MONTH')]
                # query += [PQLColumn(f"'Year '||YEAR({cel_name})", name+'_YEAR')]
            # transform missing strings to "missing"
            else: query += [PQLColumn(f"{cel_name}",name)]
            
    return query


def get_context_table(datamodel: Datamodel, case_key: dict, tables_to_include: list, additional_columns: dict = None, target: dict = None, filter: str = None):
    '''
        Returns a pd.Dataframe with the casekey as index, and all columns of the 'selected_tables' without excluded_cols; with add_cols one can add additional columns with PQL statements as the target
        Columns of type date are included as 2 columns with categories Month and Year
        input:
            datamodel(Datamodel) - the datamodel of the process
            case_key(['column_name','pql_statement']) - index of the table
            selected_tables - list of names of tables whose columns should be included 
            exlude_cols([table_name1.column_name1,table_name2.column_name2,...]) - list of names of columns to be excluded
            add_cols([['column_name1','pql_statement1'],['column_name2','pql_statement2'],...]) - columns to be included in the table
        output:
            pd.Dataframe with the casekey as index, all columns of selected tables, all columns defined in add_cols, without exlude_cols
    '''
    #define variables
    query = PQL()
    # add case table key to query
    query += [PQLColumn(f"{list(case_key.values())[0]}",list(case_key.keys())[0])]
    #Add feature columns to input table
    query += add_selected_table_columns_to_pql(datamodel, tables_to_include)
    #Add additional columns
    if additional_columns != None:
        for c in additional_columns:
            query += [PQLColumn(additional_columns.get(c),f"{c}")]
    #Add target
    if target != None:
        query += [PQLColumn(f"{list(target.values())[0]}",list(target.keys())[0])]
    #Add filter
    if filter != None:
        query += [PQLFilter(filter)]
    #Create input table with query
    context_table = datamodel.get_data_frame(query)
    #Set primary keys as index
    context_table = context_table.set_index(list(case_key.keys())[0])

    return context_table


def get_activity_count_table(datamodel: Datamodel, case_key: dict, activity_table:str, activity_name:str, filter: str = None):
    '''
        Returns a pd.Dataframe with the casekey as index, each activity with counts as columns
        input:
            datamodel(Datamodel) - the datamodel of the process
            case_key(['column_name','pql_statement']) - index of the table
            activity_table(str) - name of activity table in datamodel
            activity_name(str) - name of column with activity names
        output:
            pd.Dataframe with the casekey as index, each activity with counts as columns
    '''
    #define variables
    query = PQL()
    # add case table key to query
    query += [PQLColumn(f"{list(case_key.values())[0]}",list(case_key.keys())[0])]
    # Add activities
    query += [PQLColumn(f'"{activity_table}"."{activity_name}"', f'{activity_table}.{activity_name}')]
    query += [PQLColumn('1','ONE')]
    #Add filter
    if filter != None:
        query += [PQLFilter(filter)]
    # Create table with query
    act_table = datamodel.get_data_frame(query)
    # Create activity_count_table
    act_ind = list(case_key.keys())[0]
    activity_count_table = pd.pivot_table(act_table, values='ONE', index=[act_ind], columns=[f'{activity_table}.{activity_name}'], aggfunc=np.sum)

    return activity_count_table


def get_mfeature_count_table(datamodel: Datamodel, case_key: dict, multiple_value_features: list, filter: str = None):
    '''
        Returns a pd.Dataframe with the casekey as index and the column with an 1:N relationship to the case table one hot encoded with count as aggfunc
        input:
            datamodel(Datamodel) - the datamodel of the process
            case_key(['column_name','pql_statement']) - index of the table
            multi_val_feature(['column_name','pql_statement']) - column that should be one hot encoded
        output:
            pd.Dataframe with the casekey as index and the column with an 1:N relationship to the case table one hot encoded with count as aggfunc
    '''   
    #define variables
    query = PQL()
    # add case table key to query
    query += [PQLColumn(f"{list(case_key.values())[0]}",list(case_key.keys())[0])]
    # Add multiple value features
    if multiple_value_features != None:
        for c in multiple_value_features:
            query += [PQLColumn(multiple_value_features.get(c),f"{c}")]
    query += [PQLColumn('1','ONE')]
    #Add filter
    if filter != None:
        query += [PQLFilter(filter)]
    # Create table with query
    mfeat_table = datamodel.get_data_frame(query)
    # Create activity_count_table
    mfeature_count_table = pd.pivot_table(mfeat_table, values='ONE', index=[list(case_key.keys())[0]], columns=[f"{list(multiple_value_features.keys())[0]}"], aggfunc=np.sum)

    return mfeature_count_table


def get_days_between_table(datamodel: Datamodel, case_key: dict, events_days_between: dict, filter:str = None):
    '''
        Returns a pd.Dataframe with the casekey as index and columns that include the days between different dates
        input:
            datamodel(Datamodel) - the datamodel of the process
            case_key(dict) - index of the table
            events_days_between(dict) - dates for which the days between are calculated
            filter(str) - filter for the data
        output:
            pd.Dataframe with the casekey as index and the days between every event as a feature
    ''' 
    #define variables
    query = PQL()
    # add case table key to query
    query += [PQLColumn(f"{list(case_key.values())[0]}",list(case_key.keys())[0])]
    # Get list of date combinations
    date_pairs = [((a,events_days_between.get(a)), (b,events_days_between.get(b))) for idx, a in enumerate(list(events_days_between)) for b in list(events_days_between)[idx + 1:]]
    # Add Workdays Between Features
    for pair in date_pairs:
        query += [PQLColumn(f" WORKDAYS_BETWEEN({pair[1][1]},{pair[0][1]},WEEKDAY_CALENDAR( MONDAY TUESDAY WEDNESDAY THURSDAY FRIDAY ))",'Workdays between: '+pair[1][0]+' / '+pair[0][0])]
    #Add filter
    if filter != None:
        query += [PQLFilter(filter)]
    #Create input table with query
    days_between_table = datamodel.get_data_frame(query)
    # Set primary keys as index
    days_between_table = days_between_table.set_index(list(case_key.keys())[0])

    return days_between_table


def add_delay_indicator(input_table: pd.DataFrame, target_date:str, actual_date:str):
    '''
        Add a delay indicator (1: actual_date-target_date > 0; 0: actual_date-target_date <= 0) to the input table and returns the whole table
        input:
            input_table(pd.Dataframe) - the table to which the indicator should be added
            target_date(str) - column name of target date in input_table
            actual_date(str) - column name of actual date in input_table
        output:
            input table with delay_indicator 
    ''' 
    input_table_func = input_table.copy()
    input_table_func['deviation'] = (input_table_func[actual_date] - input_table_func[target_date]).dt.days
    # Classify into OnTime/Early/Late
    input_table_func.loc[input_table_func['deviation'] > 0, 'delay_indicator'] = 'Late'
    input_table_func.loc[input_table_func['deviation'] <= 0, 'delay_indicator'] = 'OnTime'

    return input_table_func


def drop_irrelevant_data(input_table: pd.DataFrame,target_name:str):
    '''
        Drop rows of input_table where target is None; set values with frequency < 0.001 per column to None; delete columns with no value or only one distinct value
        input:
            input_table(pd.Dataframe) - the table to which the indicator should be added
            target_name(str) - column name of target
        output:
            reduced input table without "irrelevant" data
    '''
    input_table_func = input_table.copy()
    # drop all incomplete cases
    features = [feat for feat in list(input_table_func) if feat != target_name]
    input_table_func=input_table_func.dropna(subset = [target_name])
    # delete low frequency values
    cat_features = [feat for feat in features if input_table_func[feat].dtypes != float]
    threshold = 0.001*len(input_table_func)
    for col in input_table_func[cat_features].columns:
        value_counts = input_table_func[col].value_counts()
        vals_to_remove = value_counts[value_counts <= threshold].index.values
        input_table_func[col].loc[input_table_func[col].isin(vals_to_remove)] = None
    # drop columns with only one distinct value
    input_table_func = input_table_func[[c for c in list(input_table_func) if len(input_table_func[c].unique()) > 1]]
    # drop duplicate columns
    duplicate_cols = getDuplicateColumns(input_table_func)
    input_table_func = input_table_func.drop(duplicate_cols, axis = 1)

    return input_table_func
    

def handle_missing_data(input_table: pd.DataFrame,target_name:str):
    '''
        Set None values of float columns to 0.0; set None values of columns where dtype != float to 'missing'
        input:
            input_table(pd.Dataframe) - the table to which the indicator should be added
            target_name(str) - column name of target
        output:
            input table without None values
    '''
    input_table_func = input_table.copy()
    features = [feat for feat in list(input_table_func) if feat != target_name]
    cat_features = [feat for feat in features if input_table_func[feat].dtypes != float]
    num_features = [feat for feat in features if input_table_func[feat].dtypes == float]
    daybtw_features = [feat for feat in num_features if '_TO_' in feat]
    # Convert None values of day between features to the columns median value
    input_table_func[daybtw_features] = input_table_func[daybtw_features].fillna(input_table_func.median())
    # Convert None values of float feature columns to 0.0
    input_table_func[num_features] = input_table_func[num_features].fillna(0.0)
    # Convert None values of categorical feature columns to category: 'missing'
    input_table_func[cat_features] = input_table_func[cat_features].fillna('missing')

    return input_table_func


def encode_cat_features(input_table: pd.DataFrame):
    '''
        Encode categorical features to random numbers
        input:
            input_table(pd.Dataframe) - the table which categorical features should be encoded
        output:
            input table without encoded categorical features
    '''
    input_table_func = input_table.copy()
    # Encode categorical variables
    le=LabelEncoder()
    for col in input_table_func.columns:
        if input_table_func[col].dtype == 'O':
            input_table_func[col]=le.fit_transform(input_table_func[col]).astype('str')

    return input_table_func




