#!/usr/bin/env python
# coding: utf-8


from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import shap
from catboost import Pool, CatBoostClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from pycelonis.celonis_api.event_collection.data_model import Datamodel
from utils import *


def get_rf_feature_importances(rf_model: RandomForestClassifier, X_train: pd.DataFrame, X_test: pd.DataFrame = None, y_test: pd.Series = None, eval_metric:str='impurity'):
    '''
        Returns a pd.Dataframe with the case key as index and the feature importance values; the user can choose between 'impurity' and 'permutation' for the evaluation metric
        input:
            datamodel(Datamodel) - the datamodel of the process
            X_train(Dataframe) - the train dataset
            X_test(Dataframe) - the test dataset
            y_test(Dataframe) - the test target
            eval_metric(str) - the evaluation metric on which the importance is calculated 'impurity' or 'permutation'
        output:
            pd.Dataframe with the case key as index and the feature importance values sorted by the feature importance descending
    '''
    if eval_metric == 'impurity':
        importances = rf_model.feature_importances_
        feature_importances = pd.DataFrame(rf_model.feature_importances_
                                      ,index = X_train.columns
                                      ,columns=['importance']
                                     ).sort_values('importance',ascending=False)
    elif eval_metric == 'permutation':
        if X_test == None or y_test == None:
            raise ValueError(f"For eval_metric == 'permutation', X_test and y_test inputs are necessary.")
        importances = permutation_importance(rf_model
                                        ,X_test
                                        ,y_test
                                        # ,random_state=42
                                        ,n_repeats = 3
                                        ,n_jobs=-2)
        feature_importances = pd.DataFrame(importances, index = X_train.columns, columns=['importances_mean']).sort_values('importances_mean',ascending=False)
    else:
        raise ValueError(f"Invalid eval_metric value. Possible values:['impurity','permutation']")
    
    return feature_importances


def get_selected_features(rf_model: RandomForestClassifier, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, feature_importance_method: str='impurity', threshold: float = None, feature_number_to_select: int = 100):
    '''
        Returns a list with features selected by the Random Forest Feature Importance Method limited by the number of features of an importance threshold.
        If number and threshold is set, the more restrictive rule is applied.
        input:
            rf_model(RandomForestClassifier) - trained Random Forest model
            X_train(Dataframe) - the train dataset
            X_test(Dataframe) - the test dataset
            y_test(Dataframe) - the test target
            threshold(float) - threshold of feature importance value that limits the features to be selected
            feature_number_to_select(int) - maximum number of features to be selected
            feature_importance_method(str) - the evaluation metric on which the importance is calculated 'impurity' or 'permutation'

        output:
            pd.Dataframe with the case key as index and the feature importance values sorted by the feature importance descending
    '''
    feature_importances = get_rf_feature_importances(rf_model, X_train, X_test, y_test, feature_importance_method)
    if feature_number_to_select == None and threshold != None:
        selected_features = feature_importances[feature_importances.importance>= threshold].index.to_list()
    elif feature_number_to_select != None and threshold == None:
        selected_features = feature_importances[:feature_number_to_select].index.to_list()
    elif feature_number_to_select != None and threshold != None:
        selected_features = feature_importances[:feature_number_to_select][feature_importances.importance>= threshold].index.to_list()
    else: 
        raise ValueError(f"Either a value for threshold or feature_number_to_select must be set.")
        
    return selected_features          


def get_grid_search_results(input_table: pd.DataFrame, target_name: str, features: list, cat_features: list, train_pool: Pool, learning_rate:list, depth:list, l2_leaf_reg:list):
    '''
        Returns the results of a grid search run, as well as the CatBoostClassifier object with the parameters of the best grid search result.
        input:
            input_table(DataFrame) - input data set
            target_name(str) - column name of target
            features(list) - list of feature names
            train_pool(Pool) - train Pool of CatBoost
            learning_rate(list) - parameters of learning rate that should be searched
            depth(list) - parameters of tree depths that should be searched
            l2_leaf_reg(list) - parameters of L2 regularization term that should be searched
        output:
            grid search results, CatBoostClassifier object with best parameters
    '''
    model = CatBoostClassifier(iterations=1, loss_function='Logloss')
    grid_search_label=input_table[target_name]
    grid_search_data=input_table[features]
    grid_search_pool = Pool( grid_search_data
                            ,grid_search_label
                            ,cat_features)

    grid = {'learning_rate': learning_rate,
            'depth': depth,
            'l2_leaf_reg': l2_leaf_reg,
            }

    grid_search_result = model.grid_search(grid,
                train_pool,
                cv=3,
                calc_cv_statistics=True,
                search_by_train_test_split=True,
                refit=True,
                shuffle=True,
                train_size=0.9,
                verbose=True,
                plot=True)

    model = CatBoostClassifier(
                        depth=grid_search_result["params"]["depth"],
                        learning_rate=grid_search_result["params"]["learning_rate"],
                        l2_leaf_reg=grid_search_result["params"]["l2_leaf_reg"])
                    
    return model, grid_search_result


### Model performance ###

def get_performance_measures(model,X_test,y_test):
    '''
        Returns performance measures of the model: accuracy, precision, recall, f1-score, roc auc score, classification report
        input:
            model(MLmodel) - the fitted ML model, e.g.RandomForestClassifier
            X_test(Dataframe) - the test dataset
            y_test(Dataframe) - the test target
        output:
            dictionary with accuracy, precision, recall, f1-score, roc auc score, classification report
    '''
    from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
    actual = y_test
    pred = model.predict(X_test)
    classification_report = classification_report(actual, pred, output_dict=True)
    precision_score = round(precision_score(actual, pred),4)
    recall_score = round(recall_score(actual, pred),4)
    f1_score = round(f1_score(actual, pred),4)
    accuracy_score = round(accuracy_score(actual, pred),4)
    roc_auc_score = round(roc_auc_score(actual, pred),4)
    ConfusionMatrixDisplay.from_predictions(actual, pred)

    return {'precision_score': precision_score, 'recall_score': recall_score, 'f1_score': f1_score,
    'accuracy_score': accuracy_score, 'roc_auc_score': roc_auc_score, 'classification_report': classification_report}


### SHAP Evaluation ###

def get_shapley_value_table(feature_table: pd.DataFrame, target_name: str, model, X: pd.DataFrame, y: pd.DataFrame, summary_plot: boolean = False):
    '''
        Returns pd.Dataframe with case key as index and shapley values for each feature
        input:
            feature_table(Dataframe) - table that includes cases (rows) and features (columns)
            target_name(str) - column name of target
            model(MLmodel) = the fitted ML model, e.g.CatBoostClassifier
            X(Dataframe) - the complete dataset without the target
            y(Dataframe) - the target for all data
            summary_plot(boolean) - if true, plots summary plot of shapley values
        output:
            pd.Dataframe with case key as index and shapley values for each feature
    '''
    # Get feature lists
    features = [feat for feat in list(feature_table) if feat != target_name]
    cat_features = [feat for feat in features if feature_table[feat].dtypes != float]
    # Calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Pool(X, y, cat_features),check_additivity=True)
    # Display summary plot
    if(summary_plot == True):
        shap.summary_plot(shap_values, X)
    # Build table with shapley values
    shap_table = pd.DataFrame(shap_values, columns = features, index=X.index)
    # Convert shap_values to probability values by applying sigmoid function
    # shap_table = shap_table.apply(np.vectorize(sigmoid))
    # Append prediction values
    preds = model.predict_proba(Pool(X, y, cat_features))
    preds_1 = [a_tuple[1] for a_tuple in preds]
    shap_table['pred_value'] = list(np.around(np.array(preds_1),4))

    return shap_table


def get_reasons(shap_table: pd.DataFrame, feature_table: pd.DataFrame, target_name:str, amount:int, datamodel: Datamodel = None):
    '''
        Returns the N selected dependencies per case, including the feature name, feature value, shap value, feature weight, pretty name and forecast confindence
        input:
            shap_table(Dataframe) - shap table from get_shapley_value_table()
            feature_table(Dataframe) - table that includes cases (rows) and features (columns)
            target_name(str) - column name of target
            amount(int) = amount of reasons that should be output 
            datamodel(Datamodel) - the datamodel of the process
        output:
            dependency table and top dependencies (by frequency)
    '''
    # Prepare variables
    shap_table_r = shap_table.copy()
    shap_table_cols = shap_table_r.columns.drop('pred_value')
    max_features,max_values,max_shaps,reason_weights,reasons,pretty_names = [],[],[],[],[],[]
    # Determine n features with highest shap value per case
    for num in range(amount):
        shap_table_r['MAX'+str(num+1)] = shap_table_r[shap_table_cols].T.apply(lambda x: x.nlargest(num+1)).idxmin()
        max_features = max_features+['MAX'+str(num+1)]
        max_values = max_values+['MAX'+str(num+1)+'_VALUE']
        max_shaps = max_shaps+['MAX'+str(num+1)+'_SHAP']
        reason_weights = reason_weights+['reason_weight'+str(num+1)]
        pretty_names = pretty_names+['pretty_names'+str(num+1)]
        reasons = reasons+['reason'+str(num+1)]
    # Add SHAP values for n features
    shap_table_r[max_shaps] = shap_table_r.apply(find_max_values, axis=1, amount = amount, result_type='expand')
    # Add column with sum of all postivie SHAP values
    shap_table_r['sum_shaps'] = shap_table_r[shap_table_cols][shap_table_r[shap_table_cols]>0].sum(1)
    # Add reason weight metric
    for num in range(amount):
        shap_table_r['reason_weight'+str(num+1)] = np.around((shap_table_r['MAX'+str(num+1)+'_SHAP']/shap_table_r['sum_shaps']),4)       
    # Add target to filter delayed cases
    shap_table_r = pd.concat([shap_table_r, feature_table[target_name]], axis=1, join="inner")
    shap_table_r = shap_table_r[shap_table_r[target_name] == 1.0]
    shap_table_r=shap_table_r.drop([target_name], axis=1)
    # Join Max features to input_table
    reason_table = pd.concat([feature_table, shap_table_r[['pred_value']+max_features+max_shaps+reason_weights]], axis=1, join="inner") 
    # Get values of 3 Max features
    reason_table[max_values] = reason_table.apply(find_max_values, axis=1, amount = amount, result_type='expand')
    reason_table[reasons] = reason_table.apply(concat_max_values, axis=1, amount = amount, result_type='expand')
    reason_table = reason_table[['pred_value']+max_features+max_shaps+max_values+reasons+reason_weights]
    # Name Mapping
    if datamodel != None:
        reason_table[pretty_names] = apply_name_mapping_to_series(reason_table[max_features], datamodel)
    # Determine global top reasons by counting all distinct Max1-3 reasons
    top_reasons = reason_table[reasons].stack().value_counts(ascending = False)

    return reason_table, top_reasons


def get_stat_table(feature_table: pd.DataFrame, target_name:str):
    '''
        Returns pd.Dataframe with index = feature+value and measures from descriptive statistics
        input:
            feature_table(Dataframe) - table that includes cases (rows) and features (columns)
            target_name(str) - column name of target
        output:
            pd.Dataframe with index = feature+value and measures from descriptive statistics
    '''    
    # prepare variables
    count = lambda x: len(x)
    stat_table = pd.DataFrame([])
    case_number = len(feature_table)
    delay_case_number = len(feature_table[feature_table[target_name] == 1.0].index)
    on_time_case_number = len(feature_table[feature_table[target_name] == 0.0].index)
    features = [feat for feat in list(feature_table) if feat != target_name]
    # create dataframe with delay count
    for feature in features:
        df = feature_table.assign(ind=feature_table[target_name]).pivot_table(values=target_name,index=[feature], columns=['ind'], fill_value=0, aggfunc=count)
        df = df.assign(column=feature)
        stat_table = stat_table.append(df)
    # reset index
    stat_table = stat_table.set_index('column', append = True)
    stat_table.index = stat_table.index.set_names(['Value','Column']).swaplevel()
    # rename columns
    stat_table = stat_table.rename(columns={1.0: "Late", 0.0: "OnTime"})    
    # add KPIs to dataframe
    ft_columns = list(stat_table.columns)
    # Frequency: In how many cases do the value of the feature occure in general?
    if 'frequency' in stat_table.columns:
        stat_table = stat_table.drop(columns=['frequency'])
    stat_table['frequency'] = round((stat_table.iloc[:, [ft_columns.index('Late')]].squeeze()+stat_table.iloc[:, [ft_columns.index('OnTime')]].squeeze()).div(case_number),4)
    # On Time Frequency: How many on time cases have the value of the feature?
    if 'on_time_frequency' in stat_table.columns:
        stat_table = stat_table.drop(columns=['on_time_frequency'])
    stat_table['on_time_frequency'] = round(stat_table.iloc[:, [ft_columns.index('OnTime')]].div(on_time_case_number),4)
    # Delay Frequency: How many delayed cases have the value of the feature?
    if 'delay_frequency' in stat_table.columns:
        stat_table = stat_table.drop(columns=['delay_frequency'])
    stat_table['delay_frequency'] = round(stat_table.iloc[:, [ft_columns.index('Late')]].div(delay_case_number),4)
    # update ft_columns
    ft_columns = list(stat_table.columns)
    # Delay Frequency / On Time Frequency: If > 1 --> If the case has this feature value it is more likely to be delayed than on time
    if 'DF_OTF' in stat_table.columns:
        stat_table = stat_table.drop(columns=['DF_OTF'])
    stat_table['DF_OTF'] = round(stat_table.iloc[:, [ft_columns.index('delay_frequency')]].squeeze().div(stat_table.iloc[:, [ft_columns.index('on_time_frequency')]].squeeze()),4)
    # Failure Rate: # delayed cases with this feature value / # all cases with this feature value
    if 'failure_rate' in stat_table.columns:
        stat_table = stat_table.drop(columns=['failure_rate'])
    stat_table['failure_rate'] = round(stat_table.iloc[:, [ft_columns.index('Late')]].squeeze().div(stat_table.iloc[:, [ft_columns.index('Late')]].squeeze()+stat_table.iloc[:, [ft_columns.index('OnTime')]].squeeze()),4)
    # Lift: % delays with this feature value / % delays in general
    if 'lift' in stat_table.columns:
        stat_table = stat_table.drop(columns=['lift'])
    stat_table['lift'] = round(stat_table.iloc[:, [ft_columns.index('delay_frequency')]].squeeze().div(delay_case_number/case_number),4)
    # Clean Table
    # Drop infinite values (divided by 0)
    stat_table = stat_table.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    # Sort table by lift desc
    stat_table = stat_table.sort_values(by='lift', ascending=False)
    
    return stat_table






