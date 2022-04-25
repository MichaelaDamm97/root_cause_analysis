#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import math
from pycelonis.celonis_api.pql.pql import PQL, PQLColumn
from pycelonis.celonis_api.event_collection.data_model import Datamodel




def get_feature_lists(input_table: pd.DataFrame,target_name:str):
    # define features, cat_feautures, features_with_target
    features = [feat for feat in list(input_table) if feat != target_name]
    cat_features = [feat for feat in features if input_table[feat].dtypes != float]
    num_features = [feat for feat in features if input_table[feat].dtypes == float]
    features_with_target=features+[target_name]
    
    return features, cat_features, num_features, features_with_target


def find_max_values(row,amount:int):
    max_values = []
    for num in range(amount):
        max_values.append(row[row['MAX'+str(num+1)]])

    return max_values


def concat_max_values(row,amount:int):
    max_concats = []
    for num in range(amount):
        max_concats.append(row['MAX'+str(num+1)] + ' = ' + str(row['MAX'+str(num+1)+'_VALUE']))

    return max_concats


# Source of the method getDuplicateColumns: ankthon - geeksforgeeks 07/02/2020
# https://www.geeksforgeeks.org/how-to-find-drop-duplicate-columns-in-a-pandas-dataframe/#:~:text=Code%201%3A%20Find%20duplicate%20columns,in%20the%20duplicate%20column%20set.
def getDuplicateColumns(df):
    duplicateColumnNames = set()
    for x in range(df.shape[1]):
        col = df.iloc[:, x]
        for y in range(x + 1, df.shape[1]):
            otherCol = df.iloc[:, y]
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])

    return list(duplicateColumnNames)


def apply_name_mapping_to_series(series: pd.Series, datamodel: Datamodel, lang='EN'):
    '''
        Use name mapping from data model to get pretty names for Series
    '''
    def get_mapping_dict(datamodel, lang='EN'):
        name_mapping = datamodel.name_mapping
        mapping = {}
        for map_dict in name_mapping:
            if map_dict['mappingType'] == 'column-mapping' and map_dict['language'] == lang:
                mapping[map_dict['identifier']] = map_dict['translation']
                mapping[map_dict['identifier']+'_WEEKDAY'] = map_dict['translation']+' (Weekday)'
                mapping[map_dict['identifier']+'_DAY'] = map_dict['translation']+' (Day)'
                mapping[map_dict['identifier']+'_MONTH'] = map_dict['translation']+' (Month)'
        return mapping

    mapping = get_mapping_dict(datamodel, lang)
    return series.replace(mapping) 

