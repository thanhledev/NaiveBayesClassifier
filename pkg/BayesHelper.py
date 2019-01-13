import pandas as pd
from typing import List
from pkg.FeatureAttrProbEntry import FeatureAttrProbEntry
from pkg.FeatureProbTable import FeatureProbTable
from pkg.ClassVariableEntry import ClassVariableEntry


def get_conditional_probability(class_variable_value: str, tables: List[FeatureProbTable],
                                dependant_feature: {}) -> float:

    for key in dependant_feature.keys():
        for table in tables:
            if table.featureName == key:
                return table.get(dependant_feature[key], class_variable_value)
    return 0.0


def get_class_probability(class_variable_value: str, class_variable_prob_lst:List[ClassVariableEntry]) -> float:

    for entry in class_variable_prob_lst:
        if entry.value == class_variable_value:
            return entry.prob
    return 0.0


def build_class_prob_list(dataframe: pd.DataFrame, class_name: str) -> List[ClassVariableEntry]:
    # get all unique values of the given class variable
    class_variable_values = dataframe[class_name].unique()

    # calculate probability and store to list
    class_variable_entries = []
    total_samples = dataframe.shape[0]
    class_variable_frequencies = dataframe[class_name].value_counts()

    for value in class_variable_values:
        class_variable_entries.append(ClassVariableEntry(value, class_variable_frequencies[value] / total_samples))

    for entry in class_variable_entries:
        print(entry.description())

    return class_variable_entries


def build_condition_prob_table(dataframe: pd.DataFrame, feature_name: str, class_name: str) -> FeatureProbTable:

    # get all unique values of the given class variable
    class_variable_values = dataframe[class_name].unique()

    # create table
    feature_table = FeatureProbTable(feature_name)

    # get all unique values of the given feature
    feature_values = dataframe[feature_name].unique()

    for f_value in feature_values:
        attr_prob_entry = FeatureAttrProbEntry(f_value, [])

        for c_value in class_variable_values:
            # get probability of given c_value
            c_prob_of_f_value = dataframe.iloc[dataframe.index[(dataframe[feature_name] == f_value)
                                   & (dataframe[class_name] == c_value)]].shape[0] / \
                                len(dataframe.index[dataframe[class_name] == c_value])

            attr_prob_entry.add_prob_by_class_value(ClassVariableEntry(c_value, c_prob_of_f_value))

        feature_table.append(attr_prob_entry)

    return feature_table
