import pandas as pd
import numpy as np
from operator import itemgetter
from pkg.BayesHelper import *
from pkg.FeatureProbTable import FeatureProbTable
from pkg.NaiveBayesClassifierEntry import NaiveBayesClassifierEntry


def classify_entry(entry: NaiveBayesClassifierEntry, class_variable: str, class_variable_values: [],
                   class_variable_prob_lst: [], conditional_probability_tables: List[FeatureProbTable])\
        -> NaiveBayesClassifierEntry:

    classifier_entry = entry
    class_variable_value_prob = {}

    # for each of class_variable value
    for class_value in class_variable_values:
        # calculate the probability of this class_variable value
        conditional_probability = 1
        for dependant_feature in entry.feature_vectors.keys():
            conditional_probability *= get_conditional_probability(class_value, conditional_probability_tables,
                                                   {dependant_feature: entry.feature_vectors[dependant_feature]})

        class_variable_value_prob[class_value] = conditional_probability * get_class_probability(class_value,
                                                        class_variable_prob_lst)

    highest_probability = sorted(class_variable_value_prob.items(), key=itemgetter(1), reverse=True)[0]

    classifier_entry.set_nb_class_value(class_variable, highest_probability[0])

    return classifier_entry


def naive_bayes_classifier(training_data: str, test_data: str, output: str, class_variable: str,
                           positive_values: [], negative_values: [], is_indexed: bool):

    # read csv files to dataframes
    #training_dataframe = pd.read_csv(training_data, na_values=["."])
    training_dataframe = pd.read_csv(training_data, na_values=["."], index_col=0) if is_indexed else \
        pd.read_csv(training_data, na_values=["."])

    #test_dataframe = pd.read_csv(test_data, na_values=["."])
    test_dataframe = pd.read_csv(test_data, na_values=["."], index_col=0) if is_indexed else \
        pd.read_csv(test_data, na_values=["."])

    # read all features
    dependent_features = [name for name in training_dataframe.columns.tolist() if name != class_variable]

    # get the unique values of class_variable
    class_variable_values = training_dataframe[class_variable].unique()

    # build class probability list
    class_variable_prob_lst = build_class_prob_list(training_dataframe, class_variable)

    # create conditional probability tables
    conditional_probability_tables = []

    # for each of dependent features, build its conditional probability table
    for feature in dependent_features:
        conditional_probability_tables.append(build_condition_prob_table(training_dataframe, feature, class_variable))

    # inspection values
    true_positive = true_negative = false_positive = false_negative = 0

    # for each row of the test data
    for index, row in test_dataframe.iterrows():
        # create a NaiveBayesClassifierEntry object
        test_row_dict = row.to_dict()
        feature_dict = {}
        for key, value in test_row_dict.items():
            if key != class_variable:
                feature_dict[key] = value

        entry = NaiveBayesClassifierEntry(feature_dict, {class_variable: test_row_dict[class_variable]},
                                          {class_variable: 'n/a'})

        print("Before applying NaiveBayes classification algorithm:" + entry.description())

        # begin to classifier the entry
        entry = classify_entry(entry, class_variable, class_variable_values, class_variable_prob_lst,
                               conditional_probability_tables)

        print("After applying NaiveBayes classification algorithm:" + entry.description())

        # update inspection values
        if entry.is_entry_true_positive(class_variable, positive_values):
            true_positive += 1

        if entry.is_entry_true_negative(class_variable, negative_values):
            true_negative += 1

        if entry.is_entry_false_positive(class_variable, negative_values):
            false_positive += 1

        if entry.is_entry_false_negative(class_variable, positive_values):
            false_negative += 1

        print("Inspection values: TP={} FN={} FP={} TN={}".format(true_positive, false_negative,
                                                                  false_positive, true_negative))

    # print required algorithm statistics
    print("Accuracy: {0:.0%}".format((true_positive + true_negative) /
                                     (true_positive + true_negative + false_positive + false_negative)))

    print("Precision: {0:.0%}".format(true_positive /
                                     (true_positive + false_positive)))

    print("Recall: {0:.0%}".format(true_positive /
                                (true_positive + false_negative)))
