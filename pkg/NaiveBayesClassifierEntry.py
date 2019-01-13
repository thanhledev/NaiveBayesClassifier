from typing import List, Dict


class NaiveBayesClassifierEntry:
    # Initializer / Instance attributes:
    def __init__(self, feature_vectors: {}, origin_class_value: {}, nb_class_value: {}):
        self.feature_vectors = feature_vectors
        self.origin_class_value = origin_class_value
        self.nb_class_value = nb_class_value

    def description(self):
        return "Entry {} has original classifier value of {} and NaiveBayes classifier value of" \
               "{}\n".format(self.feature_vectors, self.origin_class_value, self.nb_class_value)

    def set_feature_vectors(self, feature_vectors: {}):
        self.feature_vectors = feature_vectors

    def set_origin_class_value(self, origin_class_value: {}):
        self.origin_class_value = origin_class_value

    def set_nb_class_value(self, class_variable: str, class_variable_value: str):
        self.nb_class_value[class_variable] = class_variable_value

    def get_feature_vectors(self) -> {}:
        return self.feature_vectors

    def get_origin_class_value(self) -> {}:
        return self.origin_class_value

    def get_nb_class_value(self) -> {}:
        return self.nb_class_value

    def is_entry_true_positive(self, class_variable: str, positive_values: []) -> bool:
        return self.origin_class_value[class_variable] in positive_values \
               and self.origin_class_value[class_variable] == self.nb_class_value[class_variable]

    def is_entry_false_negative(self, class_variable: str, positive_values: []) -> bool:
        return self.origin_class_value[class_variable] in positive_values \
               and self.origin_class_value[class_variable] != self.nb_class_value[class_variable]

    def is_entry_true_negative(self, class_variable: str, negative_values: []) -> bool:
        return self.origin_class_value[class_variable] in negative_values \
               and self.origin_class_value[class_variable] == self.nb_class_value[class_variable]

    def is_entry_false_positive(self, class_variable: str, negative_values: []) -> bool:
        return self.origin_class_value[class_variable] in negative_values \
               and self.origin_class_value[class_variable] != self.nb_class_value[class_variable]
