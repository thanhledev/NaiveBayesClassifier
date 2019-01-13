from typing import List
from pkg.ClassVariableEntry import ClassVariableEntry


class FeatureAttrProbEntry:
    # Initializer / Instance attributes:
    def __init__(self, value: str, entries: List[ClassVariableEntry]):
        self.attrValue = value
        self.entries = entries

    def description(self):
        desc = "{} has the conditional probability entries below:\n".format(self.attrValue)

        for entry in self.entries:
            desc += entry.description()

        return desc

    def add_prob_by_class_value(self, entry: ClassVariableEntry):
        self.entries.append(entry)

    def get_prob_by_class_value(self, class_value: str) -> float:
        for entry in self.entries:
            if entry.value == class_value:
                return entry.prob
        return None
