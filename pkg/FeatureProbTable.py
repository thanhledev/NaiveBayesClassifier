from pkg.FeatureAttrProbEntry import FeatureAttrProbEntry


class FeatureProbTable:

    # Initializer / Instance attributes:
    def __init__(self, name: str):
        self.featureName = name
        self.attributeProbEntries = []

    def append(self, entry: FeatureAttrProbEntry):
        self.attributeProbEntries.append(entry)

    def get(self, attr_value: str, class_value: str) -> float:
        for attrEntry in self.attributeProbEntries:
            if attrEntry.attrValue == attr_value:
                return attrEntry.get_prob_by_class_value(class_value)
        return None
