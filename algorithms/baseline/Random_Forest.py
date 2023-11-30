from sklearn.ensemble import RandomForestClassifier
from algorithms.baseline.Generic import Generic

class Random_Forest(Generic):
    def __init__(self, n_estimators = 100, max_depth= None, max_features = "auto"):
        Generic.__init__(self)
        self.classifier = RandomForestClassifier(n_estimators = n_estimators, max_depth= max_depth, max_features=max_features)
        self.name = "Random_Forest"
