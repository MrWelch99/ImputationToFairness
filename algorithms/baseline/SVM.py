from sklearn.svm import SVC as SKLearn_SVM
from algorithms.baseline.Generic import Generic

class SVM(Generic):
    def __init__(self, c=1, gamma = "scale"):
        Generic.__init__(self)
        self.classifier = SKLearn_SVM(C= c, gamma = gamma, probability=True)
        self.name = "SVM"
