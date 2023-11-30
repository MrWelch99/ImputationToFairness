from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import nan_euclidean_distances
from algorithms.baseline.Generic import Generic

class KNN_Classifier(Generic):
    def __init__(self, n_neighbors=10, metric='precomputed'):
        Generic.__init__(self)
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        self.name = "KNN_Classifier"
    
    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs, privileged_vals, params):
        # remove sensitive attributes from the training set
        train_df_nosensitive = train_df.drop(columns = sensitive_attrs)
        test_df_nosensitive = test_df.drop(columns = sensitive_attrs)

        # create and train the classifier
        classifier = self.get_classifier()
        y = train_df_nosensitive[class_attr]
        X = train_df_nosensitive.drop(columns = class_attr)

        # Get nan_euclidean_dist
        X_dist = nan_euclidean_distances(X, X)
        print(X_dist)
        print(X_dist.shape)
        classifier.fit(X_dist, y)

        # get the predictions on the test set
        X_test = test_df_nosensitive.drop(class_attr, axis=1)
        X_test_dist = nan_euclidean_distances(X_test,X)
        print(X_test_dist.shape)
        predictions = classifier.predict(X_test_dist)
        prob_predictions = classifier.predict_proba(X_test_dist)
        prob_predictions = prob_predictions[:, 1]

        return predictions, prob_predictions, [], X_test

