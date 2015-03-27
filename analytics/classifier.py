# Code for the Kaggle AXA challenge
#-----------------------------------
#    Copyright (C) 2015- by
#    Frank Lin <email@>
#    Jim Simpson <email@>
#    All rights reserved.
#    BSD license.
#-----------------------------------

import os
from sklearn import ensemble, linear_model, mixture, grid_search


class Classifier(object):

    """The classifier provides a unified API to a variety of classifiers.

    This class serves the purpose of abstracting the classifier implementation
    and providing the ability to easily switch and combine different
    classifier.

    Note:
      The main public methods in this class are
      gridsearch_model(),
      train_model(), and
      predict_model()

    Args:
      None

    Attributes:
      None

    """

    def __init__(self, classifier_name=''):
        # Save classifier name
        self.classifier_name = classifier_name

        # Init classifier
        if self.classifier_name == 'RandomForestClassifier':
            # Init RandomForestClassifier
            self.clf = ensemble.RandomForestClassifier(
                random_state=None,
                n_estimators=500,
                max_features='sqrt')
        elif self.classifier_name == 'GradientBoostingClassifier':
            # Init GradientBoostingClassifier
            self.clf = ensemble.GradientBoostingClassifier()
        elif self.classifier_name == 'LogisticRegression':
            # Init LogisticRegression
            self.clf = linear_model.LogisticRegression(
                random_state=None,
                class_weight='auto')
        elif self.classifier_name == 'GMM':
            # Init GMM
            self.clf = mixture.GMM(random_state=None)
        else:
            # Error
            print "ERROR: Incorrect classifier name"

    def fit(self, X_train, y_train):
        """Fit the classifier."""

        # Fit classifier
        if self.classifier_name == 'RandomForestClassifier':
            # Fit RandomForestClassifier
            self.clf.fit(X_train, y_train)
        elif self.classifier_name == 'GradientBoostingClassifier':
            # Fit GradientBoostingClassifier
            self.clf.fit(X_train, y_train)
        elif self.classifier_name == 'LogisticRegression':
            # Fit LogisticRegression
            self.clf.fit(X_train, y_train)
        elif self.classifier_name == 'GMM':
            # Fit GMM
            self.clf.fit(X_train)
        else:
            # Error
            print "ERROR: Incorrect classifier name"

    def score(self, X, y):
        """Score the trained model."""

        # Score model
        if self.classifier_name == 'RandomForestClassifier':
            # Score RandomForestClassifier
            self.clf_score = self.clf.score(X, y)
        elif self.classifier_name == 'GradientBoostingClassifier':
            # Score GradientBoostingClassifier
            self.clf_score = self.clf.score(X, y)
        elif self.classifier_name == 'LogisticRegression':
            # Score LogisticRegression
            self.clf_score = self.clf.score(X, y)
        elif self.classifier_name == 'GMM':
            # Score GMM
            # TODO: Add xor to score GMM
            self.clf_score = 0
        else:
            # Error
            print "ERROR: Incorrect classifier name"

        return self.clf_score

    def predict(self, X_true):
        """Predict using the classifier."""

        # Predict using the classifier
        if self.classifier_name == 'RandomForestClassifier':
            # Predict using the RandomForestClassifier
            self.y_pred = self.clf.predict(X_true)
        elif self.classifier_name == 'GradientBoostingClassifier':
            # Predict using the GradientBoostingClassifier
            self.y_pred = self.clf.predict(X_true)
        elif self.classifier_name == 'LogisticRegression':
            # Predict using the LogisticRegression
            self.y_pred = self.clf.predict(X_true)
        elif self.classifier_name == 'GMM':
            # Predict using the GMM
            self.y_pred = self.clf.predict(X_true)
        else:
            # Error
            print "ERROR: Incorrect classifier name"

        return self.y_pred

    def predict_proba(self, X_true):
        """Predict probability estimates using the classifier."""

        # Predict using the classifier
        if self.classifier_name == 'RandomForestClassifier':
            # Predict using the RandomForestClassifier
            self.y_pred = self.clf.predict_proba(X_true)
        elif self.classifier_name == 'GradientBoostingClassifier':
            # Predict using the GradientBoostingClassifier
            self.y_pred = self.clf.predict_proba(X_true)
        elif self.classifier_name == 'LogisticRegression':
            # Predict using the LogisticRegression
            self.y_pred = self.clf.predict_proba(X_true)
        elif self.classifier_name == 'GMM':
            # Predict using the GMM
            self.y_pred = self.clf.predict_proba(X_true)
        else:
            # Error
            print "ERROR: Incorrect classifier name"

        return self.y_pred[:, 1]  # Pr corresponding to class 1

    def feature_importances(self):
        """Feature Importances of the classifier."""

        # Feature Importances of the classifier
        if self.classifier_name == 'RandomForestClassifier':
            # Feature Importances of the  RandomForestClassifier
            self.feature_importances = self.clf.feature_importances_
        elif self.classifier_name == 'GradientBoostingClassifier':
            # Feature Importances of the  GradientBoostingClassifier
            self.feature_importances = self.clf.feature_importances_
        elif self.classifier_name == 'LogisticRegression':
            # Feature Importances of the  LogisticRegression
            # TODO: Get correct size
            self.feature_importances = [0]
        elif self.classifier_name == 'GMM':
            # Feature Importances of the  GMM
            # TODO: Get correct size
            self.feature_importances = [0]
        else:
            # Error
            print "ERROR: Incorrect classifier name"

        return self.feature_importances

    def gridsearch_model(self, X_train, y_train, X_test, y_test):
        """Gridsearch and train the model."""

        # Gridsearch and train the model
        if self.classifier_name == 'RandomForestClassifier':
            # Gridsearch and train the RandomForestClassifier
            parameters = {
                'min_samples_split': [2, 10, 20],
                'min_samples_leaf': [1, 5, 10],
                'bootstrap': [True, False]}

            # Create gridsearch object
            clf2 = grid_search.GridSearchCV(self.clf, parameters, n_jobs=1)

            # Fit gridsearch model
            clf2.fit(X_train, y_train)

            # See best params chosen
            print clf2.best_params_

            # Score best model
            self.clf_score_train = clf2.score(X_train, y_train)
            self.clf_score_test = clf2.score(X_test, y_test)

            # Set the retrained model to this object's model
            self.clf = clf2

        elif self.classifier_name == 'GradientBoostingClassifier':
            # Gridsearch and train the GradientBoostingClassifier
            todo = 1
        elif self.classifier_name == 'LogisticRegression':
            # Gridsearch and train the LogisticRegression

            # Set parameters
            parameters = {
                'penalty': ('l1', 'l2'),
                'C': [1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'tol': [1e-11, 1e-10, 1e-9, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 0.01]}

            # Create gridsearch object
            clf2 = grid_search.GridSearchCV(self.clf, parameters, n_jobs=1)

            # Fit gridsearch model
            clf2.fit(X_train, y_train)

            # See best params chosen
            print clf2.best_params_

            # Score best model
            self.clf_score_train = clf2.score(X_train, y_train)
            self.clf_score_test = clf2.score(X_test, y_test)

            # Set the retrained model to this object's model
            self.clf = clf2

        elif self.classifier_name == 'GMM':
            # Gridsearch and train the GMM

            # Set parameters
            parameters = {
                'n_components': [1, 10, 100],
                'covariance_type': ('spherical', 'tied', 'diag', 'full'),
                'thresh': [0.1, 0.01, 0.001],
                'n_iter': [10, 100, 1000]}

            # TODO: Must pass a custom scorer to do gridsearch on GMM.

        else:
            # Error
            print "ERROR: Incorrect classifier name"

        return self.clf_score_train, self.clf_score_test

    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the model."""

        # Train the model
        if self.classifier_name == 'RandomForestClassifier':
            # Train RandomForestClassifier
            self.fit(X_train, y_train)
            # Score RandomForestClassifier
            self.clf_score_train = self.score(X_train, y_train)
            self.clf_score_test = self.score(X_test, y_test)
        elif self.classifier_name == 'GradientBoostingClassifier':
            # Train GradientBoostingClassifier
            self.fit(X_train, y_train)
            # Score GradientBoostingClassifier
            self.clf_score_train = self.score(X_train, y_train)
            self.clf_score_test = self.score(X_test, y_test)
        elif self.classifier_name == 'LogisticRegression':
            # Train LogisticRegression
            self.fit(X_train, y_train)
            # Score LogisticRegression
            self.clf_score_train = self.score(X_train, y_train)
            self.clf_score_test = self.score(X_test, y_test)
        elif self.classifier_name == 'GMM':
            # Train GMM
            self.fit(X_train, y_train)
            # Score GMM
            self.clf_score_train = self.score(X_train, y_train)
            self.clf_score_test = self.score(X_test, y_test)
        else:
            # Error
            print "ERROR: Incorrect classifier name"

        return self.clf_score_train, self.clf_score_test

    def predict_model(self, X_true, proba=False):
        """Predict using the trained model."""
        if proba:
            # Predict using the trained model
            if self.classifier_name == 'RandomForestClassifier':
                # Predict using the trained RandomForestClassifier
                self.y_pred = self.predict_proba(X_true)
            elif self.classifier_name == 'GradientBoostingClassifier':
                # Predict using the trained GradientBoostingClassifier
                self.y_pred = self.predict_proba(X_true)
            elif self.classifier_name == 'LogisticRegression':
                # Predict using the trained LogisticRegression
                self.y_pred = self.predict_proba(X_true)
            elif self.classifier_name == 'GMM':
                # Predict using the trained GMM
                self.y_pred = self.predict_proba(X_true)
            else:
                # Error
                print "ERROR: Incorrect classifier name"
        else:
            # Predict using the trained model
            if self.classifier_name == 'RandomForestClassifier':
                # Predict using the trained RandomForestClassifier
                self.y_pred = self.predict(X_true)
            elif self.classifier_name == 'GradientBoostingClassifier':
                # Predict using the trained GradientBoostingClassifier
                self.y_pred = self.predict(X_true)
            elif self.classifier_name == 'LogisticRegression':
                # Predict using the trained LogisticRegression
                self.y_pred = self.predict(X_true)
            elif self.classifier_name == 'GMM':
                # Predict using the trained GMM
                self.y_pred = self.predict(X_true)
            else:
                # Error
                print "ERROR: Incorrect classifier name"

        return self.y_pred
