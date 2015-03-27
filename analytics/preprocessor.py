# Code for the Kaggle AXA challenge
#-----------------------------------
#    Copyright (C) 2015- by
#    Frank Lin <email@>
#    Jim Simpson <email@>
#    All rights reserved.
#    BSD license.
#-----------------------------------

import os
import sys
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing


class Preprocessor(object):

    """The preprocessor applies basic transformations before the Classifier.

    This class serves the purpose of transforming X and y.

    Note:
      The main public method in this class is apply_preprocessing().

    Args:
      None

    Attributes:
      None

    """

    def __init__(self, file_path=''):
        self.status = 0

    def combine_fake_and_true(self, X_fake, X_true, y_fake, y_true):
        """Combine fake and true into single feature set."""
        try:
            X = np.vstack((X_fake, X_true))
            y = np.hstack((y_fake, y_true))
        except ValueError or UnboundLocalError:
            print "ERROR size mistmatch between fake and true."
            print "Is either file loaded from saved pickle?"
            sys.exit(1)

        return X, y

    def cast_to_float32(self, X, y):
        """Make sure X and y are type float32 and not float64."""
        X = X.astype('float32')
        y = y.astype('float32')

        return X, y

    def reorder_fake_and_true(self, X, y):
        """Reorder to intersperse fake and true data points."""
        # TODO: Remove/change random seed later
        np.random.seed(None)
        reorder = np.random.permutation(len(X))
        X = X[reorder]
        y = y[reorder]

        return X, y

    def split_train_and_test(self, X, y, test_size):
        """Split into train and test sets for cross validation."""
        # TODO: Remove/change random seed later
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=test_size, random_state=None)

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_fake, X_true):
        """Apply feature scaling to zero mean and one variance."""
        # Stack together to apply same standardization
        X = np.vstack((X_fake, X_true))

        # Apply feature scaling
        X = preprocessing.scale(X)

        # Split back
        X_fake = X[:len(X_fake)]
        X_true = X[len(X_fake):]

        return X_fake, X_true

    def apply_preprocessing(self, X_fake, X_true, y_fake, y_true):
        """Apply all preprocessing steps."""

        # Apply feature scaling
        X_fake, X_true = self.scale_features(X_fake, X_true)

        # Combine fake and true into single feature set
        X, y = self.combine_fake_and_true(X_fake, X_true, y_fake, y_true)

        # Cast from float64 to float32
        # Some scikit learners are float32 only
        X, y = self.cast_to_float32(X, y)

        # Reorder to intersperse fake and true data points
        X, y = self.reorder_fake_and_true(X, y)

        # Split into train and test sets for cross validation
        X_train, X_test, y_train, y_test = self.split_train_and_test(X, y, 0.1)

        return X_train, X_test, y_train, y_test, X_true
