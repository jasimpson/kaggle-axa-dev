# Code for the Kaggle AXA challenge
#-----------------------------------
#    Copyright (C) 2015- by
#    Frank Lin <email@>
#    Jim Simpson <email@>
#    All rights reserved.
#    BSD license.
#-----------------------------------

import os
import random
import numpy as np

import driver


class Combiner(object):

    """The combiner class stages the features from trips into a 2D numpy array.

    This class serves as an interface between the low-level Driver and Trip
    classes and the high level scikit-learn Classifier classes. The 
    scikit-learn fit/predict API expects the features to be input in the X, y
    format where X is a 2D numpy array of features and y is a 1D numpy array of 
    corresponding classes. Each column in X corresponds to a different feature.
    Each row in X and y corresponds to a data point (specific trip).

    This class also serves the purpose of generating X for both classes in y.
    The "true" class of y corresponds to features from trips correctly 
    classified as driven by the current driver. The "fake" class of y
    corresponds to features from trips classified as *not* driven by the 
    current driver.

    Note:
      The main public methods in this class are 
      get_features_for_fake_drivers() and
      get_features_for_true_driver().

    Args:
      file_path (str): String path to the drivers data root folder.

    Attributes:
      data_root_path (str): Path to data files.
      driver_object (object): Instantiation holding current driver object.
      driver_id_in_mem (int): ID of current driver loaded in driver_object.
      feature_names (list of strings): Key names of features from Trip class.

    """

    def __init__(self, file_path=''):
        # Assign data_root_path
        self.data_root_path = file_path

        # Prevent re-loading driver object
        # TODO: Change this ugliness
        self.driver_object = None
        self.driver_id_in_mem = 0

        # Feature names
        self.feature_names = None

    def get_list_of_drivers(self, driver_path):
        """Get a list of ints of drivers."""
        return sorted(
            [int(x) for x in (
                [s for s in os.listdir(driver_path)
                 if os.path.isdir(os.path.join(driver_path, s))])])

    def get_list_of_trips(self, num_of_trips):
        """Get a list of ints of trips."""
        return range(1, num_of_trips + 1)

    def get_list_of_feature_names(self):
        '''Get feature names for parsing results'''
        driver_id = 1
        driver_path = os.path.join(self.data_root_path, str(driver_id))
        self.driver_object = driver.Driver(driver_path)
        trip_id = 1
        trip_object = self.driver_object.driver[str(trip_id) + '.csv']

        # Get feature names
        self.feature_names = trip_object.features.keys()

        return self.feature_names

    def get_features_for_one_trip(self, driver_trip_tuple):
        """Get features for a single trip."""

        # Check if driver object is already in memory
        driver_id = driver_trip_tuple[0]
        if driver_id != self.driver_id_in_mem:
            driver_path = os.path.join(self.data_root_path, str(driver_id))
            self.driver_object = driver.Driver(driver_path)
            self.driver_id_in_mem = driver_id
        trip_id = driver_trip_tuple[1]
        trip_object = self.driver_object.driver[str(trip_id) + '.csv']

        return trip_object.features

    def get_features_for_multiple_trips(self, driver_trip_tuple):
        """Get features for multiple trips."""
        return map(self.get_features_for_one_trip, driver_trip_tuple)

    def get_features_for_fake_drivers(self, num_of_fake_drivers, num_of_trips_per_fake_driver):
        """Get features for the other (fake) drivers' trips."""

        # Step 1a: Get features for the other (fake) drivers' trips
        # Step 1b: Label these as negative (0)

        # Calculate this only once and use for all driver
        # Warning: This could include 1 trip from the true driver, but oh well

        # List of fake drivers and trips to form tuples
        list_of_random_fake_drivers_ids = []
        list_of_random_fake_trips_ids = []

        # Make it repeatably random
        # TODO: Remove/change random seed later
        random.seed(None)

        # From list of all drivers, randomly select num_of_fake_drivers
        list_of_all_drivers = self.get_list_of_drivers(self.data_root_path)
        list_of_random_fake_drivers = random.sample(
            list_of_all_drivers, num_of_fake_drivers)

        # From list of all trips, randomly select num_of_trips_per_fake_driver
        list_of_all_trips = self.get_list_of_trips(200)
        list_of_random_fake_trips = random.sample(
            list_of_all_trips, num_of_trips_per_fake_driver)

        # Form random fake (driver, trip) tuples
        for idx_d, driver_id in enumerate(list_of_random_fake_drivers):
            for idx_t, trip_id in enumerate(list_of_random_fake_trips):
                list_of_random_fake_drivers_ids.append(driver_id)
                list_of_random_fake_trips_ids.append(trip_id)

        list_of_random_200_driver_trip_tuples = zip(
            list_of_random_fake_drivers_ids,
            list_of_random_fake_trips_ids)

        # Get and convert from list of dict of features to np.array
        X_fake = np.array(
            map(lambda x: x.values(),
                self.get_features_for_multiple_trips(
                    list_of_random_200_driver_trip_tuples)))

        # Label these as negative
        y_fake = np.zeros(len(X_fake))

        return X_fake, y_fake

    def get_features_for_true_driver(self, driver_id):
        """Get features for selected (true) driver's 200 trips."""

        # Step 2a: Get features for selected (true) driver's 200 trips
        # Step 2b: Label all 200 of these as positive (1)

        # Make list of 200 (driver, trip) tuples
        driver_trip_tuple = zip([driver_id] * 200, self.get_list_of_trips(200))

        # Convert from list of dict of features to np.array
        X_true = np.array(
            map(lambda x: x.values(),
                self.get_features_for_multiple_trips(
                    driver_trip_tuple)))

        y_true = np.ones(len(X_true))

        return X_true, y_true
