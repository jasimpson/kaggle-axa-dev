# Code for the Kaggle AXA challenge
#-----------------------------------
#    Copyright (C) 2015- by
#    Frank Lin <email@>
#    Jim Simpson <email@>
#    All rights reserved.
#    BSD license.
#-----------------------------------

import numpy as np
import pandas as pd
import copy
import math
import featureextractor
import seriesextractor
# from pykalman import KalmanFilter


class Trip(object):

    """The Trip class extracts time series data and features from trip data.

    The time series data is stored as a Pandas DataFrame and is a 1D array
    for each time series such as distance, velocity, acceleration, jerk, etc.
    Each of these time series has a separate column in the DataFrame.

    The features data is stored as a dictionary and is a singular value for
    each key in the dictionary such as total_distance, avg_velocity, etc.
    Each of these features has a separate key in the dictionary.

    Note:
      Time series data is extracted using the SeriesExtractor class.
      Feature data is extracted using the FeatureExtractor class.

    Args:
      None

    Attributes:
      None

    """

    def __init__(self, data=None, features=None):
        if data is None:
            self.trip_data = []
        else:
            self.trip_data = data

        if features is None:
            self.features = {}

            # Instantiate series extractor
            se = seriesextractor.SeriesExtractor(self.trip_data)
            self.series = se.get_series()
            self.tmp = se.get_tmpdata()

            # Instantiate feature extractor
            # Make sure trip_data is computed before calling
            if 1:
                # To run on entire trip and for all features
                fe = featureextractor.FeatureExtractor(
                    self.series, self.tmp)
                self.features = fe.get_features()

            if 0:
                # To run on entire trip but for select features

                # Specify features wanted
                feature_list = [
                    'compute_total_dist',
                    'compute_num_stops',
                    'compute_vel_params',
                    'compute_accel_params',
                    'compute_jerk_params',
                    'compute_vel_times_accel'
                ]

                # Run on entire trip
                fe = featureextractor.FeatureExtractor(
                    self.series, self.tmp, feature_list)
                self.features = fe.get_features()

            if 0:
                # To run on select regions for select features

                """
                This mode allows you to create the same features for different
                sections of trip data.

                Instead of creating functions in FeatureExtractor like:
                    compute_vel_quantiles_straights(), and
                    compute_vel_quantiles_turns();

                Create a new series that splits trip_data into:
                    series_straight_regions, and
                    series_turn_regions;

                And set the list of features to be generated for each part as:
                    feature_list = [
                        'compute_vel_quantiles',
                    ]


                E.g. See below for generating the features:
                    compute_total_dist, and
                    compute_jerk_params;

                For each region of trip_data:
                    series_vel_bin1,
                    series_vel_bin2, and
                    series_vel_bin3.
                """
                # Bin on velocity regions
                series_vel_bin1, series_vel_bin2, series_vel_bin3 = \
                    self.bin_velocity_regions(self.series)

                # Bin for these features
                feature_list = [
                    'compute_total_dist',
                    'compute_jerk_params'
                ]

                # Get velocity bin 1 features and append to main features
                fe_vel_bin1 = featureextractor.FeatureExtractor(
                    series_vel_bin1, self.tmp, feature_list)
                features_vel_bin1 = fe_vel_bin1.get_features_renamed(
                    '_vel_bin1')
                self.features.update(
                    features_vel_bin1)

                # Get velocity bin 2 features and append to main features
                fe_vel_bin2 = featureextractor.FeatureExtractor(
                    series_vel_bin2, self.tmp, feature_list)
                features_vel_bin2 = fe_vel_bin2.get_features_renamed(
                    '_vel_bin2')
                self.features.update(
                    features_vel_bin2)

                # Get velocity bin 3 features and append to main features
                fe_vel_bin3 = featureextractor.FeatureExtractor(
                    series_vel_bin3, self.tmp, feature_list)
                features_vel_bin3 = fe_vel_bin3.get_features_renamed(
                    '_vel_bin3')
                self.features.update(
                    features_vel_bin3)

        else:
            self.features = features

    def __getitem__(self, key):
        '''Returns the item with value key'''

        # Check that the feature dictionary has the requested key
        if self.features.has_key(key):
            return self.features[key]
        else:
            print "Requested key does not exist"

    def __setitem__(self, key, value):
        self.features[key] = value

    #---BINNING TO REGIONS-----------------------------------------------------

    def bin_accelation_regions(self, series_all):
        ''' Bin acceleration into three different regions'''

        # Region boundaries
        accl_lo = 0.1
        accl_hi = 0.2

        # Bin 1 - acceleration region
        series_accl_bin1 = series_all[series_all['accel'] < accl_lo]

        # Bin 2 - acceleration region
        series_accl_bin2 = series_all[(series_all['accel'] >= accl_lo) &
                                      (series_all['accel'] < accl_hi)]

        # Bin 3 - acceleration region
        series_accl_bin3 = series_all[series_all['accel'] >= accl_hi]

        # print series_all.shape
        # print type(series_accl_bin1.shape)
        # print series_accl_bin2.shape
        # print series_accl_bin3.shape

        # Prevent empty data frames (lots of slicing and diving by zero later)
        ncols = series_all.shape[1]
        if series_accl_bin1.shape[0] == 0:
            series_accl_bin1 = pd.DataFrame(
                data=np.ones((1, ncols)), columns=list(series_all))
        if series_accl_bin2.shape[0] == 0:
            series_accl_bin2 = pd.DataFrame(
                data=np.ones((1, ncols)), columns=list(series_all))
        if series_accl_bin3.shape[0] == 0:
            series_accl_bin3 = pd.DataFrame(
                data=np.ones((1, ncols)), columns=list(series_all))

        return series_accl_bin1, series_accl_bin2, series_accl_bin3

    def bin_velocity_regions(self, series_all):
        ''' Bin velocity into three different regions'''

        # Region boundaries
        vel_lo = 13.4
        vel_hi = 24.6

        # Bin 1 - velocity region
        series_vel_bin1 = series_all[series_all['vel'] < vel_lo]

        # Bin 2 - velocity region
        series_vel_bin2 = series_all[(series_all['vel'] >= vel_lo) &
                                     (series_all['vel'] < vel_hi)]

        # Bin 3 - velocity region
        series_vel_bin3 = series_all[series_all['vel'] >= vel_hi]

        # print series_all.shape
        # print type(series_vel_bin1.shape)
        # print series_vel_bin2.shape
        # print series_vel_bin3.shape

        # Prevent empty data frames (lots of slicing and diving by zero later)
        ncols = series_all.shape[1]
        if series_vel_bin1.shape[0] == 0:
            series_vel_bin1 = pd.DataFrame(
                data=np.ones((1, ncols)), columns=list(series_all))
        if series_vel_bin2.shape[0] == 0:
            series_vel_bin2 = pd.DataFrame(
                data=np.ones((1, ncols)), columns=list(series_all))
        if series_vel_bin3.shape[0] == 0:
            series_vel_bin3 = pd.DataFrame(
                data=np.ones((1, ncols)), columns=list(series_all))

        return series_vel_bin1, series_vel_bin2, series_vel_bin3

    #---IMPROVED FEATURE METRICS-----------------------------------------------

    def compute_acceleration_blocks(self):
        if 'vel' in self.trip_data.columns and 'accel' in self.trip_data.columns:
            vel_vals = np.array(self.trip_data['vel'])
            accel_vals = np.array(self.trip_data['accel'])

            vel_vals_lt25 = np.where(vel_vals <= 11.2)[0]
            vel_vals_gt25_lt50 = np.where(
                (vel_vals > 11.2) & (vel_vals <= 22.4))[0]
            vel_vals_gt50 = np.where(vel_vals > 22.4)[0]

            if len(vel_vals_lt25) == 0:
                self.features['avg_accel_lt25'] = np.float64(0.0)
                # self.features['std_decel_lt25'] =  np.float64(0.0)
                self.features['avg_accel_lt25'] = np.float64(0.0)
                # self.features['std_decel_lt25'] =  np.float64(0.0)
            else:
                tmp_vals = accel_vals[vel_vals_lt25]
                all_accelerations = tmp_vals[np.where(tmp_vals > 0)[0]]
                all_decelerations = tmp_vals[np.where(tmp_vals < 0)[0]]
                if len(all_accelerations) == 0:
                    self.features['avg_accel_lt25'] = np.float64(0.0)
                    # self.features['std_accel_lt25'] =  np.float64(0.0)
                else:
                    self.features['avg_accel_lt25'] = np.float64(
                        np.mean(all_accelerations))
                    # self.features['std_accel_lt25'] =  np.float64(np.std(all_accelerations)/np.mean(all_accelerations))
                if len(all_decelerations) == 0:
                    self.features['avg_decel_lt25'] = np.float64(0.0)
                    # self.features['std_decel_lt25'] =  np.float64(0.0)
                else:
                    self.features['avg_decel_lt25'] = np.float64(
                        np.mean(all_decelerations))
                    # self.features['std_decel_lt25'] =  np.float64(np.std(all_decelerations)/np.mean(all_decelerations))

            if len(vel_vals_gt25_lt50) == 0:
                self.features['avg_accel_gt25_lt50'] = np.float64(0.0)
                self.features['avg_decel_gt25_lt50'] = np.float64(0.0)
                # self.features['std_accel_gt25_lt50'] =  np.float64(0.0)
                # self.features['std_decel_gt25_lt50'] =  np.float64(0.0)
            else:
                tmp_vals = accel_vals[vel_vals_gt25_lt50]
                all_accelerations = tmp_vals[np.where(tmp_vals > 0)[0]]
                all_decelerations = tmp_vals[np.where(tmp_vals < 0)[0]]
                if len(all_accelerations) == 0:
                    self.features['avg_accel_gt25_lt50'] = np.float64(0.0)
                    # self.features['std_accel_gt25_lt50'] =  np.float64(0.0)
                else:
                    self.features['avg_accel_gt25_lt50'] = np.float64(
                        np.mean(all_accelerations))
                    # self.features['std_accel_gt25_lt50'] =  np.float64(np.std(all_accelerations)/np.mean(all_accelerations))
                if len(all_decelerations) == 0:
                    self.features['avg_decel_gt25_lt50'] = np.float64(0.0)
                    # self.features['std_decel_gt25_lt50'] =  np.float64(0.0)
                else:
                    self.features['avg_decel_gt25_lt50'] = np.float64(
                        np.mean(all_decelerations))
                    # self.features['std_decel_gt25_lt50'] =  np.float64(np.std(all_decelerations)/np.mean(all_decelerations))

            if len(vel_vals_gt50) == 0:
                self.features['avg_accel_gt50'] = np.float64(0.0)
                self.features['avg_decel_gt50'] = np.float64(0.0)
                # self.features['std_accel_gt50'] =  np.float64(0.0)
                # self.features['std_decel_gt50'] =  np.float64(0.0)
            else:
                tmp_vals = accel_vals[vel_vals_gt50]
                all_accelerations = tmp_vals[np.where(tmp_vals > 0)[0]]
                all_decelerations = tmp_vals[np.where(tmp_vals < 0)[0]]
                if len(all_accelerations) == 0:
                    self.features['avg_accel_gt50'] = np.float64(0.0)
                    # self.features['std_accel_gt50'] =  np.float64(0.0)
                else:
                    self.features['avg_accel_gt50'] = np.float64(
                        np.mean(all_accelerations))
                    # self.features['std_accel_gt50'] =  np.float64(np.std(all_accelerations)/np.mean(all_accelerations))
                if len(all_decelerations) == 0:
                    self.features['avg_decel_gt50'] = np.float64(0.0)
                    # self.features['std_decel_gt50'] =  np.float64(0.0)
                else:
                    self.features['avg_decel_gt50'] = np.float64(
                        np.mean(all_decelerations))
                    # self.features['std_decel_gt50'] =  np.float64(np.std(all_decelerations)/np.mean(all_decelerations))

    def compute_acceleration_ngapproach(self):
        '''A 5 second interval post stop or roll. The driver MUST have driven 
        for at least 10 seconds above 1.0 ms post stop for this to be counted
        unless at the end or start of a time window. If the stop is at the 
        beginning and you cant reach back far enough for the average 
        deceleration, dont use it. Obviously, a stop at the very end only 
        allows for a deceleration calculation.'''

        # Find all stops that last more than a 5 second period
        if 'vel' in self.trip_data.columns and 'accel' in self.trip_data.columns:
            vel_vals = np.array(self.trip_data['vel'])

            # A true stop is considered as the stoptime parameter in seconds
            # A true stop is when the velocity is 0
            all_zero_vel = np.where(vel_vals <= 1.0)[0]

            all_zero_diff = np.diff(all_zero_vel)
            all_zero_diff[np.where(all_zero_diff < 10)[0]] = 1
            zero_stop_indices = np.where(all_zero_diff > 1)[0] + 1
            zero_stop_indices = np.append(0, zero_stop_indices)
            zero_stop_indices = np.append(
                zero_stop_indices, len(all_zero_diff))

            previous_index = all_zero_vel[0]
            for i in range(1, len(all_zero_vel)):
                cur_index = all_zero_vel[i]
                # Check if index is less than 5

                # if indices
            print all_zero_diff
            print all_zero_vel[zero_stop_indices]
            return all_zero_vel

        else:
            print 'No distance column in the dataframe to compute the stops'

    def compute_travel_type_table(self):
        '''Based on characteristic features, compute the typeof travel
            1. Travel type - Local street driving
            2. Travel type - Freeway like driving
            3. Travel type - traffic
            Label the Pandas dataframe in column ['drivingtype'] with strings
            for each type, 'local','traffic','freeway' for each type. Transitions
            between each type will be challenging to label.

            Local is indicative of stops, with some sharp angled turns (~45degrees),slow
            speed driving (less than 45 mph).

            Traffic should show no sharp angled turns, but a lot of evidence of
            low speed acceleration and decelerations.

            Highway driving or open driving should show no stopping with lower angled
            turns if any, and higher speeds (>55mph).

            One type we might consider is local backroads type driving

            velocity -> m/s to m/h is 2.237 multiplier
            avg highway acceleration = 0.26 - 0.66 ft/s2
            avg city acceleration = 1.48 - 4 ft/s2

            max speeds greater than 100 m/s are aberrations in the data'''

        pass
