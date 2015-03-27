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
import scipy as sp
import pandas as pd

from sklearn.cluster import DBSCAN


class FeatureExtractor(object):

    """The FeatureExtractor extracts features from trip data.

    This class serves the purpose of keeping all the feature extraction
    methods together. The class also makes sure each feature is strictly
    validated before passing on to the machine learning algorithm.

    Note:
      All outputs of class members are written to self.features

    Args:
      None

    Attributes:
      None

    """

    def __init__(self, trip_data, tmp_data, feature_list=None):
        self.features = {}
        self.trip_data = trip_data
        self.tmp_data = tmp_data

        if feature_list == None:
            self.compute_total_dist()
            self.compute_num_stops()
            self.compute_vel_params()
            self.compute_accel_params()

            self.compute_vel_normhist()
            self.compute_accel_normhist()
            self.compute_decel_normhist()

            self.compute_jerk_params()

            self.compute_vel_quantiles()
            self.compute_accel_quantiles()
            self.compute_heading_quantiles()
            self.compute_vel_times_accel_quantiles()
            self.compute_vel_times_accel()

            self.compute_vel_quantiles_straights()
            self.compute_accel_quantiles_straights()
            self.compute_jerk_quantiles_straights()
            self.compute_num_stops_straights()

            self.compute_turn_accel_params()
            self.compute_normal_tangent_accel_normhist()

            self.compute_stop_params()
        else:
            if 'compute_total_dist' in feature_list:
                self.compute_total_dist()
            if 'compute_num_stops' in feature_list:
                self.compute_num_stops()
            if 'compute_vel_params' in feature_list:
                self.compute_vel_params()
            if 'compute_accel_params' in feature_list:
                self.compute_accel_params()

            if 'compute_vel_normhist' in feature_list:
                self.compute_vel_normhist()
            if 'compute_accel_normhist' in feature_list:
                self.compute_accel_normhist()
            if 'compute_decel_normhist' in feature_list:
                self.compute_decel_normhist()

            if 'compute_jerk_params' in feature_list:
                self.compute_jerk_params()

            if 'compute_heading_quantiles' in feature_list:
                self.compute_heading_quantiles()
            if 'compute_vel_quantiles' in feature_list:
                self.compute_vel_quantiles()
            if 'compute_accel_quantiles' in feature_list:
                self.compute_accel_quantiles()
            if 'compute_vel_times_accel_quantiles' in feature_list:
                self.compute_vel_times_accel_quantiles()
            if 'compute_vel_times_accel' in feature_list:
                self.compute_vel_times_accel()

            if 'compute_vel_quantiles_straights' in feature_list:
                self.compute_vel_quantiles_straights()
            if 'compute_accel_quantiles_straights' in feature_list:
                self.compute_accel_quantiles_straights()
            if 'compute_jerk_quantiles_straights' in feature_list:
                self.compute_jerk_quantiles_straights()
            if 'compute_num_stops_straights' in feature_list:
                self.compute_num_stops_straights()

            if 'compute_turn_accel_params' in feature_list:
                self.compute_turn_accel_params()
            if 'compute_normal_tangent_accel_normhist' in feature_list:
                self.compute_normal_tangent_accel_normhist()

            if 'compute_stop_params' in feature_list:
                self.compute_stop_params()

    def get_features(self):
        return self.features

    def get_tmpdata(self):
        return self.tmp_data

    def get_features_renamed(self, suffix='_new'):
        # Loop through each key and add suffix to key name
        for k, v in self.features.items():
            self.features[k + suffix] = self.features.pop(k)

        return self.features

    #--------------------------------------------------------------------------
    def validate_feature(self, feature_name=''):
        ''' Validate that a feature is singular numpy float64 number'''

        # Validate that the feature exists
        try:
            assert feature_name in self.features
        except AssertionError:
            print "ERROR feature validation failed: does not exist"
            print "Check feature %s " % (feature_name)
            pass

        # Validate feature has data type numpy.float64
        try:
            assert type(self.features[feature_name]) == np.dtype('Float64')
        except AssertionError:
            print "ERROR feature validation failed: not type numpy.float64"
            print "Check feature %s " % (feature_name)
            self.features[feature_name] = np.float64(
                self.features[feature_name])
            pass

        # Validate feature is singular and not an array
        try:
            assert self.features[feature_name].size == 1
        except AssertionError:
            print "ERROR feature validation failed: not singular"
            print "Check feature %s " % (feature_name)
            pass

        # Validate feature is not nan
        try:
            assert np.isnan(self.features[feature_name]) == False
        except AssertionError:
            print "ERROR feature validation failed: is nan"
            print "Check feature %s " % (feature_name)
            self.features[feature_name] = np.nan_to_num(
                self.features[feature_name])
            pass

        # Validate feature is not inf
        try:
            assert np.isinf(self.features[feature_name]) == False
        except AssertionError:
            print "ERROR feature validation failed: is inf"
            print "Check feature %s " % (feature_name)
            self.features[feature_name] = np.nan_to_num(
                self.features[feature_name])
            pass

    #--------------------------------------------------------------------------
    def compute_total_dist(self):
        ''' Return the total distance traveled for this trip'''

        if 'dist' in self.trip_data.columns:
            all_distances = np.array(self.trip_data['dist'])
            self.features['total_distance'] = np.float64(np.sum(all_distances))

            # Avoid divide by 0
            if self.features['total_distance'] == 0:
                self.features['total_distance'] = np.float64(1.0)

            self.validate_feature('total_distance')
        else:
            print "No distance column in the dataframe to compute the total distance value"

    #--------------------------------------------------------------------------
    def compute_num_stops(self, stop_time=2):
        '''Compute the number of total stops per trip. The stop time
            parameter is in seconds, and so it finds the number of
            stops where the person does not move (velocity = 0) for
            three or more seconds.'''

        if 'dist' in self.trip_data.columns:
            #dist_vals   = np.array(self.trip_data['dist'])
            dist_vals = np.array(self.trip_data['vel'])

            # A true stop is considered as the stoptime parameter in seconds
            # A true stop is when the velocity is 0
            all_zero_vel = np.where(dist_vals <= 1.0)[0]
            all_zero_vel_diff = np.diff(all_zero_vel)
            array_matcher = [1] * (stop_time - 1)

            # Find where the arrayMatcher matches the allZeroVel array to get
            # the positions
            array_correlate = np.correlate(all_zero_vel_diff, array_matcher)

            # To catch potential end cases, add a 1 at the end of
            # array_correlate
            array_correlate = np.append(array_correlate, 1)
            all_stops = np.where(array_correlate == (stop_time - 1))[0]
            tot_stops = len(np.where(np.diff(all_stops) > 1)[0]) + 1
            if self.features['total_distance'] != 0:
                self.features['total_stops'] = np.float64(
                    tot_stops) / self.features['total_distance']
            else:
                self.features['total_stops'] = np.float64(0)

            self.validate_feature('total_stops')
        else:
            print 'No distance column in the dataframe to compute the stops'

    #--------------------------------------------------------------------------
    def compute_vel_params(self):
        ''' Compute the avg velocity, max velocity, and min velocity
            min velocity should be the minimum that is NOT 0'''

        if 'vel' in self.trip_data.columns:
            all_velocities = np.array(self.trip_data['vel'])
            vel_params_filt = all_velocities[np.where(all_velocities > 1.0)[0]]
            #vel_params_filt = all_velocities[np.where(all_velocities > 0.0)[0]]

            # If there are NO or ONE velocities in the array greater than 1.0
            if len(vel_params_filt) < 2:
                vel_params_filt = all_velocities
            self.features['avg_velocity'] = np.float64(
                np.mean(vel_params_filt))
            if self.features['avg_velocity'] == np.float64(0.0):
                self.features['std_velocity'] = np.float64(0.0)
            else:
                self.features['std_velocity'] = np.float64(
                    np.std(vel_params_filt) / np.mean(vel_params_filt))
            self.features['max_velocity'] = np.float64(np.max(vel_params_filt))

            self.validate_feature('avg_velocity')
            self.validate_feature('std_velocity')
            self.validate_feature('max_velocity')

        else:
            print "No velocity column in the dataframe to compute the velocity metrics"

    #-------------------------------------------------------------------------
    def compute_accel_params(self):
        ''' Compute the avg _acceleration, avg deceleration, max and min for both'''

        if 'accel' in self.trip_data.columns:
            all_accel = np.array(self.trip_data['accel'])
            all_accelerations = all_accel[np.where(all_accel > 0)[0]]
            all_decelerations = all_accel[np.where(all_accel < 0)[0]]
            if len(all_accelerations) == 0:
                self.features['avg_acceleration'] = np.float64(0.0)
                self.features['max_acceleration'] = np.float64(0.0)
                self.features['std_acceleration'] = np.float64(0.0)
                #self.features['time_in_acceleration'] = np.float64(0.0)
                #self.features['min_acceleration'] = np.float64(0.0)
            else:
                self.features['avg_acceleration'] = np.float64(
                    np.mean(all_accelerations))
                self.features['max_acceleration'] = np.float64(
                    np.max(all_accelerations))
                self.features['std_acceleration'] = np.float64(
                    np.std(all_accelerations) / np.mean(all_accelerations))
                self.features['time_in_acceleration'] = np.float64(
                    len(all_accelerations) / self.features['total_distance'])
                # self.features['min_acceleration'] = np.float64(
                #    np.min(all_accelerations))
            if len(all_decelerations) == 0:
                self.features['avg_deceleration'] = np.float64(0.0)
                self.features['max_deceleration'] = np.float64(0.0)
                self.features['std_deceleration'] = np.float64(0.0)
                self.features['time_in_deceleration'] = np.float64(0.0)
                #self.features['min_deceleration'] = np.float64(0.0)
            else:
                self.features['max_deceleration'] = np.float64(
                    np.min(all_decelerations))
                self.features['avg_deceleration'] = np.float64(
                    np.mean(all_decelerations))
                self.features['std_deceleration'] = np.float64(
                    np.std(all_decelerations) / np.mean(all_decelerations))
                self.features['time_in_deceleration'] = np.float64(
                    len(all_decelerations) / self.features['total_distance'])
                # self.features['min_deceleration'] = np.float64(
                #    np.min(all_decelerations))

            self.validate_feature('avg_acceleration')
            self.validate_feature('max_acceleration')
            self.validate_feature('std_acceleration')
            self.validate_feature('time_in_acceleration')
            # self.validate_feature('min_acceleration')

            self.validate_feature('avg_deceleration')
            self.validate_feature('max_deceleration')
            self.validate_feature('std_deceleration')
            self.validate_feature('time_in_deceleration')
            # self.validate_feature('min_deceleration')
        else:
            print "No acceleration column in the dataframe to compute acceleration metrics"

    #---NORMALIZED HISTOGRAM---------------------------------------------------

    def compute_vel_normhist(self, binsize=1):
        '''Uses a similar method as the paper, they used 80 bins for mph, given our
            speeds are in m/s, we will test first by using 50 bins for 0-50 m/s'''
        if 'vel' in self.trip_data.columns:
            all_velocities = np.array(self.trip_data['vel'])
            vel_params_filt = all_velocities[np.where(all_velocities > 1.0)[0]]
            # If there are NO or ONE velocities in the array greater than 1.0
            if len(vel_params_filt) < 2:
                vel_params_filt = all_velocities

            '''# Create the histogram and then normalize the values
            bins = np.arange(0, 50, binsize)
            hist, edges = np.histogram(vel_params_filt, bins)
            hist = hist / float(np.sum(hist))
            #self.features['normhist_velocity'] = np.float64(hist)
            # Calculate 3rd and 4th moments to describe shape of hist
            skewness = sp.stats.mstats.moment(hist, moment=3)
            kurtosis = sp.stats.mstats.moment(hist, moment=4)
            self.features['normhist_vel_skewness'] = np.float64(skewness)
            self.features['normhist_vel_kurtosis'] = np.float64(kurtosis)

            self.validate_feature('normhist_vel_skewness')
            self.validate_feature('normhist_vel_kurtosis')'''

            bins = np.arange(1, 35, binsize)
            # Create the histogram and then normalize the values
            hist, edges = np.histogram(vel_params_filt, bins)
            hist = np.float64(hist / float(np.sum(hist)))
            for i in range(0, len(bins) - 1):
                feature_name = 'normhist_vel_b' + str(i)
                self.features[feature_name] = np.nan_to_num(
                    np.float64(hist[i]))
                self.validate_feature(feature_name)
        else:
            print "No velocity column in the dataframe to compute the velocity metrics"

    def compute_accel_normhist(self, binsize=0.1):
        '''Compute normalized accceleration'''
        if 'accel' in self.trip_data.columns:
            accel = np.array(self.trip_data['accel'])
            all_accel = accel[np.where(accel > 0.1)[0]]
            bins = np.arange(0.1, 3.0, binsize)
            if len(all_accel) == 0:
                for i in range(0, len(bins) - 1):
                    feature_name = 'normhist_accel_b' + str(i)
                    self.features[feature_name] = np.float64(0)
                    self.validate_feature(feature_name)
            else:
                # Create the histogram and then normalize the values
                hist, edges = np.histogram(all_accel, bins)
                hist = np.float64(hist / float(np.sum(hist)))
                for i in range(0, len(bins) - 1):
                    feature_name = 'normhist_accel_b' + str(i)
                    self.features[feature_name] = np.nan_to_num(
                        np.float64(hist[i]))
                    self.validate_feature(feature_name)
        else:
            print "No velocity column in the dataframe to compute the velocity metrics"

    def compute_decel_normhist(self, binsize=0.1):
        '''Compute normalized accceleration'''
        if 'accel' in self.trip_data.columns:
            decel = np.array(self.trip_data['accel'])
            all_decel = np.abs(decel[np.where(decel < -0.1)[0]])
            bins = np.arange(0.1, 3.0, binsize)
            if len(all_decel) == 0:
                for i in range(0, len(bins) - 1):
                    feature_name = 'normhist_decel_b' + str(i)
                    self.features[feature_name] = np.float64(0)
                    self.validate_feature(feature_name)
            else:
                # Create the histogram and then normalize the value
                hist, edges = np.histogram(all_decel, bins)
                hist = np.float64(hist / float(np.sum(hist)))
                for i in range(0, len(bins) - 1):
                    feature_name = 'normhist_decel_b' + str(i)
                    self.features[feature_name] = np.float64(hist[i])
                    self.validate_feature(feature_name)
        else:
            print "No velocity column in the dataframe to compute the velocity metrics"

    #--------------------------------------------------------------------------
    def compute_jerk_params(self):
        ''' Compute the avg _acceleration, avg deceleration, max and min for both'''

        if 'jerk' in self.trip_data.columns:
            all_jerk = np.array(self.trip_data['jerk'])
            all_pos_jerk = all_jerk[np.where(all_jerk > 0)[0]]
            all_neg_jerk = all_jerk[np.where(all_jerk < 0)[0]]
            if len(all_pos_jerk) == 0:
                self.features['avg_pos_jerk'] = np.float64(0.0)
                self.features['max_pos_jerk'] = np.float64(0.0)
                self.features['pseudo_pos_jerk'] = np.float64(0.0)
            else:
                self.features['avg_pos_jerk'] = np.float64(
                    np.mean(all_pos_jerk))
                self.features['max_pos_jerk'] = np.float64(
                    np.max(all_pos_jerk))
                self.features['pseudo_pos_jerk'] = np.float64(
                    np.std(all_pos_jerk) / np.mean(all_pos_jerk))
            if len(all_neg_jerk) == 0:
                self.features['avg_neg_jerk'] = np.float64(0.0)
                self.features['max_neg_jerk'] = np.float64(0.0)
                self.features['pseudo_neg_jerk'] = np.float64(0.0)
            else:
                self.features['max_neg_jerk'] = np.float64(
                    np.min(all_neg_jerk))
                self.features['avg_neg_jerk'] = np.float64(
                    np.mean(all_neg_jerk))
                self.features['pseudo_neg_jerk'] = np.float64(
                    np.std(all_neg_jerk) / np.mean(all_neg_jerk))

            self.validate_feature('avg_pos_jerk')
            self.validate_feature('max_pos_jerk')
            self.validate_feature('pseudo_pos_jerk')
            self.validate_feature('avg_neg_jerk')
            self.validate_feature('max_neg_jerk')
            self.validate_feature('pseudo_neg_jerk')
        else:
            print "No acceleration column in the dataframe to compute acceleration metrics"

    #---QUANTILES--------------------------------------------------------------
    def compute_heading_quantiles(self):
        if 'angles' in self.trip_data.columns:
            all_angles = np.array(self.trip_data['angles'])
            diff_value = np.abs(np.diff(all_angles))
            diff_value[np.where(diff_value > 180)[0]] = 360 - \
                diff_value[np.where(diff_value > 180)[0]]
            diff_angle_rads = diff_value * (np.pi / 180)

            # If there are NO or ONE velocities in the array greater than 1.0
            self.features['p50_heading'] = np.float64(
                np.percentile(diff_angle_rads, 50))
            self.features['p5_heading'] = np.float64(
                np.percentile(diff_angle_rads, 5))
            self.features['p90_heading'] = np.float64(
                np.percentile(diff_angle_rads, 95))

            self.validate_feature('p5_heading')
            self.validate_feature('p50_heading')
            self.validate_feature('p90_heading')
        else:
            print "No heading column in the dataframe to compute the velocity metrics"

    def compute_vel_quantiles(self):
        if 'vel' in self.trip_data.columns:
            all_velocities = np.array(self.trip_data['vel'])
            vel_params_filt = all_velocities[np.where(all_velocities > 1.0)[0]]
            # If there are NO or ONE velocities in the array greater than 1.0
            if len(vel_params_filt) < 2:
                vel_params_filt = all_velocities
            self.features['p5_velocity'] = np.float64(
                np.percentile(vel_params_filt, 5))
            self.features['p25_velocity'] = np.float64(
                np.percentile(vel_params_filt, 25))
            self.features['p90_velocity'] = np.float64(
                np.percentile(vel_params_filt, 90))
            self.features['max_velocity'] = np.float64(np.max(vel_params_filt))

            self.validate_feature('p5_velocity')
            self.validate_feature('p25_velocity')
            self.validate_feature('p90_velocity')
            self.validate_feature('max_velocity')
        else:
            print "No velocity column in the dataframe to compute the velocity metrics"

    def compute_accel_quantiles(self):
        if 'accel' in self.trip_data.columns:
            all_accel = np.array(self.trip_data['accel'])
            all_accelerations = all_accel[np.where(all_accel > 0)[0]]
            all_decelerations = all_accel[np.where(all_accel < 0)[0]]
            if len(all_accelerations) == 0:
                self.features['p50_acceleration'] = np.float64(0.0)
                self.features['p25_acceleration'] = np.float64(0.0)
                self.features['p75_acceleration'] = np.float64(0.0)
                self.features['max_acceleration'] = np.float64(0.0)
                self.features['time_in_acceleration'] = np.float64(0.0)
            else:
                self.features['p50_acceleration'] = np.float64(
                    np.percentile(all_accelerations, 50))
                self.features['p25_acceleration'] = np.float64(
                    np.percentile(all_accelerations, 25))
                self.features['p75_acceleration'] = np.float64(
                    np.percentile(all_accelerations, 75))
                self.features['max_acceleration'] = np.float64(
                    np.max(all_accelerations))
                self.features['time_in_acceleration'] = np.float64(
                    len(all_accelerations))
            if len(all_decelerations) == 0:
                self.features['p50_deceleration'] = np.float64(0.0)
                self.features['p25_deceleration'] = np.float64(0.0)
                self.features['p75_deceleration'] = np.float64(0.0)
                self.features['max_deceleration'] = np.float64(0.0)
                self.features['time_in_deceleration'] = np.float64(0.0)
            else:
                self.features['p50_deceleration'] = np.float64(
                    np.percentile(all_decelerations, 50))
                self.features['p25_deceleration'] = np.float64(
                    np.percentile(all_decelerations, 25))
                self.features['p75_deceleration'] = np.float64(
                    np.percentile(all_decelerations, 75))
                self.features['max_deceleration'] = np.float64(
                    np.min(all_decelerations))
                self.features['time_in_deceleration'] = np.float64(
                    len(all_decelerations))

            self.validate_feature('p50_acceleration')
            self.validate_feature('p25_acceleration')
            self.validate_feature('p75_acceleration')
            self.validate_feature('max_acceleration')
            self.validate_feature('time_in_acceleration')

            self.validate_feature('p50_deceleration')
            self.validate_feature('p25_deceleration')
            self.validate_feature('p75_deceleration')
            self.validate_feature('max_deceleration')
            self.validate_feature('time_in_deceleration')
        else:
            print "No acceleration column in the dataframe to compute acceleration metrics"

    def compute_vel_times_accel_quantiles(self):
        # Check that the dataframe has the columns necessary to do the
        # computation
        if 'vel' in self.trip_data.columns and 'accel' in self.trip_data.columns:
            vel = np.array(self.trip_data['vel'])
            acc = np.array(self.trip_data['accel'])
            vel_times_acc = vel * acc

            # Get those positive or negative vel x acc values
            vel_prod_acc = vel_times_acc[np.where(vel_times_acc > 0)[0]]
            vel_prod_dec = vel_times_acc[np.where(vel_times_acc < 0)[0]]

            self.trip_data['velxaccel'] = vel_times_acc
            # If there are no vel_prod_acc values, then set the values to 0
            if len(vel_prod_acc) == 0:
                self.features['p25_vel_prod_acc'] = np.float64(0.0)
                self.features['p50_vel_prod_acc'] = np.float64(0.0)
                self.features['p75_vel_prod_acc'] = np.float64(0.0)
                self.features['max_vel_prod_acc'] = np.float64(0.0)
            else:
                self.features['p25_vel_prod_acc'] = np.float64(
                    np.percentile(vel_prod_acc, 25))
                self.features['p50_vel_prod_acc'] = np.float64(
                    np.percentile(vel_prod_acc, 50))
                self.features['p75_vel_prod_acc'] = np.float64(
                    np.percentile(vel_prod_acc, 75))
                self.features['max_vel_prod_acc'] = np.float64(
                    np.max(vel_prod_acc))
            if len(vel_prod_dec) == 0:
                self.features['p25_vel_prod_dec'] = np.float64(0.0)
                self.features['p50_vel_prod_dec'] = np.float64(0.0)
                self.features['p75_vel_prod_dec'] = np.float64(0.0)
                self.features['max_vel_prod_dec'] = np.float64(0.0)
            else:
                self.features['p25_vel_prod_dec'] = np.float64(
                    np.percentile(vel_prod_dec, 25))
                self.features['p50_vel_prod_dec'] = np.float64(
                    np.percentile(vel_prod_dec, 50))
                self.features['p75_vel_prod_dec'] = np.float64(
                    np.percentile(vel_prod_dec, 75))
                self.features['max_vel_prod_dec'] = np.float64(
                    np.min(vel_prod_dec))

            self.validate_feature('p25_vel_prod_acc')
            self.validate_feature('p50_vel_prod_acc')
            self.validate_feature('p75_vel_prod_acc')
            self.validate_feature('max_vel_prod_acc')

            self.validate_feature('p25_vel_prod_dec')
            self.validate_feature('p50_vel_prod_dec')
            self.validate_feature('p75_vel_prod_dec')
            self.validate_feature('max_vel_prod_dec')
        else:
            print "Cannot compute the velocity times acceleration value because \
                one of the columns is missing from the dataframe"

    #--------------------------------------------------------------------------
    def compute_vel_times_accel(self):
        ''' Compute the velocity and acceleration product avg and std
            Note that the computed acceleration does not include
            values of 0, only positive values, or negative values 
            are included in the average, standard deviation, and
            computation of the min and max.'''

        # Check that the dataframe has the columns necessary to do the
        # computation
        if 'vel' in self.trip_data.columns and 'accel' in self.trip_data.columns:
            vel = np.array(self.trip_data['vel'])
            acc = np.array(self.trip_data['accel'])
            vel_times_acc = vel * acc

            # Get those positive or negative vel x acc values
            vel_prod_acc = vel_times_acc[np.where(vel_times_acc > 0)[0]]
            vel_prod_dec = vel_times_acc[np.where(vel_times_acc < 0)[0]]

            # If there are no vel_prod_acc values, then set the values to 0
            if len(vel_prod_acc) == 0:
                self.features['avg_vel_prod_acc'] = np.float64(0.0)
                self.features['std_vel_prod_acc'] = np.float64(0.0)
                self.features['max_vel_prod_acc'] = np.float64(0.0)
            else:
                self.features['avg_vel_prod_acc'] = np.float64(
                    np.mean(vel_prod_acc))
                self.features['std_vel_prod_acc'] = np.float64(
                    np.std(vel_prod_acc) / np.mean(vel_prod_acc))
                self.features['max_vel_prod_acc'] = np.float64(
                    np.max(vel_prod_acc))
            if len(vel_prod_dec) == 0:
                self.features['avg_vel_prod_dec'] = np.float64(0.0)
                self.features['std_vel_prod_dec'] = np.float64(0.0)
                self.features['max_vel_prod_dec'] = np.float64(0.0)
            else:
                self.features['avg_vel_prod_dec'] = np.float64(
                    np.mean(vel_prod_dec))
                self.features['std_vel_prod_dec'] = np.float64(
                    np.std(vel_prod_dec) / np.mean(vel_prod_dec))
                self.features['max_vel_prod_dec'] = np.float64(
                    np.min(vel_prod_dec))

            self.validate_feature('avg_vel_prod_acc')
            self.validate_feature('std_vel_prod_acc')
            self.validate_feature('max_vel_prod_acc')
            self.validate_feature('avg_vel_prod_dec')
            self.validate_feature('std_vel_prod_dec')
            self.validate_feature('max_vel_prod_dec')
        else:
            print "Cannot compute the velocity times acceleration value because \
                one of the columns is missing from the dataframe"


#--STRAIGHTS--------------------------------------------------------------
    def compute_accel_quantiles_straights(self):
        '''Compute the acceleration and deceleration in straights in blocks'''

        if 'turn_points' in self.trip_data.columns and 'accel' in self.trip_data.columns:
            turn_pts = np.array(self.trip_data['turn_points'])
            straights = np.where(turn_pts == 1)[0]
            all_accel = np.array(self.trip_data['accel'])[straights]

            if len(all_accel) > 0:
                all_accelerations = all_accel[np.where(all_accel > 0)[0]]
                all_decelerations = all_accel[np.where(all_accel < 0)[0]]
                if len(all_accelerations) == 0:
                    self.features['p50_acceleration_st'] = np.float64(0.0)
                    self.features['p10_acceleration_st'] = np.float64(0.0)
                    self.features['p90_acceleration_st'] = np.float64(0.0)
                    self.features['time_in_acceleration_st'] = np.float64(0.0)
                else:
                    self.features['p50_acceleration_st'] = np.float64(
                        np.percentile(all_accelerations, 50))
                    self.features['p10_acceleration_st'] = np.float64(
                        np.percentile(all_accelerations, 10))
                    self.features['p90_acceleration_st'] = np.float64(
                        np.percentile(all_accelerations, 90))
                    self.features['time_in_acceleration_st'] = np.float64(
                        float(len(all_accelerations)) / float(len(straights)))
                if len(all_decelerations) == 0:
                    self.features['p50_deceleration_st'] = np.float64(0.0)
                    self.features['p10_deceleration_st'] = np.float64(0.0)
                    self.features['p90_deceleration_st'] = np.float64(0.0)
                    self.features['time_in_deceleration_st'] = np.float64(0.0)
                else:
                    self.features['p50_deceleration_st'] = np.float64(
                        np.percentile(all_decelerations, 50))
                    self.features['p10_deceleration_st'] = np.float64(
                        np.percentile(all_decelerations, 10))
                    self.features['p90_deceleration_st'] = np.float64(
                        np.percentile(all_decelerations, 90))
                    self.features['time_in_deceleration_st'] = np.float64(
                        float(len(all_decelerations)) / float(len(straights)))
            else:
                self.features['p50_deceleration_st'] = np.float64(0.0)
                self.features['p10_deceleration_st'] = np.float64(0.0)
                self.features['p90_deceleration_st'] = np.float64(0.0)
                self.features['time_in_deceleration_st'] = np.float64(0.0)
                self.features['p50_acceleration_st'] = np.float64(0.0)
                self.features['p10_acceleration_st'] = np.float64(0.0)
                self.features['p90_acceleration_st'] = np.float64(0.0)
                self.features['time_in_acceleration_st'] = np.float64(0.0)

            self.validate_feature('p10_acceleration_st')
            self.validate_feature('p50_acceleration_st')
            self.validate_feature('p90_acceleration_st')
            self.validate_feature('time_in_acceleration_st')

            self.validate_feature('p50_deceleration_st')
            self.validate_feature('p10_deceleration_st')
            self.validate_feature('p90_deceleration_st')
            self.validate_feature('time_in_deceleration_st')
        else:
            print "No acceleration column in the dataframe to compute acceleration metrics"

    def compute_vel_quantiles_straights(self):
        if 'turn_points' in self.trip_data.columns and 'vel' in self.trip_data.columns:
            turn_pts = np.array(self.trip_data['turn_points'])
            straights = np.where(turn_pts == 1)[0]
            all_vel = np.array(self.trip_data['vel'])[straights]

            if len(all_vel) > 0:
                vel_params_filt = all_vel[np.where(all_vel > 1.0)[0]]
                # If there are NO or ONE velocities in the array greater than
                # 1.0
                if len(vel_params_filt) < 2:
                    vel_params_filt = all_vel
                self.features['p50_velocity_st'] = np.float64(
                    np.percentile(vel_params_filt, 50))
                self.features['p10_velocity_st'] = np.float64(
                    np.percentile(vel_params_filt, 10))
                self.features['p90_velocity_st'] = np.float64(
                    np.percentile(vel_params_filt, 90))
            else:
                self.features['p50_velocity_st'] = np.float64(0.0)
                self.features['p10_velocity_st'] = np.float64(0.0)
                self.features['p90_velocity_st'] = np.float64(0.0)

            self.validate_feature('p50_velocity_st')
            self.validate_feature('p10_velocity_st')
            self.validate_feature('p90_velocity_st')
        else:
            print "No velocity column in the dataframe to compute the velocity metrics"

    def compute_jerk_quantiles_straights(self):
        '''Compute the acceleration and deceleration in straights in blocks'''

        if 'turn_points' in self.trip_data.columns and 'jerk' in self.trip_data.columns:
            turn_pts = np.array(self.trip_data['turn_points'])
            straights = np.where(turn_pts == 1)[0]
            all_jerk = np.array(self.trip_data['jerk'])[straights]

            if len(all_jerk) > 0:
                all_posjerk = all_jerk[np.where(all_jerk > 0)[0]]
                all_negjerk = all_jerk[np.where(all_jerk < 0)[0]]
                if len(all_posjerk) == 0:
                    self.features['p50_posjerk_st'] = np.float64(0.0)
                    self.features['p10_posjerk_st'] = np.float64(0.0)
                    self.features['p90_posjerk_st'] = np.float64(0.0)
                else:
                    self.features['p50_posjerk_st'] = np.float64(
                        np.percentile(all_posjerk, 50))
                    self.features['p10_posjerk_st'] = np.float64(
                        np.percentile(all_posjerk, 10))
                    self.features['p90_posjerk_st'] = np.float64(
                        np.percentile(all_posjerk, 90))
                if len(all_negjerk) == 0:
                    self.features['p50_negjerk_st'] = np.float64(0.0)
                    self.features['p10_negjerk_st'] = np.float64(0.0)
                    self.features['p90_negjerk_st'] = np.float64(0.0)
                else:
                    self.features['p50_negjerk_st'] = np.float64(
                        np.percentile(all_negjerk, 50))
                    self.features['p10_negjerk_st'] = np.float64(
                        np.percentile(all_negjerk, 10))
                    self.features['p90_negjerk_st'] = np.float64(
                        np.percentile(all_negjerk, 90))
            else:
                self.features['p50_negjerk_st'] = np.float64(0.0)
                self.features['p10_negjerk_st'] = np.float64(0.0)
                self.features['p90_negjerk_st'] = np.float64(0.0)
                self.features['p50_posjerk_st'] = np.float64(0.0)
                self.features['p10_posjerk_st'] = np.float64(0.0)
                self.features['p90_posjerk_st'] = np.float64(0.0)

            self.validate_feature('p10_posjerk_st')
            self.validate_feature('p50_posjerk_st')
            self.validate_feature('p90_posjerk_st')

            self.validate_feature('p50_negjerk_st')
            self.validate_feature('p10_negjerk_st')
            self.validate_feature('p90_negjerk_st')
        else:
            print "No acceleration column in the dataframe to compute acceleration metrics"

    def compute_num_stops_straights(self, stop_time=2):

        if 'turn_points' in self.trip_data.columns and 'dist' in self.trip_data.columns:
            turn_pts = np.array(self.trip_data['turn_points'])
            straights = np.where(turn_pts == 1)[0]
            dist_vals = np.array(self.trip_data['dist'])[straights]

            if len(dist_vals) > 0:
                # A true stop is considered as the stoptime parameter in seconds
                # A true stop is when the velocity is 0
                all_zero_vel = np.where(dist_vals == 0.0)[0]
                all_zero_vel_diff = np.diff(all_zero_vel)
                array_matcher = [1] * (stop_time - 1)

                # Find where the arrayMatcher matches the allZeroVel array to get
                # the positions
                array_correlate = np.correlate(
                    all_zero_vel_diff, array_matcher)

                # To catch potential end cases, add a 1 at the end of
                # array_correlate
                array_correlate = np.append(array_correlate, 1)
                all_stops = np.where(array_correlate == (stop_time - 1))[0]
                tot_stops = len(np.where(np.diff(all_stops) > 1)[0]) + 1
                if np.sum(dist_vals) != 0:
                    self.features['total_stops_st'] = np.float64(
                        float(tot_stops) / np.sum(dist_vals))
                else:
                    self.features['total_stops_st'] = np.float64(0)
            else:
                self.features['total_stops_st'] = np.float64(0)
            self.validate_feature('total_stops_st')
        else:
            print 'No distance column in the dataframe to compute the stops'


#--TURNS------------------------------------------------------------------
    def compute_turn_accel_params(self):
        ''' Compute avg accel / delta radval of start to end of turn
            Compute avg decel / delta radval of start to end of turn
            Compute avg velocity / delta radval of start to end of turn
            Compute avg velocity * radvals through a turn'''

        vel = np.array(self.trip_data['vel'])
        accel = np.array(self.trip_data['accel'])
        angles = np.array(self.trip_data['angles'])

        turn_mem_dict = dict()
        turn_segments = self.tmp_data
        avg_vel_ang_turn = list()
        avg_vel_turn = list()
        avg_accel_turn = list()
        avg_decel_turn = list()
        if len(turn_segments) == 0:
            self.features['avg_vel_turn'] = np.float64(0)
            self.features['avg_accel_turn'] = np.float64(0)
            self.features['avg_decel_turn'] = np.float64(0)
            self.features['avg_vel_ang_turn'] = np.float64(0)
        else:
            for i in turn_segments:
                if not turn_mem_dict.has_key(str(i)):
                    diff_value = np.abs(angles[i[-1]] - angles[i[0]])
                    if diff_value > 180:
                        diff_value = 360 - diff_value
                    diff_angle_rads = diff_value * (np.pi / 180)
                    avg_vel_turn.append(
                        np.power(np.mean(vel[i]), 2) * (diff_angle_rads) / np.sum(vel[i]))

                    tmp_vals = accel[i]
                    all_accelerations = tmp_vals[np.where(tmp_vals > 0)[0]]
                    all_decelerations = tmp_vals[np.where(tmp_vals < 0)[0]]
                    if len(all_accelerations) == 0:
                        avg_accel_turn.append(np.float64(0.0))
                    else:
                        avg_accel_turn.append(
                            np.mean(all_accelerations) * diff_angle_rads / np.sqrt(np.sum(vel[i])))
                    if len(all_decelerations) == 0:
                        avg_decel_turn.append(np.float64(0.0))
                    else:
                        avg_decel_turn.append(
                            np.mean(all_decelerations) * diff_angle_rads / np.sqrt(np.sum(vel[i])))
                    turn_mem_dict.update({str(i): 0})

                    allangles = np.abs(np.diff(angles[i]))
                    allvels = vel[i]
                    allangles[
                        np.where(allangles > 180)[0]] = 360 - allangles[np.where(allangles > 180)[0]]
                    avg_vel_ang_turn.append(
                        np.mean(allvels[:-1] * ((np.pi / 180) * allangles)))

            self.features['avg_vel_turn'] = np.float64(np.mean(avg_vel_turn))
            self.features['avg_accel_turn'] = np.float64(
                np.mean(avg_accel_turn))
            self.features['avg_decel_turn'] = np.float64(
                np.mean(avg_decel_turn))
            self.features['avg_vel_ang_turn'] = np.float64(
                np.mean(avg_vel_ang_turn))

            self.validate_feature('avg_vel_turn')
            self.validate_feature('avg_accel_turn')
            self.validate_feature('avg_decel_turn')
            self.validate_feature('avg_vel_ang_turn')

    def compute_normal_tangent_accel_normhist(self, binsize=0.2):
        '''Compute tangential and turn accceleration'''

        # Compute the normal hist through the trip
        accel = np.array(self.trip_data['accel'])
        vel = np.array(self.trip_data['vel'])
        vel_vals_lt25 = np.where(vel <= 13.4)[0]
        vel_vals_gt25_lt50 = np.where((vel > 13.4) & (vel <= 26.8))[0]
        vel_vals_gt50 = np.where(vel > 26.8)[0]

        if len(vel_vals_lt25) == 0:
            self.features['p05_accel_lt25'] = np.float64(0.0)
            self.features['p25_accel_lt25'] = np.float64(0.0)
            self.features['p50_accel_lt25'] = np.float64(0.0)
            self.features['p75_accel_lt25'] = np.float64(0.0)
            self.features['p95_accel_lt25'] = np.float64(0.0)
        else:
            tmp_vals = accel[vel_vals_lt25]
            self.features['p05_accel_lt25'] = np.float64(
                np.percentile(tmp_vals, 5))
            self.features['p50_accel_lt25'] = np.float64(
                np.percentile(tmp_vals, 50))
            self.features['p95_accel_lt25'] = np.float64(
                np.percentile(tmp_vals, 95))
            self.features['p75_accel_lt25'] = np.float64(
                np.percentile(tmp_vals, 75))
            self.features['p25_accel_lt25'] = np.float64(
                np.percentile(tmp_vals, 25))

        self.validate_feature('p05_accel_lt25')
        self.validate_feature('p25_accel_lt25')
        self.validate_feature('p50_accel_lt25')
        self.validate_feature('p75_accel_lt25')
        self.validate_feature('p95_accel_lt25')

        if len(vel_vals_gt25_lt50) == 0:
            self.features['p05_accel_gt25_lt50'] = np.float64(0.0)
            self.features['p50_accel_gt25_lt50'] = np.float64(0.0)
            self.features['p95_accel_gt25_lt50'] = np.float64(0.0)
            self.features['p25_accel_gt25_lt50'] = np.float64(0.0)
            self.features['p75_accel_gt25_lt50'] = np.float64(0.0)
        else:
            tmp_vals = accel[vel_vals_gt25_lt50]
            self.features['p05_accel_gt25_lt50'] = np.float64(
                np.percentile(tmp_vals, 5))
            self.features['p50_accel_gt25_lt50'] = np.float64(
                np.percentile(tmp_vals, 50))
            self.features['p95_accel_gt25_lt50'] = np.float64(
                np.percentile(tmp_vals, 95))
            self.features['p25_accel_gt25_lt50'] = np.float64(
                np.percentile(tmp_vals, 25))
            self.features['p75_accel_gt25_lt50'] = np.float64(
                np.percentile(tmp_vals, 75))

        self.validate_feature('p05_accel_gt25_lt50')
        self.validate_feature('p50_accel_gt25_lt50')
        self.validate_feature('p95_accel_gt25_lt50')
        self.validate_feature('p25_accel_gt25_lt50')
        self.validate_feature('p75_accel_gt25_lt50')

        if len(vel_vals_gt50) == 0:
            self.features['p05_accel_gt50'] = np.float64(0.0)
            self.features['p50_accel_gt50'] = np.float64(0.0)
            self.features['p95_accel_gt50'] = np.float64(0.0)
            self.features['p25_accel_gt50'] = np.float64(0.0)
            self.features['p75_accel_gt50'] = np.float64(0.0)
        else:
            tmp_vals = accel[vel_vals_gt50]
            self.features['p05_accel_gt50'] = np.float64(
                np.percentile(tmp_vals, 5))
            self.features['p50_accel_gt50'] = np.float64(
                np.percentile(tmp_vals, 50))
            self.features['p95_accel_gt50'] = np.float64(
                np.percentile(tmp_vals, 95))
            self.features['p25_accel_gt50'] = np.float64(
                np.percentile(tmp_vals, 25))
            self.features['p75_accel_gt50'] = np.float64(
                np.percentile(tmp_vals, 75))

        self.validate_feature('p05_accel_gt50')
        self.validate_feature('p50_accel_gt50')
        self.validate_feature('p95_accel_gt50')
        self.validate_feature('p25_accel_gt50')
        self.validate_feature('p75_accel_gt50')

        turn_segments = self.tmp_data
        angles = np.array(self.trip_data['angles'])
        if len(turn_segments) == 0:
            self.features['p05_tan_accel'] = np.float64(0.0)
            self.features['p50_tan_accel'] = np.float64(0.0)
            self.features['p95_tan_accel'] = np.float64(0.0)
            self.features['p25_tan_accel'] = np.float64(0.0)
            self.features['p75_tan_accel'] = np.float64(0.0)
            self.features['num_turns'] = np.float64(0.0)
        else:
            tangential_accel = []
            for i in turn_segments:
                # Calculate radius of the curvature
                ang_one = np.abs(90 - np.abs(angles[i[-1]]))
                ang_two = np.abs(90 - np.abs(angles[i[0]]))
                central_angle = (ang_one + ang_two) * (np.pi / 180)
                arc_distance = np.sum(vel[i])
                radius = arc_distance / central_angle

                # Tangential acceleration is angular acceleration * radius of
                # curvature
                allangles = np.abs(np.diff(angles[i]))
                allangles[
                    np.where(allangles > 180)[0]] = 360 - allangles[np.where(allangles > 180)[0]]
                ang_accel = np.abs(np.diff(allangles))
                ang_accel[
                    np.where(ang_accel > 180)[0]] = 360 - ang_accel[np.where(ang_accel > 180)[0]]

                tangential_accel.extend(radius * ang_accel * (np.pi / 180))
            self.features['p05_tan_accel'] = np.float64(
                np.percentile(tangential_accel, 5))
            self.features['p50_tan_accel'] = np.float64(
                np.percentile(tangential_accel, 50))
            self.features['p95_tan_accel'] = np.float64(
                np.percentile(tangential_accel, 95))
            self.features['p25_tan_accel'] = np.float64(
                np.percentile(tangential_accel, 25))
            self.features['p75_tan_accel'] = np.float64(
                np.percentile(tangential_accel, 75))
            self.features['num_turns'] = np.float64(
                len(turn_segments) / self.features['total_distance'])

        self.validate_feature('p05_tan_accel')
        self.validate_feature('p50_tan_accel')
        self.validate_feature('p95_tan_accel')
        self.validate_feature('p25_tan_accel')
        self.validate_feature('p75_tan_accel')
        self.validate_feature('num_turns')

    #--------------------------------------------------------------------------

    def compute_stop_params(self, a_lim=1.2, v_lim=6):
        '''New stops method that also gives avg distance between stops and
        duration of stops, in addition to num_stops_per_km'''
        # Check that the dataframe has the columns necessary to do the
        # computation
        if 'vel' in self.trip_data.columns and 'accel' in self.trip_data.columns:

            stops_first = self.trip_data[:1]  # First point is always a stop
            stops_last = self.trip_data[-1:]  # Last point is always a stop

            stops_mid = self.trip_data[(
                self.trip_data['accel'] < a_lim) & (
                self.trip_data['accel'] > -a_lim) & ((
                    self.trip_data['vel'] < v_lim))]

            stop_points = pd.concat([
                stops_first, stops_first, stops_mid, stops_last, stops_last])

            X = stop_points[['x', 'y']]
            X = np.array(X)

            # Run DBSCAN to cluster stops
            db = DBSCAN(eps=5, min_samples=2).fit(X)  # eps is in m's
            labels = db.labels_

            # loop through each label
            num_stops = max(labels) + 1
            stop_durations = []
            stop_centers = []
            stop_gaps = [0]

            for idx, lbl in enumerate(range(0, num_stops)):
                # get all points in this cluster
                stop_locs = [stop_points.index.tolist()[i] for i in list(
                    np.where(labels == lbl)[0])]
                # Duration of time spent at stops
                stop_durations.append(len(stop_locs))
                # Center of stop locations
                stop_centers.append(np.mean(stop_locs))
            # Distance between stops in secs and in m
            stop_gaps.extend(np.diff(np.sort(stop_centers)))

            avg_stop_duration = np.mean(stop_durations)
            avg_stop_gap = np.mean(stop_gaps)
            num_stops_per_km = (num_stops) / (
                (self.features['total_distance']) / (1000))

            self.features['avg_stop_duration'] = np.float64(avg_stop_duration)
            self.features['avg_stop_gap'] = np.float64(avg_stop_gap)
            self.features['num_stops_per_km'] = np.float64(num_stops_per_km)

            self.validate_feature('avg_stop_duration')
            self.validate_feature('avg_stop_gap')
            self.validate_feature('num_stops_per_km')

        else:
            print "Cannot compute the velocity times acceleration value \
                because one of the columns is missing from the dataframe"

    #--------------------------------------------------------------------------
