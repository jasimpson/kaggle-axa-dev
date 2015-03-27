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
import copy
import numpy as np
import scipy as sp
import pandas as pd
import scipy.signal as sg


class SeriesExtractor(object):

    """The SeriesExtractor extracts time series data from trip data.

    This class serves the purpose of keeping all the time series extraction
    methods together. The class also makes sure each time series is strictly
    validated before passing on to the FeatureExtractor.

    Note:
      All outputs of class members are written to self.trip_data

    Args:
      None

    Attributes:
      None

    """

    def __init__(self, trip_data):
        self.series = []
        self.tmp_data = []
        self.trip_data = trip_data

        self.compute_cleanxy_table()
        self.compute_distance_table()
        self.compute_accel_table()
        self.compute_jerk_table()
        self.compute_angle_table()
        self.compute_quadrants_table()

    def get_series(self):
        return self.trip_data

    def get_tmpdata(self):
        return self.tmp_data

    #-------------------------------------------------------------------------
    def compute_cleanxy_table(self):
        '''Cleans the xy of potential jumps in the data.
            Simple approach is to look for any instantaneous change 
            in the coordinates by 100 meters in x or y and to shift the 
            value. This would mean that there was a distinct 25 meter change
            per second.'''
        xvals = np.array(self.trip_data['x'])
        yvals = np.array(self.trip_data['y'])

        xindices = np.sort(np.where(np.abs(np.diff(xvals)) > 67)[0])
        yindices = np.sort(np.where(np.abs(np.diff(yvals)) > 67)[0])

        for ind in xindices:
            # Shift all values post the index value (if 0, then 1 to end)
            diffVal = xvals[ind + 1] - xvals[ind]
            self.trip_data['x'][
                ind + 1:] = self.trip_data['x'][ind + 1:] - diffVal
        for ind in yindices:
            # Shift all values post the index value (if 0, then 1 to end)
            diffVal = yvals[ind + 1] - yvals[ind]
            self.trip_data['y'][
                ind + 1:] = self.trip_data['y'][ind + 1:] - diffVal

        # Apply the Kalman filter to the x-y values
        #new_xy = np.column_stack((self.trip_data['x'],self.trip_data['y']))
        #kf     = KalmanFilter(initial_state_mean=[0,0], n_dim_obs=2)
        #kf     = kf.em(new_xy,n_iter=5)
        #(filtered_state_means, filtered_state_covariances) = kf.smooth(new_xy)
        #self.trip_data['kx'] = filtered_state_means[:,0]
        #self.trip_data['ky'] = filtered_state_means[:,1]

        self.trip_data['x'] = sg.savgol_filter(
            np.array(self.trip_data['x']), 15, 3)
        self.trip_data['y'] = sg.savgol_filter(
            np.array(self.trip_data['y']), 15, 3)

    #-------------------------------------------------------------------------
    def compute_distance_table(self):
        ''' Compute the distance from the data, matrix calcs'''

        if 'x' in self.trip_data.columns and 'y' in self.trip_data.columns:
            x1_vals = np.array(self.trip_data['x'])
            y1_vals = np.array(self.trip_data['y'])
            x2_vals = x1_vals[1:len(x1_vals)]
            y2_vals = y1_vals[1:len(y1_vals)]
            x2_vals = np.append(x2_vals, x2_vals[-1])
            y2_vals = np.append(y2_vals, y2_vals[-1])
            dist_vals = np.hypot(x2_vals - x1_vals, y2_vals - y1_vals)
            self.trip_data['dist'] = dist_vals
            self.trip_data['vel'] = dist_vals
            #self.trip_data['smooth_vel']  = smooth_dist_vals
        else:
            print 'This data frame does not have x and y values to compute distance and velocity'

    #-------------------------------------------------------------------------
    def compute_accel_table(self):
        ''' Compute the acceleration from the computed distance/velocity '''

        if 'vel' in self.trip_data.columns:
            vel = self.trip_data['vel']
            #vel   = self.trip_data['smooth_vel']
            accel = np.diff(vel)
            accel = np.append(accel, 0.0)
            self.trip_data['accel'] = accel
        else:
            print 'Cannot compute the acceleration dataframe column because no velocity column exists'

    #-------------------------------------------------------------------------
    def compute_jerk_table(self):
        '''Compute the jerk or derivative of the _acceleration value'''

        # Check to see if the trip_data object has an _acceleration column
        if 'accel' in self.trip_data.columns:
            accel = np.array(self.trip_data['accel'])
            jerk = np.diff(accel)
            # Append a 0 value to the end of the array to fit the pandas
            # dataframe
            jerk = np.append(jerk, 0.0)
            self.trip_data['jerk'] = jerk
        else:
            print "No acceleration column in the dataframe to compute the jerk value"

    #-------------------------------------------------------------------------
    def compute_angle_table(self):
        '''Compute the angle table. First, '''

        x1_vals = np.array(self.trip_data['x'])
        y1_vals = np.array(self.trip_data['y'])
        x2_vals = x1_vals[1:len(x1_vals)]
        y2_vals = y1_vals[1:len(y1_vals)]
        x2_vals = np.append(x2_vals, x2_vals[-1])
        y2_vals = np.append(y2_vals, y2_vals[-1])

        radvals = np.arctan2(y2_vals - y1_vals, x2_vals - x1_vals)
        self.trip_data['angles'] = 180 * radvals / np.pi
        #self.trip_data['radvals'] = radvals
        self.trip_data['angles'][
            len(x1_vals) - 1] = self.trip_data['angles'][len(x1_vals) - 2]
        #self.trip_data['radvals'][len(x1_vals)-1] = self.trip_data['radvals'][len(x1_vals) - 2]

    def compute_quadrants_table(self):
        # Attempts to compute curved point areas vs straight point areas based on
        # looking at changes in heading

        if 'angles' in self.trip_data.columns and 'vel' in self.trip_data.columns:
            ang = np.array(self.trip_data['angles'])
            vel = np.array(self.trip_data['vel'])
            quad = np.array(self.trip_data['angles'])
            quad[np.where((quad <= 45) & (quad > 0))[0]] = 0  # Quadrant 1
            quad[np.where((quad <= 90) & (quad > 45))[0]] = 1  # Quadrant 2
            quad[np.where((quad <= 135) & (quad > 90))[0]] = 2  # Quadrant 3
            quad[np.where((quad <= 180) & (quad > 135))[0]] = 3  # Quadrant 4
            quad[np.where((quad <= -135) & (quad > -180))[0]] = 4  # Quadrant 5
            quad[np.where((quad <= -90) & (quad > -135))[0]] = 5  # Quadrant 6
            quad[np.where((quad <= -45) & (quad > -90))[0]] = 6  # Quadrant 7
            quad[np.where((quad <= 0) & (quad > -45))[0]] = 7  # Quadrant 8
            self.trip_data['quadrants'] = quad

            # Compute turn areas vs straight areas, turn areas = 0 value,
            # straight = 1
            allindices = np.array([1] * len(quad))
            quad_changes = np.where(np.abs(np.diff(quad)) > 0)[0]

            # Find the points that should be labeled as a turn
            turnpoints = list()
            tmppoints = list()
            for i in quad_changes:
                # Do not do this if i = 1 or i = length-2 (2nd point or 2nd to
                # last point)
                if i != 1 and i != len(quad) - 2 and i not in turnpoints:

                    if (ang[i] >= 0 and ang[i + 1] >= 0):
                        if (ang[i + 1] < ang[i]):  # CW
                            multiplier = -1
                        else:  # CCW
                            multiplier = 1
                    elif (ang[i] <= 0 and ang[i + 1] <= 0):
                        if (np.abs(ang[i + 1]) > np.abs(ang[i])):  # CW
                            multiplier = -1
                        else:  # CCW
                            multiplier = 1
                    elif (ang[i] >= 0 and ang[i + 1] <= 0):
                        if (ang[i + 1] > (ang[i] - 180)):  # CW
                            multiplier = -1
                        else:
                            multiplier = 1
                    elif (ang[i] <= 0 and ang[i + 1] >= 0):
                        if (ang[i + 1] > (ang[i] + 180)):  # CW
                            multiplier = -1
                        else:
                            multiplier = 1

                    # Go backward first to find the starting point of the turn
                    # start
                    deg_change = 5 * multiplier
                    turn_indices = [i]
                    start_index = i - 1
                    next_index = i - 1
                    start_angle = ang[i - 1]
                    change_angle = np.abs(ang[next_index] - start_angle)
                    if change_angle > 180:
                        change_angle = 360 - change_angle
                    while((multiplier * deg_change) > 0.5 and
                            change_angle < 180 and
                            start_index > 0):
                        next_index -= 1
                        deg_change = ang[start_index] - ang[next_index]
                        turn_indices.append(start_index)
                        start_index = next_index
                        change_angle = np.abs(ang[next_index] - start_angle)
                        if change_angle > 180:
                            change_angle = 360 - change_angle

                    # Go forward until the end - start angle > 180 degrees
                    # Or the angle is less than 0.5 degrees
                    start_angle = ang[start_index]

                    # Go forward until you find that the angle change is > 0
                    # degrees or quad changes
                    start_index = i + 1
                    next_index = i + 1
                    deg_change = 5 * multiplier
                    change_angle = np.abs(ang[next_index] - start_angle)
                    if change_angle > 180:
                        change_angle = 360 - change_angle
                    while((multiplier * deg_change) > 0.5 and
                            change_angle < 180 and
                            start_index < (len(quad) - 1)):
                        next_index += 1
                        deg_change = ang[next_index] - ang[start_index]
                        turn_indices.append(start_index)
                        start_index = next_index
                        change_angle = np.abs(ang[next_index] - start_angle)
                        if change_angle > 180:
                            change_angle = 360 - change_angle

                    turn_indices = np.sort(np.unique(np.array(turn_indices)))
                    if (len(turn_indices) > 5 and
                            np.mean(vel[turn_indices])) > 1:
                        change_angle = np.abs(
                            ang[turn_indices[-1]] - ang[turn_indices[0]])
                        if change_angle > 180:
                            change_angle = 360 - change_angle
                        if change_angle > 45:
                            turnpoints.extend(turn_indices)
                            tmppoints.append(turn_indices)

            turnpoints = np.sort(np.unique(turnpoints)).astype('int')
            allindices[turnpoints] = 0
            self.trip_data['turn_points'] = allindices
            self.tmp_data = tmppoints
        else:
            print "No angle columnmi in the dataframe to compute the quadrants"
