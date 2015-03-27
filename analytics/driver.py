# Code for the Kaggle AXA challenge
#-----------------------------------
#    Copyright (C) 2015- by
#    Frank Lin <email@>
#    Jim Simpson <email@>
#    All rights reserved.
#    BSD license.
#-----------------------------------

import math
import pandas as pd
import numpy as np
import os
import trip
import cPickle


class Driver(object):

    '''Driver object class'''

    def __init__(self, file_path=''):
        ''' Initialize driver object and their trips'''
        self.driver = {}

        if file_path == "":
            print "No file path given, no driver was created"
        else:
            # Walk the file path given
            for file_name in os.listdir(file_path):
                trip_data = pd.read_csv(
                    os.path.join(file_path + '/', file_name))
                trip_obj = trip.Trip(trip_data)
                self.add_trip(file_name, trip_obj)

    def add_trip(self, trip_num, trip_obj):
        ''' Method adds the trip object to the driver object'''
        self.driver.update({trip_num: trip_obj})

    def rem_trip(self, trip_num, trip_data):
        ''' Method removes the trip object to the driver object'''
        self.driver.pop(trip_num)

    def save_driver(self, file_name):
        ''' Serialize and save as a Pickle object '''
        open(file_name, 'wb')
        cPickle.dump(self.driver, file_name)

    def load_driver(self, file_name):
        ''' Load in the pickle object that was saved'''
        open(file_name, 'rb')
        self.driver = cPickle.load(file_name)
