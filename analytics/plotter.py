# Code for the Kaggle AXA challenge
#-----------------------------------
#    Copyright (C) 2015- by
#    Frank Lin <email@>
#    Jim Simpson <email@>
#    All rights reserved.
#    BSD license.
#-----------------------------------

import math
import numpy
import pandas
import driver
import trip
import matplotlib.pyplot as plt

from scipy.interpolate import spline


def plotDistribution_acrossdrivers(driverobject, feature1='avg_acceleration', binsize=1):
    allfeaturevals = list()
    for allkeys in driverobject.driver.keys():
        allfeaturevals.append(driverobject.driver[allkeys].features[feature1])
    allfeaturevals = numpy.array(allfeaturevals)
    minval = math.floor(numpy.min(allfeaturevals))
    maxval = math.ceil(numpy.max(allfeaturevals))
    bins = numpy.arange(minval, maxval, binsize)
    plt.hist(allfeaturevals, bins=bins)
    plt.show()


def plotDistribution_withindrivers(driverobject, tripnum, feature1='accel', binsize=1):
    tripdata = driverobject.driver[tripnum]
    histdata = numpy.array(tripdata.datatable[feature1])
    minval = math.floor(numpy.min(histdata))
    maxval = math.ceil(numpy.max(histdata))
    bins = numpy.arange(minval, maxval, binsize)
    plt.hist(histdata, bins=bins)
    plt.show()


def plot2Dscatter_acrossdrivers(driverobject, feature1='avg_acceleration', feature2='avg_velocity'):
    feature1vals = list()
    feature2vals = list()
    for allkeys in driverobject.driver.keys():
        feature1vals.append(driverobject.driver[allkeys].features[feature1])
        feature2vals.append(driverobject.driver[allkeys].features[feature2])
    plt.plot(feature1vals, feature2vals, '.')
    plt.show()


def plot2Dscatter_withindrivers(driverobject, tripnum, feature1='accel', feature2='dist'):
    tripdata = driverobject.driver[tripnum]
    plt.plot(tripdata.datatable[feature1], tripdata.datatable[feature2], '.')
    plt.show()


def plotDistribution_alldrivers(filepath, driverlist=[], feature1='avg_acceleration', binsize=1):
    if driverlist == []:
        pass
    else:
        for driverName in driverlist:
            temppath = filepath + "/" + str(driverName)
            driverobject = driver.Driver(temppath)

            allfeaturevals = list()
            for allkeys in driverobject.driver.keys():
                allfeaturevals.append(
                    driverobject.driver[allkeys].features[feature1])
            allfeaturevals = numpy.array(allfeaturevals)
            minval = math.floor(numpy.min(allfeaturevals))
            maxval = math.ceil(numpy.max(allfeaturevals))
            bins = numpy.arange(minval, maxval, binsize)
            y, binEdges = numpy.histogram(allfeaturevals, bins=bins)
            bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
            plt.plot(bincenters, y, '-')
    plt.show()
