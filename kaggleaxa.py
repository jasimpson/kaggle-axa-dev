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

import analytics.driver
import analytics.combiner
import analytics.preprocessor
import analytics.classifier

import csv
import time
import pickle
from datetime import datetime

from joblib import Parallel, delayed
import multiprocessing


###############################################################################

def process_driver(driver_id):
    """Apply entire processing to a single driver"""

    # TODO: Store and run classifier separately for each driver
    X_true, y_true = combiner_object.get_features_for_true_driver(driver_id)

    print "Fake drivers have %d trips with %d features each" % (len(X_fake),
                                                                len(X_fake[0]))
    print "True drivers have %d trips with %d features each" % (len(X_true),
                                                                len(X_true[0]))

    ###########################################################################
    # Preprocessor section

    preprocessor_object = analytics.preprocessor.Preprocessor()

    # Apply all preprocessing steps
    X_train, X_test, y_train, y_test, X_true = \
        preprocessor_object.apply_preprocessing(
            X_fake, X_true, y_fake, y_true)

    ###########################################################################
    # Classifier section

    # Select the classifier
    classifier_name = 'RandomForestClassifier'
    #classifier_name = 'GradientBoostingClassifier'
    #classifier_name = 'LogisticRegression'
    #classifier_name = 'GMM'

    # Instantiate the classifier object
    classifier_object = analytics.classifier.Classifier(classifier_name)

    # Build the model
    if gridsearch_mode:
        # Gridsearch the model
        clf_score_train, clf_score_test = \
            classifier_object.gridsearch_model(
                X_train, y_train, X_test, y_test)
    else:
        # Train the model
        clf_score_train, clf_score_test = \
            classifier_object.train_model(
                X_train, y_train, X_test, y_test)

    # Predict using the model
    proba = True
    y_pred = classifier_object.predict_model(X_true, proba)

    # Calculate pr thresh based on desired pred class weights
    desired_ratio_of_1s_to_0s = 0.75  # enter this value
    proba_thresh = np.percentile(
        y_pred, (1 - desired_ratio_of_1s_to_0s) * 100)

    # Threshold Probabilities
    if proba:
        y_pred = [1 if (p > proba_thresh) else 0 for p in y_pred]

    # Calculate true score (will always be ratio for this method)
    actual_ratio_of_1s_to_0s = np.mean(y_pred)
    clf_score_true = 1 - np.mean(np.logical_xor(y_true, y_pred))

    # Feature Importances
    if save_feature_importances:
        clf_feature_importances = classifier_object.feature_importances()
    else:
        clf_feature_importances = []

    ###########################################################################

    print "Classifier scored %f on training dataset" % (clf_score_train)
    print "Classifier scored %f on test dataset" % (clf_score_test)
    print "Classifier scored %f on true dataset" % (clf_score_true)

    results = \
        driver_id, \
        clf_score_train, \
        clf_score_test, \
        y_pred, \
        clf_feature_importances

    return results

###############################################################################
# Set input parameters

start_time = time.time()

# Set debug mode
# Reads in features for fake drivers from previously saved pickle
# Sets list_of_all_drivers to smaller debug set
debug_mode = False

# Set gridsearch mode
# Implements gridsearch_model() instead of train_model()
gridsearch_mode = False

# Set parallel processing mode
parfor_mode = True

# Save feature importances if available
save_feature_importances = True

# TODO: Move this to config file
# Set location of data folder
data_root_path = os.path.expanduser("~") + '/data/kaggle-axa-data/drivers/'

# Fake driver parameters
num_of_fake_drivers = 200  # drivers
num_of_trips_per_fake_driver = 1  # trips/driver

###############################################################################
# Initializers

# Initialize submission list
csv_driver_ids = []
csv_trips_ids = []
csv_y_pred = []

# Initialize score averages
clf_score_train_sum = 0.0
clf_score_test_sum = 0.0
clf_score_train_avg = 0.0
clf_score_test_avg = 0.0

# Create submissions folder if it doesn't already exist
try:
    os.mkdir('submissions')
except OSError:
    pass

# Create pickles folder if it doesn't already exist
try:
    os.mkdir('pickles')
except OSError:
    pass

###############################################################################
# Combiner section

combiner_object = analytics.combiner.Combiner(data_root_path)

###############################################################################
# Prepare to loop through all drivers

# Set list of drivers to use
if debug_mode:
    list_of_all_drivers = [1, 2, 3]
else:
    list_of_all_drivers = combiner_object.get_list_of_drivers(data_root_path)

# Get num of drivers being processed
num_of_drivers = len(list_of_all_drivers)

# Get list of features being used
feature_names = combiner_object.get_list_of_feature_names()

# Get num of features being used
num_of_features = len(feature_names)

# List to store feature importances
if save_feature_importances:
    clf_feature_importances_list = np.zeros([1, num_of_features])
else:
    clf_feature_importances_list = []

print "###################################################################"

print "Using these %d features:" % (num_of_features)
print('\n'.join(fn for fn in feature_names))

print "###################################################################"
print "Processing started for %d drivers" % (num_of_drivers)

if debug_mode:
    print "Debug mode on"
else:
    print "Debug mode off"

if gridsearch_mode:
    print "Gridsearch mode on"
else:
    print "Gridsearch mode off"

if parfor_mode:
    print "Parfor mode on"
else:
    print "Parfor mode off"


# Get features for fake drivers
if debug_mode:
    X_fake = pickle.load(
        open(os.path.join('pickles', 'data_X_fake.pickle'), "rb"))
    y_fake = pickle.load(
        open(os.path.join('pickles', 'data_y_fake.pickle'), "rb"))
else:
    print "Processing fake drivers..."
    X_fake, y_fake = combiner_object.get_features_for_fake_drivers(
        num_of_fake_drivers, num_of_trips_per_fake_driver)

    pickle.dump(
        X_fake, open(os.path.join('pickles', 'data_X_fake.pickle'), "wb"))
    pickle.dump(
        y_fake, open(os.path.join('pickles', 'data_y_fake.pickle'), "wb"))

###############################################################################
# Loop through all drivers

if parfor_mode:
    # Use all cores
    num_cores = multiprocessing.cpu_count()
    print("Number of cores = " + str(num_cores))
    print "Processing drivers..."
    print "...this will take a while..."

    results = Parallel(n_jobs=num_cores, verbose=50)(delayed(
        process_driver)(driver_id) for driver_id in list_of_all_drivers)

    for index, res in enumerate(results):

        # Unzip results
        driver_id = res[0]
        clf_score_train = res[1]
        clf_score_test = res[2]
        y_pred = res[3]
        clf_feature_importances = res[4]

        # Save
        csv_driver_ids.extend([driver_id] * 200)
        csv_trips_ids.extend(range(1, 201))
        csv_y_pred.extend(y_pred)
        clf_feature_importances_list = np.vstack(
            (clf_feature_importances_list, clf_feature_importances))

        # Keep track of scores for averaging
        clf_score_train_sum = clf_score_train_sum + clf_score_train
        clf_score_test_sum = clf_score_test_sum + clf_score_test

else:
    # Use only one core
    # Iterate through each driver
    for index, driver_id in enumerate(list_of_all_drivers):

        start_loop_time = time.time()
        print "Processing driver %d - %d of %d" % (driver_id, index + 1,
                                                   num_of_drivers)

        # Apply all processing to current driver
        results = process_driver(driver_id)

        # Unzip results
        driver_id = results[0]
        clf_score_train = results[1]
        clf_score_test = results[2]
        y_pred = results[3]
        clf_feature_importances = results[4]

        # Save
        csv_driver_ids.extend([driver_id] * 200)
        csv_trips_ids.extend(range(1, 201))
        csv_y_pred.extend(y_pred)
        clf_feature_importances_list = np.vstack(
            (clf_feature_importances_list, clf_feature_importances))

        # Keep track of scores for averaging
        clf_score_train_sum = clf_score_train_sum + clf_score_train
        clf_score_test_sum = clf_score_test_sum + clf_score_test

        print "Processing driver took %f seconds to run" % (
            time.time() - start_loop_time)

###############################################################################
# Gather results from all drivers

# Save feature importances if available
if save_feature_importances:
    # Delete header (zeros) from clf_feature_importances_list array
    clf_feature_importances_list = np.delete(
        clf_feature_importances_list, (0), axis=0)
    # Average importances
    clf_feature_importances_avg = np.mean(clf_feature_importances_list, axis=0)
    # Pickle importances for later
    pickle.dump(clf_feature_importances_avg, open(os.path.join(
        'pickles', 'data_clf_feature_importances_avg.pickle'), "wb"))
    pickle.dump(feature_names, open(os.path.join(
        'pickles', 'data_feature_names.pickle'), "wb"))

# Calculate score averages
clf_score_train_avg = clf_score_train_sum / num_of_drivers
clf_score_test_avg = clf_score_test_sum / num_of_drivers

# Format submission output
out_int = zip(csv_driver_ids, csv_trips_ids, csv_y_pred)
out_str = [str(x[0]) + '_' + str(x[1]) + ',' + str(int(x[2])) for x in out_int]

# Write submission output
with open(os.path.join(
        'submissions', 'submission_{}.csv'.format(datetime.now().strftime(
            '%Y-%m-%d_%H-%M-%S'))), 'w') as writefile:
    writefile.write("driver_trip,prob\n")
    for line in out_str:
        writefile.write("%s\n" % line)

print "Classifier scored average %f on training set" % (clf_score_train_avg)
print "Classifier scored average %f on test set" % (clf_score_test_avg)

print "Entire processing took %f seconds to run" % (time.time() - start_time)

print "###################################################################"
print "###################################################################"


###############################################################################
