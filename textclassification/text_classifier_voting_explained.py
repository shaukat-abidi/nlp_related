#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  text_classifier_voting_explained.py
#  
#  Copyright 2018 Syed Shaukat Raza Abidi <abi008@demak-ep>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import sys
import os
import codecs
import numpy as np
import pickle

from common_functions import cf
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn import linear_model
from scipy.sparse import hstack
import pandas as pd

from time import time
from copy import deepcopy

# Save models here
# Output models will be saved here
model_dir = './voting_result/'

#'''
#########################################
# load data instead of reading it all
########################################

###############################################
# Client's data is stored in pickled object 
###############################################
data_stored = './pickled_data/dataread_full_v1.pk'
# File is not uploaded due to client's property (Will be emailed separately)
# Just provide the path to .pk file

data_obj = cf.data()

with open(data_stored, 'rb') as fout:
    data_obj.xdata, data_obj.ylabels, data_obj.details, data_obj.tot_train_files, data_obj.tot_test_files, data_obj.tot_positive_files, data_obj.tot_negative_files = pickle.load(fout)

#############################
# Verification of data read
##############################
print('Successfully read %d positive and negative pages' %len(data_obj.details))
print('Total train files (divide it by 2 b/c of _ref.txt and _noref.txt): %d Total test files (divide it by 2 b/c of _ref.txt and _noref.txt): %d'%(data_obj.tot_train_files, data_obj.tot_test_files) )
print('Positive files: %d Negative files: %d (both should be same)'%(data_obj.tot_positive_files, data_obj.tot_negative_files) )
assert(data_obj.tot_positive_files == data_obj.tot_negative_files)
assert(len(data_obj.xdata) != 0)
assert(len(data_obj.xdata) == len(data_obj.ylabels) == len(data_obj.details))
#''' 

#################################
# Get train/test data objects
#####################################
train_obj, test_obj = cf.func_segregate_traintest_obj(data_obj)

##############################################
# Get pos and neg counts from training object
################################################
dict_count_posnegs = cf.func_return_count(train_obj.ylabels)
print(dict_count_posnegs)

#################################################################################
# Get a dictionary with indices of positive and negative samples in train_obj
#################################################################################
posnegind_dict = cf. func_ret_posnegind_dict(train_obj)
#posnegind_dict

#################################
# Getting the stats of test set
##################################
print(cf.func_return_count(test_obj.ylabels))


#####################################
# Start datasampling here (Naive/Random sampling)
######################################
percentage_list = np.linspace(0.01,1,100)

#############################################
# Initialize dataframes to collect results
##########################################
df_filteredtrain = pd.DataFrame(columns=['Description', 'TRAIN_NEG_SAMPLING','feats','tp','tn','fp','fn','TOT_POS','TOT_NEG','Accuracy','Precision','Recall','F1-Score'])
df_test = pd.DataFrame(columns=['Description', 'TRAIN_NEG_SAMPLING','feats','tp','tn','fp','fn','TOT_POS','TOT_NEG','Accuracy','Precision','Recall','F1-Score'])
df_fulltrain = pd.DataFrame(columns=['Description', 'TRAIN_NEG_SAMPLING','feats','tp','tn','fp','fn','TOT_POS','TOT_NEG','Accuracy','Precision','Recall','F1-Score'])

# Prepare df_dict
df_dict = {}
df_dict['df_filteredtrain'] = df_filteredtrain
df_dict['df_test'] = df_test
df_dict['df_fulltrain'] = df_fulltrain


#############################################################################
# For sampled data, running all the classifiers avaliable in scikit learn
############################################################################
for _per in percentage_list:
    _select_neg_samples = np.int(np.floor(_per * dict_count_posnegs[0])) # Select these many negative random samples
    print('=' * 50)
    print('Selecting %d negative samples (%d Percent Selected)' %(_select_neg_samples, _per*100) )
    
    # Base classifiers for Voting
    clf1 = AdaBoostClassifier(n_estimators=10)
    clf2 = AdaBoostClassifier(n_estimators=100)
    clf3 = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)
    clf4 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
    clf5 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf6 = GradientBoostingClassifier(loss='exponential', n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0) 
    clf7 = linear_model.SGDClassifier(loss='hinge', penalty='None', max_iter=1000, tol=1e-3)
    clf8 = linear_model.SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, tol=1e-3)
    clf9 = linear_model.SGDClassifier(loss='hinge', penalty='l1', max_iter=1000, tol=1e-3)
    clf10 = linear_model.SGDClassifier(loss='hinge', penalty='elasticnet', max_iter=1000, tol=1e-3)
    clf11 = linear_model.SGDClassifier(loss='perceptron', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty=None)
    clf12 = linear_model.SGDClassifier(loss='perceptron', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='l2')
    clf13 = linear_model.SGDClassifier(loss='perceptron', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='l1')
    clf14 = linear_model.SGDClassifier(loss='perceptron', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='elasticnet')
    clf15 = linear_model.SGDClassifier(loss='modified_huber', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty=None)
    clf16 = linear_model.SGDClassifier(loss='modified_huber', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='l2')
    clf17 = linear_model.SGDClassifier(loss='modified_huber', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='l1')
    clf18 = linear_model.SGDClassifier(loss='modified_huber', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='elasticnet')
    clf19 = linear_model.SGDClassifier(loss='squared_hinge', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty=None)
    clf20 = linear_model.SGDClassifier(loss='squared_hinge', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='l2')
    clf21 = linear_model.SGDClassifier(loss='squared_hinge', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='l1')
    clf22 = linear_model.SGDClassifier(loss='squared_hinge', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='elasticnet')
    clf23 = linear_model.SGDClassifier(loss='log', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty=None)
    clf24 = linear_model.SGDClassifier(loss='log', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='l2')
    clf25 = linear_model.SGDClassifier(loss='log', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='l1')
    clf26 = linear_model.SGDClassifier(loss='log', eta0=1, max_iter=1000, tol=1e-3, learning_rate='optimal', penalty='elasticnet')
    clf27 = MultinomialNB()
    clf28 = RidgeClassifier(tol=1e-2, solver="sag")
    clf29 = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1)
    clf30 = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1)
    clf31 = RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=-1)
    clf32 = RandomForestClassifier(n_estimators=40, random_state=0, n_jobs=-1)
    clf33 = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    
    #################################
    # Create Voting Classifier Object
    ##################################
    clf_obj = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2), ('clf3', clf3), ('clf4', clf4), ('clf5', clf5), ('clf6', clf6), ('clf7', clf7), ('clf8', clf8), ('clf9', clf9), ('clf10', clf10), ('clf11', clf11), ('clf12', clf12), ('clf13', clf13), ('clf14', clf14), ('clf15', clf15), ('clf16', clf16), ('clf17', clf17), ('clf18', clf18), ('clf19', clf19), ('clf20', clf20), ('clf21', clf21), ('clf22', clf22), ('clf23', clf23), ('clf24', clf24), ('clf25', clf25), ('clf26', clf26), ('clf27', clf27), ('clf28', clf28), ('clf29', clf29), ('clf30', clf30), ('clf31', clf31), ('clf32', clf32), ('clf33', clf33)], voting='hard')
    description = 'Voting'

    #############
    # Save model
    ##############
    _filename = str(_per*100) + description + '.pk'
    filepath = model_dir + _filename
    #print('Saving model to: %s' %model_dir)

    ###########################
    # Prepare filtered_trainobj
    ###########################
    filtered_trainobj = cf.func_ret_filtered_obj(train_obj, posnegind_dict, _select_neg_samples)

    # Prepare Feature Dictionary
    feat_dict = {}
    feat_dict['filtered_trainobj'] = filtered_trainobj
    feat_dict['fulltrain_obj'] = train_obj
    feat_dict['test_obj'] = test_obj
    feat_dict['verbose'] = True
    feat_dict['classifier_obj'] = clf_obj
    feat_dict['description'] = _filename
    feat_dict['_per'] = _per
    feat_dict['filepath'] = filepath
    
    try:
        # #########################################
        # Get features in Dictionary
        # Unigram features are used (very simple)
        ##########################################
        
        cf.func_fit_and_return_feats(feat_dict)

        # Learning and prediction
        ##########################
        cf.func_learn_and_predict(feat_dict, df_dict, save_model=True)
        
        # Save Dataframes Instantly
        #############################
        df_filteredtrain.to_csv(model_dir+'df_filteredtrain.csv')
        df_test.to_csv(model_dir+'df_test.csv')
        df_fulltrain.to_csv(model_dir+'df_fulltrain.csv')
    except Exception as e:
        print(e)

