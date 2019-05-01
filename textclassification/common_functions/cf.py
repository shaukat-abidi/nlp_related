#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  cf.py
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

# This module contains all common functions and classes
import os
import codecs
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import hstack
from time import time
from copy import deepcopy
import pandas as pd

from collections import Counter

class data:
    def __init__(self):
        self.xdata = []
        self.ylabels = np.empty((0,), dtype=np.int64)
        self.details = []
        self.tot_train_files = 0
        self.tot_test_files = 0
        self.tot_positive_files = 0
        self.tot_negative_files = 0

def func_generate_dataobj(data_obj, base_dir, _filename, label, type_data, encoding_string):
    
    ####################################################################
    # ATTENTION: data_obj is passed by reference (no value is returned)
    ####################################################################
    
    # data_obj       : This method will populate data_obj (instance of class data)
    # base_dir        : Directory where files are located, each page of a file is one document
    # label          : Labels for file
    # encoding_string: encoding used to read text files
    # type_data      : 'train' or 'test'
    
    # init page nums
    page_num = 1
    _text = ''
    #print('Reading file: %s' %(_filename))

    # read file
    _filepath = base_dir + '/' + _filename
    _fileread = codecs.open(_filepath, 'r', encoding=encoding_string)

    for line in _fileread:
        #print(len(line), '-----', '\f' in line, '-----',ascii(line))
        _text = _text + line

        if '\f' in line:
            # Clean the text 
            _text = func_clean_text(_text)
            # Append the following to the list
            data_obj.xdata.append(_text)
            data_obj.ylabels = np.append(data_obj.ylabels, np.array([label]), axis=0)
            data_obj.details.append({'base_dir': base_dir, 'filename': _filename, 'type': type_data, 'page_number': page_num, 'label': label})


            # Reset _text
            _text = ''

            # Increment page_num
            page_num = page_num + 1
            #print('moving on to page_num: %d' %page_num)



    # Do the final step and finishing iterating over all lines

    # Append the following to the list
    # Clean the text 
    _text = func_clean_text(_text)
    data_obj.xdata.append(_text)
    data_obj.ylabels = np.append(data_obj.ylabels, np.array([label]), axis=0)
    data_obj.details.append({'base_dir': base_dir, 'filename': _filename, 'type': type_data, 'page_number': page_num, 'label': label})

    _fileread.close()

def func_generate_dataobj_in_production(_fileread):
    # data_obj       : This method will populate data_obj (instance of class data)
    # label          : Labels for file
    # encoding_string: encoding used to read text files
    # type_data      : 'deployment'
    
    # Useful objects and variables 
    data_obj = data()
    type_data = 'deployment'
    label=-1
    encoding_string = 'utf-8' #'utf-16le'
    
    # init page nums
    page_num = 1
    _text = ''
    
    for line in _fileread.splitlines():
        #print(len(line), '-----', '\f' in line, '-----',ascii(line))
        _text = _text + line

        if '\f' in line:
            # Clean the text 
            _text = func_clean_text(_text)
            # Append the following to the list
            data_obj.xdata.append(_text)
            data_obj.ylabels = np.append(data_obj.ylabels, np.array([label]), axis=0)
            data_obj.details.append({'base_dir': '/', 'filename': '/runtime_file', 'type': type_data, 'page_number': page_num, 'label': label})


            # Reset _text
            _text = ''

            # Increment page_num
            page_num = page_num + 1
            #print('moving on to page_num: %d' %page_num)



    # Do the final step and finishing iterating over all lines

    # Append the following to the list
    # Clean the text 
    _text = func_clean_text(_text)
    data_obj.xdata.append(_text)
    data_obj.ylabels = np.append(data_obj.ylabels, np.array([label]), axis=0)
    data_obj.details.append({'base_dir': '/', 'filename': '/runtime_file', 'type': type_data, 'page_number': page_num, 'label': label})
    
    # return
    return data_obj

def func_clean_text(_text):
    # Clean text
    _text = ' '.join(_text.split()) #remove whitespaces
    _text = ''.join([i if ord(i) < 128 else '' for i in _text]) #remove undesirable words
    _text = ' '.join(_text.split()) # Finally removing additional spaces
    return _text

def func_is_pos_instance(_ident_str):
    if _ident_str == 'ref.txt':
        return True
    else:
        return False

def func_is_neg_instance(_ident_str):
    if _ident_str == 'noref.txt':
        return True
    else:
        return False

def func_return_count(_array):
    _dict = Counter(_array)
    return _dict

def func_ret_posnegind_dict(passed_obj):
    # This function will return indices of all positive and negative samples available inside passed_obj
    # The result is stored in dictionary: posnegind_dict['pos_indices'] and posnegind_dict['neg_indices'] 
    assert(len(passed_obj.xdata) == len(passed_obj.ylabels))
    
    # Init dictionary
    posnegind_dict = {'pos_indices': [], 'neg_indices': []}

    for _iter in range(0,len(passed_obj.xdata)):
        if passed_obj.ylabels[_iter] == 1:
            posnegind_dict['pos_indices'].append(_iter)
        else:
            posnegind_dict['neg_indices'].append(_iter)
    
    #return
    return posnegind_dict

def func_ret_filtered_obj(passed_obj, posnegind_dict, _select_indices):
    # This function does the following: It retains all positive pages in passed_obj and select #_select_indices negative pages from passed_obj
    # This function is responsible to perform random sampling (permutation) to select negative pages. We are doing it because we have skewed data distribution
    # i.e. few positives and lots of negatives
    
    # Arguments
    # passed_obj: data object containing all pos and neg samples
    # posnegind_dict: It has two keys: 'pos_indices' and 'neg_indices' contains list. These list contains indices of pos/neg samples in passed_obj  
    # _select_indices: # of negative examples to be selected
    
    # To return: 
    # filtered_obj: downsampled version of passed_obj


    # Filtered object
    filtered_obj = data()

    # Permuted list of indices for negative samples
    list_ind = np.random.permutation(posnegind_dict['neg_indices'])
    
    # Select first _select_indices from permuted list_ind
    for _iter_sel in range(0,_select_indices):
        _iter = list_ind[_iter_sel]
        #print(list_ind[_iter])
        filtered_obj.xdata.append(passed_obj.xdata[_iter])
        filtered_obj.ylabels = np.append(filtered_obj.ylabels, np.array([passed_obj.ylabels[_iter]]), axis=0) 
        filtered_obj.details.append(passed_obj.details[_iter])
        #print(filtered_obj.details)
    
    # Select all positive samples
    for _iter in posnegind_dict['pos_indices']:
        filtered_obj.xdata.append(passed_obj.xdata[_iter])
        filtered_obj.ylabels = np.append(filtered_obj.ylabels, np.array([passed_obj.ylabels[_iter]]), axis=0) 
        filtered_obj.details.append(passed_obj.details[_iter])

    # filtered_obj: downsampled version of passed_obj
    print(Counter(filtered_obj.ylabels))
    return filtered_obj

def func_calculate_f1_score(y_gt, y_pred, _str):
    assert(len(y_gt) == len(y_pred))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    print('***********')
    print(_str)
    print('***********')
    print('len_y_gt = %d len_y_pred = %d' %(len(y_gt), len(y_pred)))
    
    # dict to return
    dict_to_return = {'tp':-1, 'fp':-1, 'fn':-1, 'tn':-1, 'tot_pos': 0, 'tot_neg': 0,
                      'tp_indices':[], 'fp_indices':[], 'fn_indices':[], 'tn_indices':[],
                      'prec': -1.0, 'rec': -1.0, 'f1_score': 0.0, 'accuracy': 0.0 }

    for iter_tok in range(0, len(y_gt)):
        #print(iter_tok)
        assert(isinstance(y_gt[iter_tok], np.int64))
        assert(isinstance(y_pred[iter_tok], np.int64))

        if(y_pred[iter_tok] == 1 and y_gt[iter_tok] == 1):
            tp = tp + 1
            dict_to_return['tp_indices'].append(iter_tok)
        if(y_pred[iter_tok] == 1 and y_gt[iter_tok] == 0):
            fp = fp + 1
            dict_to_return['fp_indices'].append(iter_tok)
        if(y_pred[iter_tok] == 0 and y_gt[iter_tok] == 1):
            fn = fn + 1
            dict_to_return['fn_indices'].append(iter_tok)
        if(y_pred[iter_tok] == 0 and y_gt[iter_tok] == 0):
            tn = tn + 1
            dict_to_return['tn_indices'].append(iter_tok)

    dict_to_return['tp'] = tp
    dict_to_return['fp'] = fp
    dict_to_return['fn'] = fn
    dict_to_return['tn'] = tn
    dict_to_return['tot_pos'] = tp + fn
    dict_to_return['tot_neg'] = tn + fp
    print('tp:%d tn:%d fp:%d fn:%d' %(tp,tn,fp,fn))
    print('tot_pos:%d tot_neg:%d' %(dict_to_return['tot_pos'], dict_to_return['tot_neg']))
    
    
    # assertion
    assert( len(y_gt) == (tp+tn+fp+fn))

    # precision/Recall/F1-Score
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = (2*precision*recall)/(precision+recall)
    print('precision = %f , recall = %f , f1_score = %f' %(precision,recall,f1_score))

    [prec, rec, f_beta, support] = precision_recall_fscore_support(y_gt, y_pred, average='binary', pos_label=1)
    f1_recalculated = (2*prec*rec)/(prec + rec)
    print('(FROM SCIKIT) precision = %f , recall = %f , f_beta = %f f_score_recalculated = %f' %(prec,rec,f_beta, f1_recalculated))
    print('accuracy = %f' %accuracy_score(y_gt, y_pred))

    # Add it to dictionary
    dict_to_return['prec'] = precision
    dict_to_return['rec'] = recall
    dict_to_return['f1_score'] = f1_score
    dict_to_return['accuracy'] = accuracy_score(y_gt, y_pred)
    
    # Return
    return dict_to_return

def func_segregate_traintest_obj(data_obj):
    # Given data_obj, it will generate two data objects
    # segregating train and test instances from data_obj
    train_obj = data()
    test_obj = data()
    for _iter in range(0, len(data_obj.xdata)):
        if data_obj.details[_iter]['type'] == 'train':
            #print(data_obj.details[_iter]['type'])
            train_obj.xdata.append(data_obj.xdata[_iter])
            train_obj.ylabels = np.append(train_obj.ylabels, np.array([data_obj.ylabels[_iter]]), axis=0) 
            train_obj.details.append(data_obj.details[_iter])


        if data_obj.details[_iter]['type'] == 'test':
            #print(data_obj.details[_iter]['type'])
            test_obj.xdata.append(data_obj.xdata[_iter])
            test_obj.ylabels = np.append(test_obj.ylabels, np.array([data_obj.ylabels[_iter]]), axis=0) 
            test_obj.details.append(data_obj.details[_iter])
            
    print('Successfully segregated %d train pages from %d total pages' %(len(train_obj.details), len(data_obj.xdata)) )
    print('Successfully segregated %d test pages from %d total pages' %(len(test_obj.details), len(data_obj.xdata)) )
    
    #Assertions
    assert(len(train_obj.xdata) != 0)
    assert(len(test_obj.xdata) != 0)
    assert( len(data_obj.xdata) == ( len(train_obj.xdata) + len(test_obj.xdata) ) )
    assert(len(train_obj.xdata) == len(train_obj.ylabels) == len(train_obj.details))
    assert(len(test_obj.xdata) == len(test_obj.ylabels) == len(test_obj.details))
    
    #return
    return train_obj, test_obj

def func_segregate_posneg_leasedocs(path_to_dirs):
    #This function populates data object and assign label to positive and negative lease documents
    #Positive leases have '_ref.txt' while negative leases have '_noref.txt'
    #
    #path_to_dirs: It is the dictionary that contains list of training directories (path_to_dirs['train']) 
    #              and test directories (path_to_dirs['test'])
    data_obj = data()
    for _iterkey in path_to_dirs.keys():
        for _dir in path_to_dirs[_iterkey]:
            for _file in os.listdir(_dir):
                # print(_file, _file.split('_'))
                _split_name = _file.split('_')
                # print(_file, '-----', _split_name, '-----', _split_name[-1])

                if _split_name[-1] == 'ref.txt' or _split_name[-1] == 'noref.txt':
                    # Check if the file is positive or negative instance
                    is_pos = func_is_pos_instance(_split_name[-1])
                    is_neg = func_is_neg_instance(_split_name[-1])

                    # Both can't be true
                    assert(is_pos != is_neg)

                    # Positive instance is picked up using 'ref.txt' ending
                    if is_pos is True:
                        print(_file, '-----', _split_name, '-----', _split_name[-1], '-----> PositiveInstance')
                        base_dir = ''.join(_dir)
                        _filename = ''.join(_file)
                        label = 1
                        type_data = ''.join(_iterkey)
                        encoding_string = 'utf8'                    
                        func_generate_dataobj(data_obj, base_dir, _filename, label, type_data, encoding_string)
                        data_obj.tot_positive_files = data_obj.tot_positive_files + 1 
                    
                    if is_neg is True:
                        print(_file, '-----', _split_name, '-----', _split_name[-1], '-----> NegativeInstance')
                        base_dir = ''.join(_dir)
                        _filename = ''.join(_file)
                        label = 0
                        type_data = ''.join(_iterkey)
                        encoding_string = 'utf8'
                        func_generate_dataobj(data_obj, base_dir, _filename, label, type_data, encoding_string)
                        data_obj.tot_negative_files = data_obj.tot_negative_files + 1

                    if type_data == 'train':
                        data_obj.tot_train_files = data_obj.tot_train_files + 1

                    if type_data == 'test':
                        data_obj.tot_test_files = data_obj.tot_test_files + 1

                        
    #return
    print('Successfully read %d positive and negative pages' %len(data_obj.details))
    print('Total train files (divide it by 2 b/c of _ref.txt and _noref.txt): %d Total test files (divide it by 2 b/c of _ref.txt and _noref.txt): %d'%(data_obj.tot_train_files, data_obj.tot_test_files) )
    print('Positive files: %d Negative files: %d (both should be same)'%(data_obj.tot_positive_files, data_obj.tot_negative_files) )
    assert(data_obj.tot_positive_files == data_obj.tot_negative_files)
    assert(len(data_obj.xdata) != 0)
    assert(len(data_obj.xdata) == len(data_obj.ylabels) == len(data_obj.details))
    return data_obj

def func_findutf8_incompatiblefiles(path_to_dirs):
    #This function prints files that have UTF8 issues
    data_obj = data()
    for _iterkey in path_to_dirs.keys():
        for _dir in path_to_dirs[_iterkey]:
            for _file in os.listdir(_dir):
                # print(_file, _file.split('_'))
                _split_name = _file.split('_')
                # print(_file, '-----', _split_name, '-----', _split_name[-1])

                if _split_name[-1] == 'ref.txt' or _split_name[-1] == 'noref.txt':
                    # Check if the file is positive or negative instance
                    is_pos = func_is_pos_instance(_split_name[-1])
                    is_neg = func_is_neg_instance(_split_name[-1])

                    # Both can't be true
                    assert(is_pos != is_neg)

                    # Positive instance is picked up using 'ref.txt' ending
                    if is_pos is True:
                        try:
                            #print(_file, '-----', _split_name, '-----', _split_name[-1], '-----> PositiveInstance')
                            base_dir = ''.join(_dir)
                            _filename = ''.join(_file)
                            label = 1
                            type_data = ''.join(_iterkey)
                            encoding_string = 'utf8'                    
                            func_generate_dataobj(data_obj, base_dir, _filename, label, type_data, encoding_string)
                            data_obj.tot_positive_files = data_obj.tot_positive_files + 1
                        except Exception as e:
                            print(base_dir + '/' + _filename)
                    
                    if is_neg is True:
                        #print(_file, '-----', _split_name, '-----', _split_name[-1], '-----> NegativeInstance')
                        try:
                            
                            base_dir = ''.join(_dir)
                            _filename = ''.join(_file)
                            label = 0
                            type_data = ''.join(_iterkey)
                            encoding_string = 'utf8'
                            func_generate_dataobj(data_obj, base_dir, _filename, label, type_data, encoding_string)
                            data_obj.tot_negative_files = data_obj.tot_negative_files + 1
                        except Exception as e:
                            print(base_dir + '/' + _filename)

def func_prepare_output_deployment_original(data_obj, pred_array, prob_array, verbosity = False):
    
    assert( len(data_obj.xdata) == len(data_obj.ylabels) == len(data_obj.details) == len(pred_array) == prob_array.shape[0])
    
    # return dictionary
    _dict = {'page_no':[], 'label':[], 'probs':[]}
    
    for _iter in range(0,len(pred_array)):
        if pred_array[_iter] == 1:
            _str = 'Ref'
            _dict['page_no'].append(data_obj.details[_iter]['page_number'])
            _dict['label'].append(_str)
            _dict['probs'].append(prob_array[_iter])
            if(verbosity is True):
                #print('Pg: %d ----- %s [%f %f]' %(data_obj.details[_iter]['page_number'], _str, prob_array[_iter][0], prob_array[_iter][1]))
                print('Pg: %d ----- %s [%f %f]' %(_dict['page_no'][_iter], _dict['label'][_iter], _dict['probs'][_iter][0], _dict['probs'][_iter][1]))
             
        if pred_array[_iter] == 0:
            _str = 'NO'
            _dict['page_no'].append(data_obj.details[_iter]['page_number'])
            _dict['label'].append(_str)
            _dict['probs'].append(prob_array[_iter])
            if(verbosity is True):
                #print('Pg: %d ----- %s [%f %f]' %(data_obj.details[_iter]['page_number'], _str, prob_array[_iter][0], prob_array[_iter][1]))
                print('Pg: %d ----- %s [%f %f]' %(_dict['page_no'][_iter], _dict['label'][_iter], _dict['probs'][_iter][0], _dict['probs'][_iter][1]))


    # Return
    return _dict

def func_prepare_output_deployment(data_obj, pred_array, prob_array, verbosity = False):
    
    assert( len(data_obj.xdata) == len(data_obj.ylabels) == len(data_obj.details) == len(pred_array) == prob_array.shape[0])
    
    # return dictionary
    _dict = {'page_no':[], 'label':[], 'probs':[], 'readable':[]}
    
    for _iter in range(0,len(pred_array)):
        if pred_array[_iter] == 1:
            _str = 'Ref'
            _dict['page_no'].append(data_obj.details[_iter]['page_number'])
            _dict['label'].append(_str)
            _dict['probs'].append(prob_array[_iter].tolist())
            _detail = 'pg:' + str(data_obj.details[_iter]['page_number']) + ', ' + _str + ', ' + str(_dict['probs'][_iter][0]) + ', ' + str(_dict['probs'][_iter][1])
            _dict['readable'].append(_detail)
            if(verbosity is True):
                #print('Pg: %d ----- %s [%f %f]' %(data_obj.details[_iter]['page_number'], _str, prob_array[_iter][0], prob_array[_iter][1]))
                print('Pg: %d ----- %s [%f %f]' %(_dict['page_no'][_iter], _dict['label'][_iter], _dict['probs'][_iter][0], _dict['probs'][_iter][1]))
             
        if pred_array[_iter] == 0:
            _str = 'NO'
            _dict['page_no'].append(data_obj.details[_iter]['page_number'])
            _dict['label'].append(_str)
            _dict['probs'].append(prob_array[_iter].tolist())
            _detail = 'pg:' + str(data_obj.details[_iter]['page_number']) + ', ' + _str + ', ' + str(_dict['probs'][_iter][0]) + ', ' + str(_dict['probs'][_iter][1])
            _dict['readable'].append(_detail)
            if(verbosity is True):
                #print('Pg: %d ----- %s [%f %f]' %(data_obj.details[_iter]['page_number'], _str, prob_array[_iter][0], prob_array[_iter][1]))
                print('Pg: %d ----- %s [%f %f]' %(_dict['page_no'][_iter], _dict['label'][_iter], _dict['probs'][_iter][0], _dict['probs'][_iter][1]))


    # Return
    return _dict

def func_set_thresh_prediction(pred_array, prob_array, thresh = 0.0, verbosity = False):
    # This function will modify predictions, if diff(probs for two classes) < thresh
    # then that sample will be considered as a positive sample
    
    # Lenght of prediction array
    len_array = len(pred_array)
    return_array = []
    if verbosity is True:
        print('Total Samples: %d' %len_array )
    
    # Assertions
    assert(prob_array.shape[0] == len_array)
    assert(prob_array.shape[1] == np.int(2))
    
    # Go through each sample
    for _iter in range(0,len_array):
        diff_in_probs = prob_array[_iter][0] - prob_array[_iter][1]
        if np.absolute(diff_in_probs) <= thresh:
            return_array.append(1)
        else:
            return_array.append(pred_array[_iter])
    
    return_array = np.array(return_array)
    
    # Return
    return return_array

def func_learn_and_predict(arg_dict, df_dict, save_model=False):
    
    # Pass a dictionary 
    classifier_obj = arg_dict['classifier_obj']
    description = arg_dict['description']
    verbose = arg_dict['verbose']
    filtered_trainobj = arg_dict['filtered_trainobj']
    train_obj = arg_dict['fulltrain_obj']
    test_obj = arg_dict['test_obj']
    _per = arg_dict['_per']
    
    # df_dict Details
    # df_dict['df_filteredtrain']
    # df_dict['df_fulltrain']
    # df_dict['df_test']:
    df_filtered_train = df_dict['df_filteredtrain']
    df_test = df_dict['df_test']
    df_full_train = df_dict['df_fulltrain']

    
    if verbose is True:
        print('-'*50)
        print('Training with %s' %(description))
    
    # Start time
    t0 = time()
    
    
    
    ########################
    ## CLF_COUNTS
    #########################
    
    # Training on Filtered dataset
    clf_counts = classifier_obj.fit(arg_dict['X_filteredTrain_counts'], filtered_trainobj.ylabels)
    arg_dict['clf_counts'] = deepcopy(clf_counts)

    # prediction on countvectorizer
    predicted_counts = clf_counts.predict(arg_dict['X_filteredTrain_counts'])
    # Show the F1-Score for on counts
    dict_to_ret = func_calculate_f1_score(filtered_trainobj.ylabels, predicted_counts, 'Training F1-Score with Counts')
    # Adding it to the dataframe
    _row = [description, _per,'COUNT',dict_to_ret['tp'], dict_to_ret['tn'], dict_to_ret['fp'], dict_to_ret['fn'], dict_to_ret['tot_pos'], dict_to_ret['tot_neg'], dict_to_ret['accuracy'], dict_to_ret['prec'], dict_to_ret['rec'], dict_to_ret['f1_score']]
    df_filtered_train.loc[len(df_filtered_train)] = _row
    
    # prediction on countvectorizer
    predicted_counts = clf_counts.predict(arg_dict['X_fullTrain_counts'])
    # Show the F1-Score for counts
    dict_to_ret = func_calculate_f1_score(train_obj.ylabels, predicted_counts, 'FULL-TRAIN-SET F1-Score with Counts')
    # Adding it to the dataframe
    _row = [description, _per,'COUNT',dict_to_ret['tp'], dict_to_ret['tn'], dict_to_ret['fp'], dict_to_ret['fn'], dict_to_ret['tot_pos'], dict_to_ret['tot_neg'], dict_to_ret['accuracy'], dict_to_ret['prec'], dict_to_ret['rec'], dict_to_ret['f1_score']]
    df_full_train.loc[len(df_full_train)] = _row
    
    # prediction on countvectorizer
    predicted_counts = clf_counts.predict(arg_dict['X_test_counts'])
    # Show the F1-Score for counts
    dict_to_ret = func_calculate_f1_score(test_obj.ylabels, predicted_counts, 'Test F1-Score with Counts')
    # Adding it to the dataframe
    _row = [description, _per,'COUNT',dict_to_ret['tp'], dict_to_ret['tn'], dict_to_ret['fp'], dict_to_ret['fn'], dict_to_ret['tot_pos'], dict_to_ret['tot_neg'], dict_to_ret['accuracy'], dict_to_ret['prec'], dict_to_ret['rec'], dict_to_ret['f1_score']]
    df_test.loc[len(df_test)] = _row
    
    
    ########################
    ## CLF_TF-IDF
    #########################
    
    clf_tfidf = classifier_obj.fit(arg_dict['X_filteredTrain_tfidf'], filtered_trainobj.ylabels)
    arg_dict['clf_tfidf'] = deepcopy(clf_tfidf)

    
    # prediction on TF-IDF
    predicted_tfidf = clf_tfidf.predict(arg_dict['X_filteredTrain_tfidf'])
    # Show the F1-Score for tf_idf
    dict_to_ret = func_calculate_f1_score(filtered_trainobj.ylabels, predicted_tfidf, 'Training F1-Score with TF-IDF')
    # Adding it to the dataframe
    _row = [description, _per,'TFIDF',dict_to_ret['tp'], dict_to_ret['tn'], dict_to_ret['fp'], dict_to_ret['fn'], dict_to_ret['tot_pos'], dict_to_ret['tot_neg'], dict_to_ret['accuracy'], dict_to_ret['prec'], dict_to_ret['rec'], dict_to_ret['f1_score']]
    df_filtered_train.loc[len(df_filtered_train)] = _row
    
    
    # prediction on TF-IDF
    predicted_tfidf = clf_tfidf.predict(arg_dict['X_fullTrain_tfidf'])
    # Show the F1-Score for tf_idf
    dict_to_ret = func_calculate_f1_score(train_obj.ylabels, predicted_tfidf, 'FULL-TRAIN-SET F1-Score with TF-IDF')
    _row = [description, _per,'TFIDF',dict_to_ret['tp'], dict_to_ret['tn'], dict_to_ret['fp'], dict_to_ret['fn'], dict_to_ret['tot_pos'], dict_to_ret['tot_neg'], dict_to_ret['accuracy'], dict_to_ret['prec'], dict_to_ret['rec'], dict_to_ret['f1_score']]
    df_full_train.loc[len(df_full_train)] = _row
    
    # prediction on TF-IDF
    predicted_tfidf = clf_tfidf.predict(arg_dict['X_test_tfidf'])
    # Show the F1-Score for tf_idf
    dict_to_ret = func_calculate_f1_score(test_obj.ylabels, predicted_tfidf, 'Test F1-Score with TF-IDF')
    _row = [description, _per,'TFIDF',dict_to_ret['tp'], dict_to_ret['tn'], dict_to_ret['fp'], dict_to_ret['fn'], dict_to_ret['tot_pos'], dict_to_ret['tot_neg'], dict_to_ret['accuracy'], dict_to_ret['prec'], dict_to_ret['rec'], dict_to_ret['f1_score']]
    df_test.loc[len(df_test)] = _row
    
    
    
    
    ########################
    ## CLF_TF-IDF_COUNTS
    #########################
    clf_tfidf_counts = classifier_obj.fit(arg_dict['X_filteredTrain_tfidf_counts'], filtered_trainobj.ylabels)
    arg_dict['clf_tfidf_counts'] = deepcopy(clf_tfidf_counts)

    
    # prediction on TF-IDF + Counts
    predicted_tfidf_counts = clf_tfidf_counts.predict(arg_dict['X_filteredTrain_tfidf_counts'])
    # Show the F1-Score for on tf_idf + counts
    dict_to_ret = func_calculate_f1_score(filtered_trainobj.ylabels, predicted_tfidf_counts, 'Training F1-Score with TF-IDF+Counts')
    # Adding it to the dataframe
    _row = [description, _per,'TFIDF+COUNT',dict_to_ret['tp'], dict_to_ret['tn'], dict_to_ret['fp'], dict_to_ret['fn'], dict_to_ret['tot_pos'], dict_to_ret['tot_neg'], dict_to_ret['accuracy'], dict_to_ret['prec'], dict_to_ret['rec'], dict_to_ret['f1_score']]
    df_filtered_train.loc[len(df_filtered_train)] = _row
    
    # prediction on TF-IDF + countvectorizer
    predicted_tfidf_counts = clf_tfidf_counts.predict(arg_dict['X_fullTrain_tfidf_counts'])
    # Show the F1-Score for tf_idf
    dict_to_ret = func_calculate_f1_score(train_obj.ylabels, predicted_tfidf_counts, 'FULL-TRAIN-SET F1-Score with TFIDF+Counts')
    _row = [description, _per,'TFIDF+COUNT',dict_to_ret['tp'], dict_to_ret['tn'], dict_to_ret['fp'], dict_to_ret['fn'], dict_to_ret['tot_pos'], dict_to_ret['tot_neg'], dict_to_ret['accuracy'], dict_to_ret['prec'], dict_to_ret['rec'], dict_to_ret['f1_score']]
    df_full_train.loc[len(df_full_train)] = _row
    
     # prediction on TF-IDF + countvectorizer
    predicted_tfidf_counts = clf_tfidf_counts.predict(arg_dict['X_test_tfidf_counts'])
    # Show the F1-Score for tf_idf + counts
    dict_to_ret = func_calculate_f1_score(test_obj.ylabels, predicted_tfidf_counts, 'Test F1-Score with TFIDF+Counts')
    _row = [description, _per,'TFIDF+COUNT',dict_to_ret['tp'], dict_to_ret['tn'], dict_to_ret['fp'], dict_to_ret['fn'], dict_to_ret['tot_pos'], dict_to_ret['tot_neg'], dict_to_ret['accuracy'], dict_to_ret['prec'], dict_to_ret['rec'], dict_to_ret['f1_score']]
    df_test.loc[len(df_test)] = _row
       

    
    # Time Elapsed
    training_duration  = time() - t0
    
    if verbose is True:
        print("Training and Evaluation finished in %fs" % (training_duration))
    
    #####################
    # SAVE MODEL
    #####################
    # Save the parameters
    if save_model is True:
        with open(arg_dict['filepath'], 'wb') as fin:
            print('Saving model to: %s' %arg_dict['filepath'])
            pickle.dump([arg_dict['count_vec'], arg_dict['tfidf_transformer'], arg_dict['clf_counts'], arg_dict['clf_tfidf'], arg_dict['clf_tfidf_counts'], df_filtered_train, df_test, df_full_train], fin)

def func_fit_and_return_feats(arg_dict):
    
    # Safest way to pass objects
    filtered_trainobj = arg_dict['filtered_trainobj']
    testobj = arg_dict['test_obj']
    fulltrain_obj = arg_dict['fulltrain_obj']
    verbose = arg_dict['verbose']
    
    if verbose is True:
        print('Filtered Train Set Samples: %d' %len(filtered_trainobj.xdata) ) 
        print('Test Set Samples: %d' %len(testobj.xdata) )
        print('Full Train Set Samples: %d' %len(fulltrain_obj.xdata) )
        
    
    # Assert
    #assert(filtered_trainobj.xdata.shape[0] <= fulltrain_obj.xdata.shape[0])
    #assert(testobj.xdata.shape[0] <= fulltrain_obj.xdata.shape[0])

    
    # Fit on Filtered_trainobj
    count_vec = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    
    # Get Count FitTransform of Filtered Trainset
    X_filteredTrain_counts = count_vec.fit_transform(filtered_trainobj.xdata)
    
    # Get TF-IDF Fit Transform of Filtered Trainset
    X_filteredTrain_tfidf = tfidf_transformer.fit_transform(X_filteredTrain_counts)
    
    # Count + TF-IDF of Filtered TrainSet
    X_filteredTrain_tfidf_counts = hstack((X_filteredTrain_tfidf, X_filteredTrain_counts))
    
    # TEST 
    X_test_counts = count_vec.transform(testobj.xdata)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    X_test_tfidf_counts = hstack((X_test_tfidf, X_test_counts))


    # FULL TRAIN
    X_fullTrain_counts = count_vec.transform(fulltrain_obj.xdata)
    X_fullTrain_tfidf = tfidf_transformer.transform(X_fullTrain_counts)
    X_fullTrain_tfidf_counts = hstack((X_fullTrain_tfidf, X_fullTrain_counts))
    

    # Populate dictionary - It will be visible outside this function
    arg_dict['count_vec'] = count_vec
    arg_dict['tfidf_transformer'] = tfidf_transformer
    arg_dict['X_filteredTrain_counts'] = X_filteredTrain_counts
    arg_dict['X_filteredTrain_tfidf'] = X_filteredTrain_tfidf
    arg_dict['X_filteredTrain_tfidf_counts'] = X_filteredTrain_tfidf_counts
    arg_dict['X_test_counts'] = X_test_counts
    arg_dict['X_test_tfidf'] = X_test_tfidf
    arg_dict['X_test_tfidf_counts'] = X_test_tfidf_counts
    arg_dict['X_fullTrain_counts'] = X_fullTrain_counts
    arg_dict['X_fullTrain_tfidf'] = X_fullTrain_tfidf
    arg_dict['X_fullTrain_tfidf_counts'] = X_fullTrain_tfidf_counts
