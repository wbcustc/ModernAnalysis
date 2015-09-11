# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:12:00 2015

@author: wangbochen
"""
#Part g
#Implement KNN Classifier
import numpy as np
from scipy import spatial as spt
class MyKNNClassifier(object):
    def __init__(self,k):
        self.k = k        
    def fit(self,train_data,train_label):
        self.kdTree = spt.KDTree(train_data)
        self.label = train_label        
    def predict(self,test_data):
        res = self.kdTree.query(test_data,self.k)
        test_label = self.label[res[1]]
        ret = []
        for e in test_label:           
            count_map = {}
            print(e)
            for label in e:
                if count_map.has_key(label):
                    count_map[label] += 1;
                else:
                    count_map[label] = 1;
            
            max_val = 0
            max_label = 0
            for e in count_map:
                if count_map[e] > max_val:
                    max_val = count_map[e]
                    max_label = e
            ret.append(max_label)      
        return np.array(ret)
