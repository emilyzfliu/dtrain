
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 22:54:05 2020

@author: Emi Z Liu
"""

from client import Client
import time
import os

c1 = Client('A')
c2 = Client('B')
# stores most recent model file from server
model_file_1 = ''
model_file_2 = ''

def read_data():
    datafiles = []
    path = 'data'
    for root, dirs, files in os.walk(path):
        for filename in files:
            if '.nii' not in filename:
                continue
            datafiles.append(os.path.join(path, filename))
    return datafiles

data = read_data()

def has_new_model(model_file, c):
    if model_file == '':
        return False
    try:
        model_file = c.load_consolidated()
        return True
    except:
        return False

def new_data():
    # need to find a way to speed this up, right now it's traversing
    # the entire directory which is very inefficient
    updated = read_data()
    if set(data) == set(updated):
        return False
    update(updated, data)
    return True

def update(updated, data):
    data = updated

while True:
    if has_new_model(model_file_1, c1):
        c1.train()
        c1.save_weights()
    else:
        if new_data():
            c1.train()
            c1.save_weights()
        time.sleep(1)
    if has_new_model(model_file_2, c2):
        #this step is not really necessary since client automatically makes them equal
        #but included in here just in case we need to change
        #(right now this step would give an error since model_file_2 is the file and c2.model is the model itself)
        #assert model_file_2 == c2.model
        c2.train()
        c2.save_weights()
    else:
        if new_data():
            c2.train()
            c2.save_weights()
        time.sleep(1)
    
