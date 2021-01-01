#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
#import numpy as np
import uuid
import glob
from make_model import sample_net

class Client():
    def __init__(self, clientid):
        self.client_id = clientid
        shape = (128, 128, 128, 1)
        self.model = sample_net(shape)
        self.load_input_output()
        self.save_prior()
        
    def load_input_output(self):
        record_file = 'tfrecord_'+self.client_id+'.tfrec'
        self.dataset = tf.data.TFRecordDataset(record_file, compression_type="GZIP")
    
    def save_prior(self):
        self.model.save_weights('prior-'+self.client_id+'.h5', save_format = 'h5')
    
    def train(self):
        _op = 'adam'
        _loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        _metrics = ['accuracy']
        self.model.compile(optimizer=_op, loss=_loss, metrics=_metrics)
        #self.model.fit(inputs=self.inputs, outputs=self.outputs, epochs=1, verbose=2)
        self.model.fit(self.dataset, epochs=1, verbose=2)
    
    def load_consolidated(self):
        list_of_files = glob.glob('/server/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        self.model.load_weights(latest_file)
        return latest_file
    
    def save_weights(self):
        filename = 'server/consolidated-'+self.client_id+'-'+uuid.uuid4().__str__()+'.h5'
        self.model.save_weights(filename, save_format = 'h5')

# example usage

# a = Client('A')
# try:
#     a.load_consolidated()
# except:
#     pass
# a.train()
# a.save_weights()

