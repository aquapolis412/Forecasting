import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib as mp
import random as rd
import argparse
import os, sys
import csv
import math
import time
import matplotlib.pyplot as pl
class County:
    def __init__(self,parsename):
        self.parsename = parsename
        dataframe = xls.parse(parsename)
        self.data = dataframe
    def disp_all(self):
        print self.dataframe
    def get_all(self):
        return self.dataframe
xls = pd.ExcelFile('ME.xls',header = None)
Zonal = []
Zonal.append(County('ME'))
ZonalNum = 1

# histo load based model stage 2
#num_epoches = 10000# training epoches for each customer samples
n_input_h = 28*24 # input size
input_seq_size_h = n_input_h
test_batch_size_h = 28 # days of a batch
valid_batch_size_h = 0
train_batch_size_h = 28
data_dim_h = 1 # same time of a week
n_output_h = 24
output_seq_size_h = 24
gap_h = 63
n_hidden_h_1= 50
n_hidden_h_2 = 50
n_hidden_h_3 = 50
tao_h = 0.5

# feature based model stage 1
#num_epoches = 30000 # training epoches for each customer samples
out_thresh = 18499
day_steps_f = 24
val_rate_f = 0.0
test_batch_size_f = test_batch_size_h*day_steps_f # days of a batch
valid_batch_size_f = valid_batch_size_h*day_steps_f
train_batch_size_f = train_batch_size_h*day_steps_f
n_output_f = 1
n_hidden_f_1 = 20
n_hidden_f_2 = 20
n_hidden_f_3 = 20
n_hidden_f_4 = 20
tao_f = 0.1
gap_test_f = 10
batch_size_f = test_batch_size_f # in this version, batch_size set same
preserve_f = 16114 ## amount of first time points without complete features
dropout_f = 0.9

# merge part stage 3
s3_input = day_steps_f
n_hidden_s3_1 = 30
n_hidden_s3_2 = 30
s3_output = 24
n_epochs = 10000

# DEMAND MATRIX 9 X LENGTH, 9: INC is total, index with 0, other substations are from 1 -> 8
rows_f = np.array(Zonal[0].data).shape[0]
columns_f = np.array(Zonal[0].data).shape[1]
database_f = np.zeros((ZonalNum, rows_f, columns_f))
for i in range(ZonalNum):
    tmp = np.array(Zonal[i].data)
    database_f[i] = tmp
database_f = np.transpose(database_f, [1, 0, 2])
totalen_f = rows_f
n_input_f = columns_f - 1
database_f[:,:,0] = database_f[:,:,0]/2500
db_f = database_f

rows = np.array(Zonal[0].data).shape[0]
columns = np.array(Zonal[0].data).shape[1]
# DEMAND MATRIX 9 X LENGTH, 9: INC is total, index with 0, other substations are from 1 -> 8
database_h = np.zeros((rows/output_seq_size_h, output_seq_size_h, columns))
tmp = np.array(Zonal[0].data)
tmp = tmp.astype(np.float32)
database_h = tmp.reshape([rows/output_seq_size_h, output_seq_size_h, columns])
totalen_h = rows/output_seq_size_h
database_h[:,:,0] = database_h[:,:,0]/2500
db_h = database_h

#define id arrays
test_id_f = np.array(test_batch_size_f)
#valid_id_f = np.array(valid_batch_size_f)
#train_id_f = np.array(totalen_f - test_batch_size_f - valid_batch_size_f)

#give values to id arrays
#rang = range(preserve_f, totalen_f - test_batch_size_f)
#valid_id_f = rd.sample(rang,valid_batch_size_f)
test_id_f = np.array(range(totalen_f - test_batch_size_f,totalen_f))
#train_id_f = set(range(preserve_f, totalen_f - test_batch_size_f)) - set(valid_id_f)

#sort three id array
#valid_id_f = np.sort(valid_id_f)
test_id_f = np.sort(test_id_f)
#train_id_f = np.array(list(train_id_f))

#define id arrays
test_id_h = np.array(test_batch_size_h)
valid_id_h = np.array(valid_batch_size_h)
train_id_h = np.array(totalen_h-test_batch_size_h-valid_batch_size_h-input_seq_size_h)

id_start = preserve_f/24 + 1
#give values to id arrays
rang_h = range(input_seq_size_h/output_seq_size_h + id_start + gap_h,totalen_h-test_batch_size_h)
valid_id_h = rd.sample(rang_h,valid_batch_size_h)
test_id_h = np.array(range(totalen_h-test_batch_size_h,totalen_h))
train_id_h = set(range(input_seq_size_h/output_seq_size_h + gap_h + id_start,totalen_h-test_batch_size_h))-set(valid_id_h)

#sort three id array
valid_id_h = np.sort(valid_id_h)
test_id_h = np.sort(test_id_h)
train_id_h = np.array(list(train_id_h))

_X_f = tf.placeholder(tf.float32, [batch_size_f, ZonalNum, n_input_f], name = "X_f")
_Y_f = tf.placeholder(tf.float32, [batch_size_f, ZonalNum, n_output_f], name = "Y_f")
_Dropout_f = tf.placeholder(tf.float32, name = "Dropout_f")
_X_h = tf.placeholder(tf.float32, [test_batch_size_h, n_input_h])
_Y_h = tf.placeholder(tf.float32, [test_batch_size_h, n_output_h])
# Create model
def MLP_f(x, _dropout, weights, biases):    
    x = tf.reshape(x, [-1, n_input_f])
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    #layer_1 = tf.nn.dropout(layer_1,_dropout)
    
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    #layer_2 = tf.nn.dropout(layer_2,_dropout)
    
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    #layer_3 = tf.nn.dropout(layer_3,_dropout)
    
    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)
    # Output layer with linear activation
    result = tf.matmul(layer_3, weights['out']) + biases['out']
    result = tf.nn.sigmoid(result)
    #result = tf.nn.dropout(result,_dropout)
    return result
def MLP_h(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    #layer_1 = tf.nn.dropout(layer_1,_dropout)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    #layer_2 = tf.nn.dropout(layer_2,_dropout)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    #layer_3 = tf.nn.dropout(layer_3,_dropout)

    # Output layer with linear activation
    result = tf.matmul(layer_3, weights['out']) + biases['out']
    result = tf.nn.sigmoid(result)
    #result = tf.nn.dropout(result,_dropout)
    return result
# MLP
weights_f = {
    'h1': tf.Variable(tf.random_normal([n_input_f, n_hidden_f_1]), name = "w_f_1"),
    'h2': tf.Variable(tf.random_normal([n_hidden_f_1, n_hidden_f_2]), name = "w_f_2"),
    'h3': tf.Variable(tf.random_normal([n_hidden_f_2, n_hidden_f_3]), name = "w_f_3"),
    'h4': tf.Variable(tf.random_normal([n_hidden_f_3, n_hidden_f_4]), name = "w_f_4"),
    'out': tf.Variable(tf.random_normal([n_hidden_f_4, n_output_f]), name = "w_o")
}
biases_f = {
    'b1': tf.Variable(tf.random_normal([n_hidden_f_1]), name = "b_f_1"),
    'b2': tf.Variable(tf.random_normal([n_hidden_f_2]), name = "b_f_2"),
    'b3': tf.Variable(tf.random_normal([n_hidden_f_3]), name = "b_f_3"),
    'b4': tf.Variable(tf.random_normal([n_hidden_f_4]), name = "b_f_4"),
    'out': tf.Variable(tf.random_normal([n_output_f]), name = "b_o")
}
weights_h = {
    'h1': tf.Variable(tf.random_normal([n_input_h, n_hidden_h_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_h_1, n_hidden_h_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_h_2, n_hidden_h_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_h_3, n_output_h]))
}
biases_h = {
    'b1': tf.Variable(tf.random_normal([n_hidden_h_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_h_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_h_3])),
    'out': tf.Variable(tf.random_normal([n_output_h]))
}
p_f = MLP_f(_X_f, _Dropout_f, weights_f, biases_f)
p_h = MLP_h(_X_h,  weights_h, biases_h)
p_f = tf.reshape(p_f, [test_batch_size_h, s3_input])
p_h = tf.reshape(p_h, [test_batch_size_h, s3_input])
_I = tf.concat(1, [p_f,p_h])
_O = _Y_h
weights_s3 = {
    'h1': tf.Variable(tf.random_normal([2*s3_input, n_hidden_s3_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_s3_1, n_hidden_s3_2])),
    #'h3': tf.Variable(tf.random_normal([n_hidden_f_2, n_hidden_f_3])),
    #'h4': tf.Variable(tf.random_normal([n_hidden_f_3, n_hidden_f_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_s3_2, s3_output]))
}
biases_s3 = {
    'b1': tf.Variable(tf.random_normal([n_hidden_s3_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_s3_2])),
    #'b3': tf.Variable(tf.random_normal([n_hidden_f_3])),
    #'b4': tf.Variable(tf.random_normal([n_hidden_f_4])),
    'out': tf.Variable(tf.random_normal([s3_output]))
}
def MLP_s3(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    #layer_1 = tf.nn.dropout(layer_1,_dropout)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    #layer_2 = tf.nn.dropout(layer_2,_dropout)
    # Output layer with linear activation
    result = tf.matmul(layer_2, weights['out']) + biases['out']
    result = tf.nn.sigmoid(result)
    #result = tf.nn.dropout(result,_dropout)
    return result

_P = MLP_s3(_I, weights_s3, biases_s3)
cost_s3 = tf.reduce_mean(tf.pow(_O - _P,2))
optimizer_s3 = tf.train.AdamOptimizer(learning_rate=0.001, beta1 = 0.8, beta2 = 0.7).minimize(cost_s3)

#n_hidden_s3_3 = 30
#n_hidden_s3_4 = 30


def maxe(predictions, targets):
    return np.max(abs(predictions-targets))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mape(predictions, targets):
    return np.mean(abs(predictions-targets)/targets)
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def test_data_gen():
    X_h = np.zeros((test_batch_size_h, n_input_h))
    Y_h = np.zeros((test_batch_size_h, n_output_h))
    X_f = np.zeros((test_batch_size_f,ZonalNum,n_input_f))
    Y_f = np.zeros((test_batch_size_f,ZonalNum,n_output_f))
    count_h = 0
    count_f = 0
    for i in test_id_f:
        Y_f[count_f] = db_f[i,:,:1]
        X_f[count_f] = db_f[i,:,1:]
        count_f = count_f + 1
    for i in test_id_h:
        Y_h[count_h,:] = (db_h[i,:,0])
        X_h[count_h,:] = (db_h[i-input_seq_size_h/output_seq_size_h-gap_h:i-gap_h,:,0]).reshape(n_input_h)
        count_h = count_h + 1
    X_h = X_h.astype(np.float32)
    Y_h = Y_h.astype(np.float32)
    X_f = X_f.astype(np.float32)
    Y_f = Y_f.astype(np.float32)
    return (X_h,Y_h,X_f,Y_f)

def train_data_gen():
    X_h = np.zeros((train_batch_size_h, n_input_h))
    Y_h = np.zeros((train_batch_size_h, n_output_h))
    X_f = np.zeros((train_batch_size_f,ZonalNum,n_input_f))
    Y_f = np.zeros((train_batch_size_f,ZonalNum,n_output_f))
    count_h = 0
    count_f = 0
    rang = range(0,train_id_h.shape[0])
    train_rd = rd.sample(rang,train_batch_size_h)
    train_rd = np.sort(train_rd)
    for i in train_rd:
        k = train_id_h[i]
        Y_h[count_h] = db_h[k,:,0]
        X_h[count_h] = (db_h[k-input_seq_size_h/output_seq_size_h-gap_h:k-gap_h,:,0]).reshape(n_input_h)
        for j in range(24):
            Y_f[count_f] = db_f[k*24+j,:,:1]
            X_f[count_f] = db_f[k*24+j,:,1:]
            count_f = count_f + 1
        count_h = count_h + 1
    X_h = X_h.astype(np.float32)
    Y_h = Y_h.astype(np.float32)
    X_f = X_f.astype(np.float32)
    Y_f = Y_f.astype(np.float32)
    return (X_h,Y_h,X_f,Y_f)

def valid_data_gen():
    X_h = np.zeros((valid_batch_size_h, n_input_h))
    Y_h = np.zeros((valid_batch_size_h, n_output_h))
    X_f = np.zeros((valid_batch_size_f,ZonalNum,n_input_f))
    Y_f = np.zeros((valid_batch_size_f,ZonalNum,n_output_f))
    count_h = 0
    count_f = 0
    rang = range(0,valid_id_h.shape[0])
    valid_rd = rd.sample(rang,train_batch_size_h)
    valid_rd = np.sort(valid_rd)
    for i in train_rd:
        k = valid_id_h[i]
        Y_h[count_h] = db_h[k,:,0]
        X_h[count_h] = (db_h[k-input_seq_size_h/output_seq_size_h-gap_h:k-gap_h,:,0]).reshape(n_input_h)
        for j in range(24):
            Y_f[count_f] = db_f[k*24+j,:,:1]
            X_f[count_f] = db_f[k*24+j,:,1:]
            count_f = count_f + 1
        count_h = count_h + 1
    X_h = X_h.astype(np.float32)
    Y_h = Y_h.astype(np.float32)
    X_f = X_f.astype(np.float32)
    Y_f = Y_f.astype(np.float32)
    return (X_h,Y_h,X_f,Y_f)
    
init = tf.initialize_all_variables()
#saver1 = tf.train.Saver({"wf1":weights_f['h1'],"wf2":weights_f['h2'],"wf3":weights_f['h3'],"wf4":weights_f['h4'],"wfo":weights_f['out'],"bf1":biases_f['b1'],"bf2":biases_f['b2'],"bf3":biases_f['b3'],"bf4":biases_f['b4'],"bfo":biases_f['out']})
#saver2 = tf.train.Saver({'wh1':weights_h['h1'],'wh2':weights_h['h2'],'wh3':weights_h['h3'],'who':weights_h['out'],'bh1':biases_h['b1'],'bh2':biases_h['b2'],'bh3':biases_h['b3'],'bho':biases_h['out']})
sess1 = tf.Session(config=tf.ConfigProto(log_device_placement=False))
sess2 = tf.Session(config=tf.ConfigProto(log_device_placement=False))
sess3 = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess1.run(init)
sess2.run(init)
sess3.run(init)
#saver1.restore(sess1, 'tmp/FBM-model.ckpt')
#saver2.restore(sess2, 'tmp/HBM-model.ckpt')
costs_train = []
costs_test = []
predlist = []
for i in range(n_epochs):
    # training process
    x_h,y_h,x_f,y_f = train_data_gen()
    y_s3 = y_h
    _, err = sess3.run([optimizer_s3, cost_s3], feed_dict = {_X_f: x_f, _Y_f: y_f, _X_h: x_h, _Y_h: y_h})
    if i % 100 == 0:
        x_h,y_h,x_f,y_f = test_data_gen()
        P, err_t = sess3.run([_P, cost_s3], feed_dict = {_X_f: x_f, _Y_f: y_f, _X_h: x_h, _Y_h: y_h})
        print "Iter " + str(i) + ", Minibatch Loss ---- Train = " + str(err) + "   ****   Test = " + str(err_t)
        P = np.reshape(np.array(P),[-1])
        costs_train.append(err)
        costs_test.append(err_t)
        predlist.append(P)
        
rmseList = []
mapeList = []
x_h,y_h,x_f,y_f = test_data_gen()
acload = np.reshape(np.array(y_f), [-1])
for prload in predlist:
    prload = np.reshape(np.array(prload), [-1])
    rmseList.append(rmse(prload,acload))
    mapeList.append(mape(prload,acload))

print "Final model MAPE = " + str(np.mean(mapeList[-101:-1]))

ac = np.array(acload)
prload = predlist[-1]
pr = np.array(np.reshape(np.array(prload), [-1]))
t = np.array(range(ac.shape[0]))

label_f0 = r"real load"
label_f1 = r"predicted load"

p1 = pl.subplot()
p1.plot(t,ac,"g",label=label_f0,linewidth=2)
p1.plot(t,pr,"r:",label=label_f1,linewidth=2)

#p1.axis([0,700,10000,25000])

p1.set_ylabel("Demand (kWh)",fontsize=14)
p1.set_xlabel("Time Point",fontsize=14)
#p1.set_title("A simple example",fontsize=18)
p1.grid(True)
p1.legend()

pl.show()

