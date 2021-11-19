#author Weining Weng
#mail weiningweng1999@gmail.com
#date 2021-11-18 16:51
#kaggle dataset: Titanic preprocess
#MLP algorthm
import os
from scipy import io
import time,datetime
import csv
import numpy as np
from numpy.core.fromnumeric import searchsorted
from tensorflow.python.ops.gen_nn_ops import softmax
from tqdm import tqdm
import random
import tensorflow as tf
import xlwt

input_data=tf.placeholder(tf.float32,[None,17],name='input-tensor')
input_label=tf.placeholder(tf.float32,[None,2],name='input-label')

atten_fc1=tf.layers.dense(
    inputs = input_data,
    units = 9,
    activation = tf.nn.tanh,
    use_bias = True)
atten=tf.layers.dense(
    inputs = atten_fc1,
    units = 17,
    activation = tf.nn.softmax)
masked_atten=tf.multiply(input_data,atten)
fc1=tf.layers.dense(
    inputs = masked_atten,
    units = 7,
    activation = tf.nn.relu,
    use_bias = True)
fc2=tf.layers.dense(
    inputs= fc1,
    units=2,
    activation=tf.nn.relu,
    use_bias= True)
classfier=tf.nn.softmax(fc2)
cross_entropy=tf.losses.softmax_cross_entropy(onehot_labels=input_label, 
                                              logits=classfier)
loss_function=tf.reduce_mean(cross_entropy,name='loss_function')
lr=0.0001
optimizer=tf.train.AdamOptimizer(lr)
train_op=optimizer.minimize(loss_function)


train_path='train_with_soc.csv'
test_path='test_with_soc.csv'
train_file=csv.reader(open(train_path))
test_file=csv.reader(open(test_path))
row_train=[row for row in train_file]
row_test=[row for row in test_file]
test_number=range(892,1310)

#generate train data and train label
train_length=len(row_train)-1
train_data=np.zeros((train_length,17))
train_label=np.zeros((train_length,2))
for i in range(0,train_length):
    for j in range(0,17):
        train_data[i,j]=int(row_train[i+1][j])
    label_pos=int(row_train[i+1][17])
    train_label[i,label_pos]=1

#generate test data
test_length=len(row_test)-1
test_data=np.zeros((test_length,17))
for i in range(0,test_length):
    for j in range(0,17):
        test_data[i,j]=int(row_test[i+1][j])

#model calculate
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
loss_func=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(0,100):#training epoch with batch_size=12
        batch_size=12
        batch_number=0
        epoch_loss=0
        pos=0
        while pos<train_length-1-batch_size:
            train_batch=train_data[pos:pos+batch_size,:]
            train_lab=train_label[pos:pos+batch_size,:]
            feed_dicts={input_data:train_batch,
                        input_label:train_lab}
            logits,loss,opt=sess.run([classfier,
                                      loss_function,
                                      train_op],
                                      feed_dict=feed_dicts)
            epoch_loss=epoch_loss+loss
            batch_number=batch_number+1
            pos=pos+12
        epoch_loss=epoch_loss/batch_number
        print(epoch_loss)
        loss_func.append(epoch_loss)
        test_batch=test_data
        test_label=np.zeros((test_length,2))
        feed_dicts={input_data:test_batch,
                    input_label:test_label}
        logits,loss,weight=sess.run([classfier,loss_function,atten],
                            feed_dict=feed_dicts)
        weight_name='./weight_2/weight_'+str(epoch)+'.npy'
        weight=np.average(weight,axis=0)
        #np.save(weight_name,weight)
        result=np.argmax(logits,axis=1)
        output=np.zeros((test_length,2))
        for i in range(0,test_length):
            output[i,0]=test_number[i]
            output[i,1]=result[i]
        name='./result_2/epoch_'+str(epoch)+'.csv'
        #np.savetxt(name,output,delimiter=',')
np.save('loss.npy',loss_func)
        
        
        







