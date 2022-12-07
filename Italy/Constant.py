#!/usr/bin/env python3

import numpy as np
#import tensorflow.keras as keras
import pandas as pd
import sys
import tensorflow.compat.v1 as tf
import timeit
import time
import csv
import datetime
import scipy.io
import scipy.optimize
from scipy import optimize
from scipy.interpolate import CubicSpline
from matplotlib.pylab import rcParams
#from statsmodels.tsa.holtwinters import SimpleExpSmothing, Holt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score
from matplotlib import pyplot as plt

tf.disable_v2_behavior()


##########################################################################################################
# load and Processing of data

df1 = pd.read_csv('world_confirmed.csv')

##########################################################################################################
# process data
today = '07/10/20' # Update this to include more data 
days = pd.date_range(start='02/27/20',end=today)
dd = np.arange(len(days))

total_cases = [df1[day.strftime('%-m/%-d/%y')].sum() for day in days] 
 
row_c=df1['Country_Region'].tolist().index('Italy')
total_cases = [df1[day.strftime('%-m/%-d/%y')][row_c] for day in days]

t = np.reshape(dd, [-1])
I = np.reshape(total_cases, [-1])

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

new_I = NormalizeData(I) # scaled btw 0 and 1

# generating more data points for training
nd = 1000
cs1 = CubicSpline(t,new_I)

Td = np.linspace(0,135,nd)

cs_I = cs1(Td)

##### The Model

class PINN_constantParam:
    # Initialize the class
    def __init__(self, t, I, layers, lb, ub):
        
        self.lb = lb
        self.ub = ub
        
        self.t = t
        
        self.I = I
        
        self.layers = layers
        
        self.weights, self.biases = self.initialize_NN(layers)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # initialize params
        self.beta = tf.Variable([100.0], dtype=tf.float32)
        self.Nf = tf.Variable([1.0], dtype=tf.float32)
        self.kappa = tf.Variable([2.0], dtype=tf.float32)
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.I_tf = tf.placeholder(tf.float32, shape=[None, self.I.shape[1]])
                
        
        self.I_predL = self.net_Logistic(self.t_tf)
        
        self.l1, self.l2 = self.net_l(self.t_tf)
        
        self.loss = tf.reduce_mean(tf.square(self.I_tf - self.I_predL)) + \
            tf.reduce_mean(tf.square(self.l1)) + \
            tf.reduce_mean(tf.square(self.l2))
        
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = self.optimizer.minimize(self.loss)
        self.loss_log = []
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, t, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sigmoid(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        #Y = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    
    def net_Logistic(self, t):
        Logistic = self.neural_net(t, self.weights, self.biases)
        I = Logistic
        return I
    
    
   
    
    def net_l(self, t):
        Nf = self.Nf
        beta = self.beta
        kappa = self.kappa
        
        I = self.net_Logistic(t)
        I_t = tf.gradients(I, t)[0]
        
        l1 = I - (Nf/(1+beta*tf.math.exp(-kappa*t))) 
        l2 = I_t - (kappa*(I-(1/Nf)*I**2))
        
        return l1, l2
    
    
    def train(self, nIter):
        tf_dict = {self.t_tf: self.t, self.I_tf: self.I}
        start_time = timeit.default_timer()

        for it in tqdm(range(nIter)):
            self.sess.run(self.train_op, tf_dict)
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                self.loss_log.append(loss_value)
                beta_value = self.sess.run(self.beta)
                kappa_value = self.sess.run(self.kappa)
                Nf_value = self.sess.run(self.Nf)
                start_time = timeit.default_timer()
        
        
    def predict(self, t_star):
        tf_dict = {self.t_tf: t_star}
        
        I_star = self.sess.run(self.I_predL, tf_dict)
        
        return I_star
        
##########################################################################################################
# training the network

niter = 25000  # number of Epochs
layers = [1, 64, 64, 64, 1]
t_data = Td.flatten()[:,None]
I_data = cs_I.flatten()[:,None]    

from sklearn.model_selection import train_test_split

# random splits

T_train, T_test, I_train, I_test = train_test_split(t_data, I_data, test_size = 0.2, random_state = 25)
T_train = np.sort(T_train, axis = 0)
T_test = np.sort(T_test, axis = 0)
I_train = np.sort(I_train, axis = 0)
I_test = np.sort(I_test, axis = 0)

# Doman bounds for train
lb =T_train.min(0)
ub = T_train.max(0)

lb1 =T_test.min(0)
ub1 =T_test.max(0)

model1 = PINN_constantParam(T_train, I_train, layers, lb, ub)
model1.train(niter)



model2 = PINN_constantParam(T_test, I_test, layers, lb1, ub1)
model2.train(niter)



#Calling the model for Training
I_predL2 = model1.predict(T_train)

mse_train_loss = model1.loss_log
rmse_train_loss = np.sqrt(mse_train_loss)

# flatten array
T0L = t.flatten()
T1L = T_train.flatten()

# re-scale data
I0L = np.min(I) + (np.max(I) - np.min(I))*new_I.flatten()
I1L = np.min(I) + (np.max(I) - np.min(I))*I_predL2.flatten()

# convert float to list
T0L = T0L[0:nd].tolist()
T1L = T1L[0:nd].tolist()
I0L = I0L[0:nd].tolist()
I1L = I1L[0:nd].tolist()


# prediction
I_predLT = model2.predict(T_test)


mse_validation_loss = model2.loss_log
rmse_validation_loss = np.sqrt(mse_validation_loss)

# flatten array
T0LT = t.flatten()
T1LT = T_test.flatten()

# re-scale data
I0LT = np.min(I) + (np.max(I) - np.min(I))*new_I.flatten()
I1LT = np.min(I) + (np.max(I) - np.min(I))*I_predLT.flatten()

# convert float to list
T0LT = T0LT[0:nd].tolist()
T1LT = T1LT[0:nd].tolist()
I0LT = I0LT[0:nd].tolist()
I1LT = I1LT[0:nd].tolist()


print("days:",*["%.8f"%(x) for x in T0LT[0:nd]])
print("time:",*["%.8f"%(x) for x in T1LT[0:nd]])
print("cases:",*["%.8f"%(x) for x in I0LT[0:nd]])
print("infectd:",*["%.8f"%(x) for x in I1LT[0:nd]])

beta_valueL = model2.sess.run(model2.beta)
kappa_valueL = model2.sess.run(model2.kappa)
Nf_valueL = model2.sess.run(model2.Nf)

# learned parameters
print("beta:",*["%.8f"%(x) for x in beta_valueL])
print("kappa:",*["%.8f"%(x) for x in kappa_valueL])
print("Nf:",*["%.8f"%(x) for x in Nf_valueL])

##########
### Error
# Coefficient of determination
corr_matrix = np.corrcoef(I0LT, I1LT[:135])
corr = corr_matrix[0,1]
R_sq = corr**2
print("R_sq:", R_sq) 

# MAPE
def mean_absolute_percentage_error(k_true, k_pred):
    k_true, k_pred = np.array(k_true), np.array(k_pred)
    return np.mean(np.abs((k_true - k_pred) / k_true))

mapep =  mean_absolute_percentage_error(I0LT, I1LT[:135])
print("MAPE:", mapep)

#EV
ev = (explained_variance_score(I0LT, I1LT[:135]))
print("EV:",ev)

#RMSE
RMSE = np.sqrt(mean_squared_error(I0LT, I1LT[:135]))
print("RMSE:", RMSE)


print("mse_train_loss:",*["%.8f"%(x) for x in mse_train_loss])
print("rmse_train_loss:",*["%.8f"%(x) for x in rmse_train_loss])


print("mse_validation_loss:",*["%.8f"%(x) for x in mse_validation_loss])
print("rmse_validation_loss:",*["%.8f"%(x) for x in rmse_validation_loss])


##########################################################################################################





    


