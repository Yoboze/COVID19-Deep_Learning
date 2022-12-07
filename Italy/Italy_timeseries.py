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
class PINN_ExpAlpha_Alpha:
    # Initialize the class
    def __init__(self, t, I,  layers1, layers2, lb, ub):
        
        self.lb = lb
        self.ub = ub
        
        self.t = t
        
        self.I = I
        
        
        
       
        
            
    
        self.layers1 = layers1
        self.layers2 = layers2
        
        
        
        
        self.weights1, self.biases1 = self.initialize_NN(layers1)
        self.weights2, self.biases2 = self.initialize_NN(layers2)
       
        
        
       
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        
        self.Nf = tf.Variable([1.0], dtype=tf.float32)
        
       
        
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.I_tf = tf.placeholder(tf.float32, shape=[None, self.I.shape[1]])
                       
        
        self.I_predA = self.net_Logistic(self.t_tf)
        self.alpha_predA = self.alphaFunc(self.t_tf)
        
        
        self.l1  = self.net_l(self.t_tf)
             
        self.loss = tf.reduce_mean(tf.square(self.I_tf - self.I_predA)) + \
                    tf.reduce_mean(tf.square(self.l1)) 
                     
            
                    
                    
                    
            
        
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
    
    def neural_net(self, t, layers1, weights1, biases1):
        num_layers = len(layers1)
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights1[l]
            b = biases1[l]
            H = tf.sigmoid(tf.add(tf.matmul(H, W), b))
        W = weights1[-1]
        b = biases1[-1]
        #Y = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    
    
    def neural_net1(self, t, layers2, weights2, biases2):
        num_layers = len(layers2)
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights2[l]
            b = biases2[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights2[-1]
        b = biases2[-1]
        #Y = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
        Y = tf.add(tf.matmul(H, W), b)
        return Y

   
    
    
    
    def net_Logistic(self, t):
        Logistic = self.neural_net(t, self.layers1, self.weights1, self.biases1)
        I = Logistic
        return I
    
    

    def alphaFunc(self,t):
        alpha = self.neural_net1(t, self.layers2, self.weights2, self.biases2)
        alp = alpha
        return alp
    
    
    def net_l(self, t):
        Nf = self.Nf
        
        
        alpha = self.alphaFunc(t)
        I = self.net_Logistic(t)
        
        I_t = tf.gradients(I, t)[0]
        
        
        #l1 = I - (Nf/(1+beta*tf.math.exp(-alpha*t)))
        
        l1 = I_t - ((alpha*I)*(1 - I/Nf))
        
        
        return l1
    
        
    def train(self, nIter):
        tf_dict = {self.t_tf: self.t, self.I_tf: self.I}
        start_time = timeit.default_timer()

        for it in tqdm(range(nIter)):
            self.sess.run(self.train_op, tf_dict)
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                self.loss_log.append(loss_value)
                Nf_value = self.sess.run(self.Nf)
                start_time = timeit.default_timer()
        
        
        
    def predict(self, t_star):
        tf_dict = {self.t_tf: t_star}
        
        
        I_star = self.sess.run(self.I_predA, tf_dict)
        
        alpha_star = self.sess.run(self.alpha_predA, tf_dict)
        
        
        return I_star,  alpha_star



#########################################################################################################
# training and testing the network

niter = 25000  # number of Epochs
layers1 = [1, 64, 64, 64, 1]
layers2 = [1, 64, 64, 64, 1]
t_data = Td.flatten()[:,None]
I_data = cs_I.flatten()[:,None]    

from sklearn.model_selection import train_test_split

# random splits

T_train, T_test, I_train, I_test = train_test_split(t_data, I_data, test_size = 0.2, random_state = 5)
T_train = np.sort(T_train, axis = 0)
T_test = np.sort(T_test, axis = 0)
I_train = np.sort(I_train, axis = 0)
I_test = np.sort(I_test, axis = 0)

# Doman bounds
lb = t_data.min(0)
ub = t_data.max(0)

lb1 =T_test.min(0)
ub1 =T_test.max(0)

model1 = PINN_ExpAlpha_Alpha(T_train, I_train, layers1, layers2, lb, ub)
model1.train(niter)


model2 = PINN_ExpAlpha_Alpha(T_test, I_test, layers1, layers2, lb1, ub1)
model2.train(niter)


#Calling the model for Training
I_predA2,  alpha_predA2 = model1.predict(T_train)

mse_train_loss = model1.loss_log
rmse_train_loss = np.sqrt(mse_train_loss)



# flatten array
T0A = t.flatten()
T1A = T_train.flatten()

# re-scale data
I0A = np.min(I) + (np.max(I) - np.min(I))*new_I.flatten()
I1A = np.min(I) + (np.max(I) - np.min(I))*I_predA2.flatten()
A1A = np.min(I) + (np.max(I) - np.min(I))*alpha_predA2.flatten()

# convert float to list
T0A = T0A[0:nd].tolist()
T1A = T1A[0:nd].tolist()
I0A = I0A[0:nd].tolist()
I1A = I1A[0:nd].tolist()
A1A = A1A[0:nd].tolist()


# prediction
I_predAT, alpha_predAT = model2.predict(T_test)


mse_validation_loss = model2.loss_log
rmse_validation_loss = np.sqrt(mse_validation_loss)


# flatten array
T0AT = t.flatten()
T1AT = T_test.flatten()

# re-scale data
I0AT = np.min(I) + (np.max(I) - np.min(I))*new_I.flatten()
I1AT = np.min(I) + (np.max(I) - np.min(I))*I_predAT.flatten()
A1AT = alpha_predAT.flatten()

# convert float to list
T0AT = T0AT[0:nd].tolist()
T1AT = T1AT[0:nd].tolist()
I0AT = I0AT[0:nd].tolist()
I1AT = I1AT[0:nd].tolist()
A1AT = A1AT[0:nd].tolist()

print("daysA:",*["%.8f"%(x) for x in T0AT[0:nd]])
print("timeA:",*["%.8f"%(x) for x in T1AT[0:nd]])
print("casesA:",*["%.8f"%(x) for x in I0AT[0:nd]])
print("infectdA:",*["%.8f"%(x) for x in I1AT[0:nd]])
print("alphaA:",*["%.8f"%(x) for x in A1AT[0:nd]])

Nf_valueA = model2.sess.run(model2.Nf)




# learned parameters
print("Nf_A:",*["%.8f"%(x) for x in Nf_valueA])


##########
### Error
# Coefficient of determination
corr_matrix = np.corrcoef(I0AT, I1AT[:135])
corr = corr_matrix[0,1]
R_sq = corr**2
print("R_sqA:", R_sq) 

# MAPE
def mean_absolute_percentage_error(k_true, k_pred):
    k_true, k_pred = np.array(k_true), np.array(k_pred)
    return np.mean(np.abs((k_true - k_pred) / k_true))

mapep =  mean_absolute_percentage_error(I0AT, I1AT[:135])
print("MAPE_A:", mapep)

#EV
ev = (explained_variance_score(I0AT, I1AT[:135]))
print("EV_A:",ev)

#RMSE
RMSE = np.sqrt(mean_squared_error(I0AT, I1AT[:135]))
print("RMSE_A:", RMSE)

print("mse_train_lossA:",*["%.8f"%(x) for x in mse_train_loss])
print("rmse_train_lossA:",*["%.8f"%(x) for x in rmse_train_loss])

print("mse_validation_lossA:",*["%.8f"%(x) for x in mse_validation_loss])
print("rmse_validation_lossA:",*["%.8f"%(x) for x in rmse_validation_loss])
##########################################################################################################

