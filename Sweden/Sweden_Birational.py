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
days = pd.date_range(start='03/11/20',end=today)
dd = np.arange(len(days))

total_cases = [df1[day.strftime('%-m/%-d/%y')].sum() for day in days] 
 
row_c=df1['Country_Region'].tolist().index('Sweden')
total_cases = [df1[day.strftime('%-m/%-d/%y')][row_c] for day in days]

t = np.reshape(dd, [-1])
I = np.reshape(total_cases, [-1])

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

new_I = NormalizeData(I) # scaled btw 0 and 1

# generating more data points for training
nd = 1000
cs1 = CubicSpline(t,new_I)

Td = np.linspace(0,122,nd)

cs_I = cs1(Td)

##### The Model
class PINN_ExpAlpha_Birational:
    # Initialize the class
    def __init__(self, t, I,  layers1, X, lb, ub):
        
        self.lb = lb
        self.ub = ub
        
        self.t = t
        
        self.I = I
        
        self.X = X
        
        #self.b = b
        
        
        
       
        
    
        self.layers1 = layers1
        
        
        self.weights1, self.biases1 = self.initialize_NN(layers1)
       
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.beta = tf.Variable([250.0], dtype=tf.float32)
        self.beta1 = tf.Variable([140.0], dtype=tf.float32)
        self.kappa = tf.Variable([9.0], dtype=tf.float32)
        self.kappa1 = tf.Variable([4.0], dtype=tf.float32)
        self.d = tf.Variable([0.1], dtype=tf.float32)
        self.d1 = tf.Variable([0.1], dtype=tf.float32)
        self.c1 = tf.Variable([1.0], dtype=tf.float32)
        self.c = tf.Variable([1.0], dtype=tf.float32)
        self.Nf = tf.Variable([1.0], dtype=tf.float32)
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.I_tf = tf.placeholder(tf.float32, shape=[None, self.I.shape[1]])
                       
        
        self.I_predB = self.net_Logistic(self.t_tf)
        self.alpha_predB = self.alphaFunc(self.t_tf)
        
        
        self.l1,self.l2  = self.net_l(self.t_tf)
             
        self.loss = tf.reduce_mean(tf.square(self.I_tf - self.I_predB)) + \
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

   
    
    
    def net_Logistic(self, t):
        Logistic = self.neural_net(t, self.layers1, self.weights1, self.biases1)
        I = Logistic
        return I
    
    
    def alphaFunc(self,t):
        X = self.X
        kappa = self.kappa
        d = self.d
        t1 = t[0:X]
        alp1 = kappa*d / (1 + d*t1)
        
        #b = self.b
        d1 = self.d1
        c1 = self.c1
        Nf = self.Nf
        kappa1 = self.kappa1
        t2 = t[X-1:]
        b1 = kappa1*d1 / (1 + d1*t2)
        b2 = 1 / (1 + ((1 -(c1/Nf))*(1 + d1*t2)**-kappa1))
        alp2 = tf.multiply(b1,b2)
        
        alp =  tf.concat([alp1,alp2],0)
        
        return alp2
    
    
    
    
    def net_l(self, t):
        X = 40
        c = self.c
        c1 = self.c1
        d = self.d
        d1 = self.d1
        beta = self.beta
        beta1 = self.beta1
        kappa = self.kappa
        kappa1 = self.kappa1
        
        
        alpha = self.alphaFunc(t)
        I = self.net_Logistic(t)
        
        I_t = tf.gradients(I, t)[0]
        
        t1 = t[0:X]
        I1 = I[0:X]
        l1 = I1 - (c/(1+beta*(1 + d*t1)**-kappa))
        
        t2 = t[X-1:]
        I2 = I[X-1:]
        l2 = I2 - (((c/(1+beta*(1 + d*X)**-kappa))-(c1/(1+beta1*(1 + d1*X)**-kappa1))+(c1/(1+beta1*(1 + d1*t2)**-kappa1))))
        
        
        
        
        return l1,l2   
    
        
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
                beta1_value = self.sess.run(self.beta1)
                d_value = self.sess.run(self.d)
                d1_value = self.sess.run(self.d1)
                c1_value = self.sess.run(self.c1)
                c_value = self.sess.run(self.c)
                kappa_value = self.sess.run(self.kappa)
                kappa1_value = self.sess.run(self.kappa1)
                Nf_value = self.sess.run(self.Nf)
                start_time = timeit.default_timer()
        
        
        
    def predict(self, t_star):
        tf_dict = {self.t_tf: t_star}
        
        
        I_star = self.sess.run(self.I_predB, tf_dict)
        
        alpha_star = self.sess.run(self.alpha_predB, tf_dict)
        
        #tt_star = self.sess.run(self.tt_pred, tf_dict)
        
        return I_star,  alpha_star

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
lb = T_train.min(0)
ub = T_train.max(0)

lb1 =T_test.min(0)
ub1 =T_test.max(0)

model1 =  PINN_ExpAlpha_Birational(T_train, I_train, layers, 1, lb, ub)
model1.train(niter)



model2 =  PINN_ExpAlpha_Birational(T_test, I_test, layers, 1, lb1, ub1)
model2.train(niter)



#Calling the model for Training
I_predR2,  alpha_predR2 = model1.predict(T_train)

mse_train_loss = model1.loss_log
rmse_train_loss = np.sqrt(mse_train_loss)


# flatten array
T0R = t.flatten()
T1R = T_train.flatten()

# re-scale data
I0R = np.min(I) + (np.max(I) - np.min(I))*new_I.flatten()
I1R = np.min(I) + (np.max(I) - np.min(I))*I_predR2.flatten()
A1R = np.min(I) + (np.max(I) - np.min(I))*alpha_predR2.flatten()

# convert float to list
T0R = T0R[0:nd].tolist()
T1R = T1R[0:nd].tolist()
I0R = I0R[0:nd].tolist()
I1R = I1R[0:nd].tolist()
A1R = A1R[0:nd].tolist()


# prediction
I_predRT, alpha_predRT = model2.predict(T_test)


mse_validation_loss = model2.loss_log
rmse_validation_loss = np.sqrt(mse_validation_loss)


# flatten array
T0RT = t.flatten()
T1RT = T_test.flatten()

# re-scale data
I0RT = np.min(I) + (np.max(I) - np.min(I))*new_I.flatten()
I1RT = np.min(I) + (np.max(I) - np.min(I))*I_predRT.flatten()
A1RT = alpha_predRT.flatten()

# convert float to list
T0RT = T0RT[0:nd].tolist()
T1RT = T1RT[0:nd].tolist()
I0RT = I0RT[0:nd].tolist()
I1RT = I1RT[0:nd].tolist()
A1RT = A1RT[0:nd].tolist()


print("daysB:",*["%.8f"%(x) for x in T0RT[0:nd]])
print("timeB:",*["%.8f"%(x) for x in T1RT[0:nd]])
print("casesB:",*["%.8f"%(x) for x in I0RT[0:nd]])
print("infectdB:",*["%.8f"%(x) for x in I1RT[0:nd]])
print("alphaB:",*["%.8f"%(x) for x in A1RT[0:nd]])


beta_valueB = model2.sess.run(model2.beta)
beta1_valueB = model2.sess.run(model2.beta1)
kappa_valueB = model2.sess.run(model2.kappa)
Nf_valueB = model2.sess.run(model2.Nf)
d_valueB = model2.sess.run(model2.d)
d1_valueB = model2.sess.run(model2.d1)
kappa1_valueB = model2.sess.run(model2.kappa1)
c1_valueB = model2.sess.run(model2.c1)
c_valueB = model2.sess.run(model2.c)

# learned parameters
print("betaB:",*["%.8f"%(x) for x in beta_valueB])
print("beta1B:",*["%.8f"%(x) for x in beta1_valueB])
print("kappaB:",*["%.8f"%(x) for x in kappa_valueB])
print("kappa1B:",*["%.8f"%(x) for x in kappa1_valueB])
print("cB:",*["%.8f"%(x) for x in c_valueB])
print("c1B:",*["%.8f"%(x) for x in c1_valueB])
print("dB:",*["%.8f"%(x) for x in d_valueB])
print("d1B:",*["%.8f"%(x) for x in d1_valueB])
print("NfB:",*["%.8f"%(x) for x in Nf_valueB])








##########
### Error
# Coefficient of determination
corr_matrix = np.corrcoef(I0RT, I1RT[:122])
corr = corr_matrix[0,1]
R_sq = corr**2
print("R_sqB:", R_sq) 

# MAPE
def mean_absolute_percentage_error(k_true, k_pred):
    k_true, k_pred = np.array(k_true), np.array(k_pred)
    return np.mean(np.abs((k_true - k_pred) / k_true))

mapep =  mean_absolute_percentage_error(I0RT, I1RT[:122])
print("MAPE_B:", mapep)

#EV
ev = (explained_variance_score(I0RT, I1RT[:122]))
print("EV_B:",ev)

#RMSE
RMSE = np.sqrt(mean_squared_error(I0RT, I1RT[:122]))
print("RMSE_B:", RMSE)

print("mse_train_lossB:",*["%.8f"%(x) for x in mse_train_loss])
print("rmse_train_lossB:",*["%.8f"%(x) for x in rmse_train_loss])

print("mse_validation_lossB:",*["%.8f"%(x) for x in mse_validation_loss])
print("rmse_validation_lossB:",*["%.8f"%(x) for x in rmse_validation_loss])
##########################################################################################################


