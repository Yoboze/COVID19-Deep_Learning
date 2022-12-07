import numpy as np
import time 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import torch
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tools.tools import add_constant
import matplotlib.dates as dates

import sys

##Data Preparation
scaler1=MinMaxScaler()
lookback = 2
def data_lstm(data, lookback, scaler1):
    """
    Input: data and time steps
    """
    ndt=scaler1.fit_transform(data)
#     ndt =data
    x_ar =[]
    y_ar =[]
    n =len(data)
    for k in range(n):
        ini = k + lookback
        if (ini)> n-1:
            break
        xs, ys =ndt[k:ini], ndt[ini]
        x_ar.append(xs)
        y_ar.append(ys)
        x, y =np.array(x_ar), np.array(y_ar) 
    
    return x,y
def split_data(data,lookback, scaler1, split):
    x, y = data_lstm(data, lookback, scaler1)
    indx =int(split*len(y))
    x_data, y_data =x, y
    x_train, y_train =x[:indx],y[:indx]
    x_test, y_test =x[indx:],y[indx:]
    return x_data, y_data, x_train, y_train, x_test, y_test

## RNN models
##LSTM class
class LSTM_model(nn.Module):

    def __init__(self, n_layers, n_hidden, in_size, out_size, drop_prob=0.2):
        super(LSTM_model, self).__init__()
        
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.in_size = in_size
        self.out_size= out_size
        #LSTM layer
        self.lstm_out = nn.LSTM(input_size=in_size, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, dropout=drop_prob)
            
        ###Fully connected layer
        self.fc = nn.Linear(n_hidden, out_size)

        
    def forward(self, x, h):
        out, h = self.lstm_out(x, h)
        out = self.fc(out[:,-1])
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden
        


class GRU_model(nn.Module):
    def __init__(self,  n_layers, n_hidden, in_size, out_size, drop_prob=0.2):
        super(GRU_model, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.in_size = in_size
        self.out_size= out_size
        # GRU layers
        self.gru = nn.GRU(input_size=in_size, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, dropout=drop_prob)

        # Fully connected layer
        self.fc = nn.Linear(n_hidden, out_size)

        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(out[:,1])
        return out, h 
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device)
        return hidden

## Train and Evaluate Models
def train_model(loader,lr,n_hidden, epochs,  n_layers, batch_size,out_size, eva_name, option="GRU"):
    in_size= next(iter(loader))[0].shape[2]
   
    # Instantiating the models
    if (option == "GRU"):
        model = GRU_model( n_layers, n_hidden, in_size, out_size)
    else:
        model = LSTM_model(n_layers, n_hidden, in_size, out_size)
    model.to(device)
    
    #Get the loss function
    loss_func = torch.nn.MSELoss()   
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    print("{} {}".format(option, eva_name))
    total_time=[]
    epc_arr=[]
    loss_arr=[]
     
    # Start training loop
    for epoch in range(1,epochs+1):
        #Get the hidden state
        st= time.time()
        loss_avg = 0.
        c= 0
        h = model.init_hidden(batch_size)
        for x, label in loader:
            c += 1
            if (option == "GRU"):
                h = h.data
            elif(option == "BiLSTM"):
                h = tuple([e.data for e in h])
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()
            
            out, h = model(x.to(device).float(), h)
            loss =  loss_func (out, label.to(device).float())
            loss.backward()
            optim.step()
            loss_avg += loss.item()
        ct= time.time()
        elapsed =ct-st
        epc_arr.append(epoch)
        loss_arr.append(loss_avg)
        total_time.append(elapsed)
        if epoch%100==0:
            print("Epoch %d/%d, %s Loss: %.3e, Time:%.2f seconds"%(epoch, epochs,eva_name, loss_avg/len(loader), elapsed))
    print('{} Total {} Time in seconds {}'.format(option, eva_name, str(sum(total_time))))
    return model, np.array(epc_arr), np.array(loss_arr)
def evaluate_model(model, x_test, y_test, x_data, y_data, x_train, y_train,scaler1, v, name):
    model.eval()
    start_time = time.time()
    inputs = torch.from_numpy(np.array(x_data))
    labs = torch.from_numpy(np.array(y_data))
    h = model.init_hidden(inputs.shape[0])
    out, h = model(inputs.to(device).float(), h)
    output=out.detach().cpu().numpy().reshape((-1,1))
    labs.numpy().reshape((-1,1))
    actual =scaler1.inverse_transform(np.array(labs))
    predicted=scaler1.inverse_transform(np.array(output))
    predicted =np.abs(predicted)
    ##Get training
    inputs_train =torch.from_numpy(np.array(x_train))
    h = model.init_hidden(inputs_train.shape[0])
    out, h = model(inputs_train.to(device).float(), h)
    output1=out.detach().cpu().numpy().reshape((-1,1))
    train_actual =scaler1.inverse_transform(np.array(y_train))
    train_pred =scaler1.inverse_transform(np.array(output1))
    
    ##Get Testing
    inputs_test =torch.from_numpy(np.array(x_test))
    h = model.init_hidden(inputs_test.shape[0])
    out, h = model(inputs_test.to(device).float(), h)
    output2=out.detach().cpu().numpy().reshape((-1,1))
    test_actual =scaler1.inverse_transform(np.array(y_test))
    test_pred =scaler1.inverse_transform(np.array(output2))
    test_pred =np.abs(test_pred)
    print("Evaluation Time: {}".format(str(time.time()-start_time)))
    
    ##Get errors
    rmse =np.sqrt(mean_squared_error(test_actual, test_pred))
    mape =np.linalg.norm((test_pred-test_actual),2)/np.linalg.norm(test_actual, 2)
    ev =1- (np.var(test_pred-test_actual)/np.var(test_actual))
    R_square = r2_score(test_actual, test_pred) 
    print('RMSE for {}-{} : {} when v={}'.format(name,model, rmse, v))
    print('MAPE for {}-{}: {} when v={}'.format(name,model,mape, v))
    print('EV for {}-{}: {} when v={}'.format(name,model,ev, v))
    print('R_sq for {}-{}: {} when v={}'.format(name,model,R_square, v))
    return actual,predicted, train_actual, train_pred, test_actual, train_pred

## Run models
torch.manual_seed(0)
np.random.seed(1234)
lr= 0.01
# in_size = 1
n_hidden = 17  #when neurons =16, 32
n_layers = 1 #when layers =2,3, 4, 5, 7
out_size = 1
n, l = 16, 2
def run_model(data, name, v, option, cs, epochs, batch_size):
	print('{} outcomes............'.format(cs))
	x_data, y_data, x_train, y_train, x_test, y_test=split_data(data, 2, scaler1, 0.8)
	train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
	test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
	train_loader = DataLoader(train_data, shuffle=True, batch_size = batch_size, drop_last=True)
	test_loader = DataLoader(test_data, shuffle=True, batch_size = 25 , drop_last=True)
	model, ep, loss_train = train_model(train_loader,lr,n_hidden, epochs, n_layers, batch_size, out_size, "Train", option = option)
	_, ep, loss_test = train_model(test_loader,lr,n_hidden, epochs, n_layers,25, out_size,"Validation", option = option)
	y_true, y_pred, tr_a, tr_p, ts_a, ts_p = evaluate_model(model, x_test, y_test, x_data, y_data, x_train, y_train, scaler1, v, name)
	return y_true, y_pred, tr_a, tr_p, ts_a, ts_p, ep, loss_train, loss_test

	


#Confidence Intervals
epochs =2500
batch_size = 32
def confidence_interval(data, M, option, cs, name, n_steps, DatName):
    pred_all =[]
    m =len(data)
    for k in range(M):#number of realizations
        print(k)
        y_inf =np.zeros((m,1))
        for j in range(m):
            y_inf[j,:] =np.random.poisson(data[j], size=(1,1))
        x_data, y_data, x_train, y_train, x_test, y_test=split_data(y_inf,2, scaler1, 0.8)
        print('####### {} bootstrap for {} #######'.format(option,cs))
        train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
        model, ep, loss_train = train_model(train_loader,lr,n_hidden, epochs, n_layers, batch_size, out_size, "Train", option = option)
        model.eval()
        km =len(data)
        x_input=y_data[km-2-n_steps:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        lst_output=[]
        i=0
        while(i<65):
            if(len(temp_input)>n_steps):
                x_input=np.array(temp_input[1:])
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                x_input = torch.from_numpy(np.array(x_input))
                h = model.init_hidden(x_input.shape[0])
                out1, h = model(x_input.to(device).float(), h)
                yhat=out1.detach().cpu().numpy().reshape((-1,1))
                
                
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                x_input = torch.from_numpy(np.array(x_input))
                h = model.init_hidden(x_input.shape[0])
                out1, h = model(x_input.to(device).float(), h)
                yhat=out1.detach().cpu().numpy().reshape((-1,1))
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i=i+1
        pred_all.append(scaler1.inverse_transform(np.array(lst_output)))
    val =np.array(pred_all)
    ym =np.mean(val, axis=0)
    var=np.std(val, axis=0)
    lower =ym -2*(var)
    upper= ym +2*(var)
    dtrange =np.arange(0,135)
    dtrange1 =np.arange(135, 200)
    
    font = 24
    fig, ax = plt.subplots() 
    ax.plot(dtrange, data, 'k--', marker='o', lw=2,label='{}-Data'.format(DatName))
    #ax.plot(dtrange1, test_data,'ob',lw=2,label='Test Data')
    ax.plot(dtrange1, ym,'r-',lw=2,label='{}-Mean-Prediction for Italy'.format(option))
    ax.fill_between(dtrange1.ravel(), lower.ravel(), upper.ravel(), facecolor='cyan', label ='Confidence Bound')
    ax.plot(dtrange1,upper, 'm-',lw=2,label='Upper 95%')
    ax.plot(dtrange1,lower, 'm-', lw=2,label='Lower 95%')
    # ax.vlines(x =date1, ymin = 0, ymax = max(data)+1e+3, colors = 'purple') 
#     ax.xaxis.set_major_locator(dates.MonthLocator(interval=2))
#     ax.xaxis.set_major_formatter(dates.DateFormatter('%m-%y'))
#     ax.xaxis.set_minor_locator(dates.DayLocator(interval=7))
    ax.legend(fontsize=22)
    ax.tick_params(axis='both', labelsize = 24)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    ax.grid(True)
    ax.set_xlabel('Days', fontsize = font)
    ax.set_ylabel('$C(t)$', fontsize = font)
    ax.set_title('Prediction',  fontsize = font)
    fig.set_size_inches(w=13,h=8.5)
#     plt.savefig(out+'con_{}_{}_{}_{}_1.png'.format(name, cs, n, l))
    plt.show()
    return ym