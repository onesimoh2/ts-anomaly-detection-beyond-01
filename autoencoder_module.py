import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from utilities import DateUtils
from fft_functions import fourier_extrapolation, fourierPrediction
import matplotlib.pyplot as plt
# te way to install pytorch is: pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

class FeatureDatasetFromDf(Dataset):


    def auto_inc(self, ipos):
        self.iPos += 1
        return ipos

    def __init__(self, *args):
        if len(args) == 8:
            df, scaler, fit_dat, columns_names, dateName, ser_pos, n_df, xrand_position = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]
        else:
            df, scaler, fit_dat, columns_names, dateName, ser_pos, n_df, self.extrpl, self.x_freqdom, self.f, self.p, self.indexes, self.ntrain, xrand_osition = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13]

        x = df 
        x = x.reset_index()
        
        self.iPos = ser_pos
        self.x_train = np.array(x[columns_names].values)
        if(fit_dat == 'true'):
            x_in = df[columns_names].values.flatten()           
            self.extrpl, self.x_freqdom, self.f, self.p, self.indexes, self.n_train = fourier_extrapolation(x_in, 0)



        forcastVals, restored_sig, trend = fourierPrediction(x, self.x_freqdom, self.f, self.p, self.indexes, self.n_train, columns_names)
        
        axFFT = plt.axes()
        tt = np.arange(0, len(trend))
        axFFT.plot(tt, np.abs(trend))
        axFFT.plot(tt, np.abs(restored_sig))
        plt.show()        
        #nDfLast = DateUtils.calc_day(x[dateName].iloc[-1]) 
        #nDf1 = int(round(nDfLast * 0.033115)) #int(nDf*0.066)
        nDf1 = n_df
        #biasRand = [xrand_position[DateUtils.calcorderFromDay(item)] for item in x[dateName]]
        
        
      


        i = 0
                     
        
        #forcastValsR = np.array(forcastVals).reshape(-1, 1)
        trend_reshape = np.array(trend).reshape(-1, 1)
        stationary = np.array(restored_sig).reshape(-1, 1)
        
        self.x_train = np.append(self.x_train, trend_reshape, axis=1)
        self.x_train = np.append(self.x_train, stationary, axis=1)

        if(fit_dat == 'true'):            
            self.x_train[:,[0,1,2]] = scaler.fit_transform(self.x_train[:,[0,1,2]].reshape(-1, 3)).reshape(-1, 3)            
        else:
            self.x_train[:,[0,1,2]] = scaler.transform(self.x_train[:,[0,1,2]].reshape(-1, 3)).reshape(-1, 3)               

        #biasAddedSeed = [((item[0]*item[0]) + (item[1]*item[1])  + (item[2]*item[2])) for item in self.x_train]
        data_vs_trend = [np.abs(item[0]-item[1]) for item in self.x_train]
        data_vs_trend_reshape = np.array(data_vs_trend).reshape(-1, 1)
        self.x_train = np.append(self.x_train, data_vs_trend_reshape, axis=1)

        #x_train have: data, train estimate, stationary estimate and difference between the first two

        #trainn.to_csv('C:/Users/ecbey/Downloads/x_train.csv')  
        self.X_train = torch.tensor(self.x_train, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx]


class autoencoder(nn.Module):

    #iitialize weights
    def initialize_weights(self):
        for n in self.modules():
            if isinstance(n, nn.Linear):
                nn.init.kaiming_uniform_(n.weight)
                nn.init.constant_(n.bias, 0)


    def __init__(self, epochs=15, batchSize=10, learningRate=1e-3, weight_decay=1e-5, layer_reduction_factor = 1.6, number_of_features = 29, seed=15000):
        super(autoencoder, self).__init__()
        #seed = 15000
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.weight_decay = weight_decay
        self.layer_reduction_factor = layer_reduction_factor
        self.number_0f_features = number_of_features
        self.output_features_by_leyers = []
        self.output_features_by_leyers.append(number_of_features)
        self.number_of_last_layer = int(round(number_of_features / 6.0))
        if self.number_of_last_layer <= 1 and number_of_features > 3:
            self.number_of_last_layer = 2

        for leyer_lev in range(5):
            reduct = int(round(self.output_features_by_leyers[leyer_lev] / layer_reduction_factor))
            if reduct < self.number_of_last_layer or reduct == self.output_features_by_leyers[-1] or reduct == 0:
                break
            last_leyer = self.output_features_by_leyers[-1]
            self.output_features_by_leyers.append(reduct)
            if(reduct == 1):
                break



        self.encoder = nn.Sequential()

        for iLayer in range(len(self.output_features_by_leyers)-1):
            self.encoder.add_module('L' + str(iLayer), nn.Linear(self.output_features_by_leyers[iLayer], self.output_features_by_leyers[iLayer + 1]))
            if iLayer + 1 < len(self.output_features_by_leyers) - 1:
                self.encoder.add_module('R' + str(iLayer), nn.ReLU())
        
        self.decoder = nn.Sequential()
        for iLayer in range(len(self.output_features_by_leyers))[::-1]:
            self.decoder.add_module('L' + str(iLayer), nn.Linear(self.output_features_by_leyers[iLayer], self.output_features_by_leyers[iLayer - 1]))
            if iLayer > 1:
                self.decoder.add_module('R' + str(iLayer), nn.ReLU())
            else:
                break

        self.initialize_weights()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)
        
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1, mode='min', verbose=True)
        self.loss = nn.MSELoss()
        #self.loss = nn.L1Loss()
