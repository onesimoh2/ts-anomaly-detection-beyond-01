import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from utilities import DateUtils, variance
from fft_functions import fourier_extrapolation, fourierPrediction
import matplotlib.pyplot as plt

class FeatureDatasetFromDf(Dataset):


    def auto_inc(self, ipos):
        self.iPos += 1
        return ipos

    def __init__(self, *args):
        if len(args) == 7:
            df, scaler, fit_dat, columns_names, dateName, ser_pos, n_df = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        else:
            df, scaler, fit_dat, columns_names, dateName, ser_pos, n_df, self.extrpl, self.x_freqdom, self.f, self.p, self.indexes, self.n_train = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]

        x = df 
        x = x.reset_index()
        
        self.iPos = ser_pos
        self.x_train = np.array(x[columns_names].values)
        if(fit_dat == 'true'):
            # create the complex coefficients and the rest of the required parameters 
            x_in = df[columns_names].values.flatten()           
            self.extrpl, self.x_freqdom, self.f, self.p, self.indexes, self.n_train = fourier_extrapolation(x_in, 0)


        # use the coefficients and the rest of the parameters to calculate the nonlinear tend and the seasonality part 
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
        seasonal_part = np.array(restored_sig).reshape(-1, 1)
        
        self.x_train = np.append(self.x_train, trend_reshape, axis=1)
        self.x_train = np.append(self.x_train, seasonal_part, axis=1)

        if(fit_dat == 'true'):            
            self.x_train[:,[0,1,2]] = scaler.fit_transform(self.x_train[:,[0,1,2]].reshape(-1, 3)).reshape(-1, 3)            
        else:
            self.x_train[:,[0,1,2]] = scaler.transform(self.x_train[:,[0,1,2]].reshape(-1, 3)).reshape(-1, 3)               

        #biasAddedSeed = [((item[0]*item[0]) + (item[1]*item[1])  + (item[2]*item[2])) for item in self.x_train]
        data_vs_trend = [np.abs(item[0]-item[1]) for item in self.x_train]
        data_vs_trend_reshape = np.array(data_vs_trend).reshape(-1, 1)
        self.x_train = np.append(self.x_train, data_vs_trend_reshape, axis=1)

        #x_train have; data, train estimate, stationary estimate and difference between the first two 

        #trainn.to_csv('C:/Users/ecbey/Downloads/x_train.csv')  
        self.X_train = torch.tensor(self.x_train, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx]


#definition of all the features for creating a model, training, testing and execution. 
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
        #defining the structure of the autoencoder, this is a general method that should fit different structure depending on the number of input nodes 
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
        #dynamically creating the structure of the autoencoder 
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

    
    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder


    def train_only(self, train_sample, max_training_loss_var_num):                               
        train_ave = []
        last_epoch_loss = []
        last_epoch_individual_loss = []
        max_training_loss = 0
        criterion_no_reduced = nn.MSELoss(reduction = 'none')

        for epoch in range(self.epochs):
            self.train() 
            train_epc = 0.0
            train_num = 0.0
            for data in train_sample:
                #predict
                output = self(data)
                # find loss
                loss = self.loss(output, data)
                loss_train_data = loss.data.item()                
                # perform back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #accumulae the losses for each element
                train_epc = train_epc + loss_train_data
                train_num =  train_num + 1
                
                #print(f'epoch {epoch + 1}/{self.epochs}, loss: {loss.data:.4f}')
                if epoch + 1 == self.epochs:
                    last_epoch_loss.append(loss_train_data)
                    #obtain the list of all individual losses in the batch
                    test_loss_list = criterion_no_reduced(output, data)
                    num_cols = test_loss_list.size(dim=1)
                    sum_all = 0.0 #calculate individual loss
                    for xsqTen in test_loss_list:
                        sum_all = 0.0
                        #add all losses for each input 
                        for xsq in xsqTen:
                            sum_all = sum_all + xsq
                        #calculate the mean for each individual element of the batch
                        indivAve = float(sum_all/num_cols)
                        last_epoch_individual_loss.append(indivAve)

                    # if max_training_loss < loss_train_data:
                    #     max_training_loss = loss_train_data
                        
            #average of the losses for a given epoch
            epoc_t_ave = train_epc/train_num
            #add all losses in an array
            train_ave.append(float(epoc_t_ave))
            #min_loss = train_epc/train_num
            print(f'******epoch {epoch + 1}, loss: {train_ave[epoch]:.4f}')

            #supply the current loss for the scheduler
            min_loss_round = round(train_ave[epoch], 4)
            self.scheduler.step(min_loss_round)
       

        #calculate the theashold to detect anomalies
        mean, var, sig = variance(last_epoch_individual_loss)
        max_training_loss = mean +  (max_training_loss_var_num * sig)
        return max_training_loss, train_ave



    def execute_evaluate(self, feature_sample, max_training_loss, index_df):
        self.eval()
        #criterion = nn.L1Loss()
        criterion = nn.MSELoss()
        indx = 0
        test_epc = 0.0
        test_num = 0.0
        detected_anomalies = []
        criterion_no_reduced = nn.MSELoss(reduction = 'none')
        #criterionNoReduced = nn.L1Loss(reduction = 'none')
        with torch.no_grad(): # Run without Autograd
            for original in feature_sample:
                output = self.forward(original)  # model can't use test to learn
                test_loss = criterion(output, original).data.item()

                test_epc = test_epc + test_loss
                test_num = test_num + 1

                indx1 = index_df.iloc[[indx], [0]]
                print('test_loss=', test_loss, ' Indx=', indx)

                if test_loss > (1 * max_training_loss) :                    
                    item = [test_loss, int(indx1['ID123'])]
                    detected_anomalies.append(item)
                    #print('          test_loss=', test_loss, ' Indx=', indx1['ID123'])
                    #pd.DataFrame(original)
                

                indx = indx + 1
    
        print('max_training_loss=', max_training_loss )
        test_loss = (test_epc/test_num)
        pcent_anomalies_detected = (len(detected_anomalies) / len(feature_sample)) * 100
        #print(f'     Validate_loss: {test_loss:.4f}')
        return detected_anomalies, pcent_anomalies_detected, test_loss

