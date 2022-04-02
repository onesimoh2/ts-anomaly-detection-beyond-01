from utilities import get_csv_from_blob, DateUtils
import pandas as pd
import numpy  as np
from datetime import date, datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statistics import mean, median
from sklearn.preprocessing import MinMaxScaler
from fft_functions import fourier_extrapolation
from autoencoder_module import FeatureDatasetFromDf, autoencoder



def catfich_1986_2001(train_test) :
    db = get_csv_from_blob('catfish.csv') 
    #make sure columns are in wright format
    db['Date']= pd.to_datetime(db['Date']) 
    db['Total'] = pd.to_numeric(db['Total'], errors='coerce')
    db['Total'] = db['Total'].astype('float')

    #resolve missing data
    db['Total'].interpolate(method='linear', inplace=True)

    COLUMN_NAMES = ['Total']
    DATE_COLUMN_NAME = 'Date'
    

    #define if there will be a train and test segmentation of the data
    TRAIN_TEST = True

    ############ first part of the time series ##################################
    DATA_FROM = date(1986,1,1) 
    DATA_TO = date(2001,1,1)

    DateUtils.inidate = DATA_FROM #define the initial date

    #data extraction
    original_data = db[(db.Date < datetime(DATA_TO.year - 1, DATA_TO.month, DATA_TO.day)) & (db.Date >= datetime(DATA_FROM.year, DATA_FROM.month, DATA_FROM.day))]
    #last part of the serie will be used as it were the new data where we are going to detect anomalies
    new_data = db[(db.Date < datetime(DATA_TO.year, DATA_TO.month, DATA_TO.day)) & (db.Date >= datetime(DATA_TO.year - 1, DATA_TO.month, DATA_TO.day))]

    

    if train_test:
        new_data.loc[176, ['Total']] = 15000 #introducing anomaly
        fig_ish = plt.figure()
        ax_fish = plt.axes()
        ax_fish.plot(original_data['Date'], original_data['Total'])
        plt.show()
        train_db, test_db = train_test_split(original_data, test_size=0.05, shuffle=False)
    else:
        
        fig_ish = plt.figure()
        ax_fish = plt.axes()
        ax_fish.plot(original_data['Date'], original_data['Total'])
        plt.show()
        train_db = original_data #first part all data
        new_data.loc[176, ['Total']] = 15000 #first part anomaly

    seed = int(median(list(train_db['Total'])))      
    np.random.seed(seed)
    xrandPosition = np.random.rand(500)
    #print(xrandPosition)
    scaler = MinMaxScaler()
    n_df = len(train_db)
    train_split = FeatureDatasetFromDf(train_db, scaler, 'true', COLUMN_NAMES, DATE_COLUMN_NAME, 1, n_df, xrandPosition)
    ipos = train_split.iPos
    extrpl, x_freqdom, f, p, indexes, n_train = train_split.extrpl, train_split.x_freqdom, train_split.f, train_split.p, train_split.indexes, train_split.n_train

    if train_test:
        test_split = FeatureDatasetFromDf(test_db, scaler, 'false', COLUMN_NAMES, DATE_COLUMN_NAME, ipos, n_df, extrpl, x_freqdom, f, p, indexes, n_train)
        ipos = test_split.iPos

    MAX_TRAINING_LOSS_VAR =  3.0 #number of sigmas to consider anomaly
    #LAYER_REDUCTION_FACTOR = 1.6 #how much each leyer of the autoencoder decrease
    LAYER_REDUCTION_FACTOR = 1.2 #how much each leyer of the autoencoder decrease
    BATCH_SIZE = int(len(train_db)/10) 
    

    number_of_features = int(train_split.X_train.size(dim=1))

    model = autoencoder(epochs = 500, batchSize = BATCH_SIZE, number_of_features = number_of_features, layer_reduction_factor = LAYER_REDUCTION_FACTOR,  seed = seed)



