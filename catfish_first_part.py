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
from torch.utils.data import DataLoader, Subset



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

    ############ first part of the time series from 1986 to 2000 ##################################
    DATA_FROM = date(1986,1,1) 
    DATA_TO = date(2001,1,1)

    DateUtils.inidate = DATA_FROM #define the initial date

    # data extraction
    original_data = db[(db.Date < datetime(DATA_TO.year - 1, DATA_TO.month, DATA_TO.day)) & (db.Date >= datetime(DATA_FROM.year, DATA_FROM.month, DATA_FROM.day))]
    #The last year of the series will be used as it were the new data used to detect anomalies 
    new_data = db[(db.Date < datetime(DATA_TO.year, DATA_TO.month, DATA_TO.day)) & (db.Date >= datetime(DATA_TO.year - 1, DATA_TO.month, DATA_TO.day))]

    
    
    if train_test:
        # select train and test and introduce the anomaly value
        new_data.loc[176, ['Total']] = 15000 #introducing anomaly
        fig_ish = plt.figure()
        ax_fish = plt.axes()
        ax_fish.plot(original_data['Date'], original_data['Total'])
        plt.show()
        train_db, test_db = train_test_split(original_data, test_size=0.05, shuffle=False)
    else:
        # select only train and introduce the anomaly value
        fig_ish = plt.figure()
        ax_fish = plt.axes()
        ax_fish.plot(original_data['Date'], original_data['Total'])
        plt.show()
        train_db = original_data #first part all data
        new_data.loc[176, ['Total']] = 10000 #introducing anomaly

    # defining the random seed
    seed = int(median(list(train_db['Total'])))      
    np.random.seed(seed)

    #define the train dataset
    scaler = MinMaxScaler()
    n_df = len(train_db)
    train_split = FeatureDatasetFromDf(train_db, scaler, 'true', COLUMN_NAMES, DATE_COLUMN_NAME, 1, n_df)
    ipos = train_split.iPos
    extrpl, x_freqdom, f, p, indexes, n_train = train_split.extrpl, train_split.x_freqdom, train_split.f, train_split.p, train_split.indexes, train_split.n_train

    if train_test:
        # define the test data set
        test_split = FeatureDatasetFromDf(test_db, scaler, 'false', COLUMN_NAMES, DATE_COLUMN_NAME, ipos, n_df, extrpl, x_freqdom, f, p, indexes, n_train)
        ipos = test_split.iPos

 
    MAX_TRAINING_LOSS_VAR =  3.0 #number of sigmas from the mean to consider a value is an anomaly
    LAYER_REDUCTION_FACTOR = 1.6 #how much each layer of the autoencoder decreases 
    #LAYER_REDUCTION_FACTOR = 1.2 #how much each layer of the autoencoder decreases 
    BATCH_SIZE = int(len(train_db)/10) 
    
    # create the data loader for testing
    data_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=False)

    number_of_features = int(train_split.X_train.size(dim=1))

    # create the model for the autoencoder
    model = autoencoder(epochs = 500, batchSize = BATCH_SIZE, number_of_features = number_of_features, layer_reduction_factor = LAYER_REDUCTION_FACTOR,  seed = seed)

    ################# TRAIN #############################################
    if train_test:
        max_training_loss, train_ave, validate_ave, last_epoch_individual_loss = model.train_validate(data_loader, test_split, MAX_TRAINING_LOSS_VAR)
        fig1 = plt.figure()
        ax1 = plt.axes()
        epoch1 = []
        i = 1
        for item in train_ave:
            epoch1.append(i)
            i += 1
        ax1.plot(epoch1, train_ave, validate_ave)
        plt.show()
        fig2 = plt.figure()
        ax2 = plt.axes()
        plt.hist(last_epoch_individual_loss)
        plt.show()

    else:
        max_training_loss, train_ave = model.train_only(data_loader, MAX_TRAINING_LOSS_VAR)
        fig1 = plt.figure()
        ax1 = plt.axes()
        epoch1 = []
        i = 1
        for item in train_ave:
            epoch1.append(i)
            i += 1
        ax1.plot(epoch1, train_ave)
        plt.show()


    #################### EXECUTE ########################################
    index_df = pd.DataFrame()

    evaluate_file = new_data
    anomaly_file_ds = FeatureDatasetFromDf(evaluate_file, scaler, 'false',COLUMN_NAMES, DATE_COLUMN_NAME, ipos, n_df, extrpl, x_freqdom, f, p, indexes, n_train)        

    index_df.insert(0,'ID123',evaluate_file.index)   

    original_file_anomalies = evaluate_file.loc[evaluate_file.index.isin(index_df['ID123']-1)]

    model.eval()
    detected_anomalies1, pcent_anomalies_detected1, test_loss1 = model.execute_evaluate(anomaly_file_ds, max_training_loss, index_df)

    df_detected_anomalies1 = pd.DataFrame(detected_anomalies1, columns=['loss_val', 'indx'])
    original_file_anomalies_df = pd.DataFrame()
    original_file_anomalies=  []
    for itemAnom in detected_anomalies1:
        indx = int(itemAnom[1])
        original_file_anomalies.append(evaluate_file.loc[indx,:])
    original_file_anomalies_df = pd.DataFrame(original_file_anomalies)
    original_file_anomalies_df.to_csv('C:/Users/ecbey/Downloads/original_file_anomalies_df.csv')  





