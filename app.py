from utilities import GetCsvFromBlob, DateUtils
import pandas as pd
from datetime import date, datetime


db = GetCsvFromBlob('catfish.csv') 
#make sure columns are in wright format
db['Date']= pd.to_datetime(db['Date']) 
db['Total'] = pd.to_numeric(db['Total'], errors='coerce')
db['Total'] = db['Total'].astype('float')

#resolve missing data
db['Total'].interpolate(method='linear', inplace=True)

DATA_FROM = date(1986,1,1) #first part of the time series
DATA_TO = date(2001,1,1)

db = db[(db.Date < datetime(DATA_TO.year, DATA_TO.month, DATA_TO.day)) & (db.Date >= datetime(DATA_FROM.year, DATA_FROM.month, DATA_FROM.day))]
DateUtils.inidate = DATA_FROM