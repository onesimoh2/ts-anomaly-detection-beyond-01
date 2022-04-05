import pandas as pd
import numpy as np
import os
from datetime import date
from functools import wraps

def get_csv_from_blob(csvN):
    ############# Read the csv ###########################  
    df = pd.read_csv('data/' + csvN) 
    return df

def singleton(orig_cls):
    orig_new = orig_cls.__new__
    instance = None

    @wraps(orig_cls.__new__)
    def __new__(cls, *args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = orig_new(cls, *args, **kwargs)
        return instance
    orig_cls.__new__ = __new__
    return orig_cls

@singleton
class DateUtils():
    inidate = date(1900,1,1)
    def calc_day(end_dat):  # returns the number of days from the initial date
        ini_dat = DateUtils.inidate
        #iniDat = date(1986,1,1)
        #iniDat = date(2001,1,1)
        dateend = end_dat.date()
        return (dateend-ini_dat).days

# returns the secuential position of the data point assuming it is a monthly data
    def calc_monthly_order_from_day(end_dat): 
        day_num = DateUtils.calc_day(end_dat)
        pos_sec = day_num / 30.4
        return int(round(pos_sec))

def variance(data):
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    sig = np.sqrt(variance)
    return mean, variance, sig