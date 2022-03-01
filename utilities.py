import pandas as pd
import numpy as np
import os
from datetime import date
from functools import wraps

def GetCsvFromBlob(csvN):
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
    def calcDay(endDat):
        iniDat = DateUtils.inidate
        #iniDat = date(1986,1,1)
        #iniDat = date(2001,1,1)
        dateend = endDat.date()
        return (dateend-iniDat).days

    def calcorderFromDay(endDat):
        dayNum = DateUtils.calcDay(endDat)
        ordSec = dayNum / 30.4
        return int(round(ordSec))