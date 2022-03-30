import numpy as np
from numpy import fft
from datetime import date
from utilities import DateUtils


def fourier_extrapolation(x, n_predict):
    n = x.size
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain

    #number of parameeters of high freq to consider
    n_param = int(n/10) #I found that this formula provides a good number
    h=np.absolute(np.sort(x_freqdom)[-n_param]) #get the value for the threshold
    #make 0 all the freqs under the threshold
    x_freqdom=[ x_freqdom[i] if np.absolute(x_freqdom[i])>=h else 0 for i in range(len(x_freqdom)) ]
    
    f = fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    #sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)

    for i in indexes:
        if x_freqdom[i] != 0: 
            ampli = np.absolute(x_freqdom[i]) / n   # amplitude
            phase = np.angle(x_freqdom[i])          # phase
            restored_sig += (ampli * np.cos(2 * np.pi * f[i] * t + phase))

    # restored_sig_for_trend = np.zeros(t.size)
    # for i in indexes[:10]:
    #     if x_freqdom[i] != 0:
    #         ampli = np.absolute(x_freqdom[i]) / n   # amplitude
    #         phase = np.angle(x_freqdom[i])          # phase
    #         restored_sig_for_trend += (ampli * np.cos(2 * np.pi * f[i] * t + phase))
    # trend = restored_sig_for_trend + p[0] * t
    reconstructed = restored_sig + p[0] * t
    #return reconstructed, x_freqdom, f, p[0], indexes, n, trend
    return reconstructed, x_freqdom, f, p[0], indexes, n

    #return restored_sig + p[0] * t, x_freqdom, f, p[0], indexes, n

    

def fourierPrediction(db, x_freqdom, f, p0, indexes, n_train, column_names):
    x = db[column_names]
    n = x.size
    tod_sampl = np.array([DateUtils.calc_monthly_order_from_day(item) for item in db['Date']])
    nTotal = tod_sampl[-1] + 1
    t = np.arange(0, nTotal)
    restored_sig = np.zeros(t.size)
    
    for i in indexes:
        if x_freqdom[i] != 0:
            ampli = np.absolute(x_freqdom[i]) / n_train   # amplitude
            phase = np.angle(x_freqdom[i])          # phase
            restored_sig += (ampli * np.cos(2 * np.pi * f[i] * t + phase))
    restored_sig_norm = restored_sig # - np.min(restored_sig)

    
    restored_sig_for_trend = np.zeros(t.size)
    for i in indexes[:10]:
        if x_freqdom[i] != 0:
            ampli = np.absolute(x_freqdom[i]) / nTrain   # amplitude
            phase = np.angle(x_freqdom[i])          # phase
            restored_sig_for_trend += (ampli * np.cos(2 * np.pi * f[i] * t + phase))

    trend = restored_sig_for_trend + p0 * t
    reconstructed = restored_sig + p0 * t
    return reconstructed[-n:], restored_sig_norm[-n:], trend[-n:]
