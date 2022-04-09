# Using Fourier Transform with Autoencoders to detect anomalies in univariate time series.

## ABSTRACT

Autoencoders are widely proposed as a method to detect anomalies, the idea here is to be able to train an autoencoder with a univariate Time Serie and then submit new values to the model letting it to predict if among those values any anomaly can be found. The problem of training an autoencoder with a time series is that the autoencoder must learn about its trend and seasonality to make accurate predictions. There are several applications using autoencoders and Recurrent Neural Networks or RNN (1) but they require us to always provide a sequence as input. In contrast, what is needed is to train the autoencoder once, and later feed it with isolated new cases to determine if they are anomalies.

## THE CHALANGE OF NOT USING AN RNN

When using a Recurrent Neural Network (RNN) you must provide the network with sequences of data each of them in the position that they appear. For example, in the sentence ‘This is a cat’ you first give to the network the word ‘This’, then ‘is’ then, then ‘a’ and finally ‘cat’. When dealing with a time series you must group the data into consecutive segments that should be given to the network in the time order that they occurred. While this order of data could provide the ability to encode the trend of the time series during training, it may not provide valuable information to the network about its seasonal properties. Since in our case we do not want to use RNN (because we do not want to depend on sequences of data when asking if a new case is an anomaly or nor), we have the challenge of how to give to the network information about both seasonal and trend that are present in the time series.

## THE FAST FOURIER TRANSFORM

The utilization  of sines and cosines suggest concepts derived from the Fourier Series. All functions of a variable, whether continuous or not, can be expanded in a series of sinusoidal functions of multiples of the variable. The Fast Fourier Transform FFT (4) is an algorithm that you can apply to decompose a time series in a sequence of complex coefficients representing different frequencies and amplitudes (this is typically known as moving from the time domain to the frequency domain). Once you have the coefficients you can, for example, filter those frequencies representing noise and then compose them back to obtain the original signal without the noise.


![image](data/catfishtotaldata.gif)
