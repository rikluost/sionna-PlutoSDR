"""
Copyright Riku Luostari rikluost@gmail.com

Layer for implementing 1T1R SDR radio over-the-air connection, utilising PlutoSDR radio. 
Inputs time domain IQ to be transmitted and and outputs the received time domain IQ data, 
readily synchronized.

PlutoSDR:

https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html

Prerequisites (tested with Ubuntu 22.04) are:

- libiio, Analog Device’s library for interfacing hardware
- libad9361-iio, AD9361 the Analog Devices RF chip
- pyadi-iio, Python API for PlutoSDR

See e.g. https://pysdr.org/content/pluto_intro.html for how to install

Limitation, the batch size must be 1!!!!!!

Tested to work with the sionna ofdm modulator and demodulator, however any other should work

This supports only 1T1R (2T2R is in the works)

"""

import adi
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
import time
from scipy import signal
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
#from pyinstrument import Profiler # for code optimisation:
plt.rcParams['font.size'] = 8.0

class SDR(Layer):
    r"""
    Layer for implementing 1T1R SDR radio over-the-air connection for Sionna, utilising PlutoSDR radio.
    Inputs time domain IQ to be transmitted and and outputs the received time domain IQ data, readily synchronized.

    Args:
        SDR_TX_IP (str): IP address of the TX SDR device
        SDR_TX_FREQ (int): TX center frequency in Hz
        SDR_TX_BANDWIDTH (int): SDR filter cutoff
        SampleRate (int): Sample rate of the SDR

    Returns:
        out (tf.tensor): received time domain IQ data, readily synchronized
        SINR (float): SINR of the received signal
        SDR_TX_GAIN (int): TX gain used in the transmission
        SDR_RX_GAIN (int): RX gain used in the reception
        fails (int): how many times the process failed to reach pearson r > self.corr_threshold
        corr (float): pearson correlation between TX and RX signal
        sdr_time (float): how long the SDR process took in seconds
    """
    def __init__(self, SDR_TX_IP, SDR_TX_FREQ, SDR_TX_BANDWIDTH, SampleRate):        
        super().__init__()

        # class variables from inputs
        self.SDR_TX_IP = SDR_TX_IP # IP address of the TX SDR device
        self.SDR_TX_FREQ = int(SDR_TX_FREQ) # TX center frequency in Hz
        self.SDR_TX_BANDWIDTH = int(SDR_TX_BANDWIDTH) # SDR filter cutoff
        self.SampleRate = SampleRate # Sample rate of the SDR

        # setup the SDR
        self.sdr_pluto = adi.Pluto(self.SDR_TX_IP) # from which IP address the PlutoSDR can be found, default 192.168.2.1
        self.sdr_pluto.sample_rate = int(self.SampleRate) # set the samplerate

        # setup SDR TX
        self.sdr_pluto.tx_lo = self.SDR_TX_FREQ # set the frequency
        self.sdr_pluto.tx_rf_bandwidth = self.SDR_TX_BANDWIDTH # set the SDR tx filter cutoff
        self.sdr_pluto.tx_destroy_buffer() # empty the tx buffer

        # SETUP sdr rx
        self.sdr_pluto.rx_lo = self.SDR_TX_FREQ # set the frequency
        self.sdr_pluto.gain_control_mode_chan0 = 'manual' # don't use AGC
        self.sdr_pluto.rx_rf_bandwidth = self.SDR_TX_BANDWIDTH # rx filter cutoff 
        self.sdr_pluto.rx_destroy_buffer() # clear any data from rx buffer
        
        # further variables
        self.corr_threshold = 0.3 # correlation threshold for TTI detection. Below 0.2 correlation sync probably not right
        self.min_attempts=10 # how many retries before giving up if above thresholds are not met (while increasing TX power each time)
      
    def call(self, SAMPLES, SDR_TX_GAIN=0, SDR_RX_GAIN=30, add_td_samples = 0, debug=False):
        now = time.time() # for measuing the duration of the process

        out_shape = list(SAMPLES.shape) # store the input tensor shape
        num_samples = SAMPLES.shape[-1] # number of samples in the input

        # remove offsets
        flat_samples = tf.reshape(SAMPLES, [-1]) # flatten the input samples
        tx_std =  tf.math.reduce_std(flat_samples) # standard deviation of the input samples
        tx_mean = np.complex64(tf.math.reduce_mean(flat_samples)) # mean of the input samples
        tx_samples = tf.math.subtract(flat_samples, tx_mean) # remove DC offset

        # scale for SDR input
        tx_samples_abs = tf.math.abs(tx_samples) # absolute values of the samples
        tx_samples_abs_max = tf.reduce_max(tx_samples_abs,0) # take the maximum value of the samples
        tx_samples_max_sample = np.float32(tx_samples_abs_max) # convert to float32
        tx_samples = tf.math.divide(tx_samples , tx_samples_max_sample) # scale the tx_samples to max 1
        tx_samples = tf.math.multiply(tx_samples, 2**14) # = 2**14 # scale the samples to 16-bit

        # create the final IQ data for transmission
        leading_zeroes = tf.zeros(500, dtype=tf.dtypes.complex64) # leading 500 zeroes for noise floor measurement
        samples_with_leading_zeros = tf.concat([leading_zeroes, tx_samples], axis=0) # add the quiet for noise mesurements
        
        # internal counters
        fails = 0 # how many times the process failed to reach pearson r > self.corr_threshold
        success = 0 #  how many times the process reached pearson r > self.corr_threshold
        
        # send some parameters to SDR
        self.sdr_pluto.tx_cyclic_buffer = True # enable cyclic buffer for TX
        self.sdr_pluto.tx_hardwaregain_chan0 = int(SDR_TX_GAIN)     
        self.sdr_pluto.rx_hardwaregain_chan0 = int(SDR_RX_GAIN)
        self.sdr_pluto.rx_buffer_size = (num_samples+500)*3 # set the RX buffer size to 3 times the number of samples
        
        self.sdr_pluto.tx_destroy_buffer() # empty TX buffer
        self.sdr_pluto.tx(samples_with_leading_zeros) # start transmitting the samples in cyclic manner

        while success == 0:        
            
            # RX samples
            self.sdr_pluto.rx_destroy_buffer() # clear the RX buffer
            rx_samples = self.sdr_pluto.rx() # receive samples from the SDR

            # convert received IQ samples to tf tensor
            rx_samples_tf = tf.convert_to_tensor(rx_samples, dtype=tf.complex64)

            # remove any offset
            rx_mean = np.complex64(tf.math.reduce_mean(rx_samples_tf)) 
            rx_samples_tf = tf.math.subtract(rx_samples_tf, rx_mean)

            # set the same stdev as in the input samples
            rx_std = tf.math.reduce_std(rx_samples_tf)
            std_multiplier = np.float16(tx_std/ rx_std)*0.9 # for calculating new multiplier for same stdev in TX and RX
            rx_samples_tf = tf.math.multiply(rx_samples_tf, std_multiplier) # set the stdev

            # calculate the correlation between TX and RX signal and find the start symbol of the first full TTI with 500 samples of noise measurements in front
            TTI_corr = signal.correlate(rx_samples_tf, flat_samples,mode='full',method='fft')
            TTI_offset = tf.math.argmax(tf.math.abs(TTI_corr[0:int(len(rx_samples_tf)/2)]))-len(flat_samples)+1 
            if TTI_offset < 500+num_samples:
                TTI_offset = TTI_offset + 500 + num_samples

            # RX TTI symbols + the additional symbols
            rx_TTI = rx_samples_tf[TTI_offset:TTI_offset+num_samples+add_td_samples] 
            
            # RX noise for SINR calculation
            rx_noise =  rx_samples_tf[TTI_offset-450:TTI_offset-50]

            # calculate the pearson correlation between complex samples_orig and rx_TTI as acceptance metric
            corr = pearsonr(tf.math.abs(flat_samples), tf.math.abs(rx_samples_tf)[TTI_offset:TTI_offset+num_samples])[0]
            
            # calculate TX power, RX power & noise power
            tx_TTI_p = tf.math.reduce_variance(flat_samples) # TX power
            noise_p = tf.math.reduce_variance(rx_noise) # noise power
            rx_TTI_p = tf.math.reduce_variance(rx_TTI) # RX signal power
            SINR = 10*tf.experimental.numpy.log10(rx_TTI_p/noise_p) # calculate SINR from received powers
            
            if debug:
                titletext = f'SINR ={SINR:1.1f}, attempt={fails+1}, TTI start index = {TTI_offset}, correlation = {corr:1.2f}, TX_p/RX_p = {tx_TTI_p/rx_TTI_p:1.2f}'
                fig, axs = plt.subplots(3, 2)
                fig.set_size_inches(16, 7)
                fig.suptitle(titletext)
                axs[0,0].plot(10*np.log10(abs(rx_samples)/max(abs(rx_samples))), label='RX_dB')
                axs[0,0].legend()
                axs[0,0].set_title('TTI received 3 times, starting at random time')
                axs[0,1].plot((abs(rx_samples_tf)), label='abs(RXsample)')
                axs[0,1].axvline(x=TTI_offset, c='r', lw=3, label='TTI start')
                axs[0,1].plot(abs(abs(TTI_corr)/np.max(abs(TTI_corr))), label='Pearson R')
                axs[0,1].legend()
                axs[0,1].set_title('Correlator for syncing the start of the second received TTI')
                
                axs[1,0].plot(np.abs(flat_samples), label='abs(TX samples)')
                axs[1,0].set_ylim(0,tx_samples_max_sample)
                axs[1,0].legend()
                axs[1,0].set_title('Transmitted signal, one TTI')
                axs[1,1].plot((abs(rx_TTI)), label='abs(RX samples)')
                axs[1,1].set_ylim(0,tx_samples_max_sample)
                axs[1,1].legend()
                axs[1,1].set_title('Received signal, one TTI, syncronized')
                
                axs[2,0].psd(flat_samples, label='TX Signal')
                axs[2,0].legend()
                axs[2,0].set_title('Transmitted signal PSD')
                axs[2,1].psd(rx_TTI, label='RX signal')
                axs[2,1].psd(rx_noise, label='noise')
                axs[2,1].legend()
                axs[2,1].set_title('Received noise PSD and signal PSD')
                plt.tight_layout()
                plt.show()
                                        
            if fails > self.min_attempts: 
                print(f"Too many sync failures_1, {fails, self.sdr_pluto.rx_hardwaregain_chan0, self.sdr_pluto.tx_hardwaregain_chan0}")
                sys.exit(1)

            # check if the correlation is reasonable to assume sync is right, if not increase power and/or rx sensitivity
            if (corr >= self.corr_threshold):
                success = 1

            else:    
                fails+=1

                if self.sdr_pluto.tx_hardwaregain_chan0 <= -5:
                    self.sdr_pluto.tx_hardwaregain_chan0  = self.sdr_pluto.tx_hardwaregain_chan0 + 5
                    SDR_TX_GAIN = self.sdr_pluto.tx_hardwaregain_chan0                     
                elif self.sdr_pluto.rx_hardwaregain_chan0 <= 40:
                    self.sdr_pluto.tx_hardwaregain_chan0 = 0
                    self.sdr_pluto.rx_hardwaregain_chan0  = self.sdr_pluto.rx_hardwaregain_chan0 + 5
                    SDR_RX_GAIN = self.sdr_pluto.rx_hardwaregain_chan0
                else :
                    self.sdr_pluto.tx_hardwaregain_chan0 = 0
                    self.sdr_pluto.rx_hardwaregain_chan0  = 40
                    SDR_TX_GAIN = self.sdr_pluto.tx_hardwaregain_chan0
                    SDR_RX_GAIN = self.sdr_pluto.rx_hardwaregain_chan0
                
        self.sdr_pluto.tx_destroy_buffer() # shut the transmitter down
                
        try :
            out_shape[-1] = out_shape[-1]+add_td_samples
            out = tf.reshape(rx_TTI, out_shape)

        except:
            print("Something failed!")
            sys.exit(1)
 
        sdr_time=time.time()-now

        return out, SINR, SDR_TX_GAIN, SDR_RX_GAIN, fails + 1, corr, sdr_time

