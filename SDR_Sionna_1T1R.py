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

Limitation, the batch size must be 1

Tested to work with the sionna ofdm modulator and demodulator, however any other should work

Current implementation supports only 1T1R, but as the HW supports 2T2R, it might be supported later. 
Note that 2T2R with pluto requires additional RF pigtails and a bit of DIY.

"""

import adi
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import time
import sys
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 9.0

class SDR(Layer):

    def __init__(self, SDR_TX_IP, SDR_TX_FREQ, RF_BANDWIDTH, SampleRate):        
        super().__init__()

        # class variables from inputs
        self.SDR_TX_IP = SDR_TX_IP # IP address of the TX SDR device
        self.SDR_TX_FREQ = int(SDR_TX_FREQ) # TX center frequency in Hz
        self.SDR_TX_BANDWIDTH = int(RF_BANDWIDTH) # SDR filter cutoff
        self.SampleRate = SampleRate # Sample rate of the SDR

        # setup the SDR
        self.sdr_pluto = adi.Pluto(self.SDR_TX_IP) # from which IP address the PlutoSDR can be found, default 192.168.2.1
        self.sdr_pluto.sample_rate = int(self.SampleRate) # set the samplerate

        # setup SDR TX
        self.sdr_pluto.tx_lo = self.SDR_TX_FREQ # set the frequency
        self.sdr_pluto.tx_rf_bandwidth = self.SDR_TX_BANDWIDTH # set the SDR tx filter cutoff
        self.sdr_pluto.tx_destroy_buffer() # empty the tx buffer

        # SETUP SDR RX
        self.sdr_pluto.rx_lo = self.SDR_TX_FREQ # set the frequency
        self.sdr_pluto.gain_control_mode_chan0 = 'manual' # don't use AGC
        self.sdr_pluto.rx_rf_bandwidth = self.SDR_TX_BANDWIDTH # rx filter cutoff 
        self.sdr_pluto.rx_destroy_buffer() # clear any data from rx buffer
        
        # further variables
        self.corr_threshold = 0.2 # min correlation threshold for TTI detection. Below 0.2 correlation sync probably not right
        self.min_attempts=10 # how many retries before giving up if above thresholds are not met (while increasing TX power each time)

    def call(self, SAMPLES, SDR_TX_GAIN=0, SDR_RX_GAIN=30, add_td_symbols = 0, debug=False, power_max_tx_scaling=1):
        now = time.time() # for measuing the duration of the process
        n_zeros = 500 # number of leading zeros for noise floor measurement
        out_shape = list(SAMPLES.shape) # store the input tensor shape
        num_samples = SAMPLES.shape[-1] # number of samples in the input
        SAMPLES = tf.reshape(SAMPLES, [-1]) # flatten the input tensor
        
        # internal counters
        fails = 0 # how many times the process failed to reach pearson r > self.corr_threshold
        success = 0 #  how many times the process reached pearson r > self.corr_threshold

        # DC offset removal from signal, stdev calculation
        def _offset_removal(samples):
            stdev =  tf.math.reduce_std(samples) # standard deviation of the input samples
            tx_mean = np.complex64(tf.math.reduce_mean(samples)) # mean of the input samples
            samples = tf.math.subtract(samples, tx_mean) # remove DC offset
            return samples, stdev # retun the samples and the stdev of the input samples

        # scale the samples for SDR input
        def _sdr_tx_scaling(tx_samples, power_max_tx_scaling):
            tx_samples_abs = tf.math.abs(tx_samples) # absolute values of the samples
            tx_samples_abs_max = np.float32(tf.reduce_max(tx_samples_abs,0)) # take the maximum value of the samples
            tx_samples = tf.math.divide(tx_samples , tx_samples_abs_max) # scale the tx_samples to max 1
            tx_samples = tf.math.multiply(tx_samples * power_max_tx_scaling, 2**14) # = 2**14 # scale the samples to 16-bit    
            return tx_samples, tx_samples_abs_max
        
        # add leading zeros for noise floor measurement
        def _add_leading_zeros(tx_samples, n_zeros=n_zeros): # e.g. 500 leading zeros 
            leading_zeroes = tf.zeros(n_zeros, dtype=tf.dtypes.complex64) # leading zeroes f
            samples_with_leading_zeros = tf.concat([leading_zeroes, tx_samples], axis=0) # add the leading zeros to the samples
            return samples_with_leading_zeros
        
        # find the start symbol and calculate the offset
        def _find_start_point(rx_samples_tf, tx_samples,): 
            TTI_corr = tf.nn.conv1d(tf.reshape(tf.math.abs(rx_samples_tf), [1, -1, 1]), filters=tf.reshape(tf.math.abs(tx_samples), [-1, 1, 1]), stride=1, padding='SAME')
            TTI_corr = tf.reshape(TTI_corr, [-1])
            TTI_offset = tf.math.argmax(TTI_corr[0:len(rx_samples_tf)-len(tx_samples)])-len(tx_samples)//2+1
            if TTI_offset < n_zeros: # 
                TTI_offset = TTI_offset + n_zeros + len(tx_samples)
            return TTI_offset.numpy()
        
        # adjust the stdev of the received samples to match the transmitted samples
        def _adjust_stdev(samples, rx_dev, tx_dev):
            std_multiplier = np.float16(tx_dev/ rx_dev)*0.9 # not sure why 0.9 is needed, but it seems to work
            samples = tf.math.multiply(samples, std_multiplier)
            return samples
        
        # prepare the tx signal, remove DC offset and scale for SDR input, add leading zeros
        tx_samples, tx_std = _offset_removal(SAMPLES)
        tx_samples, tx_max_sample = _sdr_tx_scaling(tx_samples, power_max_tx_scaling)
        tx_samples_out = _add_leading_zeros(tx_samples) 

        # prepare SDR
        self.sdr_pluto.tx_cyclic_buffer = True # enable cyclic buffer for TX
        self.sdr_pluto.tx_hardwaregain_chan0 = int(SDR_TX_GAIN) # set the TX gain
        self.sdr_pluto.rx_hardwaregain_chan0 = int(SDR_RX_GAIN) # set the RX gain
        self.sdr_pluto.rx_buffer_size = (num_samples+n_zeros)*3 # set the RX buffer size to 3 times the number of samples
       
        # start the TX and send the samples
        self.sdr_pluto.tx_destroy_buffer() # empty TX buffer
        self.sdr_pluto.tx(tx_samples_out) # start transmitting the samples in a cyclic manner

        while success == 0:        
            
            # RX samples
            self.sdr_pluto.rx_destroy_buffer() # clear the RX buffer
            rx_samples = self.sdr_pluto.rx() # receive samples from the SDR, fill the buffer

            # convert received IQ samples to tf tensor
            rx_samples_tf = tf.convert_to_tensor(rx_samples, dtype=tf.complex64)

            # remove any offset and calculate standard deviation of the received samples
            rx_samples_tf, rx_std =  _offset_removal(rx_samples_tf)

            # set the same stdev to output samples as in the original input samples
            rx_samples_tf = _adjust_stdev(rx_samples_tf, rx_std, tx_std)

            # find the start symbol
            TTI_offset = _find_start_point(rx_samples_tf, tx_samples) 

            # calculate noise floor
            rx_noise =  rx_samples_tf[TTI_offset-n_zeros+20:TTI_offset-20] 
            noise_p = tf.math.reduce_variance(rx_noise) 

            # cut the received samples to the length of the transmitted samples + additional symbols, starting from the start symbol
            rx_samples_tf = rx_samples_tf[TTI_offset:TTI_offset+num_samples+add_td_symbols]

            # calculate the received signal power
            rx_p = tf.math.reduce_variance(rx_samples_tf)

            # calculate the correlation between the transmitted and received samples
            corr = pearsonr(tf.math.abs(tx_samples), tf.math.abs(rx_samples_tf[:-add_td_symbols]))[0]

            # calculate SINR
            SINR = 10*tf.experimental.numpy.log10(rx_p/noise_p)
            
            # plot debug graphs 
            if debug:
                self._plot_debug_info(SAMPLES, rx_samples, TTI_offset, rx_samples_tf, tx_max_sample, rx_noise)
                                        
            # if the process fails too many times, give up
            if fails > self.min_attempts: 
                print(f"Too many sync failures_1, {fails, self.sdr_pluto.rx_hardwaregain_chan0, self.sdr_pluto.tx_hardwaregain_chan0}")
                sys.exit(1)

            # check if the correlation is reasonable to assume sync is right, if not increase TX power and/or RX sensitivity
            if (corr >= self.corr_threshold):
                success = 1

            # if the correlation is not good enough, increase TX power and/or RX sensitivity
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
                
        self.sdr_pluto.tx_destroy_buffer() # shut the transmitter down and ensure the buffer is empty

        # reshape the output tensor to match the input tensor shape        
        try :
            out_shape[-1] = out_shape[-1]+add_td_symbols
            out = tf.reshape(rx_samples_tf, out_shape)

        # if the process fails, exit
        except:
            print("Something failed!")
            sys.exit(1)
 
        # calculate the duration of the process
        sdr_time=time.time()-now

        # return the output tensor, SINR, TX and RX gains, number of failures, correlation and the duration of the process
        return out, SINR, SDR_TX_GAIN, SDR_RX_GAIN, fails + 1, corr, sdr_time
    
        
    def _plot_debug_info(self, tx_samples, all_rx_samples, TTI_offset, rx_samples, tx_samples_max_sample, rx_noise, save=False):
        save_path_prefix = "pics/"
        picsize = (6, 3)
        
        # Plot TTI received 3 times, starting at random time
        fig, ax = plt.subplots(figsize=picsize)
        ax.plot(10 * np.log10(np.abs(all_rx_samples) / np.max(np.abs(all_rx_samples))), label='RX_dB')
        ax.legend()
        ax.set_title('Received samples')
        if save:
            plt.savefig(f'{save_path_prefix}_plot1.png')
        plt.show()
        plt.close()

        # Plot Correlator for syncing the start of the second received TTI
        fig, ax = plt.subplots(figsize=picsize)
        ax.plot(np.abs(all_rx_samples), label='abs(RX sample)')
        ax.axvline(x=TTI_offset, c='r', lw=3, label='TTI start')
        ax.legend()
        ax.set_title('Correlator for syncing the start of the second received TTI')
        if save:
            plt.savefig(f'{save_path_prefix}_plot2.png')
        plt.show()
        plt.close()

        # Plot Transmitted signal, one TTI
        fig, ax = plt.subplots(figsize=picsize)
        ax.plot(np.abs(tx_samples), label='abs(TX samples)')
        ax.set_ylim(0, tx_samples_max_sample)
        ax.legend()
        ax.set_title('Transmitted signal, one TTI')
        if save:
            plt.savefig(f'{save_path_prefix}_plot3.png')
        plt.show()
        plt.close()

        # Plot Received signal, one TTI, synchronized
        fig, ax = plt.subplots(figsize=picsize)
        ax.plot(np.abs(rx_samples), label='abs(RX samples)')
        ax.set_ylim(0, tx_samples_max_sample)
        ax.legend()
        ax.set_title('Received signal, synchronized')
        if save:
            plt.savefig(f'{save_path_prefix}_plot4.png')
        plt.show()
        plt.close()

        # Plot Transmitted signal PSD
        fig, ax = plt.subplots(figsize=picsize)
        ax.psd(tx_samples, label='TX Signal')
        ax.legend()
        ax.set_title('Transmitted signal PSD')
        if save:
            plt.savefig(f'{save_path_prefix}_plot5.png')
        plt.show()
        plt.close()

        # Plot Received noise PSD and signal PSD
        fig, ax = plt.subplots(figsize=picsize)
        ax.psd(rx_samples, label='RX signal')
        ax.psd(rx_noise, label='Noise')
        ax.legend()
        ax.set_title('Received noise PSD and signal PSD')
        if save:
            plt.savefig(f'{save_path_prefix}_plot6.png')
        plt.show()
        plt.close()
