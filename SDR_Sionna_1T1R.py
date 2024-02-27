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
ave
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
        # Class variables from inputs
        self.SDR_TX_IP = SDR_TX_IP
        self.SDR_TX_FREQ = int(SDR_TX_FREQ)
        self.SDR_TX_BANDWIDTH = int(RF_BANDWIDTH)
        self.SampleRate = SampleRate
        self.corr_threshold = 5
        self.min_attempts = 10
        self.sdr_setup_done = False

    def setup_sdr(self):
        if not self.sdr_setup_done:
            # Setup the SDR
            self.sdr_pluto = adi.Pluto(self.SDR_TX_IP)
            self.sdr_pluto.sample_rate = int(self.SampleRate)
            self.sdr_pluto.tx_lo = self.SDR_TX_FREQ
            self.sdr_pluto.tx_rf_bandwidth = self.SDR_TX_BANDWIDTH
            self.sdr_pluto.tx_destroy_buffer()
            self.sdr_pluto.rx_lo = self.SDR_TX_FREQ
            self.sdr_pluto.gain_control_mode_chan0 = 'manual'
            self.sdr_pluto.rx_rf_bandwidth = self.SDR_TX_BANDWIDTH
            self.sdr_pluto.rx_destroy_buffer()
            self.sdr_setup_done = True
        
    def receive_samples(self):
        self.sdr_pluto.rx_destroy_buffer()
        rx_samples = self.sdr_pluto.rx()
        return np.array(rx_samples, dtype=np.complex64)
    
    def transmit_samples(self, tx_samples_out, SDR_TX_GAIN, SDR_RX_GAIN, num_samples,n_zeros):
        self.sdr_pluto.tx_cyclic_buffer = True # enable cyclic buffer for TX
        self.sdr_pluto.tx_hardwaregain_chan0 = int(SDR_TX_GAIN) # set the TX gain
        self.sdr_pluto.rx_hardwaregain_chan0 = int(SDR_RX_GAIN) # set the RX gain
        self.sdr_pluto.rx_buffer_size = (num_samples+n_zeros)*3 # set the RX buffer size to 3 times the number of samples
        self.sdr_pluto.tx_destroy_buffer() # empty TX buffer
        self.sdr_pluto.tx(tx_samples_out) # start transmitting the samples in a cyclic manner

    def close_tx(self):
        self.sdr_pluto.tx_destroy_buffer() # empty TX buffer
        self.sdr_pluto.rx_destroy_buffer()

    def update_txp(self):
        if self.sdr_pluto.tx_hardwaregain_chan0 <= -2:
            self.sdr_pluto.tx_hardwaregain_chan0 = self.sdr_pluto.tx_hardwaregain_chan0 + 2
        time.sleep(0.05)

    def call(self, SAMPLES, SDR_TX_GAIN=0, SDR_RX_GAIN=30, add_td_symbols = 0, threshold=6, debug=False, power_max_tx_scaling=1):
        now = time.time() # for measuing the duration of the process
        self.setup_sdr()
        n_zeros = 500 # number of leading zeros for noise floor measurement
        out_shape = list(SAMPLES.shape) # store the input tensor shape
        num_samples = SAMPLES.shape[-1] # number of samples in the input
        SAMPLES = tf.reshape(SAMPLES, [-1]) # flatten the input tensor
        
        # DC offset removal from signal, stdev calculation
        def _offset_removal(samples):
            stdev =  tf.math.reduce_std(samples) # standard deviation of the input samples
            tx_mean = tf.math.reduce_mean(samples) # mean of the input samples
            samples = tf.math.subtract(samples, tx_mean) # remove DC offset
            return samples, stdev # retun the samples and the stdev of the input samples

        # scale the samples for SDR input
        def _sdr_tx_scaling(tx_samples, power_max_tx_scaling):
            tx_samples_abs = tf.math.abs(tx_samples) # absolute values of the samples
            tx_samples_abs_max = tf.reduce_max(tx_samples_abs,0) # take the maximum value of the samples
            tx_samples_normalized = tx_samples / tf.cast(tx_samples_abs_max, tf.complex64)
            scaling_factor = tf.cast(power_max_tx_scaling, tf.complex64) * tf.cast(2**14, tf.complex64)
            tx_samples_scaled = tx_samples_normalized * scaling_factor
            return tx_samples_scaled, tx_samples_abs_max
        
        def _add_leading_zeros(tx_samples, n_zeros=n_zeros): # add leading zeros for noise floor measurement
            leading_zeroes = tf.zeros(n_zeros, dtype=tf.dtypes.complex64) # leading zeroes f
            samples_with_leading_zeros = tf.concat([leading_zeroes, tx_samples], axis=0) # add the leading zeros to the samples
            return samples_with_leading_zeros
        
        def _find_start_point(rx_samples_tf, tx_samples, threshold, n_zeros=n_zeros): # find the start symbol and calculate the offset

            len_tx = tx_samples.shape[0]  # Length of the transmitted samples
            len_rx = rx_samples_tf.shape[0]
                        
            # Perform cross-correlation in real and imaginary parts
            TTI_corr_real = tf.nn.conv1d(tf.reshape(tf.math.real(rx_samples_tf), [1, -1, 1]),
                                        filters=tf.reshape(tf.math.real(tx_samples), [-1, 1, 1]), stride=1, padding='SAME')
            TTI_corr_imag = tf.nn.conv1d(tf.reshape(tf.math.imag(rx_samples_tf), [1, -1, 1]),
                                        filters=tf.reshape(tf.math.imag(tx_samples), [-1, 1, 1]), stride=1, padding='SAME')

            # Combine real and imaginary parts and calculate the magnitude of the correlation
            correlation = tf.math.abs(tf.complex(TTI_corr_real, TTI_corr_imag))
            correlation = tf.reshape(correlation, [-1])

            zeros_tensor = tf.zeros([n_zeros+len_tx], dtype=tf.float32)
            sliced_tensor = correlation[n_zeros+len_tx:]
            correlation = tf.concat([zeros_tensor, sliced_tensor], axis=0)

            zeros_tensor = tf.zeros([len_tx], dtype=tf.float32)
            sliced_tensor = correlation[:-len_tx]
            correlation = tf.concat([sliced_tensor, zeros_tensor], axis=0)
            
            correlation_mean = tf.reduce_mean(correlation)
            
            # Function to find the maximum correlation offset, correlation threshold set to 0
            def find_max_offset():
                TTI_offset_max = tf.math.argmax(correlation) - len_tx // 2 + 1
                return TTI_offset_max

            # Function to find the first index where correlation exceeds the mean by a threshold
            def find_first_exceeding_threshold():
                exceed_mask = correlation > correlation_mean * threshold
                first_exceeding_index = tf.argmax(tf.cast(exceed_mask, tf.int32), axis=0)
                # Adjust the index based on the search window and offset
                return first_exceeding_index - len_tx // 2 + 1
            
            # Decide which offset to use based on the threshold
            if threshold == 0:
                TTI_offset = find_max_offset()
            else:
                TTI_offset = find_first_exceeding_threshold()
                if (TTI_offset<n_zeros) or (TTI_offset > (len_rx - n_zeros)):
                    TTI_offset = find_max_offset()

            # Access the correlation value at the found offset
            final_correlation = tf.gather(correlation, TTI_offset+len_tx//2-1)/correlation_mean

            return TTI_offset, correlation, final_correlation

        
        # adjust the stdev of the received samples to match the transmitted samples
        
        def _adjust_stdev(samples, rx_dev, tx_dev):
            std_multiplier = tf.cast(tx_dev, tf.complex64) / tf.cast(rx_dev, tf.complex64) * tf.constant(0.9, dtype=tf.complex64)
            samples = tf.math.multiply(samples, std_multiplier)
            return samples
        
        # prepare the tx signal, remove DC offset and scale for SDR input, add leading zeros
        tx_samples, tx_std = _offset_removal(SAMPLES)
        tx_samples, tx_max_sample = _sdr_tx_scaling(tx_samples, power_max_tx_scaling)
        tx_samples_out = _add_leading_zeros(tx_samples) 
        
        tf.py_function(func=self.transmit_samples, inp=[tx_samples_out,SDR_TX_GAIN, SDR_RX_GAIN, num_samples,n_zeros], Tout=[])

        rx_samples_tf = tf.ones([SAMPLES.shape[0] + add_td_symbols], dtype=tf.complex64)
        
        final_correlation = tf.constant(0, dtype=tf.float32)
        success = tf.constant(0, dtype=tf.int32)
        SINR = tf.constant(0, dtype=tf.float32)

        def loop_body(success, rx_samples_tf, SINR, final_correlation):
            rx_samples_tf_i = tf.py_function(func=self.receive_samples, inp=[], Tout=tf.complex64)
            
            rx_samples_tf_i, rx_std =  _offset_removal(rx_samples_tf_i) # remove any offset and calculate standard deviation of the received samples
            rx_samples_tf_i = _adjust_stdev(rx_samples_tf_i, rx_std, tx_std) # set the same stdev to output samples as in the original input samples

            TTI_offset, TTI_correlation, final_correlation  = _find_start_point(rx_samples_tf_i, tx_samples, threshold = threshold) 

            rx_noise =  rx_samples_tf_i[TTI_offset-n_zeros+20:TTI_offset-20] 
            noise_p = tf.math.reduce_variance(rx_noise) 
            
            rx_samples_tf = rx_samples_tf_i[TTI_offset:TTI_offset+num_samples+add_td_symbols] # cut the received samples to the length of the transmitted samples + additional symbols, starting from the start symbol
            rx_samples_tf.set_shape([SAMPLES.shape[0] + add_td_symbols])
            rx_p = tf.math.reduce_variance(rx_samples_tf) # calculate the received signal power

            SINR = 10*tf.experimental.numpy.log10(rx_p/noise_p)
 
            # plot debug graphs 
            if debug:
                self._plot_debug_info(SAMPLES, rx_samples_tf_i, TTI_offset, n_zeros, TTI_correlation, rx_samples_tf, tx_max_sample, rx_noise, save=True)

            condition1 = tf.greater_equal(final_correlation, self.corr_threshold)
            condition2 = tf.greater(SINR, 3)
            combined_condition = tf.logical_and(condition1, condition2)

            success = tf.cond(combined_condition,
                lambda: tf.constant(1),  # If the condition is True, set success to 1
                lambda: tf.constant(0)   # If the condition is False, you might want to do something else
            )

            tf.cond(
                tf.logical_not(combined_condition),
                self.update_txp,  # This function will execute if combined_condition is False
                lambda: tf.constant(0)  # This lambda does nothing but is necessary for tf.cond structure
            )


            return success, rx_samples_tf, SINR, final_correlation

        def condition(success, rx_samples_tf, SINR, final_correlation):
            return tf.equal(success, 0)
        
        success, rx_samples_tf, SINR, final_correlation = tf.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=[success, rx_samples_tf, SINR, final_correlation],)
        
        tf.py_function(func=self.close_tx, inp=[], Tout=[])
        
        # reshape the output tensor to match the input tensor shape        
        try :
            out_shape[-1] = out_shape[-1] + add_td_symbols
            out = tf.reshape(rx_samples_tf, out_shape)

        # if the process fails, exit
        except:
            print("Something failed!")
            tf.py_function(func=self.close_tx, inp=[], Tout=[])
            sys.exit(1)
 
        # calculate the duration of the process
        sdr_time=time.time()-now

        # return the output tensor, SINR, TX and RX gains, number of failures, correlation and the duration of the process
        return out, SINR, self.sdr_pluto.tx_hardwaregain_chan0, SDR_RX_GAIN, 1, final_correlation, sdr_time
    
        
    def _plot_debug_info(self, tx_samples, all_rx_samples, TTI_offset, n_zeros, TTI_correlation, rx_samples, tx_samples_max_sample, rx_noise, save=False):


        save_path_prefix = "pics/"
        picsize = (6, 3)
        
        # Plot Correlator for syncing the start of the second received TTI
        fig, ax = plt.subplots(figsize=picsize)
        ax.plot(np.abs(all_rx_samples), label='abs(RX sample)')
        ax.axvline(x=TTI_offset, c='r', lw=3, label='TTI start')
        ax.legend()
        ax.set_title('Correlator for syncing the start of a fully received OFDM block')
        if save:
            plt.savefig(f'{save_path_prefix}_plot2.png')
        plt.show()
        plt.grid()
        plt.close()

        # Plot Transmitted signal, one TTI
        fig, ax = plt.subplots(figsize=picsize)
        ax.plot(np.abs(tx_samples), label='abs(TX samples)')
        ax.set_ylim(0, tx_samples_max_sample)
        ax.legend()
        ax.set_title('Transmitted signal, one OFDM block')
        if save:
            plt.savefig(f'{save_path_prefix}_plot3.png')
        plt.show()
        plt.grid()
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
        plt.grid()
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

        #TTI_offset, TTI_offset_threshold, TTI_correlation

        fig, ax = plt.subplots(figsize=picsize)
        ax.set_title('Correlator')
        plt.plot(np.arange(-20,20), TTI_correlation[TTI_offset+len(tx_samples)//2-21:TTI_offset+len(tx_samples)//2+19 ], label='correlation')
        plt.grid()
        plt.xlabel("Samples around peak correlation")
        plt.ylabel("Complex conjugate correlation")
        plt.axvline(x=0, color = 'r', linewidth=3, label='max cor offset')
        plt.legend()
        if save:
            plt.savefig(f'{save_path_prefix}_plot7.png')
        plt.show()
