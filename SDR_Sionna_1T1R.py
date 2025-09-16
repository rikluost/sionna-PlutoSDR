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

The batch size must be 1

Tested to work with the sionna ofdm modulator and demodulator, however any other should work

This implementation supports only 1T1R.

This version comes with automatic TX power control to keep the SINR in between minSINR maxSINR (e.g. 5, and 35, correspondingly)

"""

from typing import Optional, Tuple, Union

try:
    import adi
except Exception as e:
    if "OpenSSL error" in str(e):
        try:
            from adi.ad936x import Pluto as adi_Pluto
            class adi:
                Pluto = adi_Pluto
        except Exception as e2:
            raise e
    else:
        raise e
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import time
from matplotlib import pyplot as plt
from SDR_config import SDRConfig, default_config

ComplexTensor = tf.Tensor

class SDR(Layer):
    """Optimized SDR Layer for PlutoSDR integration with Sionna.
    
    This layer provides over-the-air testing capabilities using PlutoSDR
    with automatic power control and synchronization.
    """
    
    def _get_param_or_default(self, param_value, config_value):
        """Helper method to get parameter value or default from config."""
        return param_value if param_value is not None else config_value

    def __init__(self, 
                 SDR_TX_IP: Optional[str] = None, 
                 SDR_TX_FREQ: Optional[float] = None, 
                 RF_BANDWIDTH: Optional[float] = None, 
                 SampleRate: Optional[float] = None, 
                 config: Optional[SDRConfig] = None) -> None:
        super().__init__()
        
        # Use provided config or default config
        if config is None:
            config = default_config
        self.config = config
        
        # Set matplotlib font size from config
        plt.rcParams['font.size'] = self.config.font_size
        
        # Class variables from inputs (override config if provided)
        self.SDR_TX_IP = self._get_param_or_default(SDR_TX_IP, self.config.sdr_tx_ip)
        self.SDR_TX_FREQ = int(self._get_param_or_default(SDR_TX_FREQ, self.config.sdr_tx_freq))
        
        # Calculate RF bandwidth from sample rate and config multiplier
        if RF_BANDWIDTH is not None:
            self.SDR_TX_BANDWIDTH = int(RF_BANDWIDTH)
        elif SampleRate is not None:
            self.SDR_TX_BANDWIDTH = self.config.get_rf_bandwidth(SampleRate)
        else:
            # This will be set later when SampleRate is known
            self.SDR_TX_BANDWIDTH = None
            
        self.SampleRate = SampleRate
        self.corr_threshold = self.config.corr_threshold
        self.sdr_setup_done = False

    def setup_sdr(self) -> None:
        """Initialize and configure the PlutoSDR hardware."""
        if not self.sdr_setup_done:
            # Calculate RF bandwidth if not set during initialization
            if self.SDR_TX_BANDWIDTH is None and self.SampleRate is not None:
                self.SDR_TX_BANDWIDTH = self.config.get_rf_bandwidth(self.SampleRate)
            
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
            
        
    def receive_samples(self) -> np.ndarray:
        """Receive samples from PlutoSDR with optimized buffer management."""
        self.sdr_pluto.rx_destroy_buffer()
        rx_samples = self.sdr_pluto.rx()
        return np.array(rx_samples, dtype=np.complex64)
    
    def transmit_samples(self, tx_samples_out: ComplexTensor, num_samples: int, n_zeros: int) -> None:
        """Transmit samples using PlutoSDR with optimized buffer configuration."""
        self.sdr_pluto.tx_cyclic_buffer = True # enable cyclic buffer for TX
        self.sdr_pluto.rx_buffer_size = (num_samples + n_zeros) * self.config.rx_buffer_multiplier # set the RX buffer size
        self.sdr_pluto.tx_destroy_buffer() # empty TX buffer
        self.sdr_pluto.tx(tx_samples_out) # start transmitting the samples in a cyclic manner

    def close_tx(self):
        self.sdr_pluto.tx_destroy_buffer() # empty TX buffer
        self.sdr_pluto.rx_destroy_buffer()

    def update_txp_down(self):
        if self.sdr_pluto.tx_hardwaregain_chan0 >= self.config.tx_gain_min:
            self.sdr_pluto.tx_hardwaregain_chan0 = self.sdr_pluto.tx_hardwaregain_chan0 - self.config.tx_gain_step
        
    def update_txp_up(self):
        if self.sdr_pluto.tx_hardwaregain_chan0 <= self.config.tx_gain_max:
            self.sdr_pluto.tx_hardwaregain_chan0 = self.sdr_pluto.tx_hardwaregain_chan0 + self.config.tx_gain_step
        
    def call(self, 
             SAMPLES: ComplexTensor,
             SDR_RX_GAIN: Optional[float] = None,
             add_td_symbols: Optional[int] = None, 
             threshold: Optional[float] = None,
             debug: Optional[bool] = None,
             power_max_tx_scaling: Optional[float] = None,
             minSINR: Optional[float] = None,
             maxSINR: Optional[float] = None) -> Tuple[ComplexTensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, float]:
        # Use config values if parameters not provided (simplified parameter handling)
        SDR_RX_GAIN = self._get_param_or_default(SDR_RX_GAIN, self.config.sdr_rx_gain)
        add_td_symbols = self._get_param_or_default(add_td_symbols, self.config.add_td_symbols)
        threshold = self._get_param_or_default(threshold, self.config.threshold)
        debug = self._get_param_or_default(debug, self.config.debug)
        power_max_tx_scaling = self._get_param_or_default(power_max_tx_scaling, self.config.power_max_tx_scaling)
        minSINR = self._get_param_or_default(minSINR, self.config.min_sinr)
        maxSINR = self._get_param_or_default(maxSINR, self.config.max_sinr)
        
        now = time.time() # for measuring the duration of the process
        self.setup_sdr()
        n_zeros = self.config.n_zeros # number of leading zeros for noise floor measurement
        out_shape = list(SAMPLES.shape) # store the input tensor shape
        num_samples = SAMPLES.shape[-1] # number of samples in the input
        SAMPLES = tf.reshape(SAMPLES, [-1]) # flatten the input tensor
        self.sdr_pluto.rx_hardwaregain_chan0 = SDR_RX_GAIN

        # Optimized DC offset removal and statistics calculation
        @tf.function
        def _offset_removal(samples: ComplexTensor) -> Tuple[ComplexTensor, tf.Tensor]:
            """Remove DC offset and calculate standard deviation efficiently."""
            tx_mean = tf.reduce_mean(samples)
            samples_centered = samples - tx_mean
            stdev = tf.math.reduce_std(samples_centered)
            return samples_centered, stdev

        # Optimized sample scaling for SDR input
        @tf.function
        def _sdr_tx_scaling(tx_samples: ComplexTensor, power_max_tx_scaling: float) -> Tuple[ComplexTensor, tf.Tensor]:
            """Scale samples for SDR input with optimized computation."""
            tx_samples_abs_max = tf.reduce_max(tf.abs(tx_samples))
            # Avoid division by zero
            tx_samples_abs_max = tf.maximum(tx_samples_abs_max, self.config.epsilon)
            
            scaling_factor = tf.cast(power_max_tx_scaling * (2**self.config.adc_scaling_bits), tf.complex64)
            tx_samples_scaled = (tx_samples / tf.cast(tx_samples_abs_max, tf.complex64)) * scaling_factor
            return tx_samples_scaled, tx_samples_abs_max
        
        # Optimized leading zeros addition
        @tf.function
        def _add_leading_zeros(tx_samples: ComplexTensor, n_zeros: int) -> ComplexTensor:
            """Add leading zeros for noise floor measurement efficiently."""
            leading_zeros = tf.zeros(n_zeros, dtype=tf.complex64)
            return tf.concat([leading_zeros, tx_samples], axis=0)
        
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

        
        # Optimized standard deviation adjustment
        @tf.function
        def _adjust_stdev(samples: ComplexTensor, rx_dev: tf.Tensor, tx_dev: tf.Tensor) -> ComplexTensor:
            """Adjust received samples standard deviation to match transmitted samples."""
            # Avoid division by zero
            rx_dev_safe = tf.maximum(rx_dev, self.config.epsilon)
            std_multiplier = tf.cast(tx_dev / rx_dev_safe * self.config.stdev_adjustment_factor, tf.complex64)
            return samples * std_multiplier
        
        # Prepare TX signal with optimized pipeline
        try:
            tx_samples, tx_std = _offset_removal(SAMPLES)
            tx_samples, tx_max_sample = _sdr_tx_scaling(tx_samples, power_max_tx_scaling)
            tx_samples_out = _add_leading_zeros(tx_samples, n_zeros)
            
            tf.py_function(func=self.transmit_samples, inp=[tx_samples_out, num_samples, n_zeros], Tout=[])

            # Pre-allocate output tensor with correct shape
            rx_samples_tf = tf.zeros([SAMPLES.shape[0] + add_td_symbols], dtype=tf.complex64)
        except Exception as e:
            print(f"Error in TX signal preparation: {e}")
            raise
        
        final_correlation = tf.constant(0, dtype=tf.float32)
        success = tf.constant(0, dtype=tf.int32)
        SINR = tf.constant(0, dtype=tf.float32)

        def loop_body(success, rx_samples_tf, SINR, final_correlation):
            rx_samples_tf_i = tf.py_function(func=self.receive_samples, inp=[], Tout=tf.complex64)
            
            rx_samples_tf_i, rx_std =  _offset_removal(rx_samples_tf_i) # remove any offset and calculate standard deviation of the received samples
            rx_samples_tf_i = _adjust_stdev(rx_samples_tf_i, rx_std, tx_std) # set the same stdev to output samples as in the original input samples

            TTI_offset, TTI_correlation, final_correlation  = _find_start_point(rx_samples_tf_i, tx_samples, threshold = threshold) 

            rx_noise = rx_samples_tf_i[TTI_offset-n_zeros+self.config.noise_guard_samples:TTI_offset-self.config.noise_guard_samples] 
            noise_p = tf.math.reduce_variance(rx_noise) 
            
            rx_samples_tf = rx_samples_tf_i[TTI_offset:TTI_offset+num_samples+add_td_symbols] # cut the received samples to the length of the transmitted samples + additional symbols, starting from the start symbol
            rx_samples_tf.set_shape([SAMPLES.shape[0] + add_td_symbols])
            rx_p = tf.math.reduce_variance(rx_samples_tf) # calculate the received signal power

            SINR = 10*tf.experimental.numpy.log10(rx_p/noise_p)
 
            # plot debug graphs 
            if debug:
                self._plot_debug_info(SAMPLES, rx_samples_tf_i, TTI_offset, n_zeros, TTI_correlation, rx_samples_tf, tx_max_sample, rx_noise, save=self.config.save_plots)

            condition1 = tf.greater(final_correlation, self.corr_threshold)
            condition2 = tf.greater(SINR, minSINR)
            combined_condition2 = tf.logical_and(condition1, condition2)
            
            condition3 = tf.less(SINR, maxSINR)
            combined_condition = tf.logical_and(tf.logical_and(condition1, condition2), condition3)

            tf.cond(
                tf.logical_not(combined_condition2),
                self.update_txp_up, 
                lambda: tf.constant(0)  
            )

            tf.cond(
                tf.logical_not(condition3),
                self.update_txp_down,  
                lambda: tf.constant(0)  
            )

            success = tf.cond(combined_condition,
                lambda: tf.constant(1),  
                lambda: tf.constant(0) 
            )
            
            return success, rx_samples_tf, SINR, final_correlation

        def condition(success, rx_samples_tf, SINR, final_correlation):
            return tf.equal(success, 0)
        
        success, rx_samples_tf, SINR, final_correlation = tf.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=[success, rx_samples_tf, SINR, final_correlation],)
      
        tf.py_function(func=self.close_tx, inp=[], Tout=[])
        
        # Reshape output tensor with improved error handling
        try:
            out_shape[-1] = out_shape[-1] + add_td_symbols
            out = tf.reshape(rx_samples_tf, out_shape)
        except Exception as e:
            print(f"Failed to reshape output tensor: {e}")
            tf.py_function(func=self.close_tx, inp=[], Tout=[])
            raise RuntimeError(f"SDR processing failed during output reshaping: {e}") from e
 
        # calculate the duration of the process
        sdr_time=time.time()-now

        return out, SINR, self.sdr_pluto.tx_hardwaregain_chan0, self.sdr_pluto.rx_hardwaregain_chan0, 1, final_correlation, sdr_time
    
        
    def _plot_debug_info(self, 
                         tx_samples: np.ndarray,
                         all_rx_samples: np.ndarray, 
                         TTI_offset: int,
                         n_zeros: int,
                         TTI_correlation: np.ndarray,
                         rx_samples: np.ndarray,
                         tx_samples_max_sample: float,
                         rx_noise: np.ndarray,
                         save: bool = False) -> None:
        """Generate debug plots with optimized rendering."""

        save_path_prefix = self.config.pics_path
        picsize = self.config.plot_size
        
        # Plot 1: Signal Overview with Synchronization Point
        fig, ax = plt.subplots(figsize=picsize)
        ax.plot(np.abs(all_rx_samples), label='|RX Signal|')
        ax.axvline(x=TTI_offset, color='red', linewidth=2, label='Sync Point')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Amplitude')
        ax.set_title('Received Signal Overview with Synchronization')
        ax.legend()
        ax.grid(True)
        if save:
            plt.savefig(f'{save_path_prefix}_plot1.png')
        plt.show()
        plt.close()



        # Plot 2: Transmitted Signal Analysis
        fig, ax = plt.subplots(figsize=picsize)
        
        # Convert tensors to NumPy
        tx_samples_np = tx_samples.numpy() if hasattr(tx_samples, 'numpy') else np.array(tx_samples)
        all_rx_samples_np = all_rx_samples.numpy() if hasattr(all_rx_samples, 'numpy') else np.array(all_rx_samples)
        
        # Extract the same synchronized portion that Plot 5 uses
        tx_len = len(tx_samples_np)
        if TTI_offset + tx_len <= len(all_rx_samples_np):
            tx_samples_sync = tx_samples_np  # Full TX signal (matches the sync portion)
        else:
            available_len = len(all_rx_samples_np) - TTI_offset
            tx_samples_sync = tx_samples_np[:available_len]  # Truncate to match available RX
        
        ax.plot(np.abs(tx_samples_sync), label='|TX Signal|')
        ax.set_ylim(0, tx_samples_max_sample)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Amplitude')
        ax.set_title('Transmitted Signal')
        ax.legend()
        ax.grid(True)
        if save:
            plt.savefig(f'{save_path_prefix}_plot2.png')
        plt.show()
        plt.close()


        # Plot 3: Received Signal Analysis (Synchronized)
        fig, ax = plt.subplots(figsize=picsize)
        
        ax.plot(np.abs(rx_samples), label='|RX Signal|')
        ax.set_ylim(0, tx_samples_max_sample)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Amplitude')
        ax.set_title('Received Signal (Synchronized)')
        ax.legend()
        ax.grid(True)
        if save:
            plt.savefig(f'{save_path_prefix}_plot3.png')
        plt.show()
        plt.close()

        # Plot 4: Transmitted Signal Power Spectral Density
        fig, ax = plt.subplots(figsize=picsize)
        
        # Convert to numpy if needed for PSD calculation
        tx_samples_psd = tx_samples.numpy() if hasattr(tx_samples, 'numpy') else np.array(tx_samples)
        
        ax.psd(tx_samples_psd, label='TX Signal PSD')
        ax.set_xlabel('Frequency (Normalized)')
        ax.set_ylabel('Power Spectral Density (dB/Hz)')
        ax.set_title('Transmitted Signal PSD')
        ax.legend()
        ax.grid(True)
        
        if save:
            plt.savefig(f'{save_path_prefix}_plot4.png')
        plt.show()
        plt.close()

        # Plot 5: Received Signal and Noise Power Spectral Density
        fig, ax = plt.subplots(figsize=picsize)
        
        # Convert to numpy if needed for PSD calculation
        rx_samples_psd = rx_samples.numpy() if hasattr(rx_samples, 'numpy') else np.array(rx_samples)
        rx_noise_psd = rx_noise.numpy() if hasattr(rx_noise, 'numpy') else np.array(rx_noise)
        
        ax.psd(rx_samples_psd, label='RX Signal')
        ax.psd(rx_noise_psd, label='Noise')
        
        ax.set_xlabel('Frequency (Normalized)')
        ax.set_ylabel('Power Spectral Density (dB/Hz)')
        ax.set_title('Received Signal and Noise PSD')
        ax.legend()
        ax.grid(True)
        
        if save:
            plt.savefig(f'{save_path_prefix}_plot5.png')
        plt.show()
        plt.close()

        try:
            fig, ax = plt.subplots(figsize=picsize)
            
            # Convert tensors to NumPy
            tx_samples_np = tx_samples.numpy() if hasattr(tx_samples, 'numpy') else np.array(tx_samples)
            all_rx_samples_np = all_rx_samples.numpy() if hasattr(all_rx_samples, 'numpy') else np.array(all_rx_samples)
            rx_noise_np = rx_noise.numpy() if hasattr(rx_noise, 'numpy') else np.array(rx_noise)
            
            # Extract synchronized RX samples that correspond to TX signal
            tx_len = len(tx_samples_np)
            if TTI_offset + tx_len <= len(all_rx_samples_np):
                rx_samples_sync = all_rx_samples_np[TTI_offset:TTI_offset + tx_len]
            else:
                available_len = len(all_rx_samples_np) - TTI_offset
                rx_samples_sync = all_rx_samples_np[TTI_offset:TTI_offset + available_len]
            
            # Signal power over time (windowed) using synchronized samples
            window_size = max(50, len(rx_samples_sync)//20)
            rx_power = []
            noise_power = []
            time_windows = []
            
            for i in range(0, len(rx_samples_sync)-window_size, window_size//2):
                window = rx_samples_sync[i:i+window_size]
                rx_power.append(np.mean(np.abs(window)**2))
                time_windows.append(i + window_size//2)
                
            for i in range(0, len(rx_noise_np)-window_size//4, window_size//8):
                if i+window_size//4 < len(rx_noise_np):
                    noise_window = rx_noise_np[i:i+window_size//4]
                    noise_power.append(np.mean(np.abs(noise_window)**2))
            
            # Pad noise_power to match rx_power length
            while len(noise_power) < len(rx_power):
                noise_power.append(noise_power[-1] if noise_power else 1e-12)
            noise_power = noise_power[:len(rx_power)]
            
            # SINR estimation
            sinr_values = 10*np.log10(np.array(rx_power) / (np.array(noise_power) + 1e-12))
            
            ax.plot(time_windows, sinr_values, label='SINR', marker='o', markersize=3)
            ax.axhline(y=np.mean(sinr_values), color='red', linestyle='--', 
                      label=f'Mean SINR: {np.mean(sinr_values):.1f} dB')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('SINR (dB)')
            ax.set_title('SINR over time')
            ax.legend()
            ax.grid(True)
            
            if save:
                plt.savefig(f'{save_path_prefix}_plot6.png')
            plt.show()
            plt.close()
        except Exception as e:
            print(f"Error in SINR plot: {e}")
            plt.close()



        # Plot 7: Correlation Analysis
        fig, ax = plt.subplots(figsize=picsize)
        
        correlation_range = np.arange(-20, 20)
        start_idx = TTI_offset + len(tx_samples)//2 - 21
        end_idx = TTI_offset + len(tx_samples)//2 + 19
        correlation_data = TTI_correlation[start_idx:end_idx]
        
        ax.plot(correlation_range, correlation_data, label='Correlation')
        ax.axvline(x=0, color='red', linewidth=2, label='Peak')
        
        ax.set_xlabel('Sample Offset from Peak')
        ax.set_ylabel('Correlation Magnitude')
        ax.set_title('Correlator')
        ax.legend()
        ax.grid(True)
        
        if save:
            plt.savefig(f'{save_path_prefix}_plot7.png')
        plt.show()
        plt.close()
        
        # Plot 8: AM-AM Linearity Analysis
        try:
            fig, ax = plt.subplots(figsize=picsize)
            
            # Convert tensors to NumPy
            tx_samples_np = tx_samples.numpy() if hasattr(tx_samples, 'numpy') else np.array(tx_samples)
            all_rx_samples_np = all_rx_samples.numpy() if hasattr(all_rx_samples, 'numpy') else np.array(all_rx_samples)
            
            # Extract synchronized RX samples that correspond to the same time instances as TX
            tx_len = len(tx_samples_np)
            
            # Extract the synchronized portion of RX samples
            if TTI_offset + tx_len <= len(all_rx_samples_np):
                rx_samples_sync = all_rx_samples_np[TTI_offset:TTI_offset + tx_len]
            else:
                # Handle edge case where not enough samples
                available_len = len(all_rx_samples_np) - TTI_offset
                rx_samples_sync = all_rx_samples_np[TTI_offset:TTI_offset + available_len]
                tx_samples_np = tx_samples_np[:available_len]
            
            # Calculate magnitudes for time-aligned samples
            tx_magnitude = np.abs(tx_samples_np)
            rx_magnitude = np.abs(rx_samples_sync)
            
            # Simple scatter plot
            ax.scatter(tx_magnitude, rx_magnitude, alpha=0.6, s=10, label='TX vs RX')
            
            # Add ideal linear response line
            max_mag = max(np.max(tx_magnitude), np.max(rx_magnitude))
            ideal_line = np.linspace(0, max_mag, 100)
            ax.plot(ideal_line, ideal_line, 'r--', linewidth=2, label='Ideal Linear')
            
            # Add best fit line
            if len(tx_magnitude) > 1:
                coeffs = np.polyfit(tx_magnitude, rx_magnitude, 1)
                fit_line = np.poly1d(coeffs)
                ax.plot(ideal_line, fit_line(ideal_line), 'g-', linewidth=2, 
                        label=f'Best Fit (slope={coeffs[0]:.3f})')

            
            ax.set_xlabel('TX Magnitude')
            ax.set_ylabel('RX Magnitude')
            ax.set_title('AM-AM Linearity')
            ax.legend(loc='lower right')
            ax.grid(True)
            
            ax.set_aspect('equal', adjustable='box')
            
            if save:
                plt.savefig(f'{save_path_prefix}_plot8.png')
            plt.show()
            plt.close()
        except Exception as e:
            print(f"Error in AM-AM Linearity plot: {e}")
            plt.close()
        
