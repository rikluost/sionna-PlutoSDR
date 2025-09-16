"""
SDR Configuration File for PlutoSDR-Sionna Integration

This file contains centralized configuration parameters for the PlutoSDR interface.
All SDR-related settings are defined here to provide a single point of configuration.

Copyright Riku Luostari rikluost@gmail.com
"""

# Default PlutoSDR Hardware Configuration
DEFAULT_SDR_TX_IP = "ip:192.168.1.10"  # IP address of the PlutoSDR
DEFAULT_SDR_TX_FREQ = 435.1e6  # Transmit frequency in Hz (435.1 MHz)
DEFAULT_RF_BANDWIDTH_MULTIPLIER = 1.3  # RF bandwidth as multiplier of sample rate for the PlutoSDR filter
DEFAULT_SDR_RX_GAIN = 30  # RX gain in dB (0-73 dB range)

# Power Control Parameters
DEFAULT_MIN_SINR = 5  # Minimum SINR threshold in dB
DEFAULT_MAX_SINR = 30  # Maximum SINR threshold in dB
DEFAULT_TX_GAIN_MIN = -40  # Minimum TX gain in dB
DEFAULT_TX_GAIN_MAX = -2   # Maximum TX gain in dB
DEFAULT_TX_GAIN_STEP = 2   # TX gain adjustment step in dB

# Signal Processing Parameters
DEFAULT_CORR_THRESHOLD = 10  # Correlation threshold for synchronization
DEFAULT_N_ZEROS = 500  # Number of leading zeros for noise floor measurement
DEFAULT_POWER_MAX_TX_SCALING = 1  # Maximum TX power scaling factor
DEFAULT_ADD_TD_SYMBOLS = 16  # Additional time-domain symbols for delay spread
DEFAULT_THRESHOLD = 0  # Correlation detection threshold (0 = use max correlation)

# Buffer and Timing Parameters
DEFAULT_RX_BUFFER_MULTIPLIER = 3  # RX buffer size multiplier
DEFAULT_NOISE_GUARD_SAMPLES = 20  # Guard samples around noise measurement window

# Debug and Visualization
DEFAULT_DEBUG = False  # Enable debug plots
DEFAULT_PLOT_SIZE = (6, 3)  # Default plot size for debug graphs
DEFAULT_SAVE_PLOTS = False  # Save debug plots to files
DEFAULT_PICS_PATH = "pics/"  # Path for saving debug plots

# Font configuration for plots
DEFAULT_FONT_SIZE = 9.0

class SDRConfig:
    """
    SDR Configuration class that holds all configuration parameters.
    Can be customized for different use cases while maintaining defaults.
    """
    
    def __init__(self, 
                 sdr_tx_ip=DEFAULT_SDR_TX_IP,
                 sdr_tx_freq=DEFAULT_SDR_TX_FREQ,
                 rf_bandwidth_multiplier=DEFAULT_RF_BANDWIDTH_MULTIPLIER,
                 sdr_rx_gain=DEFAULT_SDR_RX_GAIN,
                 min_sinr=DEFAULT_MIN_SINR,
                 max_sinr=DEFAULT_MAX_SINR,
                 tx_gain_min=DEFAULT_TX_GAIN_MIN,
                 tx_gain_max=DEFAULT_TX_GAIN_MAX,
                 tx_gain_step=DEFAULT_TX_GAIN_STEP,
                 corr_threshold=DEFAULT_CORR_THRESHOLD,
                 n_zeros=DEFAULT_N_ZEROS,
                 power_max_tx_scaling=DEFAULT_POWER_MAX_TX_SCALING,
                 add_td_symbols=DEFAULT_ADD_TD_SYMBOLS,
                 threshold=DEFAULT_THRESHOLD,
                 rx_buffer_multiplier=DEFAULT_RX_BUFFER_MULTIPLIER,
                 noise_guard_samples=DEFAULT_NOISE_GUARD_SAMPLES,
                 debug=DEFAULT_DEBUG,
                 plot_size=DEFAULT_PLOT_SIZE,
                 save_plots=DEFAULT_SAVE_PLOTS,
                 pics_path=DEFAULT_PICS_PATH,
                 font_size=DEFAULT_FONT_SIZE,
                 # Signal processing constants
                 adc_scaling_bits=14,
                 stdev_adjustment_factor=0.9,
                 epsilon=1e-12):
        
        # Hardware configuration
        self.sdr_tx_ip = sdr_tx_ip
        self.sdr_tx_freq = sdr_tx_freq
        self.rf_bandwidth_multiplier = rf_bandwidth_multiplier
        self.sdr_rx_gain = sdr_rx_gain
        
        # Power control
        self.min_sinr = min_sinr
        self.max_sinr = max_sinr
        self.tx_gain_min = tx_gain_min
        self.tx_gain_max = tx_gain_max
        self.tx_gain_step = tx_gain_step
        
        # Signal processing
        self.corr_threshold = corr_threshold
        self.n_zeros = n_zeros
        self.power_max_tx_scaling = power_max_tx_scaling
        self.add_td_symbols = add_td_symbols
        self.threshold = threshold
        
        # Buffer and timing
        self.rx_buffer_multiplier = rx_buffer_multiplier
        self.noise_guard_samples = noise_guard_samples
        
        # Debug and visualization
        self.debug = debug
        self.plot_size = plot_size
        self.save_plots = save_plots
        self.pics_path = pics_path
        self.font_size = font_size
        
        # Signal processing constants (previously hardcoded)
        self.adc_scaling_bits = adc_scaling_bits  # ADC resolution (2**14)
        self.stdev_adjustment_factor = stdev_adjustment_factor  # Standard deviation adjustment
        self.epsilon = epsilon  # Division-by-zero protection
    
    def get_rf_bandwidth(self, sample_rate):
        """Calculate RF bandwidth based on sample rate and multiplier"""
        return int(sample_rate * self.rf_bandwidth_multiplier)
    
    def __str__(self):
        """String representation of the configuration"""
        return f"""SDR Configuration:
  Hardware:
    - TX IP: {self.sdr_tx_ip}
    - TX Freq: {self.sdr_tx_freq/1e6:.1f} MHz
    - RF BW Multiplier: {self.rf_bandwidth_multiplier}
    - RX Gain: {self.sdr_rx_gain} dB
  
  Power Control:
    - SINR Range: {self.min_sinr}-{self.max_sinr} dB
    - TX Gain Range: {self.tx_gain_min} to {self.tx_gain_max} dB
    - TX Gain Step: {self.tx_gain_step} dB
  
  Signal Processing:
    - Correlation Threshold: {self.corr_threshold}
    - Leading Zeros: {self.n_zeros}
    - Additional TD Symbols: {self.add_td_symbols}
  
  Debug: {self.debug}"""

# Create a default configuration instance
default_config = SDRConfig()

# Convenience functions for common configurations
def create_config_for_frequency(freq_mhz, **kwargs):
    """Create configuration for a specific frequency in MHz"""
    return SDRConfig(sdr_tx_freq=freq_mhz * 1e6, **kwargs)

def create_debug_config(**kwargs):
    """Create configuration with debug enabled"""
    return SDRConfig(debug=True, save_plots=True, **kwargs)

def create_low_power_config(**kwargs):
    """Create configuration for low power operation"""
    return SDRConfig(
        min_sinr=3,
        max_sinr=20,
        power_max_tx_scaling=0.5,
        **kwargs
    )

def create_high_performance_config(**kwargs):
    """Create configuration for high performance operation"""
    return SDRConfig(
        min_sinr=10,
        max_sinr=35,
        sdr_rx_gain=20,
        corr_threshold=15,
        **kwargs
    )
