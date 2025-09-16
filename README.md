# Enabling Over-the-Air Testing: Integration of PlutoSDR into Sionna Framework 

## Introduction

This project focuses on integrating the PlutoSDR radio into the Sionna framework. The main goal is to enable testing and demonstrations of simulated systems created with Sionna through a true over-the-air connection, thereby if not eliminating the need for a simulated radio channel, providing a validation point or over-the-air demosntration capability for a research. The implementation leverages PlutoSDR, an affordable Software-Defined Radio (SDR) developed by Analog Devices.

**Important: This implementation is designed for 1T1R SISO (Single Input Single Output) operation with single SDR device only.** It uses a single PlutoSDR unit operating as both transmitter and receiver in a loopback configuration. This **excludes** the synchronization and frequency correction requirements that are necessary for dual SDR implementations (separate TX/RX units), significantly simplifying the setup while still providing valuable over-the-air demonstration and validation capabilities.

The jupyter notebook `00_Sionna_Pluto_SDR_example.ipynb` provides an end to end example on how to use the functionality.


## Functionality

The system is designed to receive modulated In-Phase and Quadrature (IQ) signals produced by the Sionna modulator. These signals are then sent through the radio interface using the Transmission (TX) port of the Software-Defined Radio (SDR). The system performs signal synchronization, analysis, and characterization as shown in the following plots.

In addition to configuring the modulator's IQ output, users have the option to set parameters such as SDR RX gain, and the number of extra unmodulated symbols. The unmodulated symbols are utilized for Signal-to-Interference-plus-Noise Ratio (SINR) estimations. The system employs a straightforward power control algorithm that maintains the SINR within predefined limits, eliminating the need to manually configure the TX power.

### RX Signal prior sync

The transmission process utilizes a cyclic transmission of the modulated signal. To facilitate the calculation of Signal-to-Noise Ratio (SINR) estimates, zero-modulated symbols are inserted between successive instances of the modulated signal.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot1.png) 

Fig 1. Received Signal Overview with Synchronization - Shows the complete received signal with the synchronization point marked in red.

### TX Signal

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot2.png) 

Fig 2. Transmitted Signal - The original modulated OFDM signal that was transmitted.

### RX Signal

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot3.png) 

Fig 3. Received Signal (Synchronized) - The received signal after synchronization, showing one OFDM block.

### TX PSD

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot4.png) 

Fig 4. Transmitted Signal PSD - Power Spectral Density of the transmitted signal showing frequency characteristics.

### RX PSD

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot5.png) 

Fig 5. Received Signal and Noise PSD - Power spectral densities comparing the received signal and noise floor.


### Correlator for sync

The transmitted and received IQ signals undergo correlation analysis to determine the start position of the modulated symbols. The offset can be determined in two ways: either by identifying the peak correlation or by using a threshold multiplier applied to the average correlation.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot7.png) 

Fig 7. Correlator - Cross-correlation analysis around the peak showing synchronization accuracy.

### TX linearity

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot8.png) 

Fig 8. AM-AM Linearity - Amplitude linearity characteristics using time-synchronized TX/RX samples for power amplifier characterization.

After the synchronisation, the received IQ signals undergo scaling to align the magnitudes with those of the original signal to ensure compatibility with the Sionna demodulator. The output format is `[IQ, SINR, SDR_TX_GAIN, SDR_RX_GAIN, fails + 1, corr, sdr_time]` where
- IQ is the IQ data in format expected by Sionna demodulator
- SINR is the measured SINR based on noise power measurement during the unmodulated symbols, and the mean power of the received and synchronised signal.
- SDR_RX_GAIN similar to above, the actual RX setting
- fails+1 is the number of repeated processes if correlation check fails. If this happens, TX power is increased each time.
- corr is the Pearson correlation of the tx and rx signals
- sdr_time is the measured time from start of the SDR process to finishing it. When debug is enabled, it takes about 1.4sec and without it takes 25ms in authors computer.


These pictures are created with the `00_Sionna_Pluto_SDR_example.ipynb` notebook located in this repository (debug=True). In the author's environment, the entire transmit-receive process typically takes around 25 milliseconds.


## Installation and Setup

### Prerequisites

**Software Requirements:**
- Python 3.8 or later
- `sionna` 1.0 or later (tested with 1.1.0)
- `tensorflow` 2.5 or later
- `pyadi-iio` for PlutoSDR interface

**Hardware Requirements:**
- PlutoSDR (ADALM-PLUTO) or compatible with Ethernet connection recommended, should work with USB too, but untested
- RF antenna or loopback cable for testing

**System Dependencies:**
For conda environments (recommended):
```bash
conda install -c conda-forge libiio pyadi-iio
```

### Quick Start

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify PlutoSDR connection:**
   ```bash
   # Test PlutoSDR connectivity (replace with your PlutoSDR IP)
   ping 192.168.1.10
   ```

3. **Basic usage example:**
   ```python
   import SDR_Sionna_1T1R as sdr
   from SDR_config import SDRConfig
   
   # Create configuration (adjust IP and frequency for your setup)
   config = SDRConfig(
       sdr_tx_ip="ip:192.168.1.10",    # Your PlutoSDR IP
       sdr_tx_freq=435.1e6,            # ISM band frequency
       debug=False,                    # Set True for debug plots
       save_plots=False               # Set True to save plots
   )
   
   # Initialize SDR with your sample rate
   SampleRate = 1.92e6  # Example: 1.92 MHz
   SDR1 = sdr.SDR(SampleRate=SampleRate, config=config)
   
   # Use with Sionna-generated signals
   x_sdr = SDR1(SAMPLES=x_time, add_td_symbols=16, threshold=25)
   ```

4. **Run the example notebook:**
   ```bash
   jupyter notebook 00_Sionna_Pluto_SDR_example.ipynb
   ```

## Configuration System

**Easy Parameter Management:**
- Hardware settings (IP address, frequency, RF gains)
- Signal processing parameters (SINR thresholds, correlation settings)
- Debug and visualization options
- Performance tuning parameters

**Built-in Configuration Presets:**
```python
# Default balanced configuration
config = SDRConfig()

# Debug configuration with plots and verbose output
config = create_debug_config(sdr_tx_ip="ip:192.168.1.10")

# Low power configuration for extended operation
config = create_low_power_config(sdr_tx_freq=435.1e6)

# Custom configuration
config = SDRConfig(
    sdr_tx_ip="ip:192.168.1.10",
    sdr_tx_freq=435.1e6,
    min_sinr=10.0,
    max_sinr=35.0,
    debug=True,
    save_plots=True
)
```


## Performance

**Typical Performance Metrics:**
- **Processing Time**: ~25ms per OFDM block (without debug plots)
- **Debug Mode**: ~1.4s per block with visualization
- **Memory Usage**: Optimized for minimal memory footprint


**Optimization Features:**
- Type hints for better IDE support and development experience
- Streamlined error handling for faster execution
- Minimal dependencies for reduced overhead
- Efficient correlation algorithms

### Limitations

- **Single SDR Operation**: Supports SISO 1T1R only (single PlutoSDR in loopback mode)
- **Batch Processing**: Batch size must be 1 
- **Execution Mode**: TensorFlow graph execution is not supported (eager execution only)
- **Connection Type**: Tested with Ethernet-connected PlutoSDR (USB possible but not tested)

## Recent Updates


- **Dependency Cleanup**: Removed unused imports (scipy, logging) for faster startup
- **Simplified Error Handling**: Streamlined error reporting with basic print statements  
- **Type Safety**: Added comprehensive type hints for better IDE support
- **Requirements Optimization**: Minimal dependency list for lighter installation
- **Updated from Sionna 0.16 to 1.x**: All imports and API calls updated for the new module structure
- **New Configuration System**: Centralized parameter management through `SDR_config.py`
- **Automatic Compatibility**: Built-in handling of OpenSSL conflicts in Jupyter notebooks


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

Hoydis, Jakob, Sebastian Cammerer, Fayçal Ait Aoudia, Avinash Vem, Nikolaus Binder, Guillermo Marcus, and Alexander Keller. “Sionna: An Open-Source Library for next-Generation Physical Layer Research.” arXiv Preprint, March 2022.

“ADALM-PLUTO Evaluation Board | Analog Devices.” Accessed November 15, 2021. https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html#eb-overview.


## Disclaimer: Legal Compliance in Radio Transmission

It is imperative to acknowledge that engaging in radio transmission activities, especially over the air, mandates strict adherence to regulatory and licensing requirements. The use of devices like the PlutoSDR for transmitting radio signals must be conducted responsibly and legally.

1. **Obtain Required Licenses**: Before initiating any radio transmission, ensure you have obtained all necessary licenses and permissions from the appropriate regulatory bodies.
2. **Follow Frequency Allocations**: Adhere to the designated frequency bands and power levels to prevent interference with licensed services and other critical communication systems.
3. **Ensure Safety**: Ensure that your transmission activities do not pose any risks to public safety or disrupt other lawful communication services.

Engaging in unauthorized radio transmission can result in severe legal consequences, including fines, legal actions, and the confiscation of equipment.


