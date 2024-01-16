# Enabling Over-the-Air Testing: Integration of PlutoSDR into Sionna Framework 

## Introduction

This project focuses on integrating the PlutoSDR radio into the Sionna framework. The main goal is to enable testing and demonstrations of simulated systems created with Sionna through a true over-the-air connection, thereby eliminating the need for a simulated radio channel. The implementation leverages PlutoSDR, an affordable Software-Defined Radio (SDR) developed by Analog Devices.

The jupyter notebook `Sionna_Pluto_SDR.ipynb` provides an end to end example on how to use the functionality, both, with eager mode, and then as integrated Keras layer integrated in a simulated system.


## Functionality

The system is designed to receive modulated In-Phase and Quadrature (IQ) signals produced by the Sionna modulator. These signals are then sent through the radio interface using the Transmission (TX) port of the Software-Defined Radio (SDR). In Figure 1, the absolute values of the complex IQ samples are illustrated, while Figure 2 displays the power spectral density (PSD) of the transmitted signal.

n addition to configuring the modulator's IQ output, users have the option to set parameters such as SDR TX gain, RX gain, and the number of extra unmodulated symbols. The unmodulated symbols are utilized for Signal-to-Interference-plus-Noise Ratio (SINR) estimations.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot3.png) 

Fig 1. Original modulated signal

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot5.png) 

Fig 2. Power Spectral Density of the transmitted signal

The transmission process utilises a cyclic transmission of the modulated signal. To facilitate the calculation of Signal-to-Noise Ratio (SINR) estimates, zero-modulated symbols are inserted between successive instances of the modulated signal. The received signal is illustrated in Figure 3.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot1.png) 

Fig 3. Received signal from three times repeated transmissions

The transmitted and received IQ signals undergo correlation analysis to determine the start position of the modulated symbols.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot2.png) 

Fig 4. Received signal with calculated start position

Upon successful synchronization, the PSD for both the received signal and the accompanying noise can be calculated. The resulting plots are visualized in Figure 6.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot6.png) 

Fig 6. Power spectral densities of the received noise during the unmodulated symbols, and modulated symbols

The offset can be determined in two ways: either by identifying the peak correlation or by using a threshold, which is a multiplier applied to the average correlation (for example, a multiplier of 6 seems to work well).

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot7.png) 

Fig 7. Correlation around the peak is shown. The blue line indicates offset detection based on the threshold, while the red line signifies offset detection based on the maximum value.

After the synchronisation, the received IQ signals undergo scaling to align the magnitudes with those of the original signal to ensure compatibility with the Sionna demodulator. The output format is `[IQ, SINR, SDR_TX_GAIN, SDR_RX_GAIN, fails + 1, corr, sdr_time]`

These pictures are created with the `Sionna_Pluto_SDR.ipynb` notebook located in this repository (debug=True). In the author's environment, the entire transmit-receive process typically takes around 25 milliseconds.


## Prerequisites

- `sionna` 16 or later
- `libiio`, Analog Device’s library for interfacing hardware
- `libad9361-iio`, AD9361 the Analog Devices RF chip
- `pyadi-iio`, Python API for PlutoSDR


### Limitations

- Currently supports SISO 1T1R only, with 2T2R functionality potentially added later. 
- Batch size must be 1.
- Graph execution is not supported.

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


