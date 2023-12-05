# sionna-PlutoSDR

## Introduction

This repository undertakes the integration of the PlutoSDR radio interface with sionna. It allows testing and demonstration over the air connection, without the need for a simulated radio channel. The PlutoSDR, a cost-effective Software-Defined Radio (SDR), is made and supplied by Analog Devices. 

In the current implementation, only 1T1R is implemented, despite that the hardware supports 2T2R. 2T2R could be implemented later.

The included jupyter notebook `Sionna_Pluto_SDR.ipynb` provides an end to end example on how to use the functionality, in eager model, and integrated into Keras model. Graph execution is not supported. 

Note: The code may well benefit from further optimization and clean-up, as the developer acknowledges not being a professional developer.

## Functionality

The system accepts modulated In-Phase and Quadrature (IQ) signals generated by the Sionna modulator. These signals are subsequently transmitted over the radio interface via the Transmission (TX) port of the radio equipment. Figure 1 shows an example of the real part of the original IQ samples and figure 2 show the power spectral density of the signal. These are created with the `Sionna_Pluto_SDR.ipynb` notebook. The input format is compatible with sionna modulator. Optional inputs include SDR TX gain, RX gain, and the number of additional unmodulated symbols.



![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot3.png) 

Fig 1. Original modulated signal

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot5.png) 

Fig 2. Power Spectral Density of the transmitted signal

The transmission process involves a continuous loop of transmitting the modulated signal. To facilitate the calculation of Signal-to-Noise Ratio (SINR) estimates, 500 zero-modulated symbols are strategically inserted between successive instances of the modulated signal. Following reception, the observed signal exhibits a characteristic appearance, as illustrated in Figure 3.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot1.png) 

Fig 3. Received signal from three times repeated transmissions

The transmitted and received signals undergo correlation analysis, and the starting point is subsequently determined through this process.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot2.png) 

Fig 4. Received signal with calculated start position

Upon successful synchronization, the Power Spectral Densities (PSD) for both the received signal and the accompanying noise can be calculated. The resulting PSD plots are visualized in Figure 6.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot6.png) 

Fig 6.

The received In-Phase and Quadrature (IQ) signals undergo scaling, aligning their magnitudes with those of the input signal. This ensures compatibility with the Sionna demodulator. The output format is [IQ, SINR, SDR_TX_GAIN, SDR_RX_GAIN, fails + 1, corr, sdr_time]

In the author's environment, the entire transmit-receive process typically spans a duration of 25 to 40 milliseconds.


## Prerequisites

- `sionna` 15.1 or later
- `libiio`, Analog Device’s library for interfacing hardware
- `libad9361-iio`, AD9361 the Analog Devices RF chip
- `pyadi-iio`, Python API for PlutoSDR
- `tensorflow_probability==0.21`


### Limitations

Batch size must be 1.
Currently supports SISO 1T1R only, with 2T2R functionality potentially added later.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

Hoydis, Jakob, Sebastian Cammerer, Fayçal Ait Aoudia, Avinash Vem, Nikolaus Binder, Guillermo Marcus, and Alexander Keller. “Sionna: An Open-Source Library for next-Generation Physical Layer Research.” arXiv Preprint, March 2022.

“ADALM-PLUTO Evaluation Board | Analog Devices.” Accessed November 15, 2021. https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html#eb-overview.

“PySDR: A Guide to SDR and DSP Using Python — PySDR: A Guide to SDR and DSP Using Python 0.1 Documentation.” Accessed November 15, 2021. https://pysdr.org/index.html.

## Disclaimer: Legal Compliance in Radio Transmission

It is imperative to acknowledge that engaging in radio transmission activities, especially over the air, mandates strict adherence to regulatory and licensing requirements. The use of devices like the PlutoSDR for transmitting radio signals must be conducted responsibly and legally.

1. **Obtain Required Licenses**: Before initiating any radio transmission, ensure you have obtained all necessary licenses and permissions from the appropriate regulatory bodies.
2. **Follow Frequency Allocations**: Adhere to the designated frequency bands and power levels to prevent interference with licensed services and other critical communication systems.
3. **Ensure Safety**: Ensure that your transmission activities do not pose any risks to public safety or disrupt other lawful communication services.

Engaging in unauthorized radio transmission can result in severe legal consequences, including fines, legal actions, and the confiscation of equipment.


