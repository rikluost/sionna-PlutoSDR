# sionna-PlutoSDR

## Introduction

This repository undertakes the integration of the PlutoSDR radio interface with sionna, presenting an avenue for testing that transcends simulated radio channels and instead leverages authentic radio interfaces. The PlutoSDR, a cost-effective Software-Defined Radio (SDR), is made and supplied by Analog Devices. 

In the current implementation, only 1T1R is implemented, despite the hardware supports 2T2R. 

The included jupyter notebook `Sionna_Pluto_SDR.ipynb` provides an example on how to use the functionality.

Note: The code may benefit from further optimization and clean-up, as the developer acknowledges not being a professional developer.

## Functionality

Inputs modulated IQ signals from sionna modulator, transmits over the radio interface through the TX port of the radio.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot3.png.png) 

Fig 1. Original modulated signal

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot3.png.png) 

Fig 2. Power Spectral Density of the transmitted signal

The signal is transmitted in loop, and 500 zero modulated symbols is placed inbetween the modulated signal for calculating SINR estimate. After reception, the signal looks as in Fig 3.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot1.png.png) 

Fig 3. Received signal from three times repeated transmissions

Signals are correlated and startpoint is calculated

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot2.png.png) 

Fig 4. Received signal with calculated start position

After synchronisation, the power spectral densities for noise and received signal can be calculated. PSD plots are shown in figure 6.

![alt text](https://github.com/rikluost/sionna-PlutoSDR/blob/main/pics/_plot6.png.png) 

Fig 6.

The received IQ signals is scaled with the same magnitude as the input signal and output is compatible for the sionna demodulator.

## Prerequisites

- `sionna` 15.1 or later
- `libiio`, Analog Device’s library for interfacing hardware
- `libad9361-iio`, AD9361 the Analog Devices RF chip
- `pyadi-iio`, Python API for PlutoSDR
- `tensorflow_probability==0.21`

## Usage

Upon successfully configuring and testing the aforementioned prerequisites, proceed to open the notebook and execute its contents.

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


