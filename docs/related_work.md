# Related work

Notes we collected while comparing ResQFog with existing research, products and patents. Accuracy numbers are as reported by each paper on its own data, so they are not directly comparable with each other or with ours.

## Research

| Work | Hardware | Where detection runs | Data / reported result | Multi-machine, alerts | Motor protection |
|---|---|---|---|---|---|
| Loukatos et al., *Sensors*, 2023 [1] | Arduino Nano 33 BLE Sense, built-in accelerometer at 100 Hz | On-device neural network (Edge Impulse, spectral features) | Lab centrifugal water pump, 5 states; 98.5 % training, 93 % on test runs | Not reported | Not reported |
| Mostafavi et al., *Sharif J. Mech. Eng.*, 2025 [2] | STM32H743 + 3-axis accelerometer | Autoencoder trained on the MCU | Centrifugal pump; >99.9 % | Not reported | Not reported |
| Arciniegas et al., *Discover Internet of Things*, 2025 [3] | Low-power MCU | TinyML with spectral analysis | Lab motor bearings; 96.5 %, 300 ms to alert | Alert generation | Not reported |
| Yang et al., CSCWD 2023 [4] | MCU + edge server | Stacked autoencoder, end-edge collaboration | CWRU; 100 % binary, 6.44 kB RAM | Not reported | Not reported |
| Kolar et al., *Tehnički Glasnik*, 2022 [5] | IIoT accelerometer, edge and fog devices | ISO 10816-1 severity rules | Industrial rotating equipment | Node-RED dashboard, MQTT | Not reported |
| Charan, SAE, 2022 [6] | ARM Cortex-M4 | Autoencoder, 7.5 kB | Vehicle vibration data; about 80 % | Not reported | Not reported |
| Román Espinoza and Pesantes Morales, thesis, 2025 [7] | ESP32 + MPU6050 | Cloud, FFT | Two motors | Cloud storage | Not reported |
| **ResQFog** | ESP32 + MPU6050 (about ₹1,050 per node) | Thresholds on the ESP32; Isolation Forest on the fog server | CWRU filtered and resampled to the MPU6050 bandwidth: 97.7 % fault-window detection, 0.7 % false alarms, AUC 0.999 | Fleet dashboard, SMS alerts with location, fault CSV recordings, offline buffer | Edge trip after 3 critical readings, remote restart |

What we think is different in ResQFog:

1. **Two detection layers.** A threshold alarm and motor trip on the ESP32 that need no network, and an ML anomaly model on the fog server that warns before the threshold is crossed.
2. **Training matched to the sensor.** Public data is low-pass filtered at 184 Hz (the MPU6050 DLPF setting we use) and resampled to 500 Hz before training, so the model only uses information our sensor can measure. With this preprocessing, shape features (crest factor, kurtosis) dropped to about 15 % detection, while band-energy features reached 97.7 %. This is a useful result for anyone using MPU6050-class sensors.
3. **Per-pump calibration.** The same model type is fitted to each pump from about 2 minutes of normal running, with a 3-of-5 vote over windows.
4. **Operational features for unmanned sites.** Store-and-forward at the edge, fault recordings, SMS alerts with the pump location sent from an ordinary SIM, and remote restart.

## Dataset

- Case Western Reserve University Bearing Data Center, https://engineering.case.edu/bearingdatacenter
- Smith, W. A., and Randall, R. B. "Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study." *Mechanical Systems and Signal Processing* 64–65 (2015). Discusses the known issues with this dataset, which we mention as a limitation.

In our checks, the normal baseline files (97–100) show the same machine peaks as the 48 kHz fault files when read at 48 kHz, so we treat them as 48 kHz recordings.

## Products and patents

- Emerson AMS / Rosemount wireless vibration transmitters, and Bently Nevada Ranger Pro: industrial wireless vibration monitoring for pumps, priced for large plants.
- US 10,317,875 B2, "Pump integrity detection, monitoring and alarm generation" (BJ Energy Solutions, granted 2019): vibration peaks compared with a baseline to detect wear or failure and shut the pump down.
- US 10,711,802 B2, "Pump monitoring" (Weir Minerals Australia, granted 2020): vibration sensor on the pump casing to determine wear.

## References

1. D. Loukatos, M. Kondoyanni, G. Alexopoulos, C. Maraveas, K. G. Arvanitis, "On-Device Intelligence for Malfunction Detection of Water Pump Equipment in Agricultural Premises: Feasibility and Experimentation," *Sensors*, 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC9860875/
2. A. Mostafavi, A. Alizadeh, A. Sadighi, "Edge-Computing-Based Anomaly Detection of Rotating Machines Using Artificial Neural Networks," *Sharif Journal of Mechanical Engineering*, vol. 41, no. 2, 2025. https://sjme.journals.sharif.edu/article_24021.html?lang=en
3. S. Arciniegas, D. Rivero, J. Piñan, E. Diaz, F. Rivas, "IoT device for detecting abnormal vibrations in motors using TinyML," *Discover Internet of Things*, vol. 5, art. 41, 2025. https://puceinvestiga.puce.edu.ec/es/publications/iot-device-for-detecting-abnormal-vibrations-in-motors-using-tiny/
4. C. Yang, Z. Lai, Y. Wang, S. Lan, L. Wang, L. Zhu, "A Novel Bearing Fault Diagnosis Method based on Stacked Autoencoder and End-edge Collaboration," CSCWD 2023. https://pure.bit.edu.cn/en/publications/a-novel-bearing-fault-diagnosis-method-based-on-stacked-autoencod/
5. D. Kolar, D. Lisjak, M. Curman, M. Pająk, "Condition Monitoring of Rotary Machinery Using Industrial IOT Framework: Step to Smart Maintenance," *Tehnički Glasnik*, vol. 16, no. 3, pp. 343–352, 2022. https://doi.org/10.31803/tg-20220517173151
6. K. S. Charan, "An Auto-Encoder Based TinyML Approach for Real-Time Anomaly Detection," SAE Technical Paper 2022-28-0406, 2022. https://saemobilus.sae.org/articles/auto-encoder-based-tinyml-approach-real-time-anomaly-detection-2022-28-0406
7. S. A. Román Espinoza, A. A. Pesantes Morales, "Diseño e implementación de un prototipo de monitoreo de vibraciones en un motor, utilizando tarjeta embebida y transmisión de datos a la nube," thesis, Universidad Politécnica Salesiana, 2025. https://dspace.ups.edu.ec/handle/123456789/31580
8. W. A. Smith, R. B. Randall, "Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study," *Mechanical Systems and Signal Processing*, vol. 64–65, 2015.
9. F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation Forest," IEEE ICDM, 2008.

Check every reference against the original paper before citing it.
