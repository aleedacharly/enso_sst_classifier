# ENSO Classification from Sea Surface Temperature Anomalies with PyTorch

This project demonstrates that neural network can recognize a large scale climate pattern directly from a raw spatial field from gridded data alone. It is an excercise towards machine learning based climate pattern recognition and ENSO forecasting, which is an active research frontier where deep learning has recently used.

## Background
The El Niño–Southern Oscillation (ENSO) is the strongest year-to-year climate fluctuation on Earth. It consists of three phases : a warm phase (El Niño), a cold phase (La Niña), and a neutral state. This both have effects on droughts, floods, monsson shifts and hurricane activities across mutliple continents.
Since ENSO is the distribution of warm water across equitorial Pacific, the signal is the SST. 
## Data
- SST : NOAA ERSSTv5 monthly sea surface temperature, 2°×2° global grid, read from netCDF with xarray
- Labels: NOAA Oceanic Niño Index (ONI), 1950–present. Each month is labelled El Niño (ONI ≥ +0.5 °C), La Niña (ONI ≤ −0.5 °C), or Neutral (in between)

## Method
1. Anomalies
2. Region: The map is cropped where the ENSO signal is concentrated.
3. Temporal split: Using training data up to 2009 and testing data from 2010
4. Models: used a feedforward model(with flattened map) and a small CNN( which preserves 2D spatial structure)

## Results
Model	|| Test accuracy	|| Notes
Majority-class baseline||	45.2%	||Always predict "Neutral"
Feedforward	||76.4%||	Balanced across classes (f1 0.74–0.80)
CNN	|| 84.0%	|| Best overall; exploits spatial structure

# Key Findings
1. Both models have more than baseline value indicating they learn patterns rather than memorizing.
2. Both models never had confusions with La Niña with El Niño
3. CNN improved accuracy from 76% to 84%

## Limitation and future directions
1. It only predicts current ENSO phase not the months ahead
2. NOAA have moved to ONI to RONI in 2026 accounting for background ocean-warming trend
3. It is only approx. 720 training months.

## Dependencies
- xrray
- netCDF4
- numpy
- pandas
- matplotlib
- torch

## Data sources
1. NOAA ERSSTv5: https://www.ncei.noaa.gov/products/extended-reconstructed-sst
2. NOAA ONI (PSL): https://psl.noaa.gov/data/correlation/oni.data
