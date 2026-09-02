# ENSO Classification from Sea Surface Temperature Anomalies
A Pytorch deep learning project classifying El Nino or Neutral months from ERSSTv5 SST anomalies.

Dependencies:
- xrray
- netCDF4
- numpy
- pandas
- matplotlib
- torch
 ## Day 1
**Goal** : classify each month as El Nino, La Nina and neutral from sea surface temperature.
**Data** : NOAA ERSSTv5 monthly SST, 2°×2° global grid (a *reconstructed*
gridded product, not raw observations), stored as netcdf, read with xarray, Longitude uses the 0–360° convention, so the Pacific sits mid-array.
**Key Ideas** : raw SST is dominated by seasonal cycle which is far larger than ENSO signal. To expose ENSO, I removed the seasonal cycle for each grid cell and calender month, and subtracted that cell's average
1991-2020 average for that month. And what remains is the anomaly which is the ENSO. I chose 1991-2020.
**Verfication** : I plotted both Decembers of 1997 and 1999 confirming the signal is present.


## Day 2 — Labels, Model, Training

**Labels:** parsed NOAA ONI index (1950–2026), thresholded at ±0.5 °C into
La Niña (0) / Neutral (1) / El Niño (2). Aligned to SST maps on the intersection
of dates (919 shared months, since ONI runs past the SST record's end).

**Input:** anomaly maps cropped to the tropical Pacific (30°S–30°N, 120–280°E),
land cells (7.5%) filled with 0 (= "no anomaly", valid because these are anomalies).

**Split:** by TIME, not randomly — train ≤2009, test ≥2010. Random splitting would
leak information because adjacent months are correlated (ENSO events span many months),
inflating accuracy dishonestly. Temporal split gives a defensible estimate.

**Model:** feedforward net (2511→128→64→3), identical to the MNIST classifier except
input and output sizes. Adam, CrossEntropyLoss, 30 epochs.

**Result:** 76.4% test accuracy vs a 45.2% majority-class baseline. Crucially, the
model NEVER confused La Niña with El Niño (it learned the sign of the anomaly);
all errors were at the weak-event/Neutral boundary — the same ambiguity a human
faces at the ±0.5 threshold. Balanced performance across classes (f1 0.74–0.80).