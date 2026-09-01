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