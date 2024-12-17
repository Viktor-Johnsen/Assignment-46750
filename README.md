# Assignment-46750
All data and scripts for the final assignment in course 46750

The following dataset are used.

Historical wind production: [Renewables Ninja](https://www.renewables.ninja). We use data for the location of Kriegers Flak, though we assume a wind farm capacity of simply 1 MW to allow for price-taker assumption.

Historical spot prices ("Spot price (DKK)"): [Energi Data Service](https://www.energidataservice.dk/tso-electricity/Elspotprices)

Historical reserve capacity and prices ("aFRR Downward regulation purchased (MW)", "aFRR Downward regulation capacity price (DKK)"): [Energi Data Service](https://www.energidataservice.dk/tso-electricity/AfrrReservesNordic)

Historical balancing market prices ("Imbalance Price (DKK)"): [Energi Data Service](https://www.energidataservice.dk/tso-electricity/RegulatingBalancePowerdata)

Data is used for the period 03/10--2024 to 06/11--2024 and for grid zone DK2 (five weeks of data where aFRR down is actually being purchased regularly).

The code is split into various stages, expressed by the filenames. 'load_data' is pretty self explanatory, as it simply loads the data into python, such that it can easily referenced in other files.

Then comes the actual implementation files for the models V1, V2, V3, V4, and V5. The file names include a small description of what is included in each version e.g. that the aFRR power market is included in V2. Each model increases in complexity from V1 to V4/V5. The file 'V4_OOS.py' is used to test the optimal solution from 'V4_aFRR_stratbal_risk.py' out-of-sample.

The two files 'Decomp-baseline.py' and 'Decomp-benders.py' are used to decompose a simpler version of V4 without strategic price offering.