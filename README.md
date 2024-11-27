# Assignment-46750
All data and scripts for the final assignment in course 46750

The following dataset are used.

Historical wind production: [Renewables Ninja](https://www.renewables.ninja). We use data for the location of Kriegers Flak, though we assume a wind farm capacity of simply 1 MW to allow for price-taker assumption.

Historical spot prices ("Spot price (DKK)"): [Energi Data Service](https://www.energidataservice.dk/tso-electricity/Elspotprices)

Historical reserve capacity and prices ("aFRR Downward regulation purchased (MW)", "aFRR Downward regulation capacity price (DKK)"): [Energi Data Service](https://www.energidataservice.dk/tso-electricity/AfrrReservesNordic)

Historical balancing market prices ("Imbalance Price (DKK)"): [Energi Data Service](https://www.energidataservice.dk/tso-electricity/RegulatingBalancePowerdata)

Data is used for the period 03/10--2024 to 06/11--2024 and for grid zone DK2 (five weeks of data where aFRR down is actually being purchased regularly).

The code is split into various stages, expressed by the filenames. 'load_data' is pretty self explanatory, as it simply loads the data into python, such that it can easily referenced in other files.
Then comes the actual implementation files, 'V1', 'V2', and 'V3', each increasing in complexity, with 'V1' being our base problem (wind farm in DA with recourse), 'V2' mainly adding the aFFR market, and strategic offering for aFFR, and 'V3' adding CVaR. 
%as well as for V3
These changes were added ontop of each other in some cases, leading to 'V2_aFFR', 'V2_aFFR_stratbal', and finally 'V2_aFFR_stratbal_chance'.