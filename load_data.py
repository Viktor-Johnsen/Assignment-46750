import pandas as pd
#import numpy as np

df_spot = pd.read_excel('./data/Elspotprices.xlsx')
df_spot.head()

df_balancing = pd.read_excel('./data/RegulatingBalancePowerdata.xlsx')
df_aFRR = pd.read_excel('./data/AfrrReservesNordic.xlsx')
df_aFRR_activation = pd.read_excel('./data/AfrrActivatedAutomatic.xlsx')
df_ninja_2022 = pd.read_csv('./data/ninja_wind_55.0000_13.0000_corrected_2022.csv', skiprows=3) # Skip first 3 rows because they are the header
df_ninja_2023 = pd.read_csv('./data/ninja_wind_55.0000_13.0000_corrected_2023.csv', skiprows=3)

print(df_spot.head())
print(df_balancing.head())
print(df_aFRR.head())
print(df_aFRR_activation.head())
print(df_ninja_2022.head())
print(df_ninja_2023.head())