import pandas as pd
import numpy as np

df_spot = pd.read_excel('./data/Elspotprices.xlsx')
df_spot.head()

df_balancing = pd.read_excel('./data/RegulatingBalancePowerdata.xlsx')
df_aFRR = pd.read_excel('./data/AfrrReservesNordic.xlsx')
df_aFRR_activation = pd.read_excel('./data/AfrrActivatedAutomatic.xlsx')
df_ninja_2022 = pd.read_csv('./data/ninja_wind_55.0000_13.0000_corrected_2022.csv', skiprows=3) # Skip first 3 rows because they are the header
df_ninja_2023 = pd.read_csv('./data/ninja_wind_55.0000_13.0000_corrected_2023.csv', skiprows=3)


p_RT = pd.concat([df_ninja_2022,df_ninja_2023], axis=0)['electricity']
p_RT = p_RT / (605*10**3) # Nominal capacity used to generate the data. Now it's between 0-1 (MW)
lambda_DA = df_spot[df_spot['PriceArea']=='DK2']['SpotPriceDKK'] # Spot price: DKK/MWh
lambda_B = df_balancing[df_balancing['PriceArea']=='DK2']['ImbalancePriceDKK'] # Balancing price: DKK/MWh (mFRR, used as aFRR)

lambda_RES = df_aFRR[df_aFRR['PriceArea']=='DK2']['aFRR_DownCapPriceDKK'] # aFRR capacity price: DKK/MW
gamma_RES = df_aFRR_activation[df_aFRR_activation['PriceArea']=='DK2']['aFRR_DownActivated'] # aFRR down-regulation activation: MW
gamma_RES = (gamma_RES > 0).astype(int) # Makes the activation binary
# "aFRR Downward regulation purchased (MW)" -- not used because we assume that they always buy
