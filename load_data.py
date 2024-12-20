import pandas as pd
import numpy as np

'''train_scenarios = 30 # Train on the first four weeks-ish
test_scenarios = 5 # Test on the following week(days)'''
train_scenarios=30
test_scenarios=5

num_scenarios = train_scenarios+test_scenarios

df_spot = pd.read_csv('./data/Elspotprices.csv', sep=';', decimal=',')
df_balancing = pd.read_csv('./data/RegulatingBalancePowerdata.csv', sep=';', decimal=',')
df_aFRR = pd.read_csv('./data/AfrrReservesNordic.csv', sep=';', decimal=',')
df_wind = pd.read_csv('./data/ElectricityBalanceNonv.csv', sep=';', decimal=',')

p_RT = df_wind['OffshoreWindPower'] # Wind power: MW
p_RT = p_RT / p_RT.max() # Normalize between 0-1
lambda_DA = df_spot[df_spot['PriceArea']=='DK2']['SpotPriceDKK'] # Spot price: DKK/MWh
lambda_B = df_balancing[df_balancing['PriceArea']=='DK2']['ImbalancePriceDKK'] # Balancing price: DKK/MWh (mFRR, used as aFRR)

lambda_RES = df_aFRR[df_aFRR['PriceArea']=='DK2']['aFRR_DownCapPriceDKK'] # aFRR capacity price: DKK/MW

print('##############\nScript is done\n##############')