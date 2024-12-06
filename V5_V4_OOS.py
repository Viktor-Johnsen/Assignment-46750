
# This file is used to test the solutions from V4 and V5 out-of-sample (OOS)

import numpy as np

# First we define the functions used to calculate the profits in each scenrio

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES# , gamma_RES
from load_data import train_scenarios, test_scenarios # 30, 7

T = 24

W_train = train_scenarios
W_test = test_scenarios
W = W_train

TT = range(T)
WW = range(W)

pi = 1/W

def evaluate_V4(DA_offers, RES_offers, alpha_RES, beta_RES,
                lambda_DA, lambda_RES, lambda_B):
    
    lambda_offer_RES = [[alpha_RES[t] * (lambda_DA[w,t+1]-lambda_DA[w,t] if t<T-1 else 0) + lambda_DA[w,t] + beta_RES[t] for t in TT] for w in WW]

    a_RES = np.zeros((W,T))
    a_RES[( np.round(lambda_offer_RES,4) < np.round(lambda_DA - lambda_B,4) ) & ( np.round(lambda_DA,4) > np.round(lambda_B,4) )] = 1
    for t in TT:
        for w in WW:
            a_RES[w,t] = min(a_RES[w,t], RES_offers[t]) # Shouldn't be 1 if less was offered

    # Set Delta correctly
    Delta_down = np.array([[ max( DA_offers[t] - p_RT[w,t] - a_RES[w,t],0) for t in TT] for w in WW] )
    
    for t in TT:
        for w in WW:
            if lambda_B[w,t] < 0:
                Delta_down[w,t] = DA_offers[t] - a_RES[w,t]

    Eprofs_w=[ sum(DA_offers[t]*lambda_DA[w,t] + 
                   RES_offers[t]*lambda_RES[w,t] - 
                   (Delta_down[w][t]+a_RES[w,t])*lambda_B[w,t] 
                   for t in TT)
                   for w in WW]

    revenue_DA_cvar = sum( sum(DA_offers * lambda_DA[w,:] * pi for w in WW) )
    revenue_RES_cvar = sum( sum(RES_offers * lambda_RES[w,:] * pi for w in WW) )
    losses_ACT_cvar =  sum( sum(-a_RES[w,:] * lambda_B[w,:] * pi for w in WW) )
    revenue_BAL_cvar= sum( sum(-Delta_down[w,:] * lambda_B[w,:] * pi for w in WW) )

    return a_RES, Delta_down, lambda_offer_RES, Eprofs_w, revenue_DA_cvar, revenue_RES_cvar, losses_ACT_cvar, revenue_BAL_cvar


import pandas as pd
import matplotlib.pyplot as plt
import ast 

# We read the solutions from V4 using the training set 

# The lists were saved as strings (easier to look at in the dataframe)
# ast.literal_eval() is used to convert from '[]' to [].

df_V4_train = pd.read_csv('plots/V4/V4_trained_model.csv')

betas = df_V4_train['Unnamed: 0'].values
betas = [0.0,0.1,0.8]

DA_offers_dict = {f'{betas[i]}':ast.literal_eval(df_V4_train['V4: DA'].iloc[i]) for i in range(len(betas))}
RES_offers_dict = {f'{betas[i]}':ast.literal_eval(df_V4_train['V4: RES'].iloc[i]) for i in range(len(betas))}
alpha_RES_dict = {f'{betas[i]}':ast.literal_eval(df_V4_train['V4: alpha_RES'].iloc[i]) for i in range(len(betas))}
beta_RES_dict = {f'{betas[i]}':ast.literal_eval(df_V4_train['V4: beta_RES'].iloc[i]) for i in range(len(betas))}

# p_RT = p_RT.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)
# lambda_B = lambda_B.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)
# lambda_DA = lambda_DA.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)
# lambda_RES = lambda_RES.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)

p_RT = p_RT.values[:W*T].reshape(W,T)
lambda_B = lambda_B.values[:W*T].reshape(W,T)
lambda_DA = lambda_DA.values[:W*T].reshape(W,T)
lambda_RES = lambda_RES.values[:W*T].reshape(W,T)

Eprofs_w_betas = {f'{beta:.1f}':[] for beta in betas}
revenue_DA_dict = {f'{beta:.1f}':[] for beta in betas} 
revenue_RES_dict = {f'{beta:.1f}':[] for beta in betas}
losses_ACT_dict = {f'{beta:.1f}':[] for beta in betas}
revenue_BAL_dict = {f'{beta:.1f}':[] for beta in betas}

for beta in betas:
    (
        a_RES_sol, Delta_down, lambda_offer_RES, 
     Eprofs_w_betas[f'{beta:.1f}'], 
     revenue_DA_dict[f'{beta:.1f}'], revenue_RES_dict[f'{beta:.1f}'], losses_ACT_dict[f'{beta:.1f}'], revenue_BAL_dict[f'{beta:.1f}']
     ) = evaluate_V4(
         DA_offers_dict[f'{beta}'], RES_offers_dict[f'{beta}'], alpha_RES_dict[f'{beta}'], beta_RES_dict[f'{beta}'],
            lambda_DA, lambda_RES, lambda_B
            )


print(f'The mean profit over all the OOS scenarios for the V4 model is:')
for beta in betas:
    print(f'for beta={beta:.1f} it is:')
    print(pi*sum(Eprofs_w_betas[f'{beta:.1f}']))

for beta in betas:   
    print(f"These are the expected revenue streams for {beta}:")
    print(f"Day-ahead market: {revenue_DA_dict[f'{beta}']:>42.2f} DKK")
    print(f"aFRR capacity market (down): {revenue_RES_dict[f'{beta}']:>31.2f} DKK")
    print(f"Money spent to buy el. back: {losses_ACT_dict[f'{beta}']:>31.2f} DKK")
    print(f"Revenue from balancing market: {revenue_BAL_dict[f'{beta}']:>29.2f} DKK")
    print(f"Summing these together yields the expected profit: {revenue_DA_dict[f'{beta}']+revenue_RES_dict[f'{beta}']+losses_ACT_dict[f'{beta}']+revenue_BAL_dict[f'{beta}']:.2f}={pi*sum(Eprofs_w_betas[f'{beta}']):.2f}")

'''
df_V5_train = pd.read_csv('plots/V5/V5_trained_model.csv')
'''