
# This file is used to test the solutions from V5 out-of-sample (OOS)

import numpy as np
import pandas as pd

# First we define the functions used to calculate the profits in each scenrio

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES# , gamma_RES
from load_data import train_scenarios, test_scenarios # 30, 7

testing = False

T = 24

W_train = train_scenarios
W_test = test_scenarios

df_V5_train = pd.read_csv('plots/V5/V5_trained_model.csv')

epsilons = [0.0, 0.1, 1.0]
#betas = [0.0,0.1,0.8]

if testing:
    W = W_test
    p_RT = p_RT.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)
    lambda_B = lambda_B.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)
    lambda_DA = lambda_DA.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)
    lambda_RES = lambda_RES.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)
else:
    W = W_train
    p_RT = p_RT.values[:W*T].reshape(W,T)
    lambda_B = lambda_B.values[:W*T].reshape(W,T)
    lambda_DA = lambda_DA.values[:W*T].reshape(W,T)
    lambda_RES = lambda_RES.values[:W*T].reshape(W,T)

TT = range(T)
WW = range(W)

pi = 1/W

def evaluate_V5(DA_offers, RES_offers,
                lambda_DA, lambda_RES, lambda_B,
                k_eps):
    a_RES = np.zeros((W,T))
    a_RES[np.round(lambda_DA,4) > np.round(lambda_B,4)] = 1
    for t in TT:
        for w in WW:
            a_RES[w,t] = min(a_RES[w,t], RES_offers[t]) # Shouldn't be 1 if less was offered

    if k_eps == 'P0':
        a_RES = np.zeros((W,T)) # -> in this case we should not ever have to activate
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

    return a_RES, Delta_down, Eprofs_w, revenue_DA_cvar, revenue_RES_cvar, losses_ACT_cvar, revenue_BAL_cvar


import matplotlib.pyplot as plt
import ast 

# We read the solutions from V4 using the training set 

# The lists were saved as strings (easier to look at in the dataframe)
# ast.literal_eval() is used to convert from '[]' to [].

DA_offers_dict = {f'P{(1-epsilon)*100:.0f}':ast.literal_eval(df_V5_train['V5: DA'].iloc[i]) for i,epsilon in zip(range(len(epsilons)),epsilons)}
RES_offers_dict = {f'P{(1-epsilon)*100:.0f}':ast.literal_eval(df_V5_train['V5: RES'].iloc[i]) for i,epsilon in zip(range(len(epsilons)),epsilons)}

keys_epsilon = [f'P{(1-epsilon)*100:.0f}' for epsilon in epsilons]
Eprofs_w_epsilons = {k:[] for k in keys_epsilon}
revenue_DA_dict = {k:[] for k in keys_epsilon} 
revenue_RES_dict = {k:[] for k in keys_epsilon}
losses_ACT_dict = {k:[] for k in keys_epsilon}
revenue_BAL_dict = {k:[] for k in keys_epsilon}

for k_eps in keys_epsilon:
    (
        a_RES_sol, Delta_down, 
     Eprofs_w_epsilons[k_eps], 
     revenue_DA_dict[k_eps], revenue_RES_dict[k_eps], losses_ACT_dict[k_eps], revenue_BAL_dict[k_eps]
     ) = evaluate_V5(
         DA_offers_dict[k_eps], RES_offers_dict[k_eps],
            lambda_DA, lambda_RES, lambda_B, k_eps
            )


print(f'The mean profit over all the OOS scenarios for the V4 model is:')
for epsilon in epsilons:
    print(f'for epsilon={epsilon:.0f} ({f"P{(1-epsilon)*100:.0f}"})it is:')
    print(pi*sum(Eprofs_w_epsilons[f'P{(1-epsilon)*100:.0f}']))

for k_eps in keys_epsilon:   
    print(f"These are the expected revenue streams for {k_eps}:")
    print(f"Day-ahead market: {revenue_DA_dict[k_eps]:>42.2f} DKK")
    print(f"aFRR capacity market (down): {revenue_RES_dict[k_eps]:>31.2f} DKK")
    print(f"Money spent to buy el. back: {losses_ACT_dict[k_eps]:>31.2f} DKK")
    print(f"Revenue from balancing market: {revenue_BAL_dict[k_eps]:>29.2f} DKK")
    print(f"Summing these together yields the expected profit: {revenue_DA_dict[k_eps]+revenue_RES_dict[k_eps]+losses_ACT_dict[k_eps]+revenue_BAL_dict[k_eps]:.2f}={pi*sum(Eprofs_w_epsilons[k_eps]):.2f}")
