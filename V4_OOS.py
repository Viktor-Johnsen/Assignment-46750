
# This file is used to test the solutions from V4 out-of-sample (OOS)

import numpy as np
import pandas as pd

# First we define the functions used to calculate the profits in each scenrio

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES# , gamma_RES
from load_data import train_scenarios, test_scenarios # 30, 5

testing = True

T = 24

W_train = train_scenarios
W_test = test_scenarios

df_V4_train = pd.read_csv('plots/V4/V4_trained_model.csv')

betas = df_V4_train['Unnamed: 0'].values
betas = [0.0,0.1,0.8]

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


import matplotlib.pyplot as plt
import ast 

# We read the solutions from V4 using the training set 

# The lists were saved as strings (easier to look at in the dataframe)
# ast.literal_eval() is used to convert from '[]' to [].

DA_offers_dict = {f'{betas[i]}':ast.literal_eval(df_V4_train['V4: DA'].iloc[i]) for i in range(len(betas))}
RES_offers_dict = {f'{betas[i]}':ast.literal_eval(df_V4_train['V4: RES'].iloc[i]) for i in range(len(betas))}
alpha_RES_dict = {f'{betas[i]}':ast.literal_eval(df_V4_train['V4: alpha_RES'].iloc[i]) for i in range(len(betas))}
beta_RES_dict = {f'{betas[i]}':ast.literal_eval(df_V4_train['V4: beta_RES'].iloc[i]) for i in range(len(betas))}

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


import matplotlib.cm as cm
x_labels = [f'w={w}' for w in WW]
x = np.arange(len(x_labels))
bar_width=0.1
colors = {betas[i]:cm.tab10.colors[i] for i in range(len(betas))}
fig, ax = plt.subplots(figsize=(8,6))
for i,beta in enumerate(betas):
     offset = (i-len(betas) / 2) * bar_width + bar_width / 2
     ax.bar(
          x + offset,
          [Eprofs_w_betas[f'{beta}'][w] for w in WW],
          width=bar_width,
          color=colors[beta],
          label=r'$\beta$='+f'{beta}'
     )
ax.set_xticks(x)
ax.set_xticklabels(x_labels,rotation=15)
ax.set_ylabel('Realized revenue [DKK]')
ax.legend()
ax.grid(axis='y',linestyle='--', alpha=.5)

plt.tight_layout()
plt.savefig('plots/V4/Step4_V4_OOS',dpi=500, bbox_inches='tight')
plt.show()
# Plot lambda_offer_RES vs lambda_B for each beta
fig, ax = plt.subplots(figsize=(8,6))
for i, beta in enumerate(betas):
    lambda_offer_RES = np.array([lambda_offer_RES[w] for w in WW])
    lambda_B_values = np.array([lambda_B[w] for w in WW])
    for t in TT:
        ax.plot(lambda_offer_RES[:, t], lambda_B_values[:, t], 'o', label=f'beta={beta:.1f}, t={t}' if i == 0 else "")

# Plot the line lambda_B = lambda_offer_RES
min_val = min(lambda_B_values.min(), lambda_offer_RES.min())
max_val = max(lambda_B_values.max(), lambda_offer_RES.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='lambda_B = lambda_offer_RES')

ax.set_xlabel('Lambda Offer RES')
ax.set_ylabel('Lambda B')
ax.set_xlim(right=2500)  # Restrict x axis to be max 2500
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('plots/V4/Lambda_offer_RES_vs_Lambda_B', dpi=500, bbox_inches='tight')
plt.show()