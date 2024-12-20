import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from IPython.display import display

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES# , gamma_RES
from load_data import train_scenarios, test_scenarios # 30, 5

T=24 #hours that we offer in
W=train_scenarios #scenarios/days, our training set

testing = False # Used for OOS

if testing: 
    W_train = W

    W_test=test_scenarios

    W = W_test
    print(p_RT.shape)
    print(lambda_B.shape)
    p_RT = p_RT.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)
    lambda_B = lambda_B.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)
    lambda_DA = lambda_DA.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)
    lambda_RES = lambda_RES.values[W_train*T:(W_train+W_test)*T].reshape(W_test,T)
else:
    p_RT = p_RT.values[:W*T].reshape(W,T)
    lambda_B = lambda_B.values[:W*T].reshape(W,T)
    lambda_DA = lambda_DA.values[:W*T].reshape(W,T)
    lambda_RES = lambda_RES.values[:W*T].reshape(W,T)

print('W,T=', W,T)

TT=np.arange(T)
WW=np.arange(W)

pi = np.ones(W)/W # Scenarios are assumed to be equiprobable
P_nom = 1 # MW

obj = np.zeros(W)
p_DAs = np.zeros((W,T))
Deltas = np.zeros((W,T))

if testing: # testing here
    M= 4111
else:
    M = max( np.max(lambda_DA-lambda_B), abs(np.min(lambda_DA-lambda_B)) ) + 936 #np.max(lambda_DA-lambda_B)*7 # Used for McCormick relaxation

alpha = 0.9 # Worried about the profits in the 10th percentile least favorable scenarios
beta = 0 # Level of risk-averseness of wind farm owner

# initiate dictionaries used to save the results for different levels of risk
# betas = np.round( np.linspace(0,1,11), 2)
betas = np.round( np.linspace(0,0.8,9), 2) # Selected range
# betas = np.array([0.0,0.1,0.8])

DA_offer = {f'{beta}': float for beta in betas}
RES_offer = {f'{beta}': float for beta in betas}
objs = {f'{beta}': float for beta in betas}
CVaR = {f'{beta}': float for beta in betas}
VaR = {f'{beta}': float for beta in betas}
Eprofs = {f'{beta}': float for beta in betas}
Eprofs_w = {f'{beta}': np.array(float) for beta in betas}
alpha_offer_RES = {f'{beta}': float for beta in betas}
beta_offer_RES = {f'{beta}': float for beta in betas}

revenue_DA_cvar = {f'{beta}':float for beta in betas}
revenue_RES_cvar= {f'{beta}':float for beta in betas}
losses_ACT_cvar ={f'{beta}':float for beta in betas}
revenue_BAL_cvar = {f'{beta}':float for beta in betas}

for beta in betas: 
    #2 Mathematical model
    model = gp.Model("V1")
    p_DA = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="p_DA")
    Delta_down = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="Delta_down")
    # New variables
    p_RES = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="p_RES")
    a_RES = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="a_RES")

    # Used to make strategic balancing price offer
    alpha_RES = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="alpha_RES")
    beta_RES = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="beta_RES")
    #lambda_offer_RES = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="lambda_offer_RES")
    g = model.addMVar((W,T), vtype=GRB.BINARY, name="g")
    phi = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="phi")

    # Used to introduce risk framework: CVaR
    zeta = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="zeta") # "Value-at-Risk"
    eta = model.addVars(W, lb=0, vtype=GRB.CONTINUOUS, name="eta") # Expected profit of each scenario "at-risk"

    model.setObjective( (1-beta) * gp.quicksum(pi[w]*
                                (p_DA[t]*lambda_DA[w,t] + 
                                 p_RES[t]*lambda_RES[w,t] - # New
                                 (Delta_down[w,t]+a_RES[w,t])*lambda_B[w,t]) # Changed
                                    for w in WW for t in TT)+
                        beta * ( zeta - 1/(1-alpha) * gp.quicksum( pi[w] * eta[w] for w in WW) ), 
                                    GRB.MAXIMIZE)

    #1st stage constraints:
    model.addConstrs((p_DA[t] <= P_nom for t in TT), name="c_Nom")
    #2nd stage constraints:
    model.addConstrs((p_RT[w,t] >= p_DA[t] - Delta_down[w,t] - a_RES[w,t] for w in WW for t in TT), name="c_RT") # Changed
    model.addConstrs((             p_DA[t] - Delta_down[w,t] - a_RES[w,t] >= 0 for w in WW for t in TT), name="c_NonnegativePhysicalDelivery") # Changed

    #Strategic offer in the balancing market
        #What we would like to do:
    # model.addConstrs((a_RES[w,t] >= p_RES[t] for w in WW for t in TT if lambda_RES_offer[w,t] <= lambda_DA[w,t]-lambda_B[w,t] and lambda_DA[w,t] > lambda_B[w,t]), name="c_Activation") # New  
        #Though this is not possible given that the strategic offer on the balancing activation price is a variable!
        #McCormick relaxation techniques are used instead:

    model.addConstrs((alpha_RES[t] * (lambda_DA[w,t+1]-lambda_DA[w,t] if t<T-1 else 0) + lambda_DA[w,t] + beta_RES[t] - M*(1-g[w,t]) <= lambda_DA[w,t] - lambda_B[w,t] for w in WW for t in TT), name='c_McCormick_7a_1')
    model.addConstrs((lambda_DA[w,t] - lambda_B[w,t] <= alpha_RES[t] * (lambda_DA[w,t+1]-lambda_DA[w,t] if t<T-1 else 0) + lambda_DA[w,t] + beta_RES[t] + M*g[w,t] for w in WW for t in TT), name='c_McCormick_7a_2')

    # difference in DA price for T[:-1] has been set to 0 as of right now.

    model.addConstrs((a_RES[w,t] <= (phi[w,t] if lambda_DA[w,t] > lambda_B[w,t] else 0) for w in WW for t in TT), name='c_McCormick_7b')
    model.addConstrs((a_RES[w,t] >= (phi[w,t] if lambda_DA[w,t] > lambda_B[w,t] else 0) for w in WW for t in TT), name='c_McCormick_7c')
    model.addConstrs((-g[w,t]*M <= phi[w,t] for w in WW for t in TT), name='c_McCormick_7d_1')
    model.addConstrs((phi[w,t] <= g[w,t]*M for w in WW for t in TT), name='c_McCormick_7d_2')
    model.addConstrs((-(1-g[w,t])*M <= phi[w,t] - p_RES[t] for w in WW for t in TT), name='c_McCormick_7e_1')
    model.addConstrs((phi[w,t] - p_RES[t] <= (1-g[w,t])*M  for w in WW for t in TT), name='c_McCormick_7e_2')

    # Without this constraint we just strategically set the balancing price offer so that we are not activated and then we are "free" to offer as much downward regulation as we want to because phi never interacts with a_RES which means that p_RES never interact with p_DA
    model.addConstrs((p_RES[t] <= p_DA[t] for t in TT), name="c_Nom_RES")

    # CVaR-constraint
    model.addConstrs(( - gp.quicksum(p_DA[t]*lambda_DA[w,t] + 
                                p_RES[t]*lambda_RES[w,t] -
                                (Delta_down[w,t]+a_RES[w,t])*lambda_B[w,t]
                                for t in TT )
                                + zeta - eta[w] <= 0 for w in WW), name="c_CVaR")


    model.optimize()

    if model.status == GRB.OPTIMAL:
        print()
        print(f'------------------- RESULTS  beta={beta} -------------------')
        optimal_objective = model.objVal # Save optimal value of objective function
        print("Optimal objective:", optimal_objective)
        p_DA_sol = [p_DA[t].x for t in TT]
        Delta_down_sol = np.array( [[Delta_down[w,t].x for t in TT] for w in WW] )
        p_RES_sol = [p_RES[t].x for t in TT]
        a_RES_sol = np.array( [[a_RES[w,t].x for t in TT] for w in WW] )
        eta_sol = [eta[w].x for w in WW]

        g_sol = np.array([[g[w,t].x for t in TT] for w in WW])
        phi_sol = np.array([[phi[w,t].x for t in TT] for w in WW])
        
        alpha_RES_sol = np.array([alpha_RES[t].x for t in TT])
        beta_RES_sol = np.array([beta_RES[t].x for t in TT])
        lambda_offer_RES = [[alpha_RES_sol[t] * (lambda_DA[w,t+1]-lambda_DA[w,t] if t<T-1 else 0) + lambda_DA[w,t] + beta_RES_sol[t] for t in TT] for w in WW]
        

        objs[f'{beta}']=optimal_objective
        Eprofs[f'{beta}']=  sum(pi[w]*
                               (p_DA_sol[t]*lambda_DA[w,t] + 
                                p_RES_sol[t]*lambda_RES[w,t] - 
                                (Delta_down_sol[w,t]+a_RES_sol[w,t])*lambda_B[w,t]) 
                            for w in WW for t in TT)
        Eprofs_w[f'{beta}']=[ sum(p_DA_sol[t]*lambda_DA[w,t] + 
                                   p_RES_sol[t]*lambda_RES[w,t] - 
                                   (Delta_down_sol[w,t]+a_RES_sol[w,t])*lambda_B[w,t] 
                                   for t in TT)
                            for w in WW]
        VaR[f'{beta}']=zeta.x
        CVaR[f'{beta}']=VaR[f'{beta}'] - 1/(1-alpha)*sum(pi[w] * eta_sol[w] for w in WW)
        
        DA_offer[f'{beta}'] = p_DA_sol
        RES_offer[f'{beta}'] = p_RES_sol
        alpha_offer_RES[f'{beta}'] = alpha_RES_sol
        beta_offer_RES[f'{beta}'] = beta_RES_sol

        revenue_DA_cvar[f'{beta}'] = sum( sum(p_DA_sol * lambda_DA[w,:] * pi[w] for w in WW) )
        revenue_RES_cvar[f'{beta}'] = sum( sum(p_RES_sol * lambda_RES[w,:] * pi[w] for w in WW) )
        losses_ACT_cvar[f'{beta}'] =  sum( sum(-a_RES_sol[w,:] * lambda_B[w,:] * pi[w] for w in WW) )
        revenue_BAL_cvar[f'{beta}']= sum( sum(-Delta_down_sol[w,:] * lambda_B[w,:] * pi[w] for w in WW) )
    else:
            print("Optimization was not successful")

lambda_offer_RES_dict = {f'{beta}':[[alpha_offer_RES[f'{beta}'][t] * (lambda_DA[w,t+1]-lambda_DA[w,t] if t<T-1 else 0) + lambda_DA[w,t] + beta_offer_RES[f'{beta}'][t] for w in WW] for t in TT] for beta in betas}

# Prints below are relevant but crowd the prompt. If the reader is curious they may be uncommented.
'''
# Where do we earn revenue?
revenue_DA =  sum( sum(p_DA_sol * lambda_DA[w,:] * pi[w] for w in WW) )
revenue_RES = sum( sum(p_RES_sol * lambda_RES[w,:] * pi[w] for w in WW) )
losses_ACT =  sum( sum(-a_RES_sol[w,:] * lambda_B[w,:] * pi[w] for w in WW) )
revenue_BAL = sum( sum(-Delta_down_sol[w,:] * lambda_B[w,:] * pi[w] for w in WW) )

print('These are the expected revenue streams:')
print(f'Day-ahead market: {revenue_DA:>42.2f} DKK')
print(f'aFRR capacity market (down): {revenue_RES:>31.2f} DKK')
print(f'Money spent to buy el. back: {losses_ACT:>31.2f} DKK')
print(f'Revenue from balancing market: {revenue_BAL:>29.2f} DKK')
print(f'Summing these together yields the expected profit: {revenue_DA+revenue_RES+losses_ACT+revenue_BAL:.2f}={optimal_objective:.2f}')

print(f'Such high balancing market offers allow us only to be activated this many times in each scenario: {np.sum( lambda_offer_RES <= lambda_B, axis=0)}')
print(f'Instead of simply: {np.sum( lambda_DA > lambda_B, axis=0)}')

print('#activated in each hour, a:\n', np.sum( a_RES_sol > 0, axis=0))
print('This does not add up with the number of times that we enforce the activation, g:\n', np.sum( g_sol > 0, axis=0))
print('Though the numbers for g match nicely with the auxiliary variable for activation, phi:\n', np.sum( phi_sol > 0, axis=0))
print('Discrepancies can be explained by the number of times where phi > a:\n', np.sum( phi_sol > a_RES_sol, axis=0))
print('These discrepancies between phi and a can be explained by the number of times where g=1 but there is no need for down-regulation, phi-condtional\n', np.sum( (phi_sol > 0) * (lambda_DA <= lambda_B), axis=0))

print('In other words, changing the balancing activation offer price works, and the conditions are that there should be a need for down-regulation and the offer price should be smaller than or equal to that of the difference between DA and BAL:')
print('This is the same as the number of times that we are activated (without equality), lambda_offer:\n', np.sum( (lambda_DA > lambda_B) * (lambda_DA - lambda_B > lambda_offer_RES), axis=0))
print('#activated in each hour, a:\n', np.sum( a_RES_sol > 0, axis=0))
print('Apart from a difference of \"1" in hour 9 for some reason')
'''

# Visualizations
import matplotlib.pyplot as plt

print('##############\nVISUALIZATION:\n##############')
import matplotlib.pyplot as plt

beta_lists=[betas,betas[1:]]

fig, ax = plt.subplots(1,2, figsize=(14,5))
ax = ax.flatten()
for i,beta_list in enumerate(beta_lists):
    for beta in beta_list:
        ax[i].scatter(CVaR[f'{beta}'], Eprofs[f'{beta}'], s=45)
    points_to_plot = beta_list #[0.8,0.9]
    for beta in points_to_plot:
        ax[i].annotate(f'beta={beta}', (CVaR[f'{beta}'], Eprofs[f'{beta}']), fontsize=13)
    ax[i].set_xlabel('CVaR [DKK]')
    ax[i].set_ylabel('Expected profit [DKK]')
    #plt.legend()
plt.tight_layout()
plt.savefig('plots/V4/Step4_V4_Markowitzs', dpi=500, bbox_inches='tight')
plt.show()

import seaborn as sns
betas_hist=[[0.0,0.1],[0.0,0.8]]
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax=ax.flatten()
colors=['tab:blue', 'tab:orange']
for k,beta_hist in enumerate(betas_hist):
    if k == 1:
        colors[1] = 'tab:olive'
    for i,beta in enumerate(beta_hist):
        sns.histplot(Eprofs_w[f'{beta}'],
                    ax=ax[k],
                    color=colors[i],
                    kde=False,
                    label=f'β={beta}',
                    alpha=.9)
        ax[k].axvline(VaR[f'{beta}'],
                    color=colors[i],
                    label=f'VaR, β={beta}',
                    linestyle='--',
                    linewidth=4,
                    alpha=1-.1*i
                    )
        ax[k].axvline(Eprofs[f'{beta}'],
                    color=colors[i],
                    label=f'Expected profit, β={beta}',
                    linestyle='-',
                    linewidth=4,
                    alpha=1-.1*i
                    )
        
    ax[k].legend()
    ax[k].set_xlabel('Expected profit [DKK]')
    ax[k].set_ylabel(f'Frequency [out of {W}]')
plt.tight_layout()
plt.savefig('plots/V4/Step4_V4_profit_hists', dpi=500, bbox_inches='tight')
plt.show()
#####

import matplotlib.cm as cm

cols = cm.tab10.colors

fig,ax = plt.subplots(figsize=(6,4),dpi=500)

markers=['s','o','x','d','p','+']
for i,beta in enumerate(betas[[0,1,-1]]):
    ax.plot(TT, DA_offer[f'{beta}'], label=r'$\beta$'+f'={beta}', marker=markers[i], color=cols[i])
    # ax.plot(TT, RES_offer[f'{beta}'], label=f'beta={beta}', marker=markers[i], color=cols2[i], alpha=.5)

ax.set_xlabel('Hour of the day [h]')

ax.set_ylabel('Offer (DA and RES) $-$ Power [MW]')

lines,labels = ax.get_legend_handles_labels()

### Used to explain the behaviour - can be removed when only showing the decisions
ax2=ax.twinx()
ax2.plot([np.mean(lambda_DA[:,t]) + np.mean(lambda_RES[:,t]) - np.mean(lambda_B[:,t]) for t in range(T)], label='E[P] 1MW DA', linestyle='dashed', color=cols[7], alpha=1.0)
ax2.set_ylabel('Expected profit of 1 MW DA over-offer [DKK]')
ax2.axhline(y=0, alpha=.8, color=cols[7], linestyle='dotted')
lines2,labels2 = ax2.get_legend_handles_labels()
plt.legend(lines+lines2,labels+labels2,loc=0)
plt.tight_layout()
plt.savefig('plots/V4/Step4_V4_decisions', dpi=500, bbox_inches='tight')
plt.show()


fig,ax = plt.subplots(figsize=(6,4),dpi=500)
ax2=ax.twinx()
for i,beta in enumerate(betas[[0,1,-1]]):
    ax.plot(TT, alpha_offer_RES[f'{beta}'], label=f'beta={beta}', marker=markers[i], color=cols[0])
    ax2.plot(TT, beta_offer_RES[f'{beta}'], label=f'beta={beta}', marker=markers[i], color=cols[1], alpha=.5)

ax.set_xlabel('Hour of the day [h]')
ax2.spines['left'].set_color(cols[0])
ax.tick_params(axis='y', colors=cols[0])
ax.yaxis.label.set_color(cols[0])
ax2.tick_params(axis='y', colors=cols[1])
ax2.spines['right'].set_color(cols[1])
ax2.yaxis.label.set_color(cols[1])
ax.set_ylabel(r' $ \alpha^{RES}_t [-]$')
ax2.set_ylabel(r' $ \beta^{RES}_t$ [DKK/MWh]')
lines,labels = ax.get_legend_handles_labels()
lines2,labels2= ax2.get_legend_handles_labels()
plt.legend(lines+lines2,labels+labels2,loc=0)
plt.tight_layout()
plt.savefig('plots/V4/Step4_V4_decisions_alphabeta_RES', dpi=500, bbox_inches='tight')
plt.show()

'''
# Looking at their boxplots
fig, ax = plt.subplots(1,2,figsize=(14,5))
beta_vals = [0.0,0.8]
ylims = (-2000,8000)
colors=['tab:blue','tab:olive']
for i in range(len(beta_vals)):
    ax[i].boxplot(lambda_offer_RES_dict[f'{beta_vals[i]}'])
    ax[i].set_xlabel('Time of day-1 [h]')
    ax[i].set_ylabel('Strategic activation price offer [DKK/MWh] - ' + r'$\beta$' + f'={beta_vals[i]}')
    ax[i].set_ylim(ylims)
    #ax[i].title('Boxplot for the scenarios in each hour of the profit made by 1 MW offer in DA')
plt.savefig('plots/V4/Step4_V4_alphabetaRES_spread', dpi=500, bbox_inches='tight')
plt.show()

# Understanding the spread within our scenarios
fig, ax = plt.subplots(2,2,figsize=(8,6))
ax = ax.flatten()
dict_params = {'$\lambda^{DA}$': lambda_DA, '$\lambda^{RES}$': lambda_RES, '$\lambda^{B}$':lambda_B, '$p^{RT}$':p_RT}
for i,k in enumerate(dict_params.keys()):
     ax[i].boxplot(dict_params[k])
     ax[i].set_title(k)
     ax[i].tick_params('x',rotation=45)
plt.tight_layout()
plt.show()
'''


plt.boxplot([lambda_DA[:,t] + lambda_RES[:,t] - lambda_B[:,t] for t in TT])
plt.xlabel('Time of day-1 [h]')
plt.ylabel('Revenue - Expected profit of 1 MW DA offer [DKK]')
plt.tight_layout()
plt.savefig('plots/V4/Step4_V4_EP1MWDA_spread', dpi=500, bbox_inches='tight')
plt.title('Boxplot for the scenarios in each hour of the profit made by 1 MW offer in DA')
plt.show()

for beta in betas:   
    print(f"These are the expected revenue streams for {beta}:")
    print(f"Day-ahead market: {revenue_DA_cvar[f'{beta}']:>42.2f} DKK")
    print(f"aFRR capacity market (down): {revenue_RES_cvar[f'{beta}']:>31.2f} DKK")
    print(f"Money spent to buy el. back: {losses_ACT_cvar[f'{beta}']:>31.2f} DKK")
    print(f"Revenue from balancing market: {revenue_BAL_cvar[f'{beta}']:>29.2f} DKK")
    print(f"Summing these together yields the expected profit: {revenue_DA_cvar[f'{beta}']+revenue_RES_cvar[f'{beta}']+losses_ACT_cvar[f'{beta}']+revenue_BAL_cvar[f'{beta}']:.2f}={Eprofs[f'{beta}']:.2f}")

import matplotlib.cm as cm
x_labels = ['Total expected profits', 'Day-ahead', 'aFRR capacity (down)', 'aFRR activation', 'Balancing']
x = np.arange(len(x_labels))
bar_width=0.1
revenue_streams = [Eprofs, revenue_DA_cvar, revenue_RES_cvar, losses_ACT_cvar, revenue_BAL_cvar]
colors = {betas[i]:cm.tab10.colors[i] for i in range(len(betas))}
fig, ax = plt.subplots(figsize=(8,6))
for i,beta in enumerate(betas):
     offset = (i-len(betas) / 2) * bar_width + bar_width / 2
     ax.bar(
          x + offset,
          [market[f'{beta}'] for market in revenue_streams],
          width=bar_width,
          color=colors[beta],
          label=r'$\beta$='+f'{beta}'
     )
ax.set_xticks(x)
ax.set_xticklabels(x_labels,rotation=15)
ax.set_ylabel('Expected revenue [DKK]')
ax.legend()
ax.grid(axis='y',linestyle='--', alpha=.5)

plt.tight_layout()
plt.savefig('plots/V4/Step4_V4_expectedProfits',dpi=500, bbox_inches='tight')
plt.show()

# Save the solution to a .csv file
import csv

alpha_offer_RES_ = {k:alpha_offer_RES[k].tolist() for k in alpha_offer_RES.keys()}
beta_offer_RES_ = {k:beta_offer_RES[k].tolist() for k in beta_offer_RES.keys()}

df_V4_train = pd.DataFrame.from_dict([DA_offer,RES_offer,alpha_offer_RES_,beta_offer_RES_])
df_V4_train = df_V4_train.T
df_V4_train.columns = ['V4: '+var for var in ['DA', 'RES', 'alpha_RES', 'beta_RES']]

# Don't overwrite it constantly - only do it when divided into training and test sets
'''
df_V4_train.to_csv("plots/V4/V4_trained_model.csv", header=True)
'''
print('##############\nScript is done\n##############')
