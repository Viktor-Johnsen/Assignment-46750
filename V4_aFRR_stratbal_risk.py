import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from IPython.display import display

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES# , gamma_RES

show_plots = True # Usedd to toggle between plotting and not plotting...

T=24 #hours that we offer in
W=30 #scenarios/days, our training set

p_RT = p_RT.values[:W*T].reshape(W,T)
lambda_B = lambda_B.values[:W*T].reshape(W,T)
lambda_DA = lambda_DA.values[:W*T].reshape(W,T)
lambda_RES = lambda_RES.values[:W*T].reshape(W,T)
# gamma_RES = gamma_RES.values[:W*T].reshape(W,T)

TT=np.arange(T)
WW=np.arange(W)

pi = np.ones(W)/W # Scenarios are assumed to be equiprobable
P_nom = 1 # MW

obj = np.zeros(W)
p_DAs = np.zeros((W,T))
Deltas = np.zeros((W,T))

#lambda_B[lambda_B <= 0] = 0 # Just used to check smth

# gamma_RES = np.ones((W,T)) # Down-regulation activated in all hours
# gamma_RES[lambda_B > lambda_DA]=0 # Down-regulation not activated in hours where balancing price is higher than DA price

M = max( np.max(lambda_DA-lambda_B), abs(np.min(lambda_DA-lambda_B)) ) + 936 #np.max(lambda_DA-lambda_B)*7 # Used for McCormick relaxation

alpha = 0.9 # Worried about the profits in the 10th percentile least favorable scenarios
beta = 0 # Level of risk-averseness of wind farm owner

# initiate dictionaries used to save the results for different levels of risk
# betas = np.round( np.linspace(0,1,11), 2)
betas = np.round( np.linspace(0,0.8,9), 2) # Selected range

DA_offer = {f'{beta}': float for beta in betas}
RES_offer = {f'{beta}': float for beta in betas}
objs = {f'{beta}': float for beta in betas}
CVaR = {f'{beta}': float for beta in betas}
VaR = {f'{beta}': float for beta in betas}
Eprofs = {f'{beta}': float for beta in betas}
Eprofs_w = {f'{beta}': np.array(float) for beta in betas}
alpha_offer_RES = {f'{beta}': float for beta in betas}
beta_offer_RES = {f'{beta}': float for beta in betas}

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
        # lambda_offer_RES === alpha_RES * ( (lambda_DA[w,t+1] if t<T-1 else lambda_DA[w,t])-lambda_DA[w,t]) + lambda_DA[w,t] + beta_RES

    # model.addConstrs((lambda_offer_RES[w,t] - M*(1-g[w,t]) <= lambda_DA[w,t] - lambda_B[w,t] for w in WW for t in TT), name='c_McCormick_7a_1')
    # model.addConstrs((lambda_DA[w,t] - lambda_B[w,t] <= lambda_offer_RES[w,t] + M*g[w,t] for w in WW for t in TT), name='c_McCormick_7a_2')
    model.addConstrs((alpha_RES[t] * (lambda_DA[w,t+1]-lambda_DA[w,t] if t<T-1 else 0) + lambda_DA[w,t] + beta_RES[t] - M*(1-g[w,t]) <= lambda_DA[w,t] - lambda_B[w,t] for w in WW for t in TT), name='c_McCormick_7a_1')
    model.addConstrs((lambda_DA[w,t] - lambda_B[w,t] <= alpha_RES[t] * (lambda_DA[w,t+1]-lambda_DA[w,t] if t<T-1 else 0) + lambda_DA[w,t] + beta_RES[t] + M*g[w,t] for w in WW for t in TT), name='c_McCormick_7a_2')

    # WHAT ABOUT THE T[:-1]??? -> I just set it to 0 as of right now.

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

        # print(f'p_DA={p_DA_sol}')
        # print()
        # print(f'p_RES={p_RES_sol}')
        # print()
        # [print(Delta_down_sol[w,:].tolist()) for w in WW]
        # print()
        # [print(a_RES_sol[w,:].tolist()) for w in WW]
        #lambda_offer_RES_sol = [[lambda_offer_RES[w,t].x for t in TT] for w in WW]
        # print('We strategically offer the balancing activation price as: ', )
        g_sol = np.array([[g[w,t].x for t in TT] for w in WW])
        phi_sol = np.array([[phi[w,t].x for t in TT] for w in WW])
        # alpha_RES_sol = np.array([[alpha_RES[w,t].x for t in TT] for w in WW])
        # beta_RES_sol = np.array([[beta_RES[w,t].x for t in TT] for w in WW])
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
        #print(f'Deltap=\n{deltap_sol[:10]}\n{deltap_sol[10:]}')
        #print(f'zB=\n{zB_sol[:10]}\n{zB_sol[10:]}')
        DA_offer[f'{beta}'] = p_DA_sol
        RES_offer[f'{beta}'] = p_RES_sol
        alpha_offer_RES[f'{beta}'] = alpha_RES_sol
        beta_offer_RES[f'{beta}'] = beta_RES_sol

    else:
            print("Optimization was not successful")

lambda_offer_RES_dict = {f'{beta}':[[alpha_offer_RES[f'{beta}'][t] * (lambda_DA[w,t+1]-lambda_DA[w,t] if t<T-1 else 0) + lambda_DA[w,t] + beta_offer_RES[f'{beta}'][t] for w in WW] for t in TT] for beta in betas}

#model.printStats()
#display(P_RT_w)
# print("Expected profit (Optimal objective):", optimal_objective)
print("Strategic balancing offer: \n", lambda_offer_RES)
# Where do we earn revenue?
revenue_DA =  sum( sum(p_DA_sol * lambda_DA[w,:] * pi[w] for w in WW) )
revenue_RES = sum( sum(p_RES_sol * lambda_RES[w,:] * pi[w] for w in WW) )
losses_ACT =  sum( sum(-a_RES_sol[w,:] * lambda_B[w,:] * pi[w] for w in WW) )
revenue_BAL = sum( sum(-Delta_down_sol[w,:] * lambda_B[w,:] * pi[w] for w in WW) )

print('These are the expected revenue streams:')
print(f'Day-ahead market: {revenue_DA:>42.2f} €')
print(f'aFRR capacity market (down): {revenue_RES:>31.2f} €')
print(f'Money spent to buy el. back: {losses_ACT:>31.2f} €')
print(f'Revenue from balancing market: {revenue_BAL:>29.2f} €')
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
#print('This is the same as the number of times that we are activated (with equality), lambda_OFFER:\n', np.sum( (lambda_DA > lambda_B) * (lambda_DA - lambda_B >= lambda_offer_RES), axis=0))
print('Apart from a difference of \"1" in hour 9 for some reason')


# Visualizations
import matplotlib.pyplot as plt

if show_plots:
        '''fig, ax=plt.subplots(figsize=(6,4),dpi=500)
        # p_RT.shape i 30 by 24 but for some reason it p_RT[t,:] is correct below and not p_RT[:,t]
        for t in range(T): ax.plot(p_RT[t,:], label='$p^{RT}_{\omega,t}$' if t == 0 else None, alpha=0.5, color='tab:red')
        ax.plot(p_DA_sol, label='$p^{DA}_t$', alpha=.8, color='tab:green')
        ax.plot(p_RES_sol, label='$p^{RES}_t$', alpha=.8, color='tab:purple')
        ax.legend(loc=0)
        ax.set_xlabel('Hour of the day [h]')
        ax.set_ylabel('Power [MW]')
        plt.title('Examining the offer decisions: p_RES and p_RT')
        plt.show()'''

        '''fig, ax=plt.subplots(figsize=(6,4),dpi=500)
        for t in range(T): ax.plot(a_RES_sol[t,:], label='$a^{RES}_{\omega,t}$' if t == 0 else None, alpha=0.5, color='tab:red')
        ax.plot(p_DA_sol, label='$p^{DA}_t$', alpha=.8, color='tab:green')
        ax.plot(p_RES_sol, label='$p^{RES}_t$', alpha=.8, color='tab:purple')
        ax.legend(loc=0)
        ax.set_xlabel('Hour of the day [h]')
        ax.set_ylabel('Power [MW]')
        plt.title('Examining the offer decisions: p_RES and a_RES')
        plt.show()'''

        fig, ax=plt.subplots(figsize=(6,4),dpi=500)
        ax.plot(p_DA_sol, label='$p^{DA}_t$', alpha=.8, color='tab:green')
        ax.plot(p_RES_sol, label='$p^{RES}_t$', alpha=.8, color='tab:purple')
        ax.plot([np.mean([p_RT[:,t]]) for t in range(T)], label='$\overline{p}^{RT}_t$', color='tab:red')

        ax2 = ax.twinx()
        ax2.plot([np.mean([lambda_DA[:,t]]) for t in range(T)] / np.array( [np.mean([lambda_B[:,t]]) for t in range(T)] ), label='$\overline{\lambda}^{DA}_t$ / $\overline{\lambda}^{B}_t$')
        ax2.plot([np.mean([lambda_RES[:,t]]) for t in range(T)] / np.array( [np.mean([lambda_DA[:,t]]) for t in range(T)] ), label='$\overline{\lambda}^{RES}_t$ / $\overline{\lambda}^{DA}_t$')

        lines, labels = ax.get_legend_handles_labels()
        lines2,labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines+lines2,labels+labels2,loc=5)

        ax.set_xlabel('Hour of the day [h]')
        ax.set_ylabel('Power [MW]')
        ax2.set_ylabel('Price ratio [-]')

        plt.title('Offers in DA and RES and the ratio between DA- and BAL-prices')
        plt.show()

        '''fig, ax=plt.subplots(figsize=(6,4),dpi=500)
        ax.plot(p_DA_sol, label='$p^{DA}_t$', alpha=.8, color='tab:green')
        ax.plot(p_RES_sol, label='$p^{RES}_t$', alpha=.8, color='tab:purple')
        ax.plot([np.mean([p_RT[:,t]]) for t in range(T)], label='$\overline{p}^{RT}_t$', color='tab:red')

        ax2 = ax.twinx()
        ax2.plot([np.mean(lambda_DA[:,t]) + np.mean(lambda_RES[:,t]) - np.mean(lambda_B[:,t]) for t in range(T)], label='E[P] 1MW DA')
        ax2.axhline(y=0, alpha=.5, color='black')

        lines, labels = ax.get_legend_handles_labels()
        lines2,labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines+lines2,labels+labels2,loc=5)

        ax.set_xlabel('Hour of the day [h]')
        ax.set_ylabel('Power [MW]')
        ax2.set_ylabel('Revenue - Expected profit of 1 MW DA offer [€]')

        plt.title('Offers in DA and RES and the ratio between DA- and BAL-prices')
        plt.show()'''

print('##############\nVISUALIZATION:\n##############')
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

beta_lists=[betas,betas[1:]]

fig, ax = plt.subplots(1,2, figsize=(14,5))
ax = ax.flatten()
for i,beta_list in enumerate(beta_lists):
    for beta in beta_list:
        ax[i].scatter(CVaR[f'{beta}'], Eprofs[f'{beta}'], s=45)
    points_to_plot = beta_list #[0.8,0.9]
    for beta in points_to_plot:
        ax[i].annotate(f'beta={beta}', (CVaR[f'{beta}'], Eprofs[f'{beta}']), fontsize=13)
    ax[i].set_xlabel('CVaR [€]')
    ax[i].set_ylabel('Expected profit [€]')
    #plt.legend()
plt.tight_layout()
plt.savefig('plots/V4/Step4_V4_Markowitzs', dpi=500, bbox_inches='tight')
plt.show()

import seaborn as sns

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

#####
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
    ax[k].set_xlabel('Expected profit [€]')
    ax[k].set_ylabel(f'Frequency [out of {W}]')
    # ax[k].set_title('Profit distributions and VaR')
plt.tight_layout()
plt.savefig('plots/V4/Step4_V4_profit_hists', dpi=500, bbox_inches='tight')
plt.show()
#####


fig,ax = plt.subplots(figsize=(6,4),dpi=500)
ax2=ax.twinx()

cols = ['b', 'r']
markers=['s','x','*','d','p','+']
for i,beta in enumerate(betas[[0,1,-1]]):
    ax.plot(TT, DA_offer[f'{beta}'], label=f'beta={beta}', marker=markers[i], color=cols[0])
    ax2.plot(TT, RES_offer[f'{beta}'], label=f'beta={beta}', marker=markers[i], color=cols[1], alpha=.5)

ax.set_xlabel('Hour of the day [h]')

ax2.spines['left'].set_color(cols[0])
ax.tick_params(axis='y', colors=cols[0])
ax.yaxis.label.set_color(cols[0])

ax2.tick_params(axis='y', colors=cols[1])
ax2.spines['right'].set_color(cols[1])
ax2.yaxis.label.set_color(cols[1])

ax.set_ylabel('DA offer $-$ Power [MW]')
ax2.set_ylabel('RES offer $-$ Power [MW]')

lines,labels = ax.get_legend_handles_labels()
lines2,labels2= ax2.get_legend_handles_labels()

### Used to explain the behaviour - can be removed when only showing the decisions
ax3 = ax.twinx()
ax3.plot([np.mean(lambda_DA[:,t]) + np.mean(lambda_RES[:,t]) - np.mean(lambda_B[:,t]) for t in range(T)], label='E[P] 1MW DA', color='k', alpha=.5)
ax3.set_ylabel('Revenue - Expected profit of 1 MW DA offer [€]')
ax3.axhline(y=0, alpha=1, color='k', linestyle='dashed')
ax3.spines.right.set_position(("axes", 1.12))
lines3,labels3 = ax3.get_legend_handles_labels()
plt.legend(lines+lines2+lines3,labels+labels2+labels3,loc=0)
### 

# plt.legend(lines+lines2,labels+labels2,loc=0)
plt.savefig('plots/V4/Step4_V4_decisions', dpi=500, bbox_inches='tight')
plt.show()

fig,ax = plt.subplots(figsize=(6,4),dpi=500)
ax2=ax.twinx()
cols = ['b', 'r']
markers=['s','x','*','d','p','+']
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
ax2.set_ylabel(r' $ \beta^{RES}_t$ [€/MWh]')
lines,labels = ax.get_legend_handles_labels()
lines2,labels2= ax2.get_legend_handles_labels()
plt.legend(lines+lines2,labels+labels2,loc=0)
plt.savefig('plots/V4/Step4_V4_decisions_alphabeta_RES', dpi=500, bbox_inches='tight')
plt.show()

# Looking at their boxplots

fig, ax = plt.subplots(1,2,figsize=(14,5))
beta_vals = [0.0,0.8]
ylims = (-2000,8000)
colors=['tab:blue','tab:olive']
for i in range(len(beta_vals)):
    ax[i].boxplot(lambda_offer_RES_dict[f'{beta_vals[i]}'])
    ax[i].set_xlabel('Time of day-1 [h]')
    ax[i].set_ylabel('Strategic activation price offer [€/MWh] - ' + r'$\beta$' + f'={beta_vals[i]}')
    ax[i].set_ylim(ylims)
    #ax[i].title('Boxplot for the scenarios in each hour of the profit made by 1 MW offer in DA')
plt.savefig('plots/V4/Step4_V4_alphabetaRES_spread', dpi=500, bbox_inches='tight')
plt.show()

print('##############\nScript is done\n##############')

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

# Or simply all-in-one go:

# np.median(np.array([lambda_DA[:,t] + lambda_RES[:,t] - lambda_B[:,t] for t in range(T)]).reshape(T*W)) = 129
# Since the median of the scenario-hour profits is only 129 and basically all points are close to 0 we can choose very low ylims
'''
plt.boxplot(np.array([lambda_DA[:,t] + lambda_RES[:,t] - lambda_B[:,t] for t in range(T)]).reshape(T*W))
plt.grid(True)
plt.show()
'''

# fig, ax = plt.subplots(2,1,figsize=(8,8))
# ax[0].boxplot([lambda_DA[:,t] + lambda_RES[:,t] - lambda_B[:,t] for t in range(T)])
# ax[0].set_title('Boxplot for the scenarios in each hour of the profit made by 1 MW offer in DA')

# ax[1].boxplot([lambda_DA[:,t] + lambda_RES[:,t] - lambda_B[:,t] for t in range(T)])
# ax[1].set_title('Smaller y-range')
# ax[1].set_ylim((-5*10**2,1*10**3))
# ax[1].axhline(y=0, color='r')

# plt.tight_layout()
# plt.show()

plt.boxplot([lambda_DA[:,t] + lambda_RES[:,t] - lambda_B[:,t] for t in TT])
plt.xlabel('Time of day-1 [h]')
plt.ylabel('Revenue - Expected profit of 1 MW DA offer [€]')
plt.savefig('plots/V4/Step4_V4_EP1MWDA_spread', dpi=500, bbox_inches='tight')
plt.title('Boxplot for the scenarios in each hour of the profit made by 1 MW offer in DA')
plt.show()
