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

TT=np.arange(T)
WW=np.arange(W)

pi = np.ones(W)/W # Scenarios are assumed to be equiprobable
P_nom = 1 # MW

obj = np.zeros(W)
p_DAs = np.zeros((W,T))
Deltas = np.zeros((W,T))

M = max( np.max(lambda_DA-lambda_B), abs(np.min(lambda_DA-lambda_B)) ) + 936 #np.max(lambda_DA-lambda_B)*7 # Used for McCormick relaxation
# M = max( np.max(lambda_DA-lambda_B), abs(np.min(lambda_DA-lambda_B)) ) + 900 # -> still activated in (w,t)=(15,10)

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

model.setObjective(gp.quicksum(pi[w]*
                               (p_DA[t]*lambda_DA[w,t] + 
                                p_RES[t]*lambda_RES[w,t] - # New
                                (Delta_down[w,t]+a_RES[w,t])*lambda_B[w,t]) # Changed
                                for w in WW for t in TT), 
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

model.optimize()

if model.status == GRB.OPTIMAL:
        print()
        print(f'-------------------   RESULTS   -------------------')
        optimal_objective = model.objVal # Save optimal value of objective function
        print("Optimal objective:", optimal_objective)
        p_DA_sol = [p_DA[t].x for t in TT]
        Delta_down_sol = np.array( [[Delta_down[w,t].x for t in TT] for w in WW] )
        p_RES_sol = [p_RES[t].x for t in TT]
        a_RES_sol = np.array( [[a_RES[w,t].x for t in TT] for w in WW] )

        print('We strategically offer the balancing activation price as: ', )
        g_sol = np.array([[g[w,t].x for t in TT] for w in WW])
        phi_sol = np.array([[phi[w,t].x for t in TT] for w in WW])
        alpha_RES_sol = np.array([alpha_RES[t].x for t in TT])
        beta_RES_sol = np.array([beta_RES[t].x for t in TT])
        lambda_offer_RES = [[alpha_RES_sol[t] * (lambda_DA[w,t+1]-lambda_DA[w,t] if t<T-1 else 0) + lambda_DA[w,t] + beta_RES_sol[t] for w in WW] for t in TT]


else:
        print("Optimization was not successful")

print("Strategic balancing offer: \n", lambda_offer_RES)
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

print(f'Such high balancing market offers allow us only to be activated this many times in each scenario: {np.sum( np.transpose(np.array(lambda_offer_RES)) <= lambda_B, axis=0)}')
print(f'Instead of simply: {np.sum( lambda_DA > lambda_B, axis=0)}')

print('#activated in each hour, a:\n', np.sum( a_RES_sol > 0, axis=0))
print('This does not add up with the number of times that we enforce the activation, g:\n', np.sum( g_sol > 0, axis=0))
print('Though the numbers for g match nicely with the auxiliary variable for activation, phi:\n', np.sum( phi_sol > 0, axis=0))
print('Discrepancies can be explained by the number of times where phi > a:\n', np.sum( phi_sol > a_RES_sol, axis=0))
print('These discrepancies between phi and a can be explained by the number of times where g=1 but there is no need for down-regulation, phi-condtional\n', np.sum( (phi_sol > 0) * (lambda_DA <= lambda_B), axis=0))

print('In other words, changing the balancing activation offer price works, and the conditions are that there should be a need for down-regulation and the offer price should be smaller than or equal to that of the difference between DA and BAL:')
print('This is the same as the number of times that we are activated (without equality), lambda_offer:\n', np.sum( (lambda_DA > lambda_B) * (lambda_DA - lambda_B > np.transpose(np.array(lambda_offer_RES))), axis=0))
print('#activated in each hour, a:\n', np.sum( a_RES_sol > 0, axis=0))
print('Apart from a difference of \"1" in hour 9 for some reason')


# Visualizations
import matplotlib.pyplot as plt

if show_plots:
        fig, ax=plt.subplots(figsize=(6,4),dpi=500)
        # p_RT.shape i 30 by 24 but for some reason it p_RT[t,:] is correct below and not p_RT[:,t]
        for t in range(T): ax.plot(p_RT[t,:], label='$p^{RT}_{\omega,t}$' if t == 0 else None, alpha=0.5, color='tab:red')
        ax.plot(p_DA_sol, label='$p^{DA}_t$', alpha=.8, color='tab:green')
        ax.plot(p_RES_sol, label='$p^{RES}_t$', alpha=.8, color='tab:purple')
        ax.legend(loc=0)
        ax.set_xlabel('Hour of the day [h]')
        ax.set_ylabel('Power [MW]')
        plt.title('Examining the offer decisions: p_RES and p_RT')
        plt.show()

        fig, ax=plt.subplots(figsize=(6,4),dpi=500)
        for t in range(T): ax.plot(a_RES_sol[t,:], label='$a^{RES}_{\omega,t}$' if t == 0 else None, alpha=0.5, color='tab:red')
        ax.plot(p_DA_sol, label='$p^{DA}_t$', alpha=.8, color='tab:green')
        ax.plot(p_RES_sol, label='$p^{RES}_t$', alpha=.8, color='tab:purple')
        ax.legend(loc=0)
        ax.set_xlabel('Hour of the day [h]')
        ax.set_ylabel('Power [MW]')
        plt.title('Examining the offer decisions: p_RES and a_RES')
        plt.show()

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

        fig, ax=plt.subplots(figsize=(6,4),dpi=500)
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
        ax2.set_ylabel('Revenue - Expected profit of 1 MW DA offer [DKK]')
        
        plt.savefig('plots/V3/Step4_V3_decisions_expP1MWDA', dpi=500, bbox_inches='tight')
        plt.title('Offers in DA and RES and the ratio between DA- and BAL-prices')
        plt.show()
print('##############\nScript is done\n##############')

fig, ax = plt.subplots(2,1,figsize=(6,6))
ax = ax.flatten()
ax[0].boxplot(lambda_offer_RES)
ax[0].set_xlabel('Time of day-1 [h]')
ax[0].set_ylabel('Price [DKK/MWh]')
ax[0].set_title('$\lambda^{offer}_t$ - Strategic activation offer prices')

ax[1].boxplot(lambda_DA-lambda_B)
ax[1].set_xlabel('Time of day-1 [h]')
ax[1].set_ylabel('Price [DKK/MWh]')
ax[1].set_title('$\lambda^{DA}_{\omega,t}-\lambda^{B}_{\omega,t}$ - Actual difference in prices')
plt.tight_layout()
plt.savefig('plots/V3/Step4_V3_lambda_offer_boxplot', dpi=500, bbox_inches='tight')
plt.show()