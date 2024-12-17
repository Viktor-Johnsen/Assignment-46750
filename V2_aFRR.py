import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from IPython.display import display

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES# , gamma_RES

show_plots = True # Used to toggle between plotting and not plotting...

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

gamma_RES = np.ones((W,T)) # Down-regulation activated in all hours
gamma_RES[lambda_B >= lambda_DA]=0 # Down-regulation not activated in hours where balancing price is higher than DA price

#2 Mathematical model
model = gp.Model("V1")
p_DA = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="p_DA")
Delta_down = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="Delta_down")
# New variables
p_RES = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="p_RES")
a_RES = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="a_RES")

model.setObjective(gp.quicksum(pi[w]*
                               (p_DA[t]*lambda_DA[w,t] + 
                                p_RES[t]*lambda_RES[w,t] - # New
                                (Delta_down[w,t]+a_RES[w,t]*gamma_RES[w,t])*lambda_B[w,t]) # Changed
                                for w in WW for t in TT), 
                                GRB.MAXIMIZE)

#1st stage constraints:
model.addConstrs((p_DA[t] <= P_nom for t in TT), name="c_Nom")
model.addConstrs((p_RT[w,t] >= p_DA[t] - Delta_down[w,t] - a_RES[w,t]*gamma_RES[w,t] for w in WW for t in TT), name="c_RT") # Changed
model.addConstrs((             p_DA[t] - Delta_down[w,t] - a_RES[w,t]*gamma_RES[w,t] >= 0 for w in WW for t in TT), name="c_NonnegativePhysicalDelivery") # Changed
model.addConstrs((a_RES[w,t] >= p_RES[t] for w in WW for t in TT), name="c_Activation") # New  

model.optimize()

if model.status == GRB.OPTIMAL:
        print()
        print(f'-------------------   RESULTS   -------------------')
        optimal_objective = model.objVal # Save optimal value of objective function
        #print("Optimal objective:", optimal_objective)
        p_DA_sol = [p_DA[t].x for t in TT]
        Delta_down_sol = np.array( [[Delta_down[w,t].x for t in TT] for w in WW] )
        p_RES_sol = [p_RES[t].x for t in TT]
        a_RES_sol = np.array( [[a_RES[w,t].x for t in TT] for w in WW] )
else:
        print("Optimization was not successful")

print("Expected profit (Optimal objective):", optimal_objective)

# Where do we earn revenue?
revenue_DA =  sum( sum(p_DA_sol * lambda_DA[w,:] * pi[w] for w in WW) )
revenue_RES = sum( sum(p_RES_sol * lambda_RES[w,:] * pi[w] for w in WW) )
losses_ACT =  sum( sum(-a_RES_sol[w,:]*gamma_RES[w,:] * lambda_B[w,:] * pi[w] for w in WW) )
revenue_BAL = sum( sum(-Delta_down_sol[w,:] * lambda_B[w,:] * pi[w] for w in WW) )

print('These are the expected revenue streams:')
print(f'Day-ahead market: {revenue_DA:>42.2f} DKK')
print(f'aFRR capacity market (down): {revenue_RES:>31.2f} DKK')
print(f'Money spent to buy el. back: {losses_ACT:>31.2f} DKK')
print(f'Revenue from balancing market: {revenue_BAL:>29.2f} DKK')
print(f'Summing these together yields the expected profit: {revenue_DA+revenue_RES+losses_ACT+revenue_BAL:.2f}={optimal_objective:.2f}')

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
        plt.savefig('plots/V2/Step4_V2_decisions', dpi=500, bbox_inches='tight')
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
        plt.savefig('plots/V2/Step4_V2_decisions_expP1MWDA', dpi=500, bbox_inches='tight')
        plt.title('Offers in DA and RES and the ratio between DA- and BAL-prices')
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

print('##############\nScript is done\n##############')