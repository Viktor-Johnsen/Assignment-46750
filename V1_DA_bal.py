import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from IPython.display import display

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES#, gamma_RES

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

#2 Mathematical model
model = gp.Model("V1")
p_DA = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="p_DA")
Delta =model.addMVar((W,T), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Delta")

model.setObjective(gp.quicksum(pi[w]*
                               (p_DA[t]*lambda_DA[w,t] + 
                                Delta[w,t]*lambda_B[w,t]) for w in WW for t in TT), 
                                GRB.MAXIMIZE)

#1st stage constraints:
model.addConstrs((p_DA[t] <= P_nom for t in TT), name="c_Nom")
model.addConstrs((p_RT[w,t] >= Delta[w,t] + p_DA[t] for w in WW for t in TT), name="c_RT")
model.addConstrs((p_DA[t] + Delta[w,t] >= 0 for w in WW for t in TT), name="c_NonnegativePhysicalDelivery")

model.optimize()

if model.status == GRB.OPTIMAL:
        print()
        print(f'-------------------   RESULTS   -------------------')
        optimal_objective = model.objVal # Save optimal value of objective function
        print("Optimal objective:", optimal_objective)
        p_DA_sol = [p_DA[t].x for t in TT]
        Delta_sol = [[Delta[w,t].x for t in TT] for w in WW]
        print(p_DA_sol)
        print()
        [print(np.array(Delta_sol[w][:]).tolist()) for w in WW]
else:
        print("Optimization was not successful")

print(np.array(p_DA_sol)[[1,2]])
t0=1
print(f'Average spot price in hour {t0} is {np.mean([lambda_DA[:,t0]]):.2f} DKK/MWh')
print(f'Average balancing price in hour {t0} is {np.mean([lambda_B[:,t0]]):.2f} DKK/MWh')

t0=2
print(f'Average price in hour {t0} is {np.mean([lambda_DA[:,t0]]):.2f} DKK/MWh')
print(f'Average balancing price in hour {t0} is {np.mean([lambda_B[:,t0]]):.2f} DKK/MWh')
# Based on the expected DA v. Balancing price in a given hour, we choose to bid all-or-nothing in that hour

print('In hours of non-negative balancing prices, there is balance between pRT and pDA & Delta because we want to max out on our available power:')
print(sum(p_RT[w,t] for t in TT for w in WW if lambda_B[w,t] >= 0))
print(sum(p_DA_sol[t] + Delta_sol[w][t] for t in TT for w in WW if lambda_B[w,t] >= 0))

import matplotlib.pyplot as plt

fig, ax=plt.subplots(figsize=(6,4),dpi=500)
ax.plot([np.mean([lambda_DA[:,t]]) for t in range(T)], label='$\overline{\lambda}^{DA}_t$')
ax.plot([np.mean([lambda_B[:,t]]) for t in range(T)], label='$\overline{\lambda}^{B}_t$')
ax2=ax.twinx()
ax2.plot(p_DA_sol, label='$p^{DA}_t$', alpha=.8, color='tab:green')

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax.set_xlabel('Hour of the day [h]')
ax.set_ylabel('Price [DKK/MWh]')
ax2.set_ylabel('DA offer [MW]')

ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.savefig('plots/V1/Step4_V1_decisions', dpi=500, bbox_inches='tight')
plt.title('Mean DA and balancing prices')
plt.show()

revenue_DA =  sum( sum(p_DA_sol * lambda_DA[w,:] * pi[w] for w in WW) )
revenue_BAL = sum( sum(Delta_sol[w][:] * lambda_B[w,:] * pi[w] for w in WW) )

print("Expected profit (Optimal objective):", optimal_objective)
print('These are the expected revenue streams:')
print(f'Day-ahead market: {revenue_DA:>42.2f} DKK')
print(f'Revenue from balancing market: {revenue_BAL:>29.2f} DKK')
print(f'Summing these together yields the expected profit: {revenue_DA+revenue_BAL:.2f}={optimal_objective:.2f}')
