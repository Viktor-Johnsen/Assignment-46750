import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from IPython.display import display

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES, gamma_RES

T=24 #hours that we offer in
W=30 #scenarios/days, our training set

p_RT = p_RT.values[:W*T].reshape(W,T)
lambda_B = lambda_B.values[:W*T].reshape(W,T)
lambda_DA = lambda_DA.values[:W*T].reshape(W,T)
lambda_RES = lambda_RES.values[:W*T].reshape(W,T)
gamma_RES = gamma_RES.values[:W*T].reshape(W,T)

TT=np.arange(T)
WW=np.arange(W)

pi = np.ones(W)/W # Scenarios are assumed to be equiprobable
P_nom = 1 # MW

obj = np.zeros(W)
p_DAs = np.zeros((W,T))
Deltas = np.zeros((W,T))

#lambda_B[lambda_B <= 0] = 0 # Just used to check smth

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
model.addConstrs((p_RT[w,t] >= - Delta_down[w,t] + p_DA[t] - a_RES[w,t] for w in WW for t in TT), name="c_RT") # Changed
model.addConstrs((p_DA[t] - Delta_down[w,t] - a_RES[w,t] >= 0 for w in WW for t in TT), name="c_NonnegativePhysicalDelivery") # Changed
model.addConstrs((a_RES[w,t] >= p_RES[t] for w in WW for t in TT), name="c_Activation") # New 
                                # '>=': 32521.61 , with non-neg B-prices
                                # '==': 31722.18 , with non-neg B-prices
                                # '>=': 32712.19
                                # '==': 31938.66  

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

        print(f'p_DA={p_DA_sol}')
        print()
        print(f'p_RES={p_RES_sol}')
        print()
        [print(Delta_down_sol[w,:].tolist()) for w in WW]
        print()
        [print(a_RES_sol[w,:].tolist()) for w in WW]
else:
        print("Optimization was not successful")
#model.printStats()
#display(P_RT_w)

print(np.array(p_DA_sol)[[1,2]])
t0=1
print(f'Average spot price in hour {t0} is {np.mean([lambda_DA[:,t0]]):.2f} DKK/MWh')
print(f'Average balancing price in hour {t0} is {np.mean([lambda_B[:,t0]]):.2f} DKK/MWh')

t0=2
print(f'Average price in hour {t0} is {np.mean([lambda_DA[:,t0]]):.2f} DKK/MWh')
print(f'Average balancing price in hour {t0} is {np.mean([lambda_B[:,t0]]):.2f} DKK/MWh')
# Based on the expected DA v. Balancing price in a given hour, we choose to bid all-or-nothing in that hour

import matplotlib.pyplot as plt

fig, ax=plt.subplots(figsize=(6,4),dpi=500)
# p_RT.shape i 30 by 24 but for some reason it p_RT[t,:] is correct below and not p_RT[:,t]
for t in range(T): ax.plot(p_RT[t,:], label='$p^{RT}_{\omega,t}$' if t == 0 else None, alpha=0.5, color='tab:red')
ax.plot(p_DA_sol, label='$p^{DA}_t$', alpha=.8, color='tab:green')
ax.plot(p_RES_sol, label='$p^{RES}_t$', alpha=.8, color='tab:purple')
ax.legend(loc=0)
ax.set_xlabel('Hour of the day [h]')
ax.set_ylabel('Price [DKK/MWh]')
plt.title('Examining the offer decisions: p_RES and p_RT')
plt.show()

fig, ax=plt.subplots(figsize=(6,4),dpi=500)
for t in range(T): ax.plot(a_RES_sol[t,:], label='$a^{RES}_{\omega,t}$' if t == 0 else None, alpha=0.5, color='tab:red')
ax.plot(p_DA_sol, label='$p^{DA}_t$', alpha=.8, color='tab:green')
ax.plot(p_RES_sol, label='$p^{RES}_t$', alpha=.8, color='tab:purple')
ax.legend(loc=0)
ax.set_xlabel('Hour of the day [h]')
ax.set_ylabel('Price [DKK/MWh]')
plt.title('Examining the offer decisions: p_RES and a_RES')
plt.show()

print("Expected profit (Optimal objective):", optimal_objective)