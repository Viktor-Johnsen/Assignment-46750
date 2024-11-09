import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from IPython.display import display

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES, alpha_RES

T=24 #hours that we offer in
W=30 #scenarios/days, our training set

p_RT = p_RT.values[:W*T].reshape(W,T)
lambda_B = lambda_B.values[:W*T].reshape(W,T)
lambda_DA = lambda_DA.values[:W*T].reshape(W,T)
lambda_RES = lambda_RES.values[:W*T].reshape(W,T)
alpha_RES = alpha_RES.values[:W*T].reshape(W,T)

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
