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
Delta_down = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="Delta_down")
# New variables
p_RES = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="p_RES")
a_RES = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="a_RES")

model.setObjective(gp.quicksum(pi[w]*
                               (p_DA[t]*lambda_DA[w,t] + 
                                p_RES[t]*lambda_RES[w,t] - # New
                                (Delta_down[w,t]+a_RES[w,t]*alpha_RES[w,t])*lambda_B[w,t]) # Changed
                                for w in WW for t in TT), 
                                GRB.MAXIMIZE)

#1st stage constraints:
model.addConstrs((p_DA[t] - p_RES[t] <= P_nom for t in TT), name="c_Nom")
model.addConstrs((p_RT[w,t] >= Delta_down[w,t] + p_DA[t] - a_RES[w,t] for w in WW for t in TT), name="c_RT") # Changed
model.addConstrs((p_DA[t] - Delta_down[w,t] - a_RES[w,t] >= 0 for w in WW for t in TT), name="c_NonnegativePhysicalDelivery") # Changed
model.addConstrs((a_RES[w,t] >= p_RES[t] for w in WW for t in TT), name="c_Activation") # New 

model.optimize()

if model.status == GRB.OPTIMAL:
        print()
        print(f'-------------------   RESULTS   -------------------')
        optimal_objective = model.objVal # Save optimal value of objective function
        print("Optimal objective:", optimal_objective)
        p_DA_sol = [p_DA[t].x for t in TT]
        Delta_down_sol = [[Delta_down[w,t].x for t in TT] for w in WW]
        p_RES_sol = [p_RES[t].x for t in TT]
        a_RES_sol = [[a_RES[w,t].x for t in TT] for w in WW]

        print(f'p_DA={p_DA_sol}')
        print()
        print(f'p_RES={p_RES_sol}')
        print()
        [print(np.array(Delta_down_sol[w][:]).tolist()) for w in WW]
        print()
        [print(np.array(a_RES_sol[w][:]).tolist()) for w in WW]
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
