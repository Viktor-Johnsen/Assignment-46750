import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from IPython.display import display

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES# , gamma_RES

def solve_sub(p_DA, p_RT, p_RES, lambda_B, lambda_RES, lambda_DA,gamma_RES, pi, T):
        # One subproblem per scenario
        submodel = gp.Model("sub")
        submodel.Params.OutputFlag = 0 # Turn off output to console
        
        Delta_down = submodel.addMVar((T), lb=0, vtype=GRB.CONTINUOUS, name="Delta_down")
        a_RES = submodel.addMVar((T), lb=0, vtype=GRB.CONTINUOUS, name="a_RES")

        # Variables to make extracting duals easier
        p_DA_out = submodel.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="p_DA_out")
        p_RES_out = submodel.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="p_RES_out")

        # Slack variable
        slack = submodel.addVar(lb=0, vtype=GRB.CONTINUOUS, name="slack")

        submodel.setObjective(gp.quicksum(
                               (p_DA_out[t]*lambda_DA[t] +
                                p_RES_out[t]*lambda_RES[t] - # New
                                (Delta_down[t]+a_RES[t]*gamma_RES[t])*lambda_B[t]) # Changed
                                for t in TT) - 100000000*slack, 
                                GRB.MAXIMIZE)

        #1st stage constraints:
        submodel.addConstrs((p_RT[t] + slack >= p_DA_out[t] - Delta_down[t] - a_RES[t]*gamma_RES[t] for t in TT), name="c_RT") # Changed
        submodel.addConstrs((             p_DA_out[t] - Delta_down[t] - a_RES[t]*gamma_RES[t] >= 0 for t in TT), name="c_NonnegativePhysicalDelivery") # Changed
        submodel.addConstrs((a_RES[t] + slack >= p_RES_out[t] for t in TT), name="c_Activation") # New  
        
        # Constraints to extract duals
        submodel.addConstrs((p_DA_out[t] == p_DA[t] for t in TT), name="c_DA_out")
        submodel.addConstrs((p_RES_out[t] == p_RES[t] for t in TT), name="c_RES_out")

        submodel.optimize()

        if submodel.status == GRB.OPTIMAL:
                optimal_objective = submodel.objVal # Save optimal value of objective function
                #print("Optimal objective:", optimal_objective)
                #Delta_down_sol = np.array( [[Delta_down[w,t].x for t in TT] for w in WW] )
                #a_RES_sol = np.array( [[a_RES[w,t].x for t in TT] for w in WW] )
                p_DA_dual = [submodel.getConstrByName(f"c_DA_out[{t}]").Pi for t in range(T)]
                p_RES_dual = [submodel.getConstrByName(f"c_RES_out[{t}]").Pi for t in range(T)]
                Delta_down_sol = [Delta_down[t].x for t in range(T)]
                a_RES_sol = [a_RES[t].x for t in range(T)]
                p_DA_out_sol = [p_DA_out[t].x for t in range(T)]
                p_RES_out_sol = [p_RES_out[t].x for t in range(T)]
                return optimal_objective, p_DA_dual, p_RES_dual

        else:
                print("Optimization was not successful")
                
                print(submodel.status)
                return None, None, None





def solve_mas(W, T, pi, p_RT, lambda_B, lambda_DA, lambda_RES, gamma_RES, P_nom):
    VMAX = 100
    WW = np.arange(W)
    TT = np.arange(T)
    UB = 1e6
    LB = -1e6
    v = 0
    
    # Initial var values
    p_DA_val = np.zeros((T,VMAX))
    p_RES_val = np.zeros((T,VMAX))

    model = gp.Model("MAS")
    model.Params.OutputFlag = 0 # Turn off output to console

    p_DA = model.addVars(T, lb=0, ub=P_nom , vtype=GRB.CONTINUOUS, name="p_DA")
    p_RES = model.addVars(T, lb=0, ub=P_nom,vtype=GRB.CONTINUOUS, name="p_RES")
    #gamma = model.addVar(lb=LB, ub=UB, vtype=GRB.CONTINUOUS, name="gamma")
    gamma = model.addVars(WW, lb=LB, ub=UB, vtype=GRB.CONTINUOUS, name="gamma")

    #model.setObjective(gamma, GRB.MAXIMIZE) # Benders objective
    model.setObjective(gp.quicksum(pi[w]*gamma[w] for w in WW), GRB.MAXIMIZE) # Benders objective

    LB_arr = np.zeros((W,VMAX))
    lambDA_arr = np.zeros((W,T, VMAX))
    lambRes_arr = np.zeros((W,T, VMAX))

    # Add Benders cuts
    #while UB - LB > 0.1:
    for v in range(VMAX):
        if v > 0:
                for w in WW:
                        LB_temp, lambDA, lambRES = solve_sub(p_DA_val[:,v-1], p_RT[w,:], p_RES_val[:,v-1], lambda_B[w,:], lambda_RES[w,:], lambda_DA[w,:], gamma_RES[w,:], pi[w], T)
                        LB_arr[w,v] = LB_temp
                        lambDA_arr[w,:,v] = lambDA
                        lambRes_arr[w,:,v] = lambRES
                sub_obj = pi[0]*np.sum(LB_arr[:,v])
                LB = max(LB, sub_obj)

                for w in WW:
                        model.addConstr(gamma[w] <= LB_arr[w,v] + 
                                        gp.quicksum(lambDA_arr[w,t,v]*(p_DA[t]-p_DA_val[t,v-1]) + 
                                        lambRes_arr[w,t,v]*(p_RES[t]-p_RES_val[t,v-1]) 
                                        for t in TT))
                #model.addConstr(gamma <= gp.quicksum(pi[w]*LB_arr[w,v] for w in WW) + 
                #                gp.quicksum(lambDA_arr[w,t,v]*(p_DA[t]-p_DA_val[t,v-1]) + 
                #                lambRes_arr[w,t,v]*(p_RES[t]-p_RES_val[t,v-1]) 
                #                for w in WW for t in TT))
        
        model.optimize()
        #model.addConstrs(p_DA[t] >=1 for t in TT)
        if model.status == GRB.OPTIMAL:
            # Update values
            UB = model.objVal
            p_DA_val[:,v] = [p_DA[t].x for t in TT]
            p_RES_val[:,v] = [p_RES[t].x for t in TT]
            #print(f"UB: {UB}, LB: {LB}")
            print(f"p_DA: {p_DA_val[:,v]}")
            print(f"p_RES: {p_RES_val[:,v]}")
            
        else:
            model.write("model.lp")
            print('Optimization of master was not successful')
            print(model.status)
            
            break
        v += 1
        print(f"The upper bound is {UB}")
        print(f"The lower bound is {LB}")
        diff = UB - LB
        print(f"The difference between UB and LB is {diff}")
        if diff < 10 and v > 2:
            break
        

    if model.status == GRB.OPTIMAL:
         return model.objVal, p_DA_val[:,v], p_RES_val[:,v]




show_plots = False # Usedd to toggle between plotting and not plotting...

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

gamma_RES = np.ones((W,T)) # Down-regulation activated in all hours
gamma_RES[lambda_B >= lambda_DA]=0 # Down-regulation not activated in hours where balancing price is higher than DA price

# Solve the MAS
obj, p_DAs, p_Res = solve_mas(W, T, pi, p_RT, lambda_B, lambda_DA, lambda_RES, gamma_RES, P_nom)
