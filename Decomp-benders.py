import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from IPython.display import display

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES# , gamma_RES

def solve_sub(beta, zeta_in, p_DA_in, p_RES_in, lambda_DA, lambda_B, lambda_RES, p_RT, w, T):
    M = 10000000
    TT = np.arange(T)
    model = gp.Model("V1")
    model.Params.OutputFlag = 0 # Turn off output
    Delta_down = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="Delta_down")
    a_RES = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="a_RES")
    
    # Used to introduce risk framework: CVaR
    eta = model.addVars(W, lb=0, vtype=GRB.CONTINUOUS, name="eta") # Expected profit of each scenario "at-risk"

    # Variables to make extracting dual values easier
    p_DA = model.addMVar((T), lb=0, vtype=GRB.CONTINUOUS, name="p_DA")
    p_RES = model.addMVar((T), lb=0, vtype=GRB.CONTINUOUS, name="p_RES")
    zeta = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="zeta") # "Value-at-Risk"
    a_slack = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="a_slack") # Slack variable for CVaR-constraint

    model.setObjective( (1-beta) * gp.quicksum(pi[w]*
                                (p_DA[t]*lambda_DA[w,t] + 
                                 p_RES[t]*lambda_RES[w,t] - # New
                                 (Delta_down[w,t]+a_RES[w,t])*lambda_B[w,t]) 
                                    for t in TT)+
                        beta * ( zeta - 1/(1-alpha) * pi[w] * eta[w] )
                        - M * a_slack, 
                                    GRB.MAXIMIZE)
    

    model.addConstrs((p_RT[w,t] >= p_DA[t] - Delta_down[w,t] - a_RES[w,t] for t in TT), name="c_RT") 
    model.addConstrs((             p_DA[t] - Delta_down[w,t] - a_RES[w,t] >= 0 for t in TT), name="c_NonnegativePhysicalDelivery") 
    model.addConstrs((a_RES[w,t] + a_slack >= p_RES[t] for t in TT), name="c_Activation") 
    
    #model.addConstrs((p_RES[t] <= p_DA[t] for t in TT), name="c_Nom_RES")

    # CVaR-constraint
    model.addConstr(( - gp.quicksum(p_DA[t]*lambda_DA[w,t] + 
                                p_RES[t]*lambda_RES[w,t] -
                                (Delta_down[w,t]+a_RES[w,t])*lambda_B[w,t]
                                for t in TT )
                                + zeta - eta[w] <= 0), name="c_CVaR")
    
    # Constraints to make extracting dual values easier
    model.addConstrs((p_DA[t] == p_DA_in[t] for t in TT), name="c_p_DA")
    model.addConstrs((p_RES[t] == p_RES_in[t] for t in TT), name="c_p_RES")
    model.addConstr((zeta == zeta_in), name="c_zeta")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.objVal, [model.getConstrByName(f"c_p_DA[{t}]").Pi for t in range(T)], [model.getConstrByName(f"c_p_RES[{t}]").Pi for t in range(T)], model.getConstrByName("c_zeta").Pi
    else:
        print('Optimization of sub was not successful')
        print(zeta_in, p_DA_in, p_RES_in)
        return None
    

def solve_mas(W, T, pi, lambda_DA, lambda_B, lambda_RES, p_RT, beta, alpha):
    UB = 100000000 # Upper bound
    LB = 0 # Lower bound
    v = 0 # Number of iterations
    zeta_val = 0
    # Initial variable values
    p_DA_val = np.zeros(T)+1
    p_RES_val = np.zeros(T)+1
    model = gp.Model("V1")
    model.Params.OutputFlag = 0 # Turn off output
    p_DA = model.addVars(T, lb=0, ub=10, vtype=GRB.CONTINUOUS, name="p_DA")
    p_RES = model.addVars(T, lb=0, ub = 10, vtype=GRB.CONTINUOUS, name="p_RES")
    gamma = model.addVar(lb= -GRB.INFINITY, vtype=GRB.CONTINUOUS, name="gamma") # Benders under-estimator
    
    # Used to introduce risk framework: CVaR
    zeta = model.addVar(lb=-100000, ub=100000, vtype=GRB.CONTINUOUS, name="zeta") # "Value-at-Risk"
    
    model.setObjective(gamma , GRB.MAXIMIZE) # Benders objective

    #1st stage constraints:
    model.addConstrs((p_DA[t] <= P_nom for t in TT), name="c_Nom")

    
    # Initialize lists
    LB_list = []
    lambDA_arr = np.zeros((W,T))
    lambRes_arr = np.zeros((W,T))
    lambZeta_arr = np.zeros(W)
    #Add cuts
    while UB - LB > 0.01:
        for w in WW:
            LB_temp, lambDA, lambRES, lambZeta = solve_sub(beta, zeta_val, p_DA_val, p_RES_val, lambda_DA, lambda_B, lambda_RES, p_RT, w, T)
            LB_list.append(LB_temp)
            lambDA_arr[w,:] = lambDA
            lambRes_arr[w,:] = lambRES
            lambZeta_arr[w] = lambZeta
        sub_obj = gp.quicksum(LB_list).getValue()
        #if v >= 1: They do this in class, but we didn't do this in large scale decomp I think. Can be added later
        model.addConstr((gamma <= sub_obj + gp.quicksum(lambDA_arr[w,t]*p_DA[t] + lambRes_arr[w,t]*p_RES[t] + lambZeta_arr[w]*zeta for w in WW for t in TT) ), name=f"c_BendersCut_{v}")
        model.optimize()
        LB = max(LB, sub_obj)
        if model.status == GRB.OPTIMAL:
            # Update values
            UB = model.objVal
            zeta_val = zeta.x
            p_DA_val = [p_DA[t].x for t in TT]
            p_RES_val = [p_RES[t].x for t in TT]
            print(f"UB: {UB}, LB: {LB}")
            print(f"zeta: {zeta_val}")
            print(f"p_DA: {p_DA_val}")
            print(f"p_RES: {p_RES_val}")
            input()
        else:
            print('Optimization of master was not successful')
            break
        v += 1
        print(f"The upper bound is {UB}")
        print(f"The lower bound is {LB}")
        diff = UB - LB
        print(f"The difference between UB and LB is {diff}")
        

    if model.status == GRB.OPTIMAL:
         return model.objVal, p_DA_val, p_RES_val, zeta.x
     
    


    

     

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

# Solve the master problem
optimal_objective, p_DA_sol, p_RES_sol, zeta_sol = solve_mas(W, T, pi, lambda_DA, lambda_B, lambda_RES, p_RT, beta, alpha)

