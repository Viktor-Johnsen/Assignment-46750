import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from IPython.display import display

from load_data import p_RT, lambda_DA, lambda_B, lambda_RES# , gamma_RES
from load_data import train_scenarios, test_scenarios # 30, 5

T=24 #hours that we offer in
W=train_scenarios #scenarios/days, our training set

testing = False # OOS testing on V5 was deemed uninteresting based on the justification provided in the report.

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

# gamma_RES = np.ones((W,T)) # Down-regulation activated in all hours
# gamma_RES[lambda_B > lambda_DA]=0 # Down-regulation not activated in hours where balancing price is higher than DA price

M = max( np.max(lambda_DA-lambda_B), abs(np.min(lambda_DA-lambda_B)) ) + 936 #np.max(lambda_DA-lambda_B)*7 # Used for McCormick relaxation
lambda_offer_fix = 0 #np.max(lambda_DA-lambda_B) # Only 1389.9 as opposed to 7458.3 above
lambda_offer_RES = lambda_offer_fix

alpha = 0.9 # Worried about the profits in the 10th percentile least favorable scenarios
beta = 0 # Level of risk-averseness of wind farm owner

# initiate dictionaries used to save the results for different levels of risk
# betas = np.round( np.linspace(0,1,11), 2)
betas = np.array([0.0,0.1,0.8]) # np.round( np.linspace(0,0.8,9), 2) # Selected range

if testing: # testing here
    betas = np.array([0.0])

M_P90 = 1

# Only used for the last epsilon in the list
DA_offer = {f'{beta}': float for beta in betas}
RES_offer = {f'{beta}': float for beta in betas}
objs = {f'{beta}': float for beta in betas}
CVaR = {f'{beta}': float for beta in betas}
VaR = {f'{beta}': float for beta in betas}
Eprofs = {f'{beta}': float for beta in betas}
Eprofs_w = {f'{beta}': np.array(float) for beta in betas}

epsilons = [0.0, 0.1, 1.0]
Eprofs_p90 = {f'P{(1-epsilon)*100:.0f}': {beta:float for beta in betas} for epsilon in epsilons}
p_DA_sol_p90 = {f'P{(1-epsilon)*100:.0f}': {beta:[] for beta in betas} for epsilon in epsilons}
p_RES_sol_p90 = {f'P{(1-epsilon)*100:.0f}': {beta:[] for beta in betas} for epsilon in epsilons}

revenue_DA_p90 = {f'P{(1-epsilon)*100:.0f}': {beta:float for beta in betas} for epsilon in epsilons}
revenue_RES_p90= {f'P{(1-epsilon)*100:.0f}': {beta:float for beta in betas} for epsilon in epsilons}
losses_ACT_p90 ={f'P{(1-epsilon)*100:.0f}': {beta:float for beta in betas} for epsilon in epsilons}
revenue_BAL_p90 = {f'P{(1-epsilon)*100:.0f}': {beta:float for beta in betas} for epsilon in epsilons}

for epsilon in epsilons:
    for beta in betas:
        #2 Mathematical model
        model = gp.Model("V1")
        p_DA = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="p_DA")
        Delta_down = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="Delta_down")
        # New variables
        p_RES = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="p_RES")
        a_RES = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="a_RES")

        # Used to make strategic balancing price offer
        # alpha_RES = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="alpha_RES")
        # beta_RES = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="beta_RES")
        g = model.addMVar((W,T), vtype=GRB.BINARY, name="g")
        phi = model.addMVar((W,T), lb=0, vtype=GRB.CONTINUOUS, name="phi")

        # Used to introduce risk framework: CVaR
        zeta = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="zeta") # "Value-at-Risk"
        eta = model.addVars(W, lb=0, vtype=GRB.CONTINUOUS, name="eta") # Expected profit of each scenario "at-risk"

        # Used to introduce joint-chance constrained programming (P90-rule)
        u = model.addMVar((W,T), vtype=GRB.BINARY, name="u")

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

        model.addConstrs(( lambda_offer_fix - M*(1-g[w,t]) <= lambda_DA[w,t] - lambda_B[w,t] for w in WW for t in TT), name='c_McCormick_7a_1')
        model.addConstrs((lambda_DA[w,t] - lambda_B[w,t] <= lambda_offer_fix + M*g[w,t] for w in WW for t in TT), name='c_McCormick_7a_2')

        # difference in DA price for T[:-1] has been set to 0 as of right now.
        
        model.addConstrs((a_RES[w,t] <= (phi[w,t] if lambda_DA[w,t] > lambda_B[w,t] else 0) for w in WW for t in TT), name='c_McCormick_7b')
        # McCormick_7c is changed
        # model.addConstrs((a_RES[w,t] - (phi[w,t] if lambda_DA[w,t] > lambda_B[w,t] else 0) >= 0 for w in WW for t in TT), name='c_McCormick_7c')
        model.addConstrs((a_RES[w,t] - (phi[w,t] if lambda_DA[w,t] > lambda_B[w,t] else 0) >= (u[w,t]-1)*M_P90 for w in WW for t in TT), name='c_McCormick_7c')

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

        # P90-constraint - slightly modified to avoid "violating" the agreement with TSO in hours where we are not even activated
        model.addConstr( gp.quicksum(u[w,t] for w in WW for t in TT)
                        >= (1-epsilon) * gp.quicksum(g[w,t] for w in WW for t in TT) )
        model.addConstrs(( u[w,t] <= g[w,t] for w in WW for t in TT), name="P90-auxiliary")

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

            u_sol = np.array([[u[w,t].x for t in TT] for w in WW])

            print(f'Saving the decisions for epsilon={epsilon} at beta={beta}')
            p_DA_sol_p90[f'P{(1-epsilon)*100:.0f}'][beta] = p_DA_sol
            p_RES_sol_p90[f'P{(1-epsilon)*100:.0f}'][beta] = p_RES_sol
            Eprofs_p90[f'P{(1-epsilon)*100:.0f}'][beta] = Eprofs[f'{beta}']

            # We save the expected profits and the decisions in DA and RES from the last beta run
            revenue_DA_p90[f'P{(1-epsilon)*100:.0f}'][beta] = sum( sum(p_DA_sol * lambda_DA[w,:] * pi[w] for w in WW) )
            revenue_RES_p90[f'P{(1-epsilon)*100:.0f}'][beta] = sum( sum(p_RES_sol * lambda_RES[w,:] * pi[w] for w in WW) )
            losses_ACT_p90[f'P{(1-epsilon)*100:.0f}'][beta] =  sum( sum(-a_RES_sol[w,:] * lambda_B[w,:] * pi[w] for w in WW) )
            revenue_BAL_p90[f'P{(1-epsilon)*100:.0f}'][beta]= sum( sum(-Delta_down_sol[w,:] * lambda_B[w,:] * pi[w] for w in WW) )

        else:
                print("Optimization was not successful")

print_additional_statements = False

if print_additional_statements:
    print(f'Such high balancing market offers allow us only to be activated this many times in each scenario: {np.sum( lambda_offer_RES <= lambda_B, axis=0)}')
    print(f'Instead of simply: {np.sum( lambda_DA > lambda_B, axis=0)}')

    print('#activated in each hour, a:\n', np.sum( a_RES_sol > 0, axis=0))
    print('This does not add up with the number of times that we enforce the activation, g:\n', np.sum( g_sol > 0, axis=0))
    print('Though the numbers for g match nicely with the auxiliary variable for activation, phi:\n', np.sum( phi_sol > 0, axis=0))
    print('Discrepancies can be explained by the number of times where phi > a:\n', np.sum( phi_sol > a_RES_sol, axis=0))
    print('These discrepancies between phi and a can be explained by the number of times where g=1 but there is no need for down-regulation, phi-condtional\n', np.sum( (phi_sol > 0) * (lambda_DA <= lambda_B), axis=0))

    print('In other words, changing the balancing activation offer price works, and the conditions are that there should be a need for down-regulation and the offer price should be smaller than or equal to that of the difference between DA and BAL:')
    print('#activated in each hour, a:\n', np.sum( a_RES_sol > 0, axis=0))
    print('This is the same as the number of times that we are activated (with equality), lambda_OFFER:\n', np.sum( (lambda_DA > lambda_B) * (lambda_DA - lambda_B >= lambda_offer_RES), axis=0))
    print('Apart from a difference of \"1" in hour 9 for some reason')


# Visualizations
import matplotlib.pyplot as plt
print('##############\nVISUALIZATION:\n##############')
import matplotlib.pyplot as plt

# Increase the number of betas for the plot below to be interesting and compare to the results from V4
'''
fig = plt.figure(figsize=(6,4),dpi=500)
for beta in betas:
    plt.scatter(CVaR[f'{beta}'], Eprofs[f'{beta}'])
points_to_plot = betas #[0.8,0.9]
for beta in points_to_plot:
    plt.annotate(f'beta={beta}', (CVaR[f'{beta}'], Eprofs[f'{beta}']))
plt.xlabel('CVaR [DKK]')
plt.ylabel(f'Expected profit [DKK] - epsilon={epsilon}')
#plt.legend()
plt.show()
'''

import seaborn as sns
import matplotlib.cm as cm

fig, ax = plt.subplots(figsize=(6,4))
cols = cm.tab10.colors
count=0
for i,k in enumerate(p_DA_sol_p90.keys()):
     for beta in betas:
        if (k, beta) in [('P100',0.0),
                         ('P100',0.1),
                         ('P90',0.0),
                         ('P90',0.1)]: 
            ax.plot(p_DA_sol_p90[k][beta], label=k+r':$\beta$'+f'={beta}'+', DA', linestyle='dashed', color=cols[count], alpha=1-.2*i)#, marker=markers[i])
            ax.plot(p_RES_sol_p90[k][beta], label=k+r':$\beta$'+f'={beta}'+', RES', linestyle='dotted', color=cols[count], alpha=.9-.2*i)#, marker=markers[i])
            count+=1

ax.set_xlabel('Time of day [h]')
ax.set_ylabel('Power offered [MW]')
ax.legend(loc=0)
plt.savefig('plots/V5/Step4_V5_decisions_betas_eps', dpi=500, bbox_inches='tight')
plt.show()

for key in [f'P{(1-epsilon)*100:.0f}' for epsilon in epsilons]:
    for beta in betas:   
        print(f'These are the expected revenue streams for {key,beta}:')
        print(f'Day-ahead market: {revenue_DA_p90[key][beta]:>42.2f} DKK')
        print(f'aFRR capacity market (down): {revenue_RES_p90[key][beta]:>31.2f} DKK')
        print(f'Money spent to buy el. back: {losses_ACT_p90[key][beta]:>31.2f} DKK')
        print(f'Revenue from balancing market: {revenue_BAL_p90[key][beta]:>29.2f} DKK')
        print(f'Summing these together yields the expected profit: {revenue_DA_p90[key][beta]+revenue_RES_p90[key][beta]+losses_ACT_p90[key][beta]+revenue_BAL_p90[key][beta]:.2f}={Eprofs_p90[key][beta]:.2f}')

if (not testing) and (print_additional_statements): 
    print('CVaR and P90-interdepence:')
    for beta in betas[1:]:
        print('P90, beta=',beta)
        print(f"{Eprofs_p90['P90'][beta]-Eprofs_p90['P100'][0.0]:.2f}")
        print(f"{(Eprofs_p90['P100'][beta]-Eprofs_p90['P100'][0.0]+Eprofs_p90['P90'][0.0]-Eprofs_p90['P100'][0.0]):.2f}")
    print('CVaR and P90-interdepence:')
    for key in list(Eprofs_p90.keys())[1:]:
        print('0.1, P?=',key)
        print(f"{Eprofs_p90[key][0.1]-Eprofs_p90['P100'][0.0]:.2f}")
        print(f"{(Eprofs_p90['P100'][0.1]-Eprofs_p90['P100'][0.0]+Eprofs_p90[key][0.0]-Eprofs_p90['P100'][0.0]):.2f}")

print('##############\nScript is done\n##############')
