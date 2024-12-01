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
lambda_offer_fix = 0#np.max(lambda_DA-lambda_B) # Only 1389.9 as opposed to 7458.3 above
lambda_offer_RES = lambda_offer_fix

alpha = 0.9 # Worried about the profits in the 10th percentile least favorable scenarios
beta = 0 # Level of risk-averseness of wind farm owner

# initiate dictionaries used to save the results for different levels of risk
# betas = np.round( np.linspace(0,1,11), 2)
betas = np.array([0.0]) # np.round( np.linspace(0,0.8,9), 2) # Selected range

M_P90 = 1
epsilon = 0.1 # Because, P90...

DA_offer = {f'{beta}': float for beta in betas}
RES_offer = {f'{beta}': float for beta in betas}
objs = {f'{beta}': float for beta in betas}
CVaR = {f'{beta}': float for beta in betas}
VaR = {f'{beta}': float for beta in betas}
Eprofs = {f'{beta}': float for beta in betas}
Eprofs_w = {f'{beta}': np.array(float) for beta in betas}

epsilons = [0.0, 0.1, 1.0]
Eprofs_p90 = {f'P{(1-epsilon)*100:.0f}': float for epsilon in epsilons}
p_DA_sol_p90 = {f'P{(1-epsilon)*100:.0f}': [] for epsilon in epsilons}
p_RES_sol_p90 = {f'P{(1-epsilon)*100:.0f}': [] for epsilon in epsilons}


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
            # lambda_offer_RES === alpha_RES * ( (lambda_DA[w,t+1] if t<T-1 else lambda_DA[w,t])-lambda_DA[w,t]) + lambda_DA[w,t] + beta_RES

        # model.addConstrs((lambda_offer_RES[w,t] - M*(1-g[w,t]) <= lambda_DA[w,t] - lambda_B[w,t] for w in WW for t in TT), name='c_McCormick_7a_1')
        # model.addConstrs((lambda_DA[w,t] - lambda_B[w,t] <= lambda_offer_RES[w,t] + M*g[w,t] for w in WW for t in TT), name='c_McCormick_7a_2')
        model.addConstrs(( lambda_offer_fix - M*(1-g[w,t]) <= lambda_DA[w,t] - lambda_B[w,t] for w in WW for t in TT), name='c_McCormick_7a_1')
        model.addConstrs((lambda_DA[w,t] - lambda_B[w,t] <= lambda_offer_fix + M*g[w,t] for w in WW for t in TT), name='c_McCormick_7a_2')

        # WHAT ABOUT THE T[:-1]??? -> I just set it to 0 as of right now.

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
            # alpha_RES_sol = np.array([alpha_RES[t].x for t in TT])
            # beta_RES_sol = np.array([beta_RES[t].x for t in TT])
            # lambda_offer_RES = [[alpha_RES_sol[t] * (lambda_DA[w,t+1]-lambda_DA[w,t] if t<T-1 else 0) + lambda_DA[w,t] + beta_RES_sol[t] for t in TT] for w in WW]
            

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

        else:
                print("Optimization was not successful")
    # We save the expected profits and the decisions in DA and RES from the last beta run
    print(f'Saving the decisions for epsilon={epsilon} at beta={beta}')
    p_DA_sol_p90[f'P{(1-epsilon)*100:.0f}'] = p_DA_sol
    p_RES_sol_p90[f'P{(1-epsilon)*100:.0f}'] = p_RES_sol
    Eprofs_p90[f'P{(1-epsilon)*100:.0f}'] = Eprofs[f'{beta}']
#model.printStats()
#display(P_RT_w)
# print("Expected profit (Optimal objective):", optimal_objective)
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
#print('This is the same as the number of times that we are activated (without equality), lambda_offer:\n', np.sum( (lambda_DA > lambda_B) * (lambda_DA - lambda_B > lambda_offer_RES), axis=0))
print('#activated in each hour, a:\n', np.sum( a_RES_sol > 0, axis=0))
print('This is the same as the number of times that we are activated (with equality), lambda_OFFER:\n', np.sum( (lambda_DA > lambda_B) * (lambda_DA - lambda_B >= lambda_offer_RES), axis=0))
print('Apart from a difference of \"1" in hour 9 for some reason')


# Visualizations
import matplotlib.pyplot as plt

if show_plots:
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

print('##############\nVISUALIZATION:\n##############')
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

fig = plt.figure(figsize=(6,4),dpi=500)
for beta in betas:
    plt.scatter(CVaR[f'{beta}'], Eprofs[f'{beta}'])
points_to_plot = betas #[0.8,0.9]
for beta in points_to_plot:
    plt.annotate(f'beta={beta}', (CVaR[f'{beta}'], Eprofs[f'{beta}']))
plt.xlabel('CVaR [€]')
plt.ylabel('Expected profit [€]')
#plt.legend()
plt.show()

''' # RISK - requires more than 1 beta-point
fig = plt.figure(figsize=(6,4),dpi=500)
for beta in betas[1:]:
    plt.scatter(CVaR[f'{beta}'], Eprofs[f'{beta}'])
points_to_plot = betas #[0.8,0.9]
for beta in points_to_plot:
    plt.annotate(f'beta={beta}', (CVaR[f'{beta}'], Eprofs[f'{beta}']))
plt.xlabel('CVaR [€]')
plt.ylabel('Expected profit [€]')
#plt.legend()
plt.show()
'''

import seaborn as sns

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

''' RISK - requires 0.0, 0.1, 0.8
fig=plt.figure(figsize=(6,4),dpi=500)
colors=['blue', 'orange']
colors2=['purple', 'red']
for i,beta in enumerate([0.0,0.1]):
      sns.histplot(Eprofs_w[f'{beta}'],
                   color=colors[i],
                   kde=False,
                   label=f'β={beta}',
                   alpha=.7)
      plt.axvline(VaR[f'{beta}'],
                  color=colors2[i],
                  label=f'VaR, β={beta}',
                  linestyle='--',
                  linewidth=4
                  )
      plt.axvline(Eprofs[f'{beta}'],
                  color=colors2[i],
                  label=f'Expected profit, β={beta}',
                  linestyle='-',
                  linewidth=4
                  )

plt.legend()
plt.xlabel('Profit [€]')
plt.ylabel(f'Frequency [out of {W}]')
plt.title('Profit distributions and VaR')
plt.show()

fig=plt.figure(figsize=(6,4),dpi=500)
colors=['blue', 'orange']
colors2=['purple', 'red']
for i,beta in enumerate([0.1,0.8]):
      sns.histplot(Eprofs_w[f'{beta}'],
                   color=colors[i],
                   kde=False,
                   label=f'β={beta}',
                   alpha=.7)
      plt.axvline(VaR[f'{beta}'],
                  color=colors2[i],
                  label=f'VaR, β={beta}',
                  linestyle='--',
                  linewidth=4
                  )
      plt.axvline(Eprofs[f'{beta}'],
                  color=colors2[i],
                  label=f'Expected profit, β={beta}',
                  linestyle='-',
                  linewidth=4
                  )
      
plt.legend()
plt.xlabel('Profit [€]')
plt.ylabel(f'Frequency [out of {W}]')
plt.title('Profit distributions and VaR')
plt.show()

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
plt.legend(lines+lines2,labels+labels2,loc=0)
plt.show()
'''

fig, ax = plt.subplots(figsize=(6,4))

cols = ['tab:blue', 'tab:orange']
markers=['*','x','s','*','p','+']

for i,k in enumerate(p_DA_sol_p90.keys()):
     ax.plot(p_DA_sol_p90[k], label=k+', DA', color=cols[0], alpha=1-.2*i, marker=markers[i])
     ax.plot(p_RES_sol_p90[k], label=k+', RES', color=cols[1], alpha=.9-.2*i, marker=markers[i])

ax.set_xlabel('Time of day [h]')
ax.set_ylabel('Power offered [MW]')

ax.legend(loc=0)

print('##############\nScript is done\n##############')
