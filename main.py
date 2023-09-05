
import numpy as np
import matplotlib.pyplot as plt
from random import sample,choice
from numpy.linalg import inv
from scipy.stats import multivariate_normal
import scipy.stats as st
from linucb_v1 import linucb_Bandit
from ts_gaussian import TS_Bandit
#import funs
from funs import *

#context_arrs = pca_contexts
#context_arrs = predictor_iter
#context_arrs = scaled_f
context_arrs = context_add_geno

NUM_PATIENTS = context_arrs.shape[0]
NUM_TRIALS = NUM_PATIENTS
NUM_ROUNDS = 30
D = context_arrs.shape[1] #dimension of feature vector

##loop over trials
ucb_accs = np.zeros((NUM_TRIALS,NUM_ROUNDS))
ucb_regrets = np.zeros((NUM_TRIALS,NUM_ROUNDS))
ts_accs = np.zeros((NUM_TRIALS,NUM_ROUNDS))
ts_regrets = np.zeros((NUM_TRIALS,NUM_ROUNDS))
base_regrets=np.zeros((NUM_TRIALS,NUM_ROUNDS))
base_accs = np.zeros((NUM_TRIALS,NUM_ROUNDS))
reg_regrets=np.zeros((NUM_TRIALS,NUM_ROUNDS))
reg_accs = np.zeros((NUM_TRIALS,NUM_ROUNDS))

#from baseline import *

for m in range(NUM_ROUNDS):
   ts_bandits = [TS_Bandit(D,0.25) for j0 in range(3)]
   ucb_bandits = [linucb_Bandit(D,1.0) for j1 in range(3)]
   perm = np.random.permutation(NUM_PATIENTS)
   for t in range(NUM_TRIALS):
      i = perm[t]
      ##actual dose level
      a = np.asscalar(response_arr[i])
      ####BASELINE
      base_r = get_as_rewards(1,a)
      base_regrets[t,m] = -base_r
      base_accs[t,m] = get_acc(1,a)

      ####linear regression
      reg_r = get_as_rewards(wcda_cleaned[i],a)
      reg_regrets[t,m] = -reg_r
      reg_accs[t,m] = get_acc(wcda_cleaned[i],a)
      
      ###feature vector
      x=context_arrs[i]
      x = np.reshape(x,(-1,1))
      
      #####Thompson Sampling
      ls_ts = [c.ts_sample(x) for c in ts_bandits]
      arr_ts = np.array(ls_ts)
      ##random breaking tie
      ts_j=np.random.choice(np.where(np.isclose(arr_ts, arr_ts.max()))[0]) 
      ts_r = get_as_rewards(ts_j,a)
      ts_accs[t,m] = get_acc(ts_j,a)
      ts_regrets[t,m] = -ts_r
      ts_bandits[ts_j].update(x,ts_r)

      #####LinearUCB
      ls_ucb = [u.get_linucb(x)[0] for u in ucb_bandits]
      ucb_j = np.argmax(ls_ucb)
      #arr_ucb = np.array(ls_ucb)
      ##random breaking tie
      ucb_r = get_as_rewards(ucb_j,a)
      ucb_accs[t,m] = get_acc(ucb_j,a)
      ucb_regrets[t,m] = -ucb_r
      ucb_bandits[ucb_j].update(x,ucb_r)

base_cum_regrets = np.apply_along_axis(np.cumsum,0,base_regrets)
base_acc_rates = np.apply_along_axis(calc_accuracy,0,base_accs)
reg_cum_regrets = np.apply_along_axis(np.cumsum,0,reg_regrets)
reg_acc_rates = np.apply_along_axis(calc_accuracy,0,reg_accs)
ts_cum_regrets = np.apply_along_axis(np.cumsum,0,ts_regrets)
ts_acc_rates = np.apply_along_axis(calc_accuracy,0,ts_accs)
ucb_cum_regrets = np.apply_along_axis(np.cumsum,0,ucb_regrets)
ucb_acc_rates = np.apply_along_axis(calc_accuracy,0,ucb_accs)

#fig = plt.figure()
plt.plot(base_cum_regrets[:,0], label='baseline_regrets')
plt.plot(reg_cum_regrets[:,0], label='regression_regrets')
plt.plot(ts_cum_regrets[:,0], label='ts_cum_regrets')
plt.plot(ucb_cum_regrets[:,0], label='ucb_cum_regrets')
plt.legend()
# Add title and axis names
plt.xlabel('samples')
plt.ylabel('regrets')
plt.title('[0 -1] rewards')
plt.show()

plt.plot(base_acc_rates[:,0], label='baseline_accuracy')
plt.plot(reg_acc_rates[:,0], label='regression_accuracy')
plt.plot(ts_acc_rates[:,0], label='ts_accuracy')
plt.plot(ucb_acc_rates[:,0], label='ucb_accuracy')
plt.legend()
# Add title and axis names
plt.xlabel('samples')
plt.ylabel('accuracy')
plt.title('[0 -1] rewards')
plt.show()


mean_base=np.mean(base_acc_rates[-1])
mean_reg=np.mean(reg_acc_rates[-1])
mean_ts=np.mean(ts_acc_rates[-1])
mean_ucb=np.mean(ucb_acc_rates[-1])

ts_interval=calc_interval(ts_acc_rates[-1])
ucb_interval=calc_interval(ucb_acc_rates[-1])


import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc


def plot_mean_and_CI(mean, lb, ub, label0,color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), lb, ub,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean, label=label0)


base_result = np.apply_along_axis(my_calc_interval,1,base_acc_rates)
reg_result = np.apply_along_axis(my_calc_interval,1,reg_acc_rates)
ts_result = np.apply_along_axis(my_calc_interval,1,ts_acc_rates)
ucb_result = np.apply_along_axis(my_calc_interval,1,ucb_acc_rates)

# plot the data
#fig = plt.figure(1, figsize=(7, 2.5))
plot_mean_and_CI(base_result[:,1], base_result[:,0], base_result[:,2],'base avg accuracy,ending: %1.4f' %(mean_base), color_mean='darkturquoise', color_shading='turquoise')
plot_mean_and_CI(reg_result[:,1], reg_result[:,0], reg_result[:,2],'reg avg accuracy,ending: %1.4f' %(mean_reg), color_mean='darkorange', color_shading='orange')
plot_mean_and_CI(ts_result[:,1], ts_result[:,0], ts_result[:,2],'ts avg accuracy,ending: %1.4f' %(mean_ts), color_mean='darkgreen', color_shading='green')
plot_mean_and_CI(ucb_result[:,1], ucb_result[:,0], ucb_result[:,2],'ucb avg accuracy,ending: %1.4f' %(mean_ucb), color_mean='darkred', color_shading='red')
plt.title('[0 -5 -10] reward structure')
plt.xlabel('samples')
plt.ylabel('accuracy')
plt.tight_layout()
plt.grid()
plt.legend()
plt.show()
# save to csv file
#np.savetxt('ts_asymmetric_regret.csv', ts_cum_regrets, delimiter=',')
#np.savetxt('ts_asymmetric_acc.csv', ts_acc_rates, delimiter=',')
#np.savetxt('ucb_asymmetric_regret.csv', ucb_cum_regrets, delimiter=',')
#np.savetxt('ucb_asymmetric_acc.csv', ucb_acc_rates, delimiter=',')