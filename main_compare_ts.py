
import numpy as np
import matplotlib.pyplot as plt
from random import sample,choice
from numpy.linalg import inv
from scipy.stats import multivariate_normal
import scipy.stats as st
from linucb_v1 import linucb_Bandit
from ts_gaussian import TS_Bandit
import funs

#context_arrs = pca_contexts
#context_arrs = predictor_iter
#context_arrs = scaled_f
context_arrs = context_add_geno

NUM_PATIENTS = context_arrs.shape[0]
NUM_TRIALS = NUM_PATIENTS
NUM_ROUNDS = 1
D = context_arrs.shape[1] #dimension of feature vector

##loop over trials
ts_accs1 = np.zeros((NUM_TRIALS,NUM_ROUNDS))
ts_regrets1 = np.zeros((NUM_TRIALS,NUM_ROUNDS))
ts_accs2 = np.zeros((NUM_TRIALS,NUM_ROUNDS))
ts_regrets2 = np.zeros((NUM_TRIALS,NUM_ROUNDS))
base_regrets=np.zeros((NUM_TRIALS,NUM_ROUNDS))
base_accs = np.zeros((NUM_TRIALS,NUM_ROUNDS))

for m in range(NUM_ROUNDS):
   ts_bandits1 = [TS_Bandit(D,0.25) for j0 in range(3)]
   ts_bandits2 = [TS_Bandit(D,1) for j0 in range(3)]
   perm = np.random.permutation(NUM_PATIENTS)
   for t in range(NUM_TRIALS):
      i = perm[t]
      ##actual dose level
      a = np.asscalar(response_arr[i])
      ####BASELINE
      base_r = pull(1,a)
      base_regrets[t,m] = 1-base_r
      base_accs[t,m] = get_acc(1,a)
      
      ###feature vector
      x=context_arrs[i]
      x = np.reshape(x,(-1,1))
      
      #####Thompson Sampling1
      ls_ts1 = [c.ts_sample(x) for c in ts_bandits1]
      ts_j1 = np.argmax(ls_ts1)
      ts_r1 = pull(ts_j1,a)
      ts_accs1[t,m] = get_acc(ts_j1,a)
      ts_regrets1[t,m] = 1-ts_r1
      ts_bandits1[ts_j1].update(x,ts_r1)
      
      #####Thompson Sampling2
      ls_ts2 = [c.ts_sample(x) for c in ts_bandits2]
      ts_j2 = np.argmax(ls_ts2)
      ts_r2 = pull(ts_j2,a)
      ts_accs2[t,m] = get_acc(ts_j2,a)
      ts_regrets2[t,m] = 1-ts_r2
      ts_bandits2[ts_j2].update(x,ts_r2)


base_cum_regrets = np.apply_along_axis(np.cumsum,0,base_regrets)
base_acc_rates = np.apply_along_axis(calc_accuracy,0,base_accs)
ts_cum_regrets1 = np.apply_along_axis(np.cumsum,0,ts_regrets1)
ts_acc_rates1 = np.apply_along_axis(calc_accuracy,0,ts_accs1)
ts_cum_regrets2 = np.apply_along_axis(np.cumsum,0,ts_regrets2)
ts_acc_rates2 = np.apply_along_axis(calc_accuracy,0,ts_accs2)

#fig = plt.figure()
plt.plot(base_cum_regrets[:,0], label='baseline_regrets')
plt.plot(ts_cum_regrets1[:,0], label='ts, v=0.25')
plt.plot(ts_cum_regrets2[:,0], label='ts, v=1')
plt.legend()
# Add title and axis names
plt.xlabel('samples')
plt.ylabel('regrets')
plt.title('[1 0] rewards')
plt.show()

plt.plot(base_acc_rates[:,0], label='baseline_accuracy')
plt.plot(ts_acc_rates1[:,0], label='ts v=0.25')
plt.plot(ts_acc_rates2[:,0], label='ucb v=1')
plt.legend()
# Add title and axis names
plt.xlabel('samples')
plt.ylabel('accuracy')
plt.title('[1 0] rewards')
plt.show()


mean_base=np.mean(base_acc_rates[-1])
mean_ts1=np.mean(ts_acc_rates1[-1])
mean_ts2=np.mean(ts_acc_rates2[-1])


import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc


def plot_mean_and_CI(mean, lb, ub, label0,color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), lb, ub,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean, label=label0)


base_result = np.apply_along_axis(my_calc_interval,1,base_acc_rates)
ts_result1 = np.apply_along_axis(my_calc_interval,1,ts_acc_rates1)
ts_result2 = np.apply_along_axis(my_calc_interval,1,ts_acc_rates2)

# plot the data
#fig = plt.figure(1, figsize=(7, 2.5))
plot_mean_and_CI(base_result[:,1], base_result[:,0], base_result[:,2],'base avg accuracy,ending: %1.4f' %(mean_base), color_mean='darkturquoise', color_shading='turquoise')
plot_mean_and_CI(ts_result1[:,1], ts_result1[:,0], ts_result1[:,2],'ts v =0.25,ending: %1.4f' %(mean_ts1), color_mean='darkgreen', color_shading='green')
plot_mean_and_CI(ts_result2[:,1], ts_result2[:,0], ts_result2[:,2],'ts v =1.0,ending: %1.4f' %(mean_ts2), color_mean='darkred', color_shading='red')
plt.title('Performance comparison over 30 Runs')
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