
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
NUM_ROUNDS = 30
D = context_arrs.shape[1] #dimension of feature vector

##loop over trials
ucb_accs1 = np.zeros((NUM_TRIALS,NUM_ROUNDS))
ucb_regrets1 = np.zeros((NUM_TRIALS,NUM_ROUNDS))
ucb_accs2 = np.zeros((NUM_TRIALS,NUM_ROUNDS))
ucb_regrets2 = np.zeros((NUM_TRIALS,NUM_ROUNDS))
ucb_accs3 = np.zeros((NUM_TRIALS,NUM_ROUNDS))
ucb_regrets3 = np.zeros((NUM_TRIALS,NUM_ROUNDS))
base_regrets=np.zeros((NUM_TRIALS,NUM_ROUNDS))
base_accs = np.zeros((NUM_TRIALS,NUM_ROUNDS))

for m in range(NUM_ROUNDS):
   ucb_bandits1 = [linucb_Bandit(D,0.5) for j1 in range(3)]
   ucb_bandits2 = [linucb_Bandit(D,1) for j1 in range(3)]
   ucb_bandits3 = [linucb_Bandit(D,1.5) for j1 in range(3)]
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
      
      #####LinearUCB1
      ls_ucb1 = [u.get_linucb(x)[0] for u in ucb_bandits1]
      ucb_j1 = np.argmax(ls_ucb1)
      ucb_r1 = pull(ucb_j1,a)
      ucb_accs1[t,m] = get_acc(ucb_j1,a)
      ucb_regrets1[t,m] = 1-ucb_r1
      ucb_bandits1[ucb_j1].update(x,ucb_r1)
      #####LinearUCB2
      ls_ucb2 = [u.get_linucb(x)[0] for u in ucb_bandits2]
      ucb_j2 = np.argmax(ls_ucb2)
      ucb_r2 = pull(ucb_j2,a)
      ucb_accs2[t,m] = get_acc(ucb_j2,a)
      ucb_regrets2[t,m] = 1-ucb_r2
      ucb_bandits2[ucb_j2].update(x,ucb_r2)
      #####LinearUCB3
      ls_ucb3 = [u.get_linucb(x)[0] for u in ucb_bandits3]
      ucb_j3 = np.argmax(ls_ucb3)
      ucb_r3 = pull(ucb_j3,a)
      ucb_accs3[t,m] = get_acc(ucb_j3,a)
      ucb_regrets3[t,m] = 1-ucb_r3
      ucb_bandits3[ucb_j3].update(x,ucb_r3)

base_cum_regrets = np.apply_along_axis(np.cumsum,0,base_regrets)
base_acc_rates = np.apply_along_axis(calc_accuracy,0,base_accs)
ucb_cum_regrets1 = np.apply_along_axis(np.cumsum,0,ucb_regrets1)
ucb_acc_rates1 = np.apply_along_axis(calc_accuracy,0,ucb_accs1)
ucb_cum_regrets2 = np.apply_along_axis(np.cumsum,0,ucb_regrets2)
ucb_acc_rates2 = np.apply_along_axis(calc_accuracy,0,ucb_accs2)
ucb_cum_regrets3 = np.apply_along_axis(np.cumsum,0,ucb_regrets3)
ucb_acc_rates3 = np.apply_along_axis(calc_accuracy,0,ucb_accs3)

#fig = plt.figure()
plt.plot(base_cum_regrets[:,0], label='baseline_regrets')
plt.plot(ucb_cum_regrets1[:,0], label='alpha=0.50')
plt.plot(ucb_cum_regrets2[:,0], label='alpha=1.0')
plt.plot(ucb_cum_regrets3[:,0], label='alpha=1.50')
plt.legend()
# Add title and axis names
plt.xlabel('samples')
plt.ylabel('regrets')
plt.title('[1 0] rewards')
plt.show()

plt.plot(base_acc_rates[:,0], label='baseline_accuracy')
plt.plot(ucb_acc_rates1[:,0], label='alpha=0.5')
plt.plot(ucb_acc_rates2[:,0], label='alpha=1.0')
plt.plot(ucb_acc_rates3[:,0], label='alpha=1.5')
plt.legend()
# Add title and axis names
plt.xlabel('samples')
plt.ylabel('accuracy')
plt.title('[1 0] rewards + pca')
plt.show()


mean_base=np.mean(base_acc_rates[-1])
mean_ucb1=np.mean(ucb_acc_rates1[-1])
mean_ucb2=np.mean(ucb_acc_rates2[-1])
mean_ucb3=np.mean(ucb_acc_rates3[-1])

#ts_interval=calc_interval(ts_acc_rates[-1])
#ucb_interval=calc_interval(ucb_acc_rates[-1])


import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc


def plot_mean_and_CI(mean, lb, ub, label0,color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), lb, ub,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean, label=label0)


base_result = np.apply_along_axis(my_calc_interval,1,base_acc_rates)
ucb_result1 = np.apply_along_axis(my_calc_interval,1,ucb_acc_rates1)
ucb_result2 = np.apply_along_axis(my_calc_interval,1,ucb_acc_rates2)
ucb_result3 = np.apply_along_axis(my_calc_interval,1,ucb_acc_rates3)

# plot the data
#fig = plt.figure(1, figsize=(7, 2.5))
plot_mean_and_CI(base_result[:,1], base_result[:,0], base_result[:,2],'base avg accuracy,ending: %1.4f' %(mean_base), color_mean='darkturquoise', color_shading='turquoise')
plot_mean_and_CI(ucb_result1[:,1], ucb_result1[:,0], ucb_result1[:,2],'ucb alpha =0.5,ending: %1.4f' %(mean_ucb1), color_mean='darkorange', color_shading='orange')
plot_mean_and_CI(ucb_result2[:,1], ucb_result2[:,0], ucb_result2[:,2],'ucb alpha =1.0,ending: %1.4f' %(mean_ucb2), color_mean='darkgreen', color_shading='green')
plot_mean_and_CI(ucb_result3[:,1], ucb_result3[:,0], ucb_result3[:,2],'ucb alpha =1.5,ending: %1.4f' %(mean_ucb3), color_mean='darkred', color_shading='red')
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