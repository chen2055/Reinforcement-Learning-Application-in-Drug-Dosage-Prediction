import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample,choice
from numpy.linalg import inv
from scipy.stats import multivariate_normal

class TS_Bandit:
  def __init__(self,D,v):
    self.B = np.identity(D)
    self.f = np.zeros((D,1))
    self.mu_hat = np.zeros((D,1))
    self.v = v
  
  def ts_sample(self,x):
    gaussian_mean=self.mu_hat.flatten()
    mu_tilde = multivariate_normal.rvs(gaussian_mean, self.v**2 * np.linalg.inv(self.B))
    mu_reward = x.T.dot(mu_tilde)
    return mu_reward

  def update(self, x, r):
    self.B = self.B + np.outer(x, x)
    self.f = self.f+ r * x
    self.mu_hat = np.linalg.inv(self.B).dot(self.f)
"""
def pull(es_a,real_a):
    # get real reward
    if es_a == real_a :
       r = 1
    else:
       r = 0
    return r

NUM_PATIENTS = context_arrs.shape[0]
NUM_TRIALS = NUM_PATIENTS
NUM_ROUNDS = 20
D = context_arrs.shape[1] #dimension of feature vector

##loop over trials
rewards = np.zeros((NUM_TRIALS,NUM_ROUNDS))
regrets = np.zeros((NUM_TRIALS,NUM_ROUNDS))
#perms = np.zeros((NUM_TRIALS,NUM_ROUNDS))

for m in range(NUM_ROUNDS):
   bandits = [TS_Bandit(D) for j in range(3)]
   perm = np.random.permutation(NUM_PATIENTS)
   #perms[:,m]=perm
   for t in range(NUM_TRIALS):
      i = perm[t]
      x=context_arrs[i]
      x = np.reshape(x,(-1,1))
      a = np.asscalar(response_arr[i])
      ls_ts = [c.ts_sample(x) for c in bandits]
      na_inds = [index for index,element in enumerate(ls_ts) if np.isnan(element)]
      if len(na_inds)>0:
         print("NAs appeared in iteration: ", t)
   
      arr_ts = np.array(ls_ts)
      ##random breaking tie
      j=np.random.choice(np.where(np.isclose(arr_ts, arr_ts.max()))[0]) 
      r = pull(j,a)
      rewards[t,m] = r
      regrets[t,m] = 1-r
      bandits[j].update(x,r)


cumulative_regrets = np.apply_along_axis(np.cumsum,0,regrets)
def calc_accuracy(b):
   return np.cumsum(b)/(np.arange(NUM_TRIALS) + 1)
win_rates = np.apply_along_axis(calc_accuracy,0,rewards)
# save to csv file
#np.savetxt('ts_gaussian_winrates.csv', win_rates, delimiter=',')
#cumulative_rewards = np.cumsum(rewards)
#win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
plt.plot(win_rates, label='mean_reward')
plt.legend()
plt.show()

a = win_rates[-1]
import scipy.stats as st
t_interval=st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
"""