
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample,choice
from numpy.linalg import inv

class linucb_Bandit:
  def __init__(self, D, alpha):
    self.A = np.identity(D)
    self.b = np.zeros((D,1))
    self.alpha = alpha

  def get_linucb(self,x):
    # get linucb scores
    theta = np.matmul(inv(self.A),self.b)
    uncertainty = np.asscalar(self.alpha * np.sqrt(x.T.dot(inv(self.A)).dot(x)))
    reward = np.asscalar(theta.transpose().dot(x))
    #reward = np.matmul(theta.transpose(),x)
    score = reward + uncertainty
    return score,reward,uncertainty

  def update(self, x,r):
    self.A = self.A + np.matmul(x,x.transpose())
    self.b = self.b + r*x



"""
def pull(es_a,real_a):
    # get real reward
    if es_a == real_a :
       r = 1
    else:
       r = 0
    return r

def update_alpha(t):
   #alpha = 1/(t+2)
   alpha = ALPHA
   return alpha


##randomly assign action to each patient
#a0 =np.zeros((response_arr.shape[0], 1))
#for i in range(response_arr.shape[0]):
#    a0[i,0] = choice(range(2))


NUM_PATIENTS = context_arrs.shape[0]
#NUM_PATIENTS = len(context_arrs)
NUM_TRIALS = 200
D = context_arrs.shape[1] #dimension of feature vector
#D = context_arrs[0].shape[1] #dimension of feature vector
ALPHA = 0.25

bandits = [linucb_Bandit(D) for j in range(3)]
##loop over trials
#total_rewards = 0
#mean_reward=[]
rewards = np.zeros(NUM_TRIALS)
#rewards = np.zeros((NUM_TRIALS,3))
regrets = np.zeros(NUM_TRIALS)
#alpha = ALPHA
#v_uncertainty = []
#v_reward = []

for t in range(NUM_TRIALS):
   i = t % NUM_PATIENTS
   #i=choice(range(NUM_PATIENTS))
   #x=context_arrs[i,:].T
   x=context_arrs[i]
   x = np.reshape(x,(-1,1))
   a = np.asscalar(response_arr[i])
   #alpha = update_alpha(t)
   #ls_ucb = [get_linucb(c.A,c.regress(), x, alpha)[0] for c in bandits]
   ls_ucb = [u.get_linucb(x)[0] for u in bandits]
   #ls_reward = [linucb(c.A,c.regress(), x, alpha)[1] for c in bandits]
   #ls_uncertainty = [linucb(c.A,c.regress(), x, alpha)[2] for c in bandits]
   #v_reward.append(ls_reward[0])
   #v_uncertainty.append(ls_uncertainty[0])
   #na_inds=np.nonzero(pd.isnull(ls_ucb))
   na_inds = [index for index,element in enumerate(ls_ucb) if np.isnan(element)]
   if len(na_inds)>0:
      print("NAs appeared in iteration: ", t)
   
   arr_ucb = np.array(ls_ucb)
   ##random breaking tie
   #j=np.random.choice(np.where(arr_ucb==arr_ucb.max())[0]) 
   j=np.random.choice(np.where(np.isclose(arr_ucb, arr_ucb.max()))[0]) 
   #j = np.argmax(ls_ucb)
   r = pull(j,a)
   rewards[t] = r
   regrets[t] = 1-r
   #total_rewards = total_rewards + r
   #mean_reward.append(total_rewards/(t+1))
   bandits[j].update(x,r)
   #context_arrs[i,0]=j


cumulative_rewards = np.cumsum(rewards)
win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
plt.plot(win_rates, label='mean_reward')
plt.legend()
plt.show()

cumulative_rewards2 = np.cumsum(rewards2)
win_rates2 = cumulative_rewards2/ (np.arange(NUM_PATIENTS) + 1)
plt.plot(win_rates2, label='mean_reward2')
plt.legend()
plt.show()

cumulative_regrets = np.cumsum(regrets)
exp_regrets = cumulative_regrets / (np.arange(NUM_TRIALS) + 1)

plt.plot(win_rates, label='mean_reward')
plt.legend()
plt.show()

plt.plot(exp_regrets, label='regrets')


_ = plt.hist(response_arr[0:400,:], bins='auto')
_ = plt.hist(response_arr[400:,:], bins='auto')

_ = plt.hist(context_arrs[0:400,1], bins='auto')
_ = plt.hist(context_arrs[400:,1], bins='auto')

As = [bandits[j].A for j in range(3)]
bs = [bandits[j].b for j in range(3)]
Thetas = [bandits[j].regress() for j in range(3)]

###Summary for now
##constant alpha is better than 1/t???
##min max scaler is better than normaliation???
"""