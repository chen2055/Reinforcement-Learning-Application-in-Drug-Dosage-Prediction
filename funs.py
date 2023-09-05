import numpy as np
def pull(es_a,real_a):
    # get real reward
    if es_a == real_a :
       r = 1
    else:
       r = 0
    return r

def real_pull(es_a,real_a):
    # get real reward
    return -abs(es_a - real_a)

def get_real_rewards(es_a,real_a):
    # get real reward
    if es_a == real_a :
       r = 0
    else:
       r = -abs(es_a-real_a)/2
    return r

def get_as_rewards(es_a,real_a):
    # get real reward
    if es_a == real_a :
       r = 0
    elif abs(es_a-real_a) ==2:
       r = -10
    else:
       r = -5
    return r

def get_acc(es_a,real_a):
    # get accuracy
    if es_a == real_a :
       acc = 1
    else:
       acc = 0
    return acc

def get_binary_rewards(es_a,real_a):
    # get real reward
    if es_a == real_a :
       r = 0
    else:
       r = -1
    return r

def calc_accuracy(b):
   return np.cumsum(b)/(np.arange(NUM_TRIALS) + 1)

def calc_interval(a):
   temp=st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
   return temp[0],np.mean(a),temp[1]


def my_calc_interval(a):
   n = len(a)
   lb = np.mean(a) - 1.96*np.std(a)/np.sqrt(n)
   ub = np.mean(a) + 1.96*np.std(a)/np.sqrt(n)
   return lb,np.mean(a),ub