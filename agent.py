import numpy as np
from visualize_train import draw_value_image, draw_policy_image

# left, right, up, down
ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]


class AGENT:         # start class AGENT
    def __init__(self, env, is_upload=False):

        self.ACTIONS = ACTIONS          # load left, right, up, down
        self.env = env
        HEIGHT, WIDTH = env.size()      #  determine gridworld's size
        self.state = [0,0]          # set initialization point

        if is_upload:
            dp_results = np.load('./result/dp.npz')       #set saving directory
            self.values = dp_results['V']       # load state value
            self.policy = dp_results['PI']      # load policy
        else:
            self.values = np.zeros((HEIGHT, WIDTH))     # initialize value
            self.policy = np.zeros((HEIGHT, WIDTH,len(self.ACTIONS)))+1./len(self.ACTIONS)  # initial policy
                                                                                        # as equiprobable policy




    def policy_evaluation(self, iter, env, policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        new_state_values = np.zeros((HEIGHT, WIDTH))    #make empty matrix for new state values
        iteration = 0   #initialize interactin value

        #***************************************************
        theta = 0.0001      #set small positive number of theta for accuracy of estimation
        delta = theta       #initialize delta for loop
        old_state_values = self.values.copy()       #save current state values
        while delta >= theta:       #if delta is smaller than theta, it stop
            delta = 0       #initialize delta as 0
            for i in range(HEIGHT):
                for j in range (WIDTH):
                    s = i, j        # set variable i, j as state
                    if not env.is_terminal([i,j]):      #check if state is terminal
                        temp = 0        #to save new state values
                        for a in range (len(policy[i][j])):
                            (next_i, next_j), r = env.interaction([i, j], ACTIONS[a])   #p(r,s'|s,a)
                            next_i, next_j = int(next_i), int(next_j)       #change array to int
                            temp += policy[i,j,a] * (r + discount * old_state_values[next_i, next_j])   #save
                        delta = max(delta, abs(temp - old_state_values[i, j]))  #update delta with state value difference
                        new_state_values[i,j] = temp
            #print(new_state_values)     #check new state values
            old_state_values = new_state_values.copy()      # update state values
            #print( )    # for spacing
        #***************************************************

        draw_value_image(iter, np.round(new_state_values, decimals=2), env=env)
        return new_state_values, iteration





    def policy_improvement(self, iter, env, state_values, old_policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        policy = old_policy.copy()      # save current policy

        #***************************************************
        policy_stable = True        # set the flag to stop the iteration
        new_policy = old_policy.copy()      # start with old policy
        #print(policy_stable)        # check current flag

        for i in range (HEIGHT):
            for j in range (WIDTH):
                s = i,j         # set i, j as current state
                act = np.zeros(len(ACTIONS))    # for empty action matrix
                for a in range(len(ACTIONS)):
                    (next_i, next_j), r = env.interaction(s, ACTIONS[a]) #p(r,s'|s,a)
                    next_i, next_j = int(next_i), int(next_j) # convert array to int
                    act[a] += state_values[next_i, next_j]  #save state values of each action
                best = np.argmax(act)       # find best action value
                new_policy[s] = np.eye(len(ACTIONS))[best]  # save best action for each state
                if not np.array_equal(policy[s], new_policy[s]):    #if the policy is not improved
                    policy_stable = False   # change the flag False and return to evaluation
            #print(policy_stable)    # check current flag
        #print(policy)   # check old policy
        #print(new_policy)       #check new policy
        policy = new_policy     #update better policy
        #***************************************************

        print('policy stable {}:'.format(policy_stable))
        draw_policy_image(iter, np.round(policy, decimals=2), env=env)
        return policy, policy_stable

    def policy_iteration(self):
        iter = 1    #set first number of iteration
        #print(iter)     #check number of iteration
        while (True):   # loop forever before the flag stops
            self.values, iteration = self.policy_evaluation(iter, env=self.env, policy=self.policy)     #policy evaluation
            #print('eval ok')        #check if evaluation is completed
            self.policy, policy_stable = self.policy_improvement(iter, env=self.env, state_values=self.values,
                                                       old_policy=self.policy, discount=1.0)    #policy improvement
            #print('imp ok')         #check if improvement is completed
            iter += 1   #increase iter for next iteration
            if policy_stable == True:       #if policy doesn't improved ( already optimal policy)
                break       #stop the iteration and finish.
        np.savez('./result/dp.npz', V=self.values, PI=self.policy)
        return self.values, self.policy



    def get_action(self, state):
        i,j = state
        return np.random.choice(len(ACTIONS), 1, p=self.policy[i,j,:].tolist()).item()


    def get_state(self):
        return self.state

