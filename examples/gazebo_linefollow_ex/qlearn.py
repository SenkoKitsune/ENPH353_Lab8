import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.
        with (open(filename, "rb")) as openfile:
            try:
                self.q = pickle.load(openfile)
            except EOFError:
                print("Error")

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        with open(filename, 'wb') as savefile:
            pickle.dump(self.q, savefile)

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        choice = random.random()
        if(choice < self.epsilon):
            choice_action = random.choice(self.actions)
        
        else:
            # :3 :3 :3 :3 :3 :3 :3 :3 :3
            # Choose the action with the highest Q value
            
            maxQ = 0.0
            if(state in self.q):
                maxQ = max(self.q[state])  # Find the maximum Q value
        
            # Handle ties: get all actions with the maximum Q value
            best_actions = []
            for action in self.actions:
                if self.q.get((state, action), 0.0) == maxQ:
                    best_actions.append(action)
            
            if(len(best_actions) > 0):
                choice_action = random.choice(best_actions)  # Randomly select among the best actions
            else:
                choice_action = random.choice(self.actions)

        if(return_q):
            return choice_action, self.q.get[(state,choice_action)]
        else:
            return choice_action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   reward

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
        if((state1, action1) not in self.q):
           self.q[(state1, action1)] = 0.0 

        current_q = self.q[(state1, action1)]

        maxQ = 0.0

        if(state2 in self.q):
            maxQ = max(self.q[state2])

        self.q[(state1,action1)] = current_q + self.alpha * (reward + self.gamma * maxQ - current_q)
