def forward(self, obs):
    alpha = np.zeros((len(obs), self.num_states))
        # your code here!
    curr_alpha = np.log(self.pi.reshape((1, self.num_states))) + np.log(self.B[:, obs[0]].reshape(1, self.num_states))
    alpha[0] = curr_alpha
    for t, obs_t in enumerate(obs[1:]):
        transition_probs = np.zeros((1, self.num_states))
        for j in range(self.num_states):
            inner_sum = curr_alpha + np.log(self.A[:,j].reshape((1,self.num_states)))
            transition_probs[0, j] = logsumexp(inner_sum)
            if transition_probs[0, j] == float("-inf"):
                print("PROBLEM IN HERE")
                print(t, j)

        emission_probs = np.log(self.B[:, obs_t].reshape(1, self.num_states))
        curr_alpha = transition_probs + emission_probs
        alpha[t+1] = curr_alpha
        
    # print("Alpha: ", alpha)
    return alpha