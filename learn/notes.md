# Master Thesis

The code example below presents the transition functions and reward signals of the "Bandit Walk" (Deterministic) and the "Bandit Slippery Walk" (Stochastic) presented in the book.

```py
P = {
    0: {
        # Here, the MDP would define the possible
        # actions in state 0.
    },
    1: {
        # The action of a state are defined as follows:
        # action: [
        #   (probability of transition,
        #   next state,
        #   reward,
        #   indication of whether next state is terminal)]
        0: [(1.0, 0, 0.0, True)],
        1: [(1.0, 2, 1.0, True)],
        # The two actions above have deterministic transitions,
        # stochastic transitions would look as follows:
        0: [(0.8, 0, 0.0, True), (0.2, 2, 1.0, True)],
        1: [(0.8, 2, 1.0, True), (0.2, 0, 0.0, True)]
        # Here arriving in state 2 provides a reward of +1, 
        # while state 0 provides no rewards, the stochastic
        # transitions mean that selecting action 1 (going right 
        # in book) does not guarantee that the agent arrives in
        # the rewarding state 2.
    }
}
```

The transition function and reward signal of "Frozen Lake" is not presented because too large but can be loaded using 

```py
import gymnasium as gym

env = gym.make('FrozenLake8x8-v1')
P = env.unwrapped.P
```

The value function for every state of an MDP can be approximated using the policy-evaluation algorithm

```py
import numpy as np

def policy_evaluation(pi, P, gamma = 1.0, theta = 1e-10):
    """
    The policy evaluation algorithm only requires the specific policy 'pi' of interest
    and the MDP 'P' the policy runs on.
    """
    prev_V = np.zeros(len(P))
    while True:
    # The forever loop that stops when the improvement between 'prev_V' and 'V' is smaller
    # than the value of the argument 'theta'.
        V = np.zeros(len(P))
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        """
        Here, the inclusion of '(not done)' ensure the value of any next state when
        landing on a terminal state is always 0 to avoid infinite sums.
        """
        if np.max(np.abs(prev_V - V)) < theta:
            """
            If the state-value function stop changing (enough), we says that they have
            converged and stop the algorithm.
            """
            break   
        prev_V = V.copy()
    return V
```

after determine the value of a policy, one might desire to find a policy with a better value as given by the state value function,
this is done using the policy-improvement algorithm below

```py 
import numpy as np

def policy_improvement(V, P, gamma = 1.0):
    """
    The arguments are the state-value function 'V' of the policy you want to improve,
    the MDP 'P', and optionally the discount factor 'gamma'.
    """
    Q = np.zeros((len(P), len(P[0])), dtype = np.float64)
    
    for s in range(len(P)):
        for a in range(len(P[s])):
            """
            We loop through the states, actions, and transitions to compute the Q-function.
            """
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    
    # We obtain a greedy policy by take for each state the action that maximizes
    # the Q-function.
    new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis = 1))}[s]
    return new_pi
```

The policy-iteration algorithm takes us from any arbitrary to an optimal one

```py 
import numpy as np

def policy_iteration(P, gamma = 1.0, theta = 1e-10):
    # We create a list of random actions and map those to states
    # to create a randomly generated policy.
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s : a for s, a in enumerate(random_actions)}[s]
    
    while True:
        old_pi = {s : pi(s) for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(pi, P, gamma, theta)
        
        if old_pi == {s: pi(s) for s in range(len(P))}:
            break
            
    return V, pi
```

