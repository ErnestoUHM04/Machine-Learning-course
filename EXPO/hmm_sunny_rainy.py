import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

# Step 1
states = ["Rainy", "Cloudy", "Sunny"]
n_states = len(states)
print('Number of hidden states :',n_states)

observations = ["Sad", "Happy"]
n_observations = len(observations)
print('Number of observations  :',n_observations)

# Step 2
state_probability = np.array([0.218, 0.273, 0.509])
print("State probability: ", state_probability)

transition_probability = np.array([[0.5, 0.3, 0.2],
                                   [0.4, 0.2, 0.4],
                                   [0.0, 0.3, 0.7]])
print("\nTransition probability:\n", transition_probability)
emission_probability= np.array([[0.9, 0.1],
                                [0.6, 0.4],
                                [0.2, 0.8]])
print("\nEmission probability:\n", emission_probability)

# Step 3
model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = state_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# Step 4                         HAPPY HAPPY SAD SAD HAPPY SAD
observations_sequence = np.array([1, 1, 0, 0, 1, 0]).reshape(-1, 1)

# Step 5
# Predict the most likely sequence of hidden states
hidden_states = model.predict(observations_sequence)
print("Most likely hidden states:", hidden_states)

# Step 6
log_probability, hidden_states = model.decode(observations_sequence,
                                              lengths = len(observations_sequence),
                                              algorithm ='viterbi' )

print('Log Probability :',log_probability)
print("Most likely hidden states:", hidden_states)

# Step 7
sns.set_style("whitegrid")
plt.plot(hidden_states, '-o', label="Hidden State")
plt.xlabel('Time step')
plt.ylabel('Most Likely Hidden State')
plt.title("Sunny or Rainy")
plt.legend()
plt.show()