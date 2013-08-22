import numpy as np

num_visible_units = 7
num_hidder_units = 3
learning_rate = 0.2

weight_matrix = np.random.rand(num_visible_units + 1, num_hidder_units)

"""
Input data:

Movies = [Harry Potter, LOTR, 007, Commando, Terminator, MI, Gladiator]

User 1 likes magical Movies
User 2 likes action Movies
User 3 likes Oscar Movies
User 4 likes magical / oscar Movies
User 5 likes sci fi Movies
User 6 likes mission based action movies 
User 7 likes oscar movies 
"""

data = np.array([
	  [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
	  [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
	  [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
	  [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
	  [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
	  [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
	  [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
	  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
	  [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
	  [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	])

def sigmoidal(given_matrix): return 1 / (1 + np.exp(-given_matrix))

def create_state_matrix(data_point, hidden_states): return np.dot(np.array([data_point]).transpose(), np.array([hidden_states]))

for epoch in range(5000):

	for data_point in data:
		
		# Reality phase of CD
		hidden_units_activation = sigmoidal(np.dot(data_point, weight_matrix)) > (np.random.rand(1, num_hidder_units)/2)[0]
		positive_positions = create_state_matrix(data_point, hidden_units_activation)

		# Dreaming phase of CD				
		visible_units_activation = sigmoidal(np.dot(hidden_units_activation, np.transpose(weight_matrix))) > np.random.rand(1, num_visible_units+1)[0]
		negative_positions = create_state_matrix(visible_units_activation, hidden_units_activation)

		# Update weights accordingle
		weight_matrix += learning_rate * (positive_positions - negative_positions)

	print "DayDreaming completed for %d epoch" % (epoch)

print weight_matrix

while True:
	user_preference = np.array(eval(raw_input("Enter user Preference : ")))
	hidden_units_activation = sigmoidal(np.dot(user_preference, weight_matrix)) > (np.random.rand(1, num_hidder_units)/2)[0]
	visible_units_activation = sigmoidal(np.dot(hidden_units_activation, np.transpose(weight_matrix))) > np.random.rand(1, num_visible_units+1)[0]
	print visible_units_activation

	# Sample user preference :  [1.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0]
	# Sample user preference :  [1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]