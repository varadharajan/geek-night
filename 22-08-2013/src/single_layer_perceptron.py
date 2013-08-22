import cPickle
import numpy

datasetFile = open("../data/mnist.pkl")
train_set, valid_set, test_set = cPickle.load(datasetFile)
alpha = 1


print "Initializing 784 x 10 weight matrix"
weightMatrix = numpy.random.rand(784, 10)
bias = numpy.random.rand(1,10)

def create_t_vector(digit):
	t_vector = numpy.zeros(shape=(1,10))
	t_vector[0, digit] = 1
	return t_vector[0]

# Run for 10 epoch
for epoch in range(0, 10):
	
	classified_correctly = 0
	
	for id, entry in enumerate(train_set[0]):
		
		digit = train_set[1][id]
		weight_updated = False		
		y_vector = [ 1 if energy > 0 else 0 for energy in (entry.dot(weightMatrix) + bias)[0]]
		t_vector = create_t_vector(digit)	

		for id, result in enumerate(y_vector):			
			if result != t_vector[id]: 
				weight_updated = True
			 	weightMatrix[:,id] = weightMatrix[:,id] + (alpha * (t_vector[id] - result) * entry)
			 	bias[0, id] = bias[0, id] + alpha * (t_vector[id] - result)
		
		if not weight_updated: classified_correctly = classified_correctly + 1

	total = len(train_set[0])
	error = float(total - classified_correctly) / float(total) * 100.0	
	print "Epoch : %d | Correctly Classified Entries : %d | Total Entries : %d | Error Percentage : %f " % (epoch, classified_correctly, total, error)	

print "Training for 10 Epochs completed"	
print "Testing with validation set"

classified_correctly = 0
for id, entry in enumerate(valid_set[0]):
	y_vector = [1 if energy > 0 else 0 for energy in (entry.dot(weightMatrix) + bias)[0]]
	digit = valid_set[1][id]
	classified_digit = y_vector.index(max(y_vector))
	if classified_digit == digit : classified_correctly = classified_correctly + 1

total = len(valid_set[0])
error = float(total - classified_correctly) / float(total) * 100.0	
print "Correctly Classified Entries : %d | Total Entries : %d | Error Percentage : %f" % (classified_correctly, total, error)	