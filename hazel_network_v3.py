weight = [9.0,0,0,0]
activation = [10,0,0,0]

#result = 0

#for w , a in zip(weight, activation):
 #   result += w * a
import numpy as np

weights = np.array([1, 2, 3])
activations = np.array([4, 5, 6])

result = np.dot(weights, activations)


print(result) 
