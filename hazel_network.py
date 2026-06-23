# Hazel Neural Network v1
# creator: Haoze Li

#Training data
xs = [1,2,3,4]
ys = [1,4,9,16]

# First Layer Parameters
w1 = 0.5
b1 = 0.0

w2 = -0.5
b2 = 0.0

# Output Layer
w3 = 0.5
b3 = 0.0

#Learning rate
lr = 0.01

#Relu activtion
def relu(x):
    if x>0:
        return x
    else:
        return 0
    
# Relu Derivative
def  relu_derivative(x):
    if x>0:
        return 1
    else:
        return 0
    

#Training
for epoch in range(1000):
    total_loss = 0
    for x, y_true in zip(xs, ys):

#Forward Propagation
# Hidden neuron 1
     h1 = w1 * x + b1
     a1 = relu(h1)

# Hidden neuron 2
     h2 = w2 * x + b2
     a2 = relu(h2)

# Output neuron
     y_pred = w3 * (a1 + a2) + b3


# Loss
     loss = (y_pred - y_true) **2
     total_loss += loss

# Backpropagation
     dloss_dypred = 2 * (y_pred - y_true)

# Output Layer Gradients
     dypred_dw3 = a1 + a2
     dypred_db3 = 1

     dloss_dw3 = dloss_dypred * dypred_dw3
     dloss_db3 = dloss_dypred * dypred_db3

# Hidden Layer Gradients
     dypred_da1 = w3
     da1_dh1 = relu_derivative(h1)
     dh1_dw1 = x
     dh1_db1 = 1

     dloss_dw1 = dloss_dypred * dypred_da1 * da1_dh1 * dh1_dw1
     dloss_db1 = dloss_dypred * dypred_da1 * da1_dh1 * dh1_db1
 
     dypred_da2 = w3
     da2_dh2 = relu_derivative(h2)
     dh2_dw2 = x
     dh2_db2 = 1

     dloss_dw2 = dloss_dypred * dypred_da2 * da2_dh2 * dh2_dw2
     dloss_db2 = dloss_dypred * dypred_da2 * da2_dh2 * dh2_db2

# Update Parameters
     w1 -= lr * dloss_dw1
     w2 -= lr * dloss_dw2
     w3 -= lr * dloss_dw3
     b1 -= lr * dloss_db1
     b2 -= lr * dloss_db2
     b3 -= lr * dloss_db3

     if epoch % 100 ==0:
        print("epoch: ", epoch)   
        print("loss: ", total_loss)
        print()

print("Hazel Neural Network Complete")



#Text
x = 5

h1 = relu(w1 * x + b1)
h2 = relu(w2 * x + b2)

y_pred = w3 * (h1 + h2) + b3

print("x =", x)
print("Hazel predicts =", y_pred)


