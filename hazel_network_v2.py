# Hazel Neural Network v1
# creator: Haoze Li

#Training data
xs = [1,2,3,4]
ys = [1,4,9,16]

# First Layer Parameters
w1 = 0.5
b1 = -1.0

w2 = 0.1
b2 = -3.0

# Output Layer
w31 = 0.2              #change
w32 = 0.8              #change
b3 = 0.0

#Learning rate
lr = 0.0101

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
for epoch in range(10000):
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
     y_pred = w31 * a1 +w32 * a2 + b3          #change


# Loss
     loss = (y_pred - y_true) **2
     total_loss += loss

# Backpropagation
     dloss_dypred = 2 * (y_pred - y_true)

# Output Layer Gradients
     dypred_dw31 = a1
     dypred_dw32 = a2
     dypred_db3 = 1

     dloss_dw31 = dloss_dypred * dypred_dw31 #change
     dloss_dw32 = dloss_dypred * dypred_dw32 #change
     dloss_db3 = dloss_dypred * dypred_db3

# Hidden Layer Gradients
     dypred_da1 = w31
     da1_dh1 = relu_derivative(h1)
     dh1_dw1 = x
     dh1_db1 = 1

     dloss_dw1 = dloss_dypred * dypred_da1 * da1_dh1 * dh1_dw1
     dloss_db1 = dloss_dypred * dypred_da1 * da1_dh1 * dh1_db1
 
     dypred_da2 = w32
     da2_dh2 = relu_derivative(h2)
     dh2_dw2 = x
     dh2_db2 = 1

     dloss_dw2 = dloss_dypred * dypred_da2 * da2_dh2 * dh2_dw2
     dloss_db2 = dloss_dypred * dypred_da2 * da2_dh2 * dh2_db2

# Update Parameters
     w1 -= lr * dloss_dw1
     w2 -= lr * dloss_dw2
     w31 -= lr * dloss_dw31    #change
     w32 -= lr * dloss_dw32    #change
     b1 -= lr * dloss_db1
     b2 -= lr * dloss_db2
     b3 -= lr * dloss_db3

    if epoch % 1000 ==0:
     print("epoch: ", epoch)   
     print("loss: ", total_loss)
     print()

print("Hazel Neural Network Complete")



#Text
x = 4



h1 = w1 * x + b1
a1 = relu(h1)
h2 = w2 * x + b2
a2 = relu(h2)

y_pred = w31 * h1 + w32 * h2 + b3 #change

print("x =", x)
print("Hazel predicts =", y_pred)

print()

print("h1 =", h1)
print("h2 =", h2)

print()

print("a1 =", a1)
print("a2 =", a2)

print()

print("===== Hazel Brain =====")

print("w1 =", w1)
print("b1 =", b1)

print("w2 =", w2)
print("b2 =", b2)

print("w31 =", w31)
print("w32 =", w32)

print("b3 =", b3)