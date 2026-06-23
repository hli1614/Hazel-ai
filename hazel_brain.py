# creator: Haoze Li
# I hope Hazel learn "y=2x+1"
# My first Lab code

# Data
xs = [1,2,3,4]
ys = [1,4,9,16]

#Offset(bias)
w = 0.0
b = 0.0

#Learn rate (学习率)
lr = 0.01

#Start
for epoch in range(1000):
    total_loss = 0
    for x, y_ture in zip(xs, ys):
        #Prediction
        y_pred = w * x +b
        #Calculation error
        loss = (y_pred - y_ture) **2
        #Cumulative error
        total_loss =+ loss
        #Calculate the gradient
        dloss_dypred = 2 * (y_pred - y_ture)
        dypred_dw = x
        dypred_db = 1

        dloss_dw = dloss_dypred * dypred_dw
        dloss_db = dloss_dypred * dypred_db

        #updata data
        w -= lr * dloss_dw
        b -= lr * dloss_db

        #每100轮打印1次
    if epoch % 100 == 0:
        print("epoch", epoch)
        print("loss", total_loss)
        print("w =", w)
        print("b =", b)
        print()

print("Hazel learning complete")
print("Fianl w =", w)
print("Fianl b =", b)

#text
x = 10
print("当 x = 10 时")
print("Hazel pred y =", w * x + b)






