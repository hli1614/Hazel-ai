import torch
import torch.nn as nn
import torch.optim as optim

# Data  1Batch, 1features
x = torch.tensor([[3.0]])
y = torch.tensor([[10.0]])

# Model  prediction = weight * x + bias
model = nn.Linear(1,1)

# Loss function
#Loss = (prediction - target) ** 2

criterion = nn.MSELoss()

# OPtimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("训练前")
print("weight:", model.weight.item())
print("bias", model.bias.item())

#Training loop
for epoch in range(100):
    #清除上一轮的梯度
    optimizer.zero_grad()

    #Forward
    prediction = model(x)

    #Loss
    loss = criterion(prediction,y)

    #Backward
    loss.backward()

    #Updata
    optimizer.step()

    #Print by 10
    if epoch % 10 == 0:
        print(
            f"epoch: {epoch}, "
            f"prediction: {prediction.item():.4f}, "
            f"loss: {loss.item():.4f}"
        )

print("\nAfter training")
print("Weigth: ", model.weight.item())
print("Bias: ", model.bias.item())

with torch.no_grad():
    final_predicyion = model(x)

print("Final text: ", final_predicyion.item())



