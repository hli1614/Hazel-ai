import torch

w = torch.tensor(2.0, requires_grad=True)

x = torch.tensor(3.0)

y = w * x

target = torch.tensor(10.0)

loss = (y - target) ** 2

loss.backward()


print("y =", y)
print("loss =", loss)
print("grad =", w.grad)
print()


print(w.grad)