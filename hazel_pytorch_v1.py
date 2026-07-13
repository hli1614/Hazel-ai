import torch
import torch.nn as nn

#x = torch.tensor([1, 2, 3])

# W = torch.tensor([
#     [1,2,3],
#     [4,5,6]
# ])

#B = torch.tensor([
   ##@[3]
#])

#print(A @ B)

#print(x)

#print(type(x))

#print(x.shape)

layer = nn.Linear(3,2)

x = torch.tensor([
    [1.0,2.0,3.0]
])
print()

print("Input =")
print(x)

print()

print("Output =")
print(layer(x))
print()

print(layer.weight)
print(layer.weight.shape)
print()

print(layer.bias)
print(layer.bias.shape)
print()