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

with torch.no_grad():
    layer.weight = torch.nn.Parameter(
        torch.tensor([
            [1.0,2.0,3.0],
            [4.0,5.0,6.0]
        ])
    )

    layer.bias = torch.nn.Parameter(
        torch.tensor([
            10.0,
            20.0
        ])
    )

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