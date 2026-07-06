import torch
x = torch.tensor([1,2,3])
print(x)
print(type(x))
print(x.shape)
print(x * 2)

A = torch.tensor([
    [1,2,3],
    [4,5,6]
])
B = torch.tensor([
    [1],
    [2],
    [3]
])
print(A @ B)
print((A @ B).shape)