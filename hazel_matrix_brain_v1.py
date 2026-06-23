import numpy as np

# w = np.array([1,2,3])
# print(w)
# print(type(w))
# print(w.shape)
# print()

# a = np.array([1,2,3])

# b = np.array([[1,2,3]])

# c = np.array([
#     [1],
#     [2],
#     [3]
# ])

#print(a.shape)
#print(b.shape)
 #print(c.shape)
 
#print()
#text
W = np.array([
    [1,2,3],
    [4,5,6]
])

X = np.array([
    [1],
    [0],
    [3]
])

b = np.array([
    [-30],
    [20]
])

H = W @ X + b
A = np.maximum(0,H)

print(H)
print(H.shape)

print()

print("A =")
print(A)
print("A shape = " )
print(A.shape)
print()

W2 = np.array([[2,3]])
b2 = np.array([[0]])

Y = W2 @ A + b2

print("Y =")
print(Y)

print("Y shape =")
print(Y.shape)