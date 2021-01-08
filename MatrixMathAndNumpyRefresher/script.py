import numpy as np

# Scalars
s = np.array(5)
s.shape

x = s + 3
x

# Vectors

v = np.array([1,2,3])
v
v.shape
v[1:]

# Matrices

m = np.array([[1,2,3], [4,5,6], [7,8,9]])
m.shape

# Tensor

t = np.array([[[[1],[2]],[[3],[4]],[[5],[6]]],[[[7],[8]], [[9],[10]],[[11],[12]]],[[[13],[14]],[[15],[16]],[[17],[17]]]])
t.shape

# Changing shapes
v = np.array([1,2,3,4])
x = v.reshape(1,4)

# Add new dimension
v[None, :].shape
v.shape

## Vectorzied vs for loop:

values = [1,2,3,4,5]
for i in range(len(values)):
    values[i] += 5
    
values = [1,2,3,4,5]
values = np.array(values) + 5

m *= 0
m

## Errors

a = np.array([[1,3],[5,7]])
c = np.array([[2,3,6],[4,5,9],[1,8,7]])

a.shape
c.shape

try: 
    a + c
except:
    print("Soft error")


# Matrix product

m = np.array([[1,2,3], [4,5,6], [7,8,9]])
m2 = np.array([[1,2,3], [4,5,6], [7,8,9]])

m * m2


# 

np.dot([1,2,3], [1,2,3])
np.dot([0, 2, 4, 6], [1, 7, 13, 19])

m * m2
m
np.dot(m, m2)

### Numpy matrix multiplication
a = np.array([[1,2,3,4],[5,6,7,8]])
a
# displays the following result:
# array([[1, 2, 3, 4],
#        [5, 6, 7, 8]])
a.shape
# displays the following result:
# (2, 4)

b = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
b
# displays the following result:
# array([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 11, 12]])
b.shape
# displays the following result:
# (4, 3)

a
b

c = np.matmul(a, b)
c
# displays the following result:
# array([[ 70,  80,  90],
#        [158, 184, 210]])
c.shape
# displays the following result:
# (2, 3)

b.transpose()
