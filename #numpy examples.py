#numpy examples

import numpy as np
import matplotlib.pyplot as plt

def add_arrays(arr1, arr2):
    return np.add(arr1, arr2)

def multiply_arrays(arr1, arr2):
    return np.multiply(arr1, arr2)

def subtract_arrays(arr1, arr2):
    return np.subtract(arr1, arr2)

def divide_arrays(arr1, arr2):
    return np.divide(arr1, arr2)

def create_array(fill_value):
    return np.array(fill_value)


ma = create_array (([4,7], [8,9]))
mb = create_array (([1,2], [3,4]))

#access elements
print(ma[0, 0])  # Output: 4
print(mb[0, 1])  # Output: 2

#replace elements
ma[0, 0] = 10
print(ma[0, 0])  # Output: 10

mb[1, 1] = 20
print(mb[1, 1])  # Output: 20

plt.plot(ma, label='ma')
plt.plot(mb, label='mb')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of ma and mb')

plt.legend()
plt.show()