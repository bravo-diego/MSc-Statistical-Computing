# Array Broadcasting in NumPy

# Term 'broadcasting' describes how numpy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is broadcast across the larger array so that they have compatible shapes. 

import numpy as np
from numpy import array

# The simplest broadcasting example occurs when an array and a scalar value are combined in an operation.

a = array([1.0, 2.0, 3.0])
b = 3.0
print(a * b) 

# A two dimensional array multiplied by a one dimensional array results in broadcasting if number of 1-d array elements matches the number of 2-d array columns; in this case c is added to each row of the product a*b

a = array([[1, 2], [3, 4]])
b = array([[0, 1], [2, 3]])
c = array([7, 9])
print((a*b) + c)



