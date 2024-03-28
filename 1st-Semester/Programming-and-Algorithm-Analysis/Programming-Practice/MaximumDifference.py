# Given an array of integers, find the maximum difference between two different elements in the array. 

n = input() # array length 
N = int(n)

numbers = input() # array elements (e.g. '2 9 3 6 1' with an n value of 5)
lst = numbers.split() # list of numbers as string
for i in range(N):
    lst[i] = int(lst[i]) # list of numbers as int
    
dlst = [] 
for j in range(N - 1):
    for k in range(j+1, N):
        diff = (lst[j] - (lst[k]))
        dlst.append(abs(diff))

print(len(dlst)) # total elements in difference list
print(max(dlst)) # maximum difference
    
