#  Given an sorted array of integers, and knowing there is a missing number on it, find that number.

n = input() # array length 
array = input() # sorted array of length N

N = int(n)

lst = array.split()
for i in range(N):
    lst[i] = int(lst[i])

i = 0
position = i
for i in range(N):
    if lst[position] != i + 1: # if position doesn't match with the actual value the missing number is i + 1
        number = i + 1 
        break
    else:
        position += 1
        
print(number) # print missing number
