# Given an array of length n, and given m queries, where each query consists of an integer i, find the i-th number in the array. 

n = input() # array length 
N = int(n)

array = input() # array elements (e.g. '4 2 8 1 5')
lst = array.split()
for i in range(N):
    lst[i] = int(lst[i])

def sort(array): # sort the array 
	for i in range(len(array) - 1):
	    for j in range(len(array) - 1 - i):
	        if array[j] > array[j + 1]:
	            temp = array[j + 1] 
	            array[j + 1] = array[j]
	            array[j] = temp
	return array
	
sorted_lst = sort(lst) 

m = input() # number of queries 
M = int(m)

index_lst = []
for i in range(M):
     ind = input() # input each query 
     IND = int(ind)
     index_lst.append(IND)
    
for j in range(M):
    for k in range(N):
        if index_lst[j] == (k + 1):
            print(sorted_lst[k])
            

