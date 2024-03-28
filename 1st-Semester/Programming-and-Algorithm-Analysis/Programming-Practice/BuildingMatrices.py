# Given an integer N (input), build two square matrices N by N such that: 

	# elements of matrix A are all ones except the elements in the diagonal, where elements are 2; 
	
	# elements of matrix B are all ones except the elements in the diagonal, where elements are 3.

n = input()
N = int(n)

# matrix A
for c in range(N): # columns
    for r in range(N): # rows
        if(c == r):
            print(2, end=' ')
        else:
            print(1, end=' ')
            if(r == (N - 1)):
                print("") # linebreak
print("\n")

# matrix B
for c in range(N): # columns
    for r in range(N): # rows
        if(c == r):
            print(3, end=' ')
        else:
            print(1, end=' ')
            if(r == (N - 1)):
                print("") # linebreak
print("\n")
