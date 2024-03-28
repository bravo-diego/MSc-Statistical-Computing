# Write a function that given an input n, sums all non negative integers up to n.

def sumint(n):
    if n == 0: # base case 
        return 0
    return n + sumint(n - 1) # recursive call 
    
print(sumint(5)) # expected value 15 
print(sumint(7)) # expected value 28
