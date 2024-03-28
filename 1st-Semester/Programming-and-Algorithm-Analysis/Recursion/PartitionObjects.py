# Write a function that counts the number of ways you can partition n objects using parts up to m (assuming m >= 0).

def partitions(n, m):
    if n == 0: # base case no. 1
        return 1
    elif m == 0 or n < 0: # base case no. 2
        return 0
    return partitions(n - m, m) + partitions(n, m - 1) # recursive call
    
print(partitions(7, 3)) # expected value 8
print(partitions(9, 5)) # expected value 23
