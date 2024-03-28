# Write a function that takes two inputs n and m and outputs the number of unique paths from the top left corner to bottom right corner of a n x m grid. 

    # Note: you can only move down or right 1 unit at a time.

def upaths(n, m):
    if n == 1 or m == 1: # base case
        return 1
    return upaths(n, m - 1) + upaths(n - 1, m) # recursive call

print(upaths(3, 4)) # expected value 10
print(upaths(4, 4)) # expected value 20
