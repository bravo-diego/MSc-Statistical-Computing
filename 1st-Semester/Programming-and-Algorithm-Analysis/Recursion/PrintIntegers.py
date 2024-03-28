# Write a function that prints the numbers 1 to n without using loops

def prInt(n):
    if n == 1: # base case
        print(n, end = " ") 
    else:
        prInt(n - 1) # recursive call
        print(n, end = " ")
    
print(prInt(5)) # expected value 1, 2, 3, 4, 5
print(prInt(10)) # expected value 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

