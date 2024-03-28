# A prime number (or prime) is a natural number greater than 1 that has no positive divisors other than 1 and itself. Two prime numbers are called twin primes if there is present only one composite number between them. Or we can also say two prime numbers whose difference is two are called twin primes. 

# Given an integer N, find all the pairs of twin primes (p, p + 2) where p + 2 <= N. 

c = input() # M integers
C = int(c)

numbers = []
for i in range(C):
    n = input()
    N = int(n)
    numbers.append(N)

primos = []
i = 2
for i in range(max(numbers)): # find all prime numbers 
    c = 0
    for j in range(1, i + 1):
        if (i % j == 0):
            c += 1
    if (c == 2):
        primos.append(i) # list of prime numbers
         
for k in range(len(numbers)):
    sublist = []
    for n in range(len(primos)):
        if (primos[n] <= numbers[k]):
            sublist.append(primos[n]) 
    for l in range(len(sublist) - 1):
        for m in range(l + 1, len(sublist)):
            diff = sublist[l] - sublist[m]
            if(abs(diff) == 2):
                print(sublist[l], sublist[m]) # twin primes
    print("")
    
    
        
