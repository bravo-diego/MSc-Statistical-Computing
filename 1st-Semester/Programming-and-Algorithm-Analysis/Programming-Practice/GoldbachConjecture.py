# Given an even integer N greater than 2, find two prime numbers p and q such that p + q = N. Note: p and q are not consecutive numbers.

n = input()
N = int(n) # integer N

primos = []
i = 2
for i in range(N): # find all prime numbers less than N
    c = 0
    for j in range(1, i + 1):
        if (i % j == 0):
            c += 1
    if (c == 2):
        primos.append(i) # list of prime numbers

print(primos)

flag = 0
for k in range(len(primos)):
    if (flag == 1):
        break
    for l in range(k+1, len(primos)):
        if (primos[k] + primos[l] == N):
            print(primos[k], primos[l])
            flag += 1
            break
