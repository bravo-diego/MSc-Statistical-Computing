# A palindrome is a word, verse or sentence or a number that reads the same backward or forward. Given an integer n, find the smaller palindrome divisible by n.

n = input()

N = int(n)
copy = N

copystring = str(copy)
string = str(N)

flag = 0

if len(copystring) == 1 and (copy % N) == 0: # check for individual numbers
    print(copy)
    flag = 1

copy += N
copystring = str(copy)

while flag == 0:
    if copy % N == 0 and copystring == copystring[::-1]:
        print(copy)
        flag += 1
    copy += N # check for all divisible numbers by n
    copystring = str(copy)
