# Given a string, find all the palindromic sequences and count how many different palindromic sequences are in the string. 

s = input() # string

words = 0

if s == s[::-1]: # verify if s is a palindromic sequences itself
    words += 1

i = 1
for i in range(len(s)): # find palindromic subsequences in s
    if s[:-i] == s[:-i][::-1]:
        words += 1

characters = []
for char in  s:
    if char not in characters:
        characters.append(char) # unique characters in the string

words += len(characters)

print(words)
