# Given a mathematical expression (string) determine whether it's correctly written or not. A expression is considered correct if all the if all the parentheses match and are sorted. Note: to simplify the task, the expression will only contain the following characters '(', ')', '{', '}', '[', and ']'.

s = input() # mathematical expression

co = 0 # no. open brackets
cc = 0 # no. closed brackets
b = 0 # no. brackets 
sb = 0 # no. square brackets
wb = 0 # no. weird brackets (curly braces) 

for i in s:
	if i == '(' or i == '[' or i == '{':
	    co += 1
	    if i == '(':
	        b += 1
	    if i == '[':
	        sb += 1
	    if i == '{':
	        wb += 1
	    i == '.' # point out the character
	elif i == ')' or i == ']' or i == '}':
	    cc += 1
	    if i == ')':
	        b += 1
	    if i == ']':
	        sb += 1
	    if i == '}':
	        wb += 1
	    i == '-' # point out the character
	    
if (co + cc) % 2 == 0 and b % 2 == 0 and sb % 2 == 0 and wb % 2 == 0 :
    print("True")
else:
    print("False")
