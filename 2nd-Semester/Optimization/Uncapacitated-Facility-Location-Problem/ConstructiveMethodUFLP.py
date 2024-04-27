# Uncapacited facility location problem (UFLP).

# The uncapacitated facility location problem ( UFLP) involves locating an undetermined number of facilities to minimize the sum of the (annualized) fixed setup costs and the variable costs of serving the market demand from these facilities. UFLP is also known as the “simple” facility location problem SFLP, where both the alternative facility locations and the customer zones are considered discrete points on a plane or a road network.

# Constructive Method

# A rapid and straightforward method to obtain a solution is to generate it randomly among the set of all feasible solutions.

import numpy as np

def costs(locations, clients): # generate variable and fixed costs random matrices
	variable_costs = np.random.randint(clients*10, size = (locations, clients)) # variable costs (distances)
	fixed_costs = np.random.randint(clients*10, size = locations) # fixed costs
	return variable_costs, fixed_costs

def constructive_method(locations, clients, variable_costs): 
	x = np.zeros((locations, clients)) # empty clients matrix
	y = np.zeros(locations) # empty locations vector
	r = np.random.rand(locations) # random number vector
	
	idx = [] # empty list to store open locations indices
	
	for i in range(locations): # decision rule
		if r[i] < 0.5: # if r value is less than 0.5 then enable ith location, otherwise keep it closed
			y[i] = 1
			idx.append(i) # list of open locations indices
	
	if len(idx) == 0: # if the are no open locations return the empty x matrix and y vector
		return x, y
					
	l = 0
	for i in range(clients): # for loop to ensure every client is assigned to a location
		open_locations = sum(y) # no. open locations
		loc = idx[l] # index of open locations
		if y[loc] == 1: # if location is open then assign client to nearest location
			client = np.argmin(variable_costs[loc])
			x[loc][client] = 1 # assign client to ith location
			for j in range(locations): # for loop to avoid clients assign to > 1 location
				for k in range(clients): 
					if k == client:
						variable_costs[j][k] = 1000 
						break
		l += 1
		if l >= open_locations: # reset l value
			l = 0
    
	return y, x
	
def objective_function(x, variable_costs, fixed_costs, y): # evaluate objective function with feasible solution obtained with constructive_method function
	variable_cost = 0
	for i in range(len(variable_costs)):
		for j in range(len(variable_costs[i])):
			variable_cost += variable_costs[i][j] * x[i][j] # variable costs
	print(variable_cost) 
	fixed_cost = np.sum(np.multiply(fixed_costs, y)) # fixed costs
	print(fixed_cost)
	total_cost = variable_cost + fixed_cost # total cost
	return total_cost
    
m = 4 # no. locations
n = 3 # no. clients

vcosts, fcosts = costs(m, n) # variable and fixed costs matrices
print(vcosts, fcosts)

c = vcosts.copy() # make a copy of variable and fixed costs matrices to evaluate objective function
f = fcosts.copy()

y, x = constructive_method(m, n, vcosts) # feasible solution
print(y, x) 

if np.sum(y) != 0:
	z = objective_function(x, c, f, y) # evaluate objective function
	print(z)
else:
	print("There are no open locations.")

