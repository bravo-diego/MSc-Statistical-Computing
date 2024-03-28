# Given an integer N that represents the number of locations that one person needs to travel, and the locations represented by a coordinate, calculate the total distance the delivery man has to travel around all points ending in the original point (i.e. coordinate '0 0'). 

u = input() # no of locations 
U = int(u)

xy = [] # empty list to store coordinates
for i in range(U):
    n = input() # input coordinate (e.g. '3 1')
    xy.append([])
    coo = n.split()
    for j in range(len(coo)):
        coo[j] = int(coo[j])
    for k in range(len(coo)):
        xy[i].append(coo[k])

def distance(point1, point2): # function to calculate the distance between two points 
    d = ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2) **(0.5)
    return d

total = 0 # total distance
for l in range(len(xy)):      
    if(l == len(xy) - 1):
        total = total + (distance(xy[0], xy[len(xy) - 1])) # distance from last to first location
    else:
        total = total + distance(xy[l], xy[l+1]) # distance between locations
    
print("{:.2f}".format(total))
    
