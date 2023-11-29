# adapted from https://github.com/tpawelski/graph-coloring-lp - MIT License
from pulp import *


n = 30

# a nested list rep for an adjacency matrix
adj_mat =[]
for i in range(0,n):
    adj_mat.append([0]*n)

edges = []
for i in range(n -1):
  edges.append((i, i+1))
edges.append((3, 5))

for edge in edges:
    r=edge[0]
    c=edge[1]
    adj_mat[r][c] = 1
    adj_mat[c][r]= 1

max_colour=10
nodes = range(30)
y=range(max_colour)



#initializes lp problem
lp = LpProblem("Coloring_Problem",LpMinimize)
# The problem variables are created
# variables c_ij to indicate whether node i is colored by color j;
c_ij = LpVariable.dicts("c",(nodes,y),0,1,LpInteger)
#variables to indicate whether colour j was used
w_j = LpVariable.dicts("w",y,0,1,LpInteger)

#objective is the sum of yj over all j
obj = lpSum(w_j[j] for j in y)
lp += obj, "Objective_Function"

#constraint s.t. each node uses exactly 1 color
for r in nodes:
    col_sum=0.0
    for j in y:
        col_sum += c_ij[r][j]
    lp += col_sum==1,""

#constraint s.t. adjacent nodes do not have the same color
for row in range(0,len(adj_mat)):
    for col in range(0, len(adj_mat)):
        if adj_mat[row][col]==1:
            for j in y:
                lp += c_ij[row][j] + c_ij[col][j] <= 1,""

#constraint s.t. if node i is assigned color k, color k is used
for i in nodes:
    for j in y:
        lp += c_ij[i][j] <= w_j[j],""

#constraint for upper bound on # of colors used
lp += lpSum(w_j[j] for j in y)<= n

#solves lp and prints optimal solution/objective value
lp.solve()
status = str(LpStatus[lp.status])
print("Solution: "+ status)

print("Optimal Solution:")
print("c_ij=1 values:")
for i in nodes:
	for j in y:
		if c_ij[i][j].value() == 1:
		          print(c_ij[i][j])

print("Number colours used: ", value(lp.objective))




# propagated from original source - adapted from https://github.com/tpawelski/graph-coloring-lp
# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
