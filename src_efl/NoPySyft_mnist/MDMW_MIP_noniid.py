import pandas as pd
import numpy as np
from mip import *


ALPHA = 0.7

#------------reading wPoint file

path = '../../output/mnist/z_ass/'
wPoint_file = "wPCA_MLP2_G10_partition_noniid90_nclient300.csv"

dfw = pd.read_csv(path + wPoint_file)
mean_wPoint = np.mean(dfw[["pc1", "pc2"]].values, axis = 0)
v = dfw[["pc1", "pc2"]].values - mean_wPoint

#------------verifying data points

print("mean_wPoint: ", mean_wPoint)
print("sum_w.x: ", np.sum(dfw["pc1"].values))
print("sum_w.y: ", np.sum(dfw["pc2"].values))
print("sum_w: ", np.sum(dfw[["pc1", "pc2"]].values, axis = 0))
print("sum_v: ", np.sum(v, axis = 0))

#------------setting the model: Multi - Dimension Multi - Way

n, m, k = len(v), len(v[0]), 30                                        #number of points, and number of dimensions, number of clusters

mip_model = Model("MDMWay_MIP", solver_name = GRB, sense = MINIMIZE)    #sense = MINIMIZE

#-----add variables

x = [mip_model.add_var(var_type=BINARY) for i in range(n*k)]             
x.append(mip_model.add_var())

#-----coefficients of objective function

c = np.zeros(n * k + 1)
c[n * k] = 1
mip_model.objective = minimize(xsum(c[i] * x[i] for i in range(n * k + 1))) #objective function

#-----add equality constraints: n constraints, each client is assigned 1 time

Aeq = np.zeros(n * k + 1)

for i in range(n):
	inds = []
	for j in range(k):
		Aeq[j * n + i] = 1
		inds.append(j * n + i)

	mip_model += xsum(Aeq[u] * x[u] for u in inds) == 1 #<= c


#-----add inequality constraints: k constraints, each cluster cannot be empty

A = np.zeros(n * k + 1)

for j in range(k):
	inds = []
	for i in range(n):
		A[j * n + i] = 1
		inds.append(j * n + i)
	mip_model += xsum(A[u] * x[u] for u in inds) >= ALPHA * n/k #1 #<= c

#-----add inequality constraints: mk(k-1)/2 constraints, for each dimension, all the differences between the sums of ...  < diameter 

A = np.zeros(n * k + 1)

#Aieq = np.zeros( m * k * (k -1))
con = 0
for l in range(m):
	for j1 in range(k-1):
		for j2 in range(j1+1, k, 1):
			inds = []
			for i in range(n):
				A[j1 * n + i], A[j2 * n + i] = v[i][l], -v[i][l]
				inds.append(j1 * n + i)
				inds.append(j2 * n + i)
			A[n * k] = -1
			inds.append(n * k)
			#Aieq[con] = A.copy()
			#con += 1
			mip_model += xsum(A[u] * x[u] for u in inds)  <= 0 #<= c


#-----add inequality constraints: mk(k-1)/2 constraints, for each dimension, all the differences between the sums of ...  < diameter 

A = np.zeros(n * k + 1)

for l in range(m):
	for j1 in range(k-1):
		for j2 in range(j1+1, k, 1):
			inds = []
			for i in range(n):
				A[j1 * n + i], A[j2 * n + i] = -v[i][l], v[i][l]
				inds.append(j1 * n + i)
				inds.append(j2 * n + i)
			A[n * k] = -1
			inds.append(n * k)
			#Aieq[con] = A.copy()
			#con += 1
			
			mip_model += xsum(A[u] * x[u] for u in inds)  <= 0 #<= c

mip_model.max_gap = 0.05
status = mip_model.optimize(max_seconds = 3600 * 10)



if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(mip_model.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {}'.format(mip_model.objective_value, mip_model.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {}'.format(mip_model.objective_bound))
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
	print('solution:')
	for va in mip_model.vars:
		if abs(va.x) > 1e-6: # only printing non-zeros
			print('{} : {}'.format(va.name, va.x))
	z = np.zeros(n, dtype = int)
	for u in range(n * k):
		if (x[u].x >= 0.99):
			j, i = int(u / n), int(u % n)			
			z[i] = j

	z_file = "z_MIP_" + wPoint_file + ".part" + str(k)
	dfz = pd.DataFrame()
	dfz["z"] = z
	dfz.to_csv(path + z_file, header = False, index = False)

	print("saved file: ", path + z_file)
	print("objective function values: ", x[k * n].x)

	print("checking inequality constrains...")


	subset_sum = np.zeros((m, k))

	for l in range(m):
		for j in range(k):
			s = 0
			for i in range(n):
				if z[i] == j:
					s += v[i][l]
			subset_sum[l][j] = s	

	count_violation = 0
	for l in range(m):
		for j1 in range(k-1):
			for j2 in range(j1+1, k, 1):
				if abs(subset_sum[l][j1] - subset_sum[l][j2]) > x[n*k].x:

					count_violation += 1
					print("violation in dimension, clusters: ", l, j1, " ", j2)
	
	print("#violation: ", count_violation)









				