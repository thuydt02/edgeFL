# to compute z from z_metis
# initialize with the z from metis
# try to swap 2 clients in 2 clusters to make the objective function y better

import pandas as pd
import numpy as np
from mip import *

EPSILON = 1e-6
NUM_ITERATIONS = 100


#---------------supporters
def get_diameter(sub_sum):
	k = len(sub_sum)
	m = len(sub_sum[0])
	y = 0
	#for j1 in range(k-1):
	#	for j2 in range(j1+1, k, 1):
	#		for l in range(m):
	#			if abs(subset_sum[j1][l] - subset_sum[j2][l]) > y:
	#				y = abs(subset_sum[j1][l] - subset_sum[j2][l])
	
	for l in range(m):
		diff = max(sub_sum[:, l]) - min(sub_sum[:, l])
		if diff > y:
			y = diff

	return y


#---------------end of supporters





#----MAIN-----------------------
#------------reading wPoint file

path = '../../output/mnist/z_ass/'
wPoint_file = "wPCA_MLP2_G10_partition_noniid90_nclient300.csv"
z_metis_file = "g_nw_d_minkowski_p1.0_MLP2_G10_partition_noniid90_nclient300.npy.part.30"

dfw = pd.read_csv(path + wPoint_file)
mean_wPoint = np.mean(dfw[["pc1", "pc2"]].values, axis = 0)
v = dfw[["pc1", "pc2"]].values - mean_wPoint

#-----------reading z file
z_metis = pd.read_csv(path + z_metis_file, header = None, index_col = None)
z_metis = z_metis.values.squeeze()
print("z_metis.shape: ", z_metis.shape )

#c----------compute the value of the objective function
n, m, k = len(v), len(v[0]), 30

clusters_list = [[] for j in range(k)]
for i in range(n):
	clusters_list[z_metis[i]].append(i)

subset_sum = np.zeros((k, m))

for j in range(k):
	for l in range(m):
		s = 0
		for i in clusters_list[j]:
			s += v[i][l]
		subset_sum[j][l] = s
		
y =  get_diameter(subset_sum)
y0 = y
print("initially y0 = ", y0)

#------consider each pair of clusters. we will swap a client in a cluster for another client in another cluster
#------if this swap can reduce the objective function's value

num_swap = 0
for t in range(NUM_ITERATIONS):
	count = 0
	for j1 in range(k - 1):
		for j2 in range(j1 + 1, k):
			swap_found = False
			for i1 in clusters_list[j1]:
				for i2 in clusters_list[j2]:
					sub_sum1, sub_sum2 = np.zeros(m), np.zeros(m)
					found = False
					for l in range (m):
						sub_sum1[l] = subset_sum[j1][l] - v[i1][l] + v[i2][l]
						sub_sum2[l] = subset_sum[j2][l] + v[i1][l] - v[i2][l]

						for j in range(k):
							if (abs(sub_sum1[l] - subset_sum[j][l]) >= y) or (abs(sub_sum2[l] - subset_sum[j][l]) >= y):
								found = True
								break
						if found:
							break
					if not found: # (j1, i1) and (j2, i2) are candidates to swap
						tmp = subset_sum[0:j1].tolist()
						tmp.append(sub_sum1)
						tmp = tmp + subset_sum[j1 + 1: j2].tolist()
						tmp.append(sub_sum2)
						tmp = tmp + subset_sum[j2 + 1: k].tolist()
						new_y = get_diameter(np.asarray(tmp))
						if new_y < y - EPSILON:
							# swap
							y = new_y
							swap_found = True
							clusters_list[j1].remove(i1)
							clusters_list[j1].append(i2)
							clusters_list[j2].remove(i2)
							clusters_list[j2].append(i1)
							subset_sum[j1] = sub_sum1
							subset_sum[j2] = sub_sum2
							count += 1
					if swap_found:
						break
				if swap_found:
					break
	if count == 0:
		break
	num_swap += count
	print("iteration, num_swap, y: ", t, ", ", num_swap, ", ", y)
	
print("y0, y, diff: ", y0, " ", y, " ",  y0 - y)

z = np.zeros(n, dtype = int)
for j in range(k):
	for i in clusters_list[j]:
		z[i] = j

z_file = z_metis_file + ".swap.NUM_ITERATIONS."+ str(t) + ".csv"
dfz = pd.DataFrame()
dfz["z"] = z
dfz.to_csv(path + z_file, header = False, index = False)

print("saved file: ", path + z_file)


		

