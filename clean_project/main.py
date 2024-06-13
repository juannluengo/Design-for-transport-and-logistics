from map import get_map, plot_map
from hubs_satellites import  optimize_hubs_satellites,plot_optimized_solution,generate_initial_solution
import matplotlib.pyplot as plt



# From hubs_satellites import optimize_hubs_satellites
import pandas as pd
import numpy as np
delivery, pickup = get_map()
plot_map(delivery, pickup, plot_B=True)
# delivery_unified=delivery[0][['X','Y']]._append(delivery[1][['X','Y']], ignore_index=True)
# pickup_unified=pickup[0][['X','Y']]._append(pickup[1][['X','Y']], ignore_index=True)
# print('DELIVERY POINTS ARE:')
# print(delivery_unified)
# print('PICKUP POINTS ARE:')
# print(pickup_unified)

# all_the_points=delivery_unified._append(pickup_unified, ignore_index=True).to_numpy()
# print('ALL THE POINTS AS NUMPY ARRAY ARE:')
# print(all_the_points)

# plt.scatter(delivery_unified.to_numpy()[:,0], delivery_unified.to_numpy()[:,1], color='g')
# plt.scatter(pickup_unified.to_numpy()[:,0], pickup_unified.to_numpy()[:,1], color='m')
# plt.show()

points = [np.vstack((d[['X','Y']], p[['X','Y']])) for d, p in zip(delivery, pickup)]
print(points)
#
#optimized_hubs, optimizd_satellites = optimize_hubs_satellites(points)
#hubs,satellites, clusters= generate_initial_solution(points)
optimized_hubs, optimized_satellites, clusters = optimize_hubs_satellites(points)

print(len(optimized_hubs))
for x in optimized_hubs:
    print(len(x))

plot_optimized_solution(points, optimized_hubs, optimized_satellites, clusters)