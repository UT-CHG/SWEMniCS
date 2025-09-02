import numpy as np
import matplotlib.pyplot as plt

cm_to_m = .01
# width of rectangle
r_width = 16.3*cm_to_m
#height of rectangle object
r_height = 8.0*cm_to_m
L = 6.0078
#original expirement value
#H = 24.0*cm_to_m
#extended domain
H = r_height*11.0

#base case
h_b_val: float = 14.0/100.0
h_b = lambda x: h_b_val * (x[:,0] < 3.26) + (h_b_val + .0404 * (x[:,0] - 3.26)) * (x[:,0] >= 3.26)

#generating grid points
npx = 101
npy = 11
npoints = npx*npy
x = np.zeros((npoints,3))
just_x = np.linspace(0,L,npx)
just_y = np.linspace(0,H,npy)
x[:,0] = np.tile(just_x,npy)
x[:,1] = np.repeat(just_y,npx)
x[:,2] = -h_b(x)
#adding noise
mean = 0.0
std = .01
x[:,2] += np.random.normal(loc=mean,scale=std,size=npoints)
#plot 2d scattered data set
# Creating plot
# Creating figure
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
ax.plot_trisurf(x[:,0], x[:,1], x[:,2], cmap='viridis', edgecolor='none')
plt.show()