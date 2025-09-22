import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys

def read_into_np_array(file_base: str, var_name: str )->np.ndarray:
    with h5py.File(file_base+"_"+var_name+".h5", 'r') as f:
    # Assuming 'your_dataset_name' is a dataset within the file
        dataset = f[var_name][:]
    return dataset


#read in file
file_base = sys.argv[1]
# file for h is file_base+_h.h5
# file for u is file_base+_u.h5
# file for v is file_base+_v.h5
h_data = read_into_np_array(file_base, 'h')
bathy_data = read_into_np_array(file_base, 'bathy')
u_data = read_into_np_array(file_base, 'u')
v_data = read_into_np_array(file_base, 'v')
nt,ny,nx = h_data.shape


#plot to make sure this is correct
cm_to_m = .01
# width of rectangle
r_width = 16.3*cm_to_m
#height of rectangle object
r_height = 8.0*cm_to_m
L = 6.0078
x_start = 1.4
x_end = 3.8
#original expirement value
#H = 24.0*cm_to_m
#extended domain
H = r_height*11.0

x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)
X,Y = np.meshgrid(x,y)
#fix scaling of color map
h_data_min = np.nanmin(h_data)
h_data_max = np.nanmax(h_data)
u_data_min = np.nanmin(u_data)
u_data_max = np.nanmax(u_data)
v_data_min = np.nanmin(v_data)
v_data_max = np.nanmax(v_data)
bathy_data_min = np.nanmin(bathy_data)
bathy_data_max = np.nanmax(bathy_data)

nlevels = 100
levels_h = np.linspace(h_data_min,h_data_max,nlevels)
levels_bathy = np.linspace(bathy_data_min,bathy_data_max,nlevels)
levels_u = np.linspace(u_data_min,u_data_max,nlevels)
levels_v = np.linspace(v_data_min,v_data_max,nlevels)

#bathymetry is just one step
#img = plt.contourf(X,Y,bathy_data,levels=levels_bathy,cmap='coolwarm')
#time index: note t = index*dt where dt = .02
dt = 0.1/5.0
t_ind = nt-1
img = plt.contourf(X,Y,h_data[t_ind],levels=levels_h,cmap='coolwarm')
#img = plt.contourf(X,Y,u_data[t_ind],levels=levels_u,cmap='coolwarm')
#img = plt.contourf(X,Y,v_data[t_ind],levels=levels_v,cmap='coolwarm')
plt.axis('scaled')
plt.colorbar(img)
plt.title(f"Solution at {t_ind*dt} seconds")
plt.show()