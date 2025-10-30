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
print(f"Data dimensions: nt={nt}, ny={ny}, nx={nx}")

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
nan_mask = np.isnan(h_data[0])
# write out x and y coords
# and mask of where nans are
np.save("x_coords.npy", X)
np.save("y_coors.npy", Y)
np.save("mesh_hole.npy",nan_mask)

print(X.shape)
print(h_data[0].shape)


print(np.load("mesh_hole.npy"))