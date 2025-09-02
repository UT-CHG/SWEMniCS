import sys
import adios2
import numpy as np

def is_inside_same_side(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """
    Checks if a point p is inside the triangle defined by vertices a, b, and c.
    This uses the "same-side" technique, which is computationally efficient.
    
    Args:
        p: A tuple (x, y) representing the point to check.
        a, b, c: Tuples (x, y) representing the vertices of the triangle.
        
    Returns:
        True if the point is inside or on the edge of the triangle, False otherwise.
    """
    # Helper function to calculate the sign of the cross-product.
    # The sign determines which side of a line a point is on.
    def _sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    # Calculate the sign for the point p with respect to each edge.
    d1 = _sign(p, a, b)
    d2 = _sign(p, b, c)
    d3 = _sign(p, c, a)

    # Check if the signs are all non-negative or all non-positive.
    # A point is inside if it has the same sign relative to all edges.
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    # If the point has both positive and negative signs, it's outside.
    # Otherwise, it's inside or on an edge.
    return not (has_neg and has_pos)

def find_elem(p: np.ndarray, connectivity: np.ndarray, coords: np.ndarray)->int:
    #simple linear search for now
    isFound = False
    ncell = connectivity.shape[0]
    nit = 0
    while(not isFound and nit < ncell):
         isFound = is_inside_same_side(p, coords[connectivity[nit][1]], coords[connectivity[nit][2]], coords[connectivity[nit][3]])
         nit+=1
    if (nit == ncell):
         nit = -9999
    else:
         nit-=1
    return nit

def find_elems(p_array: np.ndarray, connectivity: np.ndarray, coords:np.ndarray)->np.ndarray:
    out_indices = np.zeros(p_array.shape[0],dtype=np.int32)
    for i,p in enumerate(p_array):
        out_indices[i] = find_elem(p,connectivity,coords)
    return out_indices

adios_folder=sys.argv[1]
with adios2.Stream(adios_folder+"/h.bp", "r") as s:
        #print(dir(s))
        #print(s.num_steps())
        for step in s: 
            # Iterate through available steps
            # Access a variable by name
            #print(dir(step))
            #print(step.available_variables())
            #print(step.available_attributes())
            my_variable = step.read("depth")
            # my_variable will now contain the data for "MyVariable" in the current step
            # You can then process or analyze my_variable as needed
            #print(my_variable.shape)
            #find coords of this variable
        #just save coords at end
        my_coords = step.read("geometry")
        my_connectivity = step.read("connectivity")

print(my_coords.shape)
print(my_connectivity.shape)
print(np.amax(my_connectivity))
print(my_variable.shape)

#now we can do linear interpolation if we wish, a little tricky because DG
#do intermediate step by finding which cell each point is contained in
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
npx = 101
npy = 11
stations = np.zeros((npx*npy,3))
just_x = np.linspace(0,L,npx)
just_y = np.linspace(0,H,npy)
stations[:,0] = np.tile(just_x,npy)
stations[:,1] = np.repeat(just_y,npx)
cell_nos = find_elems(stations, my_connectivity, my_coords)
#vary slow