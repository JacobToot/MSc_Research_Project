def quasi_crystal_1d(lattice_spacing = 1, alpha_deg = 45, number_of_points = 100, width = 1):
    """Generates a 1D quasi-crystal by projecting points from a 2D lattice onto a line at an angle alpha to the x-axis.
    
    Imports: 
        numpy as np.

    Args:
        lattice_spacing (float, optional): spacing between points in the 2D lattice. Defaults to 1.
        alpha_deg (float, optional): angle in degrees between the projection line and the x-axis. Defaults to 45.
        number_of_points (int, optional): number of points to generate in the quasi-crystal. Defaults to 100.
        width (float, optional): width of the strip in the 2D lattice from which points are projected. Defaults to 1.
    
    Returns:
        quasi_crystal (np.ndarray): array of distances of projected points from the origin.
        points (np.ndarray): array of original 2D lattice points within the strip.
        proj_points (np.ndarray): array of projected 2D points onto the line."""
    
    slope = np.tan(alpha_deg * np.pi / 180) # compute slope of line from angle
    y_intercept = np.cos(alpha_deg * np.pi / 180) * width # y-intercept of line

    points = [] # initialising list of points

    min_i = - int(np.ceil(width / lattice_spacing)) # minimum value of i to prevent missing points that project from x<0 to x=>0 from final sequence
    max_i = 3 * number_of_points # maximum value of i to ensure enough points (arbitrary limit, may change later)

    for i in range (min_i, max_i): 

        lower_limit = i * lattice_spacing * slope
        upper_limit = lower_limit + y_intercept
        
        for j in range (10 * number_of_points):

            if lower_limit < j * lattice_spacing < upper_limit: # check if point is in strip
                points.append([i * lattice_spacing, j * lattice_spacing]) 
            
            elif upper_limit < j * lattice_spacing:
                break # break inner loop if point is above strip

    points = np.array(points) 
    quasi_crystal = []
    proj_points = []

    for i in range(len(points)):

        x = points[i,0]  
        y = points[i,1]

        proj_x = (slope * y + x) / (slope ** 2 + 1) # project x coordinate of point onto line y = slope * x
        
        if proj_x <0:
            continue # only store values proj_x=>0
        else: 
            proj_y = proj_x * slope
            proj_points.append([proj_x, proj_y])
            quasi_crystal.append(np.sqrt(proj_x ** 2 + proj_y ** 2))
    
    proj_points = np.array(proj_points)
    quasi_crystal = np.array(quasi_crystal) 
    
    return quasi_crystal, points, proj_points