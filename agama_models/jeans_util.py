import numpy as np, matplotlib.pyplot as plt, agama


def cartesian_to_spherical(x, y, z, vx, vy, vz):
    N = len(x)
    sin_theta = np.zeros(N)
    cos_theta = np.zeros(N)

    sin_phi = np.zeros(N)
    cos_phi = np.zeros(N)

    sin_theta = np.sqrt(x**2 + y**2) / np.sqrt(x**2 + y**2 + z**2)
    cos_theta = z / np.sqrt(x**2 + y**2 + z**2)
    sin_phi = y / np.sqrt(x**2 + y**2)
    cos_phi = x / np.sqrt(x**2 + y**2)

    v_r = np.zeros(N)
    v_theta = np.zeros(N)
    v_phi = np.zeros(N)


    for i in range(N):
        conversion_matrix = [[sin_theta[i] * cos_phi[i], sin_theta[i] * sin_phi[i],  cos_theta[i]],
                             [cos_theta[i] * cos_phi[i], cos_theta[i] * sin_phi[i], -sin_theta[i]],
                             [       -sin_phi[i]      ,        cos_phi[i]       ,        0    ]]
        vel_mat = [ [vx[i]], [vy[i]], [vz[i]] ]
        sphere_vels = np.matmul(conversion_matrix, vel_mat)

        v_r[i]     = sphere_vels[0]
        v_theta[i] = sphere_vels[1]
        v_phi[i]   = sphere_vels[2]

    return v_r, v_theta, v_phi


def format_dataset(gal_data):
    """
    Calculate spherical velocities and radii from cartesian kinematics
    gal_data formatted as x,y,z,vx,vy,vz
    """
    x = np.asarray(gal_data[:,0])
    y = np.asarray(gal_data[:,1])
    z = np.asarray(gal_data[:,2])
    vx = np.asarray(gal_data[:,3])
    vy = np.asarray(gal_data[:,4])
    vz = np.asarray(gal_data[:,5])

    r = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))

    v_r, v_theta, v_phi = cartesian_to_spherical(x, y, z, vx, vy, vz)

    # rvel_sq = np.asarray([i**2 for i in v_r])
    # tvel_sq = np.asarray([i**2 for i in v_theta])
    # pvel_sq = np.asarray([i**2 for i in v_phi])

    return r, v_r**2, v_theta**2, v_phi**2


def getaxes(pos, mass, radius):
    evec = np.eye(3)   # initial guess for axes orientation
    axes = np.ones(3)  # and axis ratios; these are updated at each iteration
    while True:
        # use particles within the elliptical radius less than the provided value
        ellpos  = pos[:,0:3].dot(evec) / axes
        filter  = np.sum(ellpos**2, axis=1) < radius**2
        inertia = pos[filter,0:3].T.dot(pos[filter,0:3] * mass[filter,None])
        val,vec = np.linalg.eigh(inertia)
        order   = np.argsort(-val)  # sort axes in decreasing order
        evec    = vec[:,order]         # updated axis directions
        axesnew = (val[order] / np.prod(val)**(1./3))**0.5  # updated axis ratios, normalized so that ax*ay*az=1
        #print evec,axesnew,sum(filter)
        if sum(abs(axesnew-axes))<0.01: break
        axes    = axesnew
    return axes, np.sum(mass[filter])
