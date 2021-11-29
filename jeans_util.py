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

    conversion_matrix = [[sin_theta * cos_phi, sin_theta * sin_phi,  cos_theta],
                            [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta],
                            [       -sin_phi      ,        cos_phi       ,        0    ]]

    velMatrix = [ [vx], [vy], [vz] ]

    sphereVels = np.matmul(conversion_matrix, velMatrix)

    v_r = sphereVels[0]
    v_theta = sphereVels[1]
    v_phi = sphereVels[2]

    return v_r, v_theta, v_phi


def format_dataset(gal_data):
    """
    convert a x,y,zpos and x,y,zvel dataset to r,t,pvel_squared with radii
    gal_data formatted as xpos, ypos, zpos, xvel, yvel, zvel

    returns rvel_sq, tvel_sq, pvel_sq, radii
    """
    xpos = np.asarray(gal_data[:,0])
    ypos = np.asarray(gal_data[:,1])
    zpos = np.asarray(gal_data[:,2])
    xvel = np.asarray(gal_data[:,3])
    yvel = np.asarray(gal_data[:,4])
    zvel = np.asarray(gal_data[:,5])

    radii = np.sqrt((xpos ** 2) + (ypos ** 2) + (zpos ** 2))

    rVel, tVel, pVel = cartesian_to_spherical(xpos, ypos, zpos, xvel, yvel, zvel)

    rvel_sq = np.asarray([i**2 for i in rVel])
    tvel_sq = np.asarray([i**2 for i in tVel])
    pvel_sq = np.asarray([i**2 for i in pVel])

    return rvel_sq, tvel_sq, pvel_sq, radii

