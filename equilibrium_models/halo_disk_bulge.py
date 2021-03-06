"""
Construction of a three-component halo-disk-bulge equilibrium model of a galaxy.
Adapted from example_self_consistent_model.py in Agama (https://github.com/GalacticDynamics-Oxford/Agama)
This example differs in that it has a somewhat simpler structure (only a single stellar disk
component, no stellar halo or gas disk).
"""

import os, sys, numpy as np, matplotlib.pyplot as plt, astropy.units as u
from configparser import RawConfigParser
import agama, jeans_util as util
np.random.seed(42)

# print some diagnostic information after each iteration
def printoutInfo(model, iteration):
    densDisk = model.components[0].getDensity()
    densBulge= model.components[1].getDensity()
    densHalo = model.components[2].getDensity()
    pt0 = (2.0, 0, 0)
    pt1 = (2.0, 0, 0.25)
    pt2 = (0.0, 0, 2.0)
    print("Disk  total mass=%g, rho(R=2,z=0)=%g, rho(R=2,z=0.25)=%g" % \
        (densDisk.totalMass(), densDisk.density(pt0), densDisk.density(pt1)))
    print("Bulge total mass=%g, rho(R=0.5,z=0)=%g" % \
        (densBulge.totalMass(), densBulge.density(0.4, 0, 0)))
    print("Halo  total mass=%g, rho(R=2,z=0)=%g, rho(R=0,z=2)=%g" % \
        (densHalo.totalMass(), densHalo.density(pt0), densHalo.density(pt2)))
    # report only the potential of stars+halo, excluding the potential of the central BH (0th component)
    # pot0 = model.potential.potential(0,0,0) - model.potential[0].potential(0,0,0)


if len(sys.argv) == 1:
    # True: give the halo a Cuddeford-Osipkov-Merrit velocity anisotropy profile
    osipkov_merrit = False # abbreviated to OM in filenames
    # True: also run variant with disk stars contaminating halo sample
    disk_contam = True     # abbreviated to DC in filenames
elif len(sys.argv) == 2 and ('OM' == sys.argv[1] or 'COM' == sys.argv[1]):
    osipkov_merrit = True
    disk_contam = False
elif len(sys.argv) > 2:
    exit("Provide 'COM' or no argument to run with or without a Cuddeford-Osipkov-Merrit velocity anisotropy profile")
else:
    exit("Could not understand command line argument")

print(f"Running halo-disk-bulge simulation{' with radially varying velocity anisotropy' if osipkov_merrit else ''}")
home_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
write_path = home_path + "/data/halo_disk_bulge/"
if not os.path.exists(write_path):
    os.makedirs(write_path)
    print(f"created output directory at data/halo_disk_bulge/")

# read parameters from the INI file
ini_path = home_path + "/equilibrium_models/HDB.ini"
ini = RawConfigParser()
#ini.optionxform=str  # do not convert key to lowercase
ini.read(ini_path)
iniPotenHalo  = dict(ini.items("Potential halo"))
iniPotenBulge = dict(ini.items("Potential bulge"))
iniPotenDisk  = dict(ini.items("Potential disk"))
iniDFDisk     = dict(ini.items("DF disk"))
iniSCMHalo    = dict(ini.items("SelfConsistentModel halo"))
iniSCMBulge   = dict(ini.items("SelfConsistentModel bulge"))
iniSCMDisk    = dict(ini.items("SelfConsistentModel disk"))
iniSCM        = dict(ini.items("SelfConsistentModel"))

# initialize the SelfConsistentModel object (only the potential expansion parameters)
model = agama.SelfConsistentModel(**iniSCM)

# create initial density profiles of all components
densityDisk  = agama.Density(**iniPotenDisk)
densityBulge = agama.Density(**iniPotenBulge)
densityHalo  = agama.Density(**iniPotenHalo)

# add components to SCM - at first, all of them are static density profiles
model.components.append(agama.Component(density=densityDisk,  disklike=True))
model.components.append(agama.Component(density=densityBulge, disklike=False))
model.components.append(agama.Component(density=densityHalo,  disklike=False))

# compute the initial potential
model.iterate()
printoutInfo(model, 'init')

# construct the DF of the disk component, using the initial (non-spherical) potential
dfDisk  = agama.DistributionFunction(potential=model.potential, **iniDFDisk)
# initialize the DFs of spheroidal components using the Eddington inversion formula
# for their respective density profiles in the initial potential
dfBulge = agama.DistributionFunction(type='QuasiSpherical', potential=model.potential, density=densityBulge)
if osipkov_merrit:
    dfHalo  = agama.DistributionFunction(type='QuasiSpherical', potential=model.potential, density=densityHalo, beta0=0.2, r_a=40)
else:
    dfHalo  = agama.DistributionFunction(type='QuasiSpherical', potential=model.potential, density=densityHalo)


print("\033[1;33m**** STARTING ITERATIVE MODELLING ****\033[0m\nMasses (computed from DF): " \
    "Mdisk=%g, Mbulge=%g, Mhalo=%g" % (dfDisk.totalMass(), dfBulge.totalMass(), dfHalo.totalMass()))

# replace the initially static SCM components with the DF-based ones
model.components[0] = agama.Component(df=dfDisk,  disklike=True,  **iniSCMDisk)
model.components[1] = agama.Component(df=dfBulge, disklike=False, **iniSCMBulge)
model.components[2] = agama.Component(df=dfHalo,  disklike=False, **iniSCMHalo)

# do a few more iterations to obtain the self-consistent density profile for both disks
for i in range(5):
    print("\033[1;37mStarting iteration #%d\033[0m" % i)
    model.iterate()
    printoutInfo(model, 'iter%d'%i)

densDisk = model.components[0].getDensity()
densBulge= model.components[1].getDensity()
densHalo = model.components[2].getDensity()

print("\033[1;33mCreating an N-body representation of the model\033[0m")

# now create genuinely self-consistent models of both components,
# by drawing positions and velocities from the DF in the given (self-consistent) potential
print("Sampling halo DF")
pos, mass = agama.GalaxyModel(potential=model.potential, df=dfHalo,  af=model.af).sample(300000)
print("Sampling disk DF")
pos_d, mass_d = agama.GalaxyModel(potential=model.potential, df=dfDisk,  af=model.af).sample(160000)
print("Sampling bulge DF")
pos_b, mass_b = agama.GalaxyModel(potential=model.potential, df=dfBulge,  af=model.af).sample(40000)

print("\033[1;33mPreparing files for input into jeans routines\033[0m")
x  = pos[:,0]
y  = pos[:,1]
z  = pos[:,2]
vx = pos[:,3]
vy = pos[:,4]
vz = pos[:,5]

# calculate spherical velocities and radius from cartesian kinematics
r, vr_sq, vtheta_sq, vphi_sq = util.format_dataset(np.transpose([x, y, z, vx, vy, vz]))

sorter = np.argsort(r)
x = x[sorter]; y = y[sorter]; z = z[sorter]
vx = vx[sorter]; vy = vy[sorter]; vz = vz[sorter]
vr_sq = vr_sq[sorter]; vtheta_sq = vtheta_sq[sorter]; vphi_sq = vphi_sq[sorter]
r = r[sorter]; mass = mass[sorter]

# Define conversions from simulation units to physical units
HDB_length = u.def_unit('s_length', 2.5 * u.kpc)  # density scale radius = 10 sim lengths = 25 kpc (Bland-Hawthorn&Gerhard 2016)
HDB_mass = u.def_unit('s_mass', 4.1e10 * u.Msun)  # sum of MW thin and thick disk stellar mass = 4.1*10^10 Msun (Bland-Hawthorn&Gerhard 2016)
HDB_time = u.def_unit('s_time', 2.905e14 * u.s)   # determined by two previous given G=1

def convert(dat, cur, dest):
    # converts dat value from current unit cur to destination unit dest
    return (dat*cur).to(dest).value

# Convert units of all quantites from simulation units to kpc, km/s, Msun
# Simulation run with G = 1 in simulation units
x       = convert(x,  HDB_length, u.kpc)
y       = convert(y,  HDB_length, u.kpc)
z       = convert(z,  HDB_length, u.kpc)
vx      = convert(vx, HDB_length/HDB_time, u.km/u.s)
vy      = convert(vy, HDB_length/HDB_time, u.km/u.s)
vz      = convert(vz, HDB_length/HDB_time, u.km/u.s)
r       = convert(r,  HDB_length, u.kpc)
mass    = convert(mass, HDB_mass, u.Msun)
vr_sq     = convert(vr_sq,     (HDB_length/HDB_time)**2, (u.km/u.s)**2)
vtheta_sq = convert(vtheta_sq, (HDB_length/HDB_time)**2, (u.km/u.s)**2)
vphi_sq   = convert(vphi_sq,   (HDB_length/HDB_time)**2, (u.km/u.s)**2)

np.savetxt(
    fname=f"{write_path}HDB{'_OM' if osipkov_merrit else ''}_prejeans.csv",
    X=np.stack([x, y, z, vx, vy, vz,
                mass, r, vr_sq, vtheta_sq, vphi_sq], axis=1),
    delimiter=',', header=" x, y, z [kpc], vx, vy, vz [km/s], mass [Msun], gc_radius, vr_sq, vtheta_sq, vphi_sq [km2/s2]"
)

r_d = (pos_d[:,0]**2 + pos_d[:,1]**2 + pos_d[:,2]**2)**0.5
r_b = (pos_b[:,0]**2 + pos_b[:,1]**2 + pos_b[:,2]**2)**0.5

r_d    = convert(r_d,  HDB_length, u.kpc)
r_b    = convert(r_b,  HDB_length, u.kpc)
mass_d = convert(mass_d,  HDB_mass, u.Msun)
mass_b = convert(mass_b,  HDB_mass, u.Msun)

r_mix = np.concatenate((r, r_d, r_b))
m_mix = np.concatenate((mass, mass_d, mass_b))

sorter = np.argsort(r_mix)
r_mix = r_mix[sorter]; m_mix = m_mix[sorter]

# Need this to include all components
np.savetxt(
    fname=f"{write_path}HDB{'_OM' if osipkov_merrit else ''}_true.csv",
    X=np.stack([r_mix[::10], np.cumsum(m_mix)[::10]], axis=1),
    delimiter=',', header=" r [kpc], M(<r)_true [Msun]"
)

if disk_contam:
    # also write HDB variant with halo sample contaminated by disk particles
    print("Creating mock with contamination of halo by disk particles")
    print("Sampling disk DF")
    idxs = np.random.choice(np.arange(len(mass_d)), int(len(mass_d)/4), replace=False)

    x_d    = convert(pos_d[:,0][idxs], HDB_length, u.kpc)
    y_d    = convert(pos_d[:,1][idxs], HDB_length, u.kpc)
    z_d    = convert(pos_d[:,2][idxs], HDB_length, u.kpc)
    vx_d   = convert(pos_d[:,3][idxs], HDB_length/HDB_time, u.km/u.s)
    vy_d   = convert(pos_d[:,4][idxs], HDB_length/HDB_time, u.km/u.s)
    vz_d   = convert(pos_d[:,5][idxs], HDB_length/HDB_time, u.km/u.s)
    mass_d = convert(mass_d[idxs], HDB_mass, u.Msun)

    r_d, vr_sq_d, vtheta_sq_d, vphi_sq_d = util.format_dataset(np.transpose([x_d, y_d, z_d, vx_d, vy_d, vz_d]))

    x_mix = np.concatenate((x, x_d))
    y_mix = np.concatenate((y, y_d))
    z_mix = np.concatenate((z, z_d))

    vx_mix = np.concatenate((vx, vx_d))
    vy_mix = np.concatenate((vx, vx_d))
    vz_mix = np.concatenate((vx, vx_d))

    m_mix = np.concatenate((mass, mass_d))
    r_mix = np.concatenate((r, r_d))

    vr_sq_mix     = np.concatenate((vr_sq, vr_sq_d))
    vtheta_sq_mix = np.concatenate((vtheta_sq, vtheta_sq_d))
    vphi_sq_mix   = np.concatenate((vphi_sq, vphi_sq_d))

    sorter = np.argsort(r_mix)
    x_mix = x_mix[sorter]; y_mix = y_mix[sorter]; z_mix = z_mix[sorter]
    vx_mix = vx_mix[sorter]; vy_mix = vy_mix[sorter]; vz_mix = vz_mix[sorter]
    vr_sq_mix = vr_sq_mix[sorter]; vtheta_sq_mix = vtheta_sq_mix[sorter]; vphi_sq_mix = vphi_sq_mix[sorter]
    r_mix = r_mix[sorter]; m_mix = m_mix[sorter]

    np.savetxt(
        fname=f"{write_path}HDB{'_OM' if osipkov_merrit else ''}_DC_prejeans.csv",
        X=np.stack([x_mix, y_mix, z_mix, vx_mix, vy_mix, vz_mix,
                    m_mix, r_mix, vr_sq_mix, vtheta_sq_mix, vphi_sq_mix], axis=1),
        delimiter=',', header=" x, y, z [kpc], vx, vy, vz [km/s], mass [Msun], gc_radius, vr_sq, vtheta_sq, vphi_sq [km2/s2]"
    )
    # No need to write another _true.csv file, it will be shared with the non-DC one

print(f"\033[1;33mFINISHED WITH halo_disk_bulge\033[0m\n")
