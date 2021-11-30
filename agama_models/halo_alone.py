#!/usr/bin/python
import os, sys, numpy as np, matplotlib.pyplot as plt, astropy.units as u, agama
import jeans_util as util

q = 0.0

if len(sys.argv) != 2:
    exit("Provide only the desired axis ratio")
else:
    q = float(sys.argv[1])

# Precomputed double power-law distribution function parameters for set axis ratios
if q == 1.0:
    halo_DF_params = dict(
        type     = 'doublepowerlaw',
        J0       = 1.2,
        slopeIn  = 1.6,
        slopeOut = 5.0,
        steepness= 1.3,
        coefJrIn = 1.55,
        coefJzIn = 0.7,
        coefJrOut= 1.15,
        coefJzOut= 0.9,
        norm     = 1.0
    )
elif q == 0.9:
    halo_DF_params = dict(
        type     = 'doublepowerlaw',
        J0       = 1.2,
        slopeIn  = 1.6,
        slopeOut = 5.0,
        steepness= 1.3,
        coefJrIn = 1.4,
        coefJzIn = 0.85,
        coefJrOut= 1.25,
        coefJzOut= 1.05,
        norm     = 1.0
    )
elif q == 0.8:
    halo_DF_params = dict(
        type     = 'doublepowerlaw',
        J0       = 1.2,
        slopeIn  = 1.6,
        slopeOut = 5.0,
        steepness= 1.3,
        coefJrIn = 1.2,
        coefJzIn = 1.05,
        coefJrOut= 0.975,
        coefJzOut= 1.25,
        norm     = 1.0
    )
elif q == 0.7:
    halo_DF_params = dict(
        type     = 'doublepowerlaw',
        J0       = 1.2,
        slopeIn  = 1.6,
        slopeOut = 5.0,
        steepness= 1.3,
        coefJrIn = 1.0,
        coefJzIn = 1.25,
        coefJrOut= 0.875,
        coefJzOut= 1.45,
        norm     = 1.0
    )
elif q == 0.6:
    halo_DF_params = dict(
        type     = 'doublepowerlaw',
        J0       = 1.2,
        slopeIn  = 1.6,
        slopeOut = 5.0,
        steepness= 1.3,
        coefJrIn = 0.8,
        coefJzIn = 1.45,
        coefJrOut= 0.775,
        coefJzOut= 1.65,
        norm     = 1.0
    )
else:
    # Iteratively adjust parameters for precomputed axis ratios to generate
    # parameters for a new axis ratio
    exit("Parameters for desired axis ratio unknown")

print(f"Running halo-alone simulation with q={q}")
if not os.path.exists("../data/halo_alone/"):
    os.makedirs("../data/halo_alone/")
    print(f"created output directory at ../data/halo_alone/")

# compute the mass and rescale norm to get the total mass = 1
halo_DF_params['norm'] /= float(agama.DistributionFunction(**halo_DF_params).totalMass())
# create distribution function object
halo_DF  = agama.DistributionFunction(**halo_DF_params)

# initial guess for the density profile
halo_dens = agama.Potential(type='Dehnen', mass=1, scaleRadius=10)

# Halo alone only has a halo component, create an Agama self consistent model object with just a halo
SCM_params = dict(
    rminSph=0.01,
    rmaxSph=100.,
    sizeRadialSph=30,
    lmaxAngularSph=8
)
halo_comp = agama.Component(df=halo_DF, density=halo_dens, disklike=False, **SCM_params)
SCM = agama.SelfConsistentModel(**SCM_params)
SCM.components=[halo_comp]

# iterate the model and plot the progression of the density profile
r=np.logspace(-20.,20.,200)
xyz=np.vstack((r,r*0,r*0)).T
plt.plot(r, halo_dens.density(xyz), label='Init density', color='k')

for i in range(1):
    SCM.iterate()
    print('Iteration %i, Phi(0)=%g, Mass=%g' % \
        (i, SCM.potential.potential(0,0,0), SCM.potential.totalMass()))
    plt.plot(r, SCM.potential.density(xyz), label='Iteration #'+str(i))
plt.legend(loc='lower left')
plt.xlabel("r")
plt.ylabel(r'$\rho$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-5, 2e2)
plt.xlim(0.01, 30)
plt.savefig(f"../data/halo_alone/density_evol_{q}.pdf", bbox_inches='tight')
plt.cla()

# Calculate axis ratios
# E1 method described in Zemp et al. 2011
pos, mass = agama.GalaxyModel(SCM.potential, halo_DF).sample(300000)

sphrad = np.sum(pos**2, axis=1)**0.5
order  = np.argsort(sphrad)
cummass= np.cumsum(mass[order])
nbins  = 20
indbin = np.searchsorted(cummass, np.linspace(0.04, 0.99, 20) * cummass[-1])
binrad = sphrad[order][indbin]
print("#radius\tmass   \ty/x    \tz/x")
qs = np.zeros(nbins)
for i in range(nbins):
    axes, binmass = util.getaxes(pos, mass, binrad[i])
    print("%.3g\t%.3g\t%.3f\t%.3f" % (binrad[i], binmass, axes[1]/axes[0], axes[2]/axes[0]))
    qs[i] = axes[2]/axes[0]

print("Preparing files for Jeans routine input")
x  = pos[:,0]
y  = pos[:,1]
z  = pos[:,2]
vx = pos[:,3]
vy = pos[:,4]
vz = pos[:,5]

# calculate spherical velocities and radius from cartesian kinematics
r, v_r_sq, v_theta_sq, v_phi_sq = util.format_dataset(np.transpose([x, y, z, vx, vy, vz]))

sorter = np.argsort(r)
x = x[sorter]; y = y[sorter]; z = z[sorter]
vx = vx[sorter]; vy = vy[sorter]; vz = vz[sorter]
v_r_sq = v_r_sq[sorter]; v_theta_sq = v_theta_sq[sorter]; v_phi_sq = v_phi_sq[sorter]
r = r[sorter]; mass = mass[sorter]

# Define conversions from simulation units to physical units
HA_length = u.def_unit('HA_length', 2.5*u.kpc)    # density scale radius = 10 sim lengths = 25 kpc (Bland-Hawthorn&Gerhard 2016)
HA_mass   = u.def_unit('HA_mass', 1.3e12*u.Msun)  # MW virial mass = 1.3*10^12 Msun (Bland-Hawthorn&Gerhard 2016)
HA_time   = u.def_unit('HA_time', 5.162e13*u.s)   # determined by two previous given G=1

def convert(dat, cur, dest):
    # converts dat value from current unit cur to destination unit dest
    return (dat*cur).to(dest).value

# Convert units of all quantites from simulation units to kpc, km/s, Msun
# Simulation run with G = 1 in simulation units
x       = convert(x,  HA_length, u.kpc)
y       = convert(y,  HA_length, u.kpc)
z       = convert(z,  HA_length, u.kpc)
vx      = convert(vx, HA_length/HA_time, u.km/u.s)
vy      = convert(vy, HA_length/HA_time, u.km/u.s)
vz      = convert(vz, HA_length/HA_time, u.km/u.s)
r       = convert(r,  HA_length, u.kpc)
mass    = convert(mass, HA_mass, u.Msun)
v_r_sq     = convert(v_r_sq,     (HA_length/HA_time)**2, (u.km/u.s)**2)
v_theta_sq = convert(v_theta_sq, (HA_length/HA_time)**2, (u.km/u.s)**2)
v_phi_sq   = convert(v_phi_sq,   (HA_length/HA_time)**2, (u.km/u.s)**2)

np.savetxt(
    fname=f"../data/halo_alone/halo_alone_{q}_prejeans.csv",
    X=np.stack([x, y, z, vx, vy, vz,
                mass, r, v_r_sq, v_theta_sq, v_phi_sq], axis=1),
    delimiter=',', header=" x, y, z [kpc], vx, vy, vz [km/s], mass [Msun], gc_radius, vr_sq, vtheta_sq, vphi_sq [km2/s2]"
)

np.savetxt(
    fname=f"../data/halo_alone/halo_alone_{q}_true.csv",
    X=np.stack([r[::300], np.cumsum(mass)[::300]], axis=1),
    delimiter=',', header=" r [kpc], M(<r)_true [Msun]"
)

print(f"FINISHED WITH halo_alone q={np.median(qs):.2f}\n")
