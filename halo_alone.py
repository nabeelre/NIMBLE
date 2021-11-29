#!/usr/bin/python
import agama, numpy as np, os, sys, matplotlib.pyplot as plt
import jeans_util as uitl

q = 0.0

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
    exit("Parameters for desired axis ratio unknown")

print(f"Running halo-alone simulation with q={q}")

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
r=np.logspace(-20.,20.)
xyz=np.vstack((r,r*0,r*0)).T
plt.plot(r, halo_dens.density(xyz), label='Init density', color='k')

for i in range(11):
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
plt.show()

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
    axes, binmass = getaxes(pos, mass, binrad[i])
    print("%.3g\t%.3g\t%.3f\t%.3f" % (binrad[i], binmass, axes[1]/axes[0], axes[2]/axes[0]))
    qs[i] = axes[2]/axes[0]

print("Preparing files for Jeans routine input")
x  = pos[:,0]
y  = pos[:,1]
z  = pos[:,2]
vx = pos[:,3]
vy = pos[:,4]
vz = pos[:,5]

vr, vtheta, vphi, r = util.format_dataset(np.transpose([x, y, z, vx, vy, vz]))

sorter = np.argsort(r)
x = x[sorter]; y = y[sorter]; z = z[sorter]
vx = vx[sorter]; vy = vy[sorter]; vz = vz[sorter]
vr = vr[sorter]; vtheta = vtheta[sorter]; vphi = vphi[sorter]
r = r[sorter]; mass = mass[sorter]

np.savetxt(
    fname=f"halo_alone_{q}_all.csv",
    X=np.stack([r, mass], axis=1),
    delimiter=',', header="radius,mass"
)

np.savetxt(
    fname=f"halo_alone_{q}_full.csv",
    X=np.stack([x, y, z, vx, vy, vz,
                mass, r, vr, vtheta, vphi], axis=1),
    delimiter=',', header="x,y,z,vx,vy,vz,mass,radius,rvelsq,tvelsq,pvelsq"
)

print("SUMMARY:")
print(f"halo_alone with q={np.median(qs):.2f}")
print(f"total particle count {len(r)}")
print(f"total mass {np.sum(mass):.3f}")


