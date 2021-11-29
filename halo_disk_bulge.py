#!/usr/bin/python
"""
Example of construction of a three-component disk-bulge-halo equilibrium model of a galaxy.
The approach is explained in example_self_consistent_model.py;
this example differs in that it has a somewhat simpler structure (only a single stellar disk
component, no stellar halo or gas disk) and adds a central supermassive black hole.
Another modification is that the halo and the bulge are represented by 'quasi-isotropic' DF:
it is a spherical isotropic DF that is constructed using the Eddington inversion formula
for the given density profile in the spherically-symmetric approximation of the total potential.
This DF is then expressed in terms of actions and embedded into the 'real', non-spherical
potential, giving rise to a somewhat different density profile; however, it is close enough
to the input one. Then a few more iterations are needed to converge towards a self-consistent
model.
"""

import agama, numpy, os, sys, matplotlib.pyplot as plt

from configparser import RawConfigParser

# write out the circular velocity curve for the entire model and per component
def writeRotationCurve(filename, potentials, names):
    radii = numpy.logspace(-3.0, 2.0, 101)
    xyz   = numpy.column_stack((radii, radii*0, radii*0))
    vcomp2= numpy.column_stack([-potential.force(xyz)[:,0] * radii for potential in potentials])
    vtot2 = numpy.sum(vcomp2, axis=1)
    numpy.savetxt(filename, numpy.column_stack((radii, vtot2**0.5, vcomp2**0.5)), fmt="%.6g", header="radius\tVcTotal\t"+"\t".join(names))

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

if __name__ == "__main__":
    # read parameters from the INI file
    # increase iterations

    iniFileName = os.path.dirname(os.path.realpath(sys.argv[0])) + "/SCM_HDB.ini"
    ini = RawConfigParser()
    #ini.optionxform=str  # do not convert key to lowercase
    ini.read(iniFileName)
    iniPotenHalo  = dict(ini.items("Potential halo"))
    iniPotenBulge = dict(ini.items("Potential bulge"))
    iniPotenDisk  = dict(ini.items("Potential disk"))
    iniPotenBH    = dict(ini.items("Potential BH")) # BH
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
    potentialBH  = agama.Potential(**iniPotenBH) # BH

    # add components to SCM - at first, all of them are static density profiles
    model.components.append(agama.Component(density=densityDisk,  disklike=True))
    model.components.append(agama.Component(density=densityBulge, disklike=False))
    model.components.append(agama.Component(density=densityHalo,  disklike=False))
    model.components.append(agama.Component(potential=potentialBH)) # BH

    # compute the initial potential
    model.iterate()
    printoutInfo(model,'init')

    # construct the DF of the disk component, using the initial (non-spherical) potential
    dfDisk  = agama.DistributionFunction(potential=model.potential, **iniDFDisk)
    # initialize the DFs of spheroidal components using the Eddington inversion formula
    # for their respective density profiles in the initial potential
    dfBulge = agama.DistributionFunction(type='QuasiSpherical', potential=model.potential, density=densityBulge)
    dfHalo  = agama.DistributionFunction(type='QuasiSpherical', potential=model.potential, density=densityHalo)

    print("\033[1;33m**** STARTING ITERATIVE MODELLING ****\033[0m\nMasses (computed from DF): " \
        "Mdisk=%g, Mbulge=%g, Mhalo=%g" % (dfDisk.totalMass(), dfBulge.totalMass(), dfHalo.totalMass()))

    # replace the initially static SCM components with the DF-based ones
    model.components[0] = agama.Component(df=dfDisk,  disklike=True,  **iniSCMDisk)
    model.components[1] = agama.Component(df=dfBulge, disklike=False, **iniSCMBulge)
    model.components[2] = agama.Component(df=dfHalo,  disklike=False, **iniSCMHalo)

    # do a few more iterations to obtain the self-consistent density profile for both disks
    for iteration in range(1,5):
        print("\033[1;37mStarting iteration #%d\033[0m" % iteration)
        model.iterate()
        printoutInfo(model, 'iter%d'%iteration)

    densDisk = model.components[0].getDensity()
    densBulge= model.components[1].getDensity()
    densHalo = model.components[2].getDensity()

    writeRotationCurve("rotcurve_test", (
        model.potential[0], # potential of the BH
        model.potential[2], # potential of the disk
        agama.Potential(type='Multipole', lmax=6, density=densBulge),  # -"- bulge
        agama.Potential(type='Multipole', lmax=6, density=densHalo) ), # -"- halo
        ('BH', 'Disk', 'Bulge', 'Halo') )

    # export model to an N-body snapshot
    print("\033[1;33mCreating an N-body representation of the model\033[0m")
    format = 'text'  # one could also use 'nemo' or 'gadget' here

    # now create genuinely self-consistent models of both components,
    # by drawing positions and velocities from the DF in the given (self-consistent) potential
    print("Sampling disk DF")
    agama.writeSnapshot("HDB5_disk", \
        agama.GalaxyModel(potential=model.potential, df=dfDisk,  af=model.af).sample(160000), format)
    print("Sampling bulge DF")
    agama.writeSnapshot("HDB5_bulge", \
        agama.GalaxyModel(potential=model.potential, df=dfBulge, af=model.af).sample(40000), format)
    print("Sampling halo DF")
    # note: use a 10x larger particle mass for halo than for bulge/disk
    agama.writeSnapshot("HDB5_halo", \
        agama.GalaxyModel(potential=model.potential, df=dfHalo,  af=model.af).sample(300000), format)
