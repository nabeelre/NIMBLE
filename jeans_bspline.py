import sys, os, numpy as np, matplotlib.pyplot as plt, matplotlib.ticker as tk, agama


def rho_eval(S, r):
    """
    Calculates rho(r)

    S(log r) is log of tracer count
        (intended to be used with agama.splineLogDensity and weights=1)
    r is point or array of points where rho should be evaluated
    """
    return np.exp(S(np.log(r))) / (4.0 * np.pi * (r**3))


def dlnrho_dlnr_eval(S, r):
    """
    Calcualtes dlnrho / dlnr

    S(log r) is log of tracer count
        (intended to be used with agama.splineLogDensity and weights=1)
    r is point or array of points where rho should be evaluated
    """
    return (S(np.log(r), der=1) - 3)


def fractional_err(r, r_true, M, M_true):
    frc_error = np.zeros(len(r))

    for i in range(len(r)):
        match_idx = (np.abs(r_true - r[i])).argmin()
        frc_error[i] = (M[i]-M_true[match_idx])/M_true[match_idx]
    return frc_error


# Make intermediate plots (velocity and density profiles) and diagnostic print statements
VERBOSE = True

G = 4.3e-6 # kpc km2 Msun-1 s-2

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Cannot understand command line args")
    exit()

filepath = sys.argv[1]

TRUE_PROVIDED = len(sys.argv) == 3
if TRUE_PROVIDED:
    true_filepath = sys.argv[2]

if "_prejeans.csv" in filepath:
    # files with '_prejeans.csv' have the following properties:
    # positions in kpc, velocities in km/s, mass in Msun
    # arranged as [x, y, z, vx, vy, vz, m, gc_radius, vr_sq, vtheta_sq, vphi_sq]
    data = np.genfromtxt(fname=filepath, delimiter=',', skip_header=1)

    radii     = data[:, 7]
    vr_sq     = data[:, 8]
    vtheta_sq = data[:, 9]
    vphi_sq   = data[:,10]

    dataset_name = os.path.basename(filepath)[0:-len("_prejeans.csv")]
else:
    print("add handling for non '_prejeans.csv' files")
    # Make sure to match expected units or adjust G and the plot labels
    exit()

# If you change these presets to custom knot configurations make sure to update
# fig3-5.py to read it those results.
if "halo_alone" in dataset_name:
    min_rad = 1    # kpc
    max_rad = 70   # kpc

    min_knot  = 1  # kpc
    max_knot  = 70 # kpc
    num_knots = 6
elif "halo_disk_bulge" in dataset_name:
    min_rad = 1    # kpc
    max_rad = 100  # kpc

    min_knot  = 5  # kpc
    max_knot  = 80 # kpc
    num_knots = 6
elif "m12" in dataset_name:
    min_rad = 1    # kpc
    max_rad = 100  # kpc

    min_knot  = 5  # kpc
    max_knot  = 80 # kpc
    num_knots = 5
else:  # edit B-spline and plotting parameters for custom file here
    min_rad = 1    # kpc
    max_rad = 100  # kpc

    min_knot  = 5  # kpc
    max_knot  = 80 # kpc
    num_knots = 6

if VERBOSE:
    print(f"Found {len(radii)} particles in {filepath}")
    results_filepath = f"results/{dataset_name}/"
    if not os.path.exists(results_filepath):
        os.makedirs(results_filepath)
        print(f"created output directory for results at {results_filepath}")
else:
    results_filepath = ""

if TRUE_PROVIDED:
    if "_true.csv" in true_filepath:
        # files with '_true.csv' have the following properties:
        # radius in kpc, mass in Msun
        # columns arranged as [r, Menc] where Menc is the true enclosed mass
        # rows are sorted by increasing radius
        true_data = np.genfromtxt(fname=true_filepath, delimiter=',', skip_header=1)

        true_r = true_data[:,0]
        true_M = true_data[:,1]
    else:
        print("read custom formatted true file here")
        # Make sure to match expected units (see header) or adjust G and plot labels
        exit()

    if VERBOSE:
        print(f"Found provided true file at {true_filepath}")

knots = np.linspace(np.log(min_knot), np.log(max_knot), num_knots)
r = np.linspace(min_rad, max_rad, 200)
q = np.log(r)

if 'halo_alone' in filepath:
    rad_mask = [(radii >= min_knot) * (radii <= max_knot)]
    radii = radii[tuple(rad_mask)]; vr_sq = vr_sq[tuple(rad_mask)];
    vtheta_sq = vtheta_sq[tuple(rad_mask)]; vphi_sq = vphi_sq[tuple(rad_mask)]

# Fit cubic spline in log to square of each spherical velocity component
vr_bspline     = agama.splineApprox(knots, np.log(radii), vr_sq)
vtheta_bspline = agama.splineApprox(knots, np.log(radii), vtheta_sq)
vphi_bspline   = agama.splineApprox(knots, np.log(radii), vphi_sq)

vr_fit     = vr_bspline(q)
vtheta_fit = vtheta_bspline(q)
vphi_fit   = vphi_bspline(q)

beta = 1 - ( (vtheta_fit + vphi_fit) / (2*vr_fit) )

dvr = vr_bspline(q, der=1)

if VERBOSE:
    plt.figure(figsize=(12.5,8))
    plt.scatter(radii, vr_sq, marker='.', c='navy', alpha=0.25, rasterized=True)
    plt.plot(r, vr_fit, c='gold')
    plt.title(r"$V_r^2$(q) vs r with B-Spline fit", fontsize=16)
    plt.xlabel("Radius [kpc]", fontsize=14)
    plt.ylabel(r"Velocity squared [km$^2$/s$^2$]", fontsize=16)
    plt.xlim([-1, max_rad*1.1])
    plt.xticks(fontsize=14)
    plt.savefig(f"{results_filepath}vr.pdf", dpi=200, bbox_inches='tight')

    plt.figure(figsize=(12.5,8))
    plt.scatter(radii, vtheta_sq, marker='.', c='navy', alpha=0.25, rasterized=True)
    plt.plot(r, vtheta_fit, c='gold')
    plt.title('$V_\\theta^2$(q) vs r with B-Spline fit', fontsize=16)
    plt.xlabel("Radius [kpc]", fontsize=12)
    plt.ylabel(r"Velocity squared [km$^2$/s$^2$]", fontsize=16)
    plt.xlim([-1, max_rad*1.1])
    plt.xticks(fontsize=14)
    plt.savefig(f"{results_filepath}vtheta.pdf", dpi=200, bbox_inches='tight')

    plt.figure(figsize=(12.5,8))
    plt.scatter(radii, vphi_sq, marker='.', c='navy', alpha=0.25, rasterized=True)
    plt.plot(r, vphi_fit, c='gold')
    plt.title('$V_\\phi^2$(q) vs r with B-Spline fit', fontsize=16)
    plt.xlabel("Radius [kpc]", fontsize=12)
    plt.ylabel(r"Velocity squared [km$^2$/s$^2$]", fontsize=16)
    plt.xlim([-1, max_rad*1.1])
    plt.xticks(fontsize=14)
    plt.savefig(f"{results_filepath}vphi.pdf", dpi=200, bbox_inches='tight')

    plt.figure(figsize=(12.5,8))
    plt.plot(r, beta, c='navy')
    plt.axhline(0, c='gold', label=r"$\beta=0$")
    plt.xlabel("Radius [kpc]", fontsize=16)
    plt.ylabel(r"Anisotropy ($\beta$)", fontsize=18)
    plt.ylim([-1.0, 1.0])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(prop={'size':20})
    plt.savefig(f"{results_filepath}beta.pdf", dpi=200, bbox_inches='tight')

S = agama.splineLogDensity(knots, x=np.log(radii), w=np.ones(len(radii)))
rho = rho_eval(S, r)
dlnrho_dlnr = dlnrho_dlnr_eval(S, r)

if VERBOSE:
    fig, ax = plt.subplots(figsize=(12.5,8))
    plt.loglog(r, rho) #, base=np.e)
    ax.set_title(r"log Density profile - log($\rho$) vs log($r$)", fontsize=16)
    ax.set_xlabel("radius [kpc]", fontsize=16)
    ax.set_ylabel(r"$\rho$ [kpc$^{-3}$]", fontsize=16)
    ax.xaxis.set_major_formatter(tk.ScalarFormatter())
    ax.yaxis.set_major_formatter(tk.ScalarFormatter())
    plt.savefig(f"{results_filepath}ln_rho.pdf", dpi=200, bbox_inches='tight')

# Split Spherical Jeans Eq (Eq. 1 in Rehemtulla et al. 2022) into three terms
a = -dlnrho_dlnr
b = -dvr/vr_fit
c = -2*beta

M_jeans = (vr_fit * r / G) * (a + b + c)

if VERBOSE:
    if TRUE_PROVIDED:
        fig = plt.figure(figsize=(12,8))
        axs = fig.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw=dict(hspace=0, height_ratios=[3,1]))

        axs[0].plot(r, M_jeans, c='navy', linewidth=2.0, label='Jeans Estimated Mass')
        axs[0].plot(true_r, true_M, c='gold', linestyle='dashed', label='True Cumulative Mass')
        axs[1].plot(r, fractional_err(r, true_r, M_jeans, true_M), c='navy', linewidth=2.0, label='Error in Jeans Estimate')

        axs[1].axhline(0, c='k', linewidth=0.5)
        axs[1].axhline(0.2, c='k', linewidth=0.5, linestyle='dashdot')
        axs[1].axhline(-0.2, c='k', linewidth=0.5, linestyle='dashdot')
        axs[0].set_title(dataset_name + r" $M_{Jeans}(<r)$ from B-Splines", fontsize=16, pad=20)
        axs[0].set_xlim([0, max_rad*1.1])
        axs[1].set_xlim([0, max_rad*1.1])
        axs[0].set_ylabel(r"M(<r) [$M_{\odot}$]", size=18)
        axs[1].set_xlabel('Radius [kpc]', size=16)
        axs[1].set_ylabel('Fractional Error', size=16)
        axs[0].legend(prop={'size': 16})
        axs[1].set_ylim([-0.4, 0.4])
        plt.savefig(f"{results_filepath}Mjeans.pdf", dpi=200, bbox_inches='tight')
    else:
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(r, M_jeans, c='navy', linestyle='dashed', linewidth=2.0, label='Jeans Estimated Mass')
        ax.set_title(dataset_name + r" $M_{Jeans}(<r)$ from B-Splines", fontsize=16, pad=20)
        ax.set_xlim([0, max_rad*1.1])
        ax.set_ylabel(r"M(<r) [$M_{\odot}$]", size=18)
        ax.set_xlabel('Radius [kpc]', size=16)
        ax.legend(prop={'size': 16})
        plt.savefig(f"{results_filepath}Mjeans.pdf", dpi=200, bbox_inches='tight')

np.savetxt(
    fname=f"{results_filepath}{dataset_name}_{min_knot:.0f}-{max_knot:.0f}-{int(num_knots)}.csv",
    X=np.stack([r, M_jeans], axis=1), delimiter=',',
    header="radius [kpc], jeans_mass [Msun]"
)

if VERBOSE:
    print(f"Finished modeling {dataset_name}\n")
