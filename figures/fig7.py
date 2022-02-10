import sys, os, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 28,
})
plt.rcParams['axes.linewidth'] = 3

def read_true(filepath):
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    r = data[:,0]; M = data[:,1]

    # plt.plot(r, M)
    # plt.xlim([0,180])
    # plt.show()

    return r, M

def fractional_err(r, r_true, M, M_true):
    frc_error = np.zeros(len(r))

    for i in range(len(r)):
        match_idx = (np.abs(r_true - r[i])).argmin()
        frc_error[i] = (M[i]-M_true[match_idx])/M_true[match_idx]
    return frc_error

home_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
write_path = home_path + "/figures/"

all_sims = ['m12f', 'm12i', 'm12m']
all_lsrs = ['LSR0', 'LSR1', 'LSR2']

sims_to_plot = ['m12f', 'm12i', 'm12m']
lsrs_to_plot = ['LSR0']
lsrs_to_plot = [lsr.upper() for lsr in lsrs_to_plot]

datasets_count = len(sims_to_plot)*len(lsrs_to_plot)
if datasets_count == 0:
    print("Cannot select no sims or lsrs to plot")
    exit()

gaia_release = ""
if len(sys.argv) == 1:
    gaia_release = 'dr3'
elif len(sys.argv) == 2:
    gaia_release = sys.argv[1].lower()
    assert(gaia_release in ['dr3', 'dr4', 'dr5'])
else:
    print("Could not understand command line arguments")
    exit()

rs           = np.zeros((9,201))
Menc_lows    = np.zeros((9,201))
Menc_meds    = np.zeros((9,201))
Menc_upps    = np.zeros((9,201))
percerr_lows = np.zeros((9,201))
percerr_meds = np.zeros((9,201))
percerr_upps = np.zeros((9,201))

r_trues   = [[],[],[]]
M_trues   = [[],[],[]]

for i, sim in enumerate(all_sims):
    if sim not in sims_to_plot:
        r_trues[i] = 0; M_trues[i] = 0;
    else:
        r_trues[i], M_trues[i] = read_true(f"{home_path}/data/{sim}/{sim}_true.csv")

for sim_idx, sim in enumerate(all_sims):
    for lsr_idx, lsr in enumerate(all_lsrs):
        iter = lsr_idx+sim_idx*len(all_lsrs)
        if (lsr in lsrs_to_plot) and (sim in sims_to_plot):
            rs[iter], Menc_lows[iter], Menc_meds[iter], Menc_upps[iter], _, _, _ = np.loadtxt(f"{home_path}/results/deconv_{sim}_{lsr}_{gaia_release}/Menc_beta_final.csv", delimiter=',', skiprows=1, unpack=True)
            percerr_lows[iter] = 100*fractional_err(rs[iter], r_trues[sim_idx], Menc_lows[iter], M_trues[sim_idx])
            percerr_meds[iter] = 100*fractional_err(rs[iter], r_trues[sim_idx], Menc_meds[iter], M_trues[sim_idx])
            percerr_upps[iter] = 100*fractional_err(rs[iter], r_trues[sim_idx], Menc_upps[iter], M_trues[sim_idx])

# For displaying label in plot
if gaia_release == 'dr3': gaia_release = '(E)DR3'

fig = plt.figure(figsize=(15,20))
gs = fig.add_gridspec(4, hspace=0, height_ratios=[3,1,1,1])
axs = gs.subplots(sharex=True)
marker = ['*', '*', '']
vert_shift = [0.6e12, 0.45e12, 0]
colors = ['#4a0078', '#bf6c06', '#097575']

# Loops to plot all lines and bands
for sim_idx, sim in enumerate(all_sims):
    for lsr_idx, lsr in enumerate(all_lsrs):
        iter = lsr_idx+sim_idx*len(all_lsrs)
        # Jeans estimate line
        axs[0].plot(rs[iter], (Menc_meds[iter]+vert_shift[sim_idx])/1e12, c=colors[sim_idx], linewidth=2.5, zorder=10)
        # Uncertainty band around Jeans estimate
        axs[0].fill_between(rs[iter], (Menc_lows[iter]+vert_shift[sim_idx])/1e12, (Menc_upps[iter]+vert_shift[sim_idx])/1e12, color=colors[sim_idx], alpha=0.3, lw=0, zorder=10)

        # Percent error line
        axs[sim_idx+1].plot(rs[iter], percerr_meds[iter], c=colors[sim_idx], linewidth=2)
        # Uncertainty band around percent error
        axs[sim_idx+1].fill_between(rs[iter], percerr_lows[iter], percerr_upps[iter], color=colors[sim_idx], alpha=0.3, lw=0)
    # True mass profile line
    axs[0].plot(r_trues[sim_idx], (M_trues[sim_idx]+vert_shift[sim_idx])/1e12, c=colors[sim_idx], linewidth=2.5, linestyle='dashed')

# Legend elements for main panel
axs[0].plot([], [], linewidth=4, c='gray', label="Jeans Estimate")
axs[0].plot([], [], linewidth=2.5, c='gray', linestyle="dashed", label="True")
axs[0].fill_between([], [], [], color='lightgray', label=r"$\pm1\sigma$")

# Labels on jeans estimate lines
axs[0].text(x=62, y=0.1, s="*vertically shifted for clarity", fontsize=24)
if "m12f" in sims_to_plot:
    axs[0].text(x=4.5, y=0.72, s="m12f*", rotation=30, fontsize=24, c=colors[0], zorder=15)
if "m12i" in sims_to_plot:
    axs[0].text(x=10, y=0.60, s="m12i*", rotation=25, fontsize=24, c=colors[1], zorder=15)
if "m12m" in sims_to_plot:
    axs[0].text(x=11, y=0.13, s="m12m", rotation=31, fontsize=24, c=colors[2], zorder=15)

axs[0].text(x=53 if gaia_release=='(E)DR3' else 57, y=1.49, s=f"Deconvolution with {gaia_release.upper()} Errors", rotation=0, fontsize=24, c='k')
axs[0].legend(loc='upper left', fontsize=22, frameon=False)

axs[0].set_ylabel(r"$M(< r)$ [$10^{12}$ $M_{\odot}$]", size=30, labelpad=15)
axs[3].set_xlabel(r'$r$ [kpc]', size=38, labelpad=10)

# Main panel ticks
axs[0].set_xlim([0,110])
axs[0].set_ylim([0, 1.6])
axs[0].yaxis.set_major_locator(MultipleLocator(0.3))
axs[0].yaxis.set_minor_locator(MultipleLocator(0.15))
axs[0].xaxis.set_minor_locator(MultipleLocator(10))
axs[0].tick_params(direction='in', length=8, width=3, left=True, right=True, top=True, labelsize='medium', pad=10)
axs[0].tick_params(direction='in', length=4, width=3, which='minor', left=True, right=True, top=True)
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)

# Legend elements for error panels
axs[1].plot([], [], linewidth=3, c='gray', label="Median Error")
axs[1].fill_between([], [], [], color='lightgray', label=r"$\pm1\sigma$")
axs[1].legend(loc='upper right', fontsize=22, frameon=False)

# Labeling for each error panel
if "m12f" in sims_to_plot:
    axs[1].text(x=2, y=22, s="m12f", fontsize=24, c=colors[0])
if "m12i" in sims_to_plot:
    axs[2].text(x=2, y=22, s="m12i", fontsize=24, c=colors[1])
if "m12m" in sims_to_plot:
    axs[3].text(x=2, y=22, s="m12m", fontsize=24, c=colors[2])

axs[2].set_ylabel('% Error', size=34)

# Limits, and ticks for error panels
for i in range(1,4):
    # Vertical axis limits and guide lines
    axs[i].axhline(0, c='k', linewidth=1.0)
    axs[i].axhline(20., c='gray', linewidth=0.5)
    axs[i].axhline(-20., c='gray', linewidth=0.5)
    axs[i].set_ylim([-40., 40.])
    axs[i].set_xlim([0,110])

    # Ticks
    axs[i].set_yticks([-20., 0, 20.])
    axs[i].set_yticklabels(["-20%", "0", "+20%"])
    axs[i].set_yticks([-30., -10., 10., 30.], minor=True)
    axs[i].xaxis.set_minor_locator(MultipleLocator(10))
    axs[i].tick_params(direction='in', axis='both', length=7, width=3, bottom=True, left=True, right=True, top=True, pad=10)
    axs[i].tick_params(axis='x', labelsize='medium')
    axs[i].tick_params(axis='y', labelsize='medium')
    axs[i].tick_params(direction='in', length=5, width=3, which='minor', bottom=True, left=True, right=True, top=True)

plt.savefig(f'{write_path}/{gaia_release.upper()}_deconv.pdf', bbox_inches='tight')
print("Finished plotting deconvolution results\n")
plt.cla()
