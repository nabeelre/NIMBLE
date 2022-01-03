import os, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 28,
})
plt.rcParams['axes.linewidth'] = 3

home_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
write_path = home_path + "/figures/"

qs = [0.6, 0.8, 1.0]

r_jeans = [[],[],[]]
r_trues = [[],[],[]]
M_jeans = [[],[],[]]
M_trues = [[],[],[]]

def read(idx, j_file, t_file):
    jeans_dat = np.loadtxt(j_file, delimiter=',', skiprows=1)
    true_dat = np.loadtxt(t_file, delimiter=',', skiprows=1)

    r_jeans[idx] = jeans_dat[:,0]
    r_trues[idx] = true_dat [:,0]
    M_jeans[idx] = jeans_dat[:,1]
    M_trues[idx] = true_dat [:,1]

def fractional_err(r, r_true, M, M_true):
    frc_error = np.zeros(len(r))

    for i in range(len(r)):
        match_idx = (np.abs(r_true - r[i])).argmin()
        frc_error[i] = (M[i]-M_true[match_idx])/M_true[match_idx]
    return frc_error

for i in range(len(qs)):
    read(i, f'{home_path}/results/halo_alone_{qs[i]}/halo_alone_{qs[i]}_1-70-6.csv', f'{home_path}/data/halo_alone/halo_alone_{qs[i]}_true.csv')

fig = plt.figure(figsize=(15,10), dpi=250)
gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])
axs = gs.subplots(sharex=True)
colors = ['#931621', '#157F1F', '#4a0078']
vert_shift = [0, 0.17e12, 0.4e12]
marker = ['', '*', '*']
label_pos = [(68, 0.3), (68, 0.45), (68, 0.6)]

for i in range(len(qs)):
    axs[0].plot(r_jeans[i], (M_jeans[i]+vert_shift[i])/1e12, c=colors[i], linewidth=4.5)
    axs[0].plot(r_trues[i], (M_trues[i]+vert_shift[i])/1e12, c=colors[i], linewidth=2.5, linestyle='dashed')
    axs[1].plot(r_jeans[i], 100*fractional_err(r_jeans[i], r_trues[i], M_jeans[i], M_trues[i]), c=colors[i], linewidth=4.0)

    axs[0].text(x=label_pos[i][0], y=label_pos[i][1], s=rf"$q={qs[i]}{marker[i]}$", fontsize=24, c=colors[i])

axs[0].text(x=47, y=0.1, s="*vertically shifted for clarity", fontsize=24)

axs[0].plot([], [], linewidth=4, c='gray', label="Jeans Estimate")
axs[0].plot([], [], linewidth=2.5, c='gray', linestyle="dashed", label="True")
axs[0].legend(loc='upper left', fontsize=22, frameon=False)

axs[1].axhline(0, c='k', linewidth=1.0)
axs[1].axhline(10., c='k', linewidth=0.5)
axs[1].axhline(-10., c='k', linewidth=0.5)

axs[0].set_ylim([0, 1.9])
axs[1].set_ylim([-20., 20.])
axs[1].set_xlim([0,80])

axs[0].set_ylabel(r"$M(< r)$ [$10^{12}$ $M_{\odot}$]", size=30, labelpad=15)
axs[1].set_xlabel(r'$r$ [kpc]', size=38, labelpad=10)
axs[1].set_ylabel('% Error', size=34)

axs[1].set_yticks([-10., 0, 10.])
axs[1].set_yticklabels(["-10%", "0", "+10%"])
axs[1].set_yticks([-15., -5., 5., 15.], minor=True)
axs[1].tick_params(direction='in', axis='both', length=8, width=3, bottom=True, left=True, right=True, pad=10)
axs[1].tick_params(axis='x', labelsize='medium')
axs[1].tick_params(axis='y', labelsize='small')
axs[1].tick_params(direction='in', length=4, width=3, which='minor', bottom=True, left=True, right=True)

axs[0].yaxis.set_major_locator(MultipleLocator(0.3))
axs[0].yaxis.set_minor_locator(MultipleLocator(0.15))
axs[0].xaxis.set_minor_locator(MultipleLocator(5))
axs[0].tick_params(direction='in', length=8, width=3, left=True, right=True, top=True, labelsize='medium', pad=10)
axs[0].tick_params(direction='in', length=4, width=3, which='minor', left=True, right=True, top=True)
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)

plt.savefig(f'{write_path}/fig3.pdf', bbox_inches='tight')
plt.cla()
