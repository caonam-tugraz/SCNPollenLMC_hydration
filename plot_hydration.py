import os
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import numpy as np
import statistics
import timeit
import pickle
import utils
from matplotlib.ticker import StrMethodFormatter

output = "."
transform = "hydration"
os.makedirs(f'{output}/figs/{transform}/', exist_ok=True)


def viz(arch, widths, layers, dimensions):
    # colors = plt.cm.gist_rainbow(np.linspace(0, 1, 8))
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, 5))

    fixed_params = [0, 0.5, 1.0]

    plt.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

    for l in layers:
        for w in widths:
            # One4All
            file_name = f'{output}/output/{transform}/One4All/{arch}_{l}_{w}/acc.npy'
            acc_one4all = pickle.loads(np.load(file_name))

            # Inverse
            # file_name = f'{output}/output/{transform}/Inverse/{arch}_{l}_{w}/acc.npy'
            # acc_inverse = pickle.loads(np.load(file_name))

            # LinearConnect
            file_name = f'{output}/output/{transform}/LinearConnect/{arch}_{l}_{w}/acc.npy'
            acc_lincon = np.array(pickle.loads(np.load(file_name))['linearconnect'])[:, 1]

            # One4one
            # file_name = f'{output}/output/{transform}/One4One/{arch}_{l}_{w}/acc.npy'
            # acc_one4one = pickle.loads(np.load(file_name))

            theta = [0, 0.5, 1.0]

            # Plot acc
            fig = plt.figure()
            fig.tight_layout()
            fig, ax = plt.subplots()
            ax.plot(theta, acc_one4all, label='One4All', color='grey', lw=3)
            # ax.plot(theta, acc_inverse, label='Inverse', color='cyan', lw=3)
            ax.plot(theta, acc_lincon, label='LMC', color='k', lw=3)
            for i, d in zip(range(len(dimensions)), dimensions):
                file_name = f'{output}/output/{transform}/SCN/hhn{arch}_{l}_{w}_{d}/acc.npy'
                acc_hhn = pickle.loads(np.load(file_name))
                ax.plot(theta, acc_hhn['acc'], label=f'SCN D={d}', color=colors[i])


            ax.grid(True)
            plt.title(f'{transform} - {arch} - pollen', fontsize=16, pad=20)
            # plt.legend(ncol=2, prop={'size': 10})
            plt.legend(bbox_to_anchor=(0.85, -0.1), ncol=2, prop={'size': 10})
            ax.tick_params(axis='x', which='major', labelsize=12)
            ax.tick_params(axis='y', which='major', labelsize=12)
            ax.set_xticks([0, 0.5, 1.0])
            
            plt.savefig(f"./figs/{transform}/viz_acc_{arch}_{l}_{w}.png", bbox_inches='tight', dpi=300)

            # Beta space
            # fig = plt.figure()
            # fig.tight_layout()
            # fig, ax = plt.subplots(1, len(dimensions), sharey=True)
            # plt.rc('font', **{'size': 4})
            # for d_id, d in zip(range(len(dimensions)), dimensions):
            #     file_name = f'{output}/output/{transform}/SCN/hhn{arch}_{l}_{w}_{d}/acc.npy'
            #     acc_hhn = pickle.loads(np.load(file_name))
            #     cols = cm.gist_rainbow(np.linspace(0, 1, d))
            #     for i in range(len(acc_hhn['beta_space'][0])):
            #         yline = acc_hhn['beta_space'][:, i]
            #         ax[d_id].plot(theta, yline, alpha=0.5, color=cols[i], label=r'$\beta_{%d}$' % (i + 1))
            #     ax[d_id].tick_params(axis='x', which='major', labelsize=4)
            #     ax[d_id].tick_params(axis='y', which='major', labelsize=4)
            #     ax[d_id].legend(loc='upper right', prop={'size': 3})
            #     ax[d_id].axes.xaxis.labelpad = -1
            #     ax[d_id].set_xlabel(r'$\alpha$', fontsize=4)
            #     ax[d_id].grid(True, linestyle=':')
            #     ax[d_id].set_title(f'D={d}')
            # for d_id in range(len(dimensions)):
            #     ax[d_id].axis('square')
            #     # ax[d_id].set_aspect(0.5)
            #     ax[d_id].autoscale(enable=True, axis='both', tight=True)
            # plt.savefig(f"{output}/figs/{transform}/viz_beta_{arch}_{l}_{w}.png", bbox_inches='tight', dpi=600)

def dacc(arch, widths, layers):
    dimensions = [1, 2, 3, 5, 8]
    fixed_setting = [0.2, 0.5, 1.0, 1.5, 2.0]

    plt.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

    for l in layers:
        for w in widths:
            # One4All
            file_name = f'{output}/output/{transform}/One4All/{arch}_{l}_{w}/acc.npy'
            acc_one4all = pickle.loads(np.load(file_name))
            print(f"One4All acc={np.mean(acc_one4all)}")

            # Inverse
            file_name = f'{output}/output/{transform}/Inverse/{arch}_{l}_{w}/acc.npy'
            acc_inverse = pickle.loads(np.load(file_name))
            print(f"Inverse acc={np.mean(acc_inverse)}")

            # LinearConnect
            file_name = f'{output}/output/{transform}/LinearConnect/{arch}_{l}_{w}/acc.npy'
            acc_lincon = np.array(pickle.loads(np.load(file_name))['linearconnect'])[:, 1]
            print(f"LinearConnect acc={sum(acc_lincon) / len(acc_lincon)}")

            # One4one
            file_name = f'{output}/output/{transform}/One4One/{arch}_{l}_{w}/acc.npy'
            acc_one4one = pickle.loads(np.load(file_name))
            print(f"One4Onne acc={sum(acc_one4one.values()) / len(acc_one4one)}")

            # HHN
            acc_hhn = []
            for d in dimensions:
                file_name = f'{output}/output/{transform}/SCN/hhn{arch}_{l}_{w}_{d}/acc.npy'
                a_hhn = pickle.loads(np.load(file_name))
                acc_hhn.append(a_hhn['acc'])
                print(f"D={d}, acc={np.mean(a_hhn['acc'])}")

            # helper function
            labels = []
            def add_label(violin, label):
                color = violin["bodies"][0].get_facecolor().flatten()
                labels.append((mpatches.Patch(color=color), label))

            # Plot D vs acc, with One4All, Inverse, One4One
            fig = plt.figure()
            fig.tight_layout()
            fig, ax = plt.subplots()
            add_label(ax.violinplot(acc_hhn, dimensions, widths=1), 'SCNs')

            add_label(ax.violinplot(acc_one4all, [-1], widths=1), 'One4All')
            add_label(ax.violinplot(acc_inverse, [dimensions[-1] + 2], widths=1), 'Inverse')
            add_label(ax.violinplot(acc_lincon, [dimensions[-1] + 3], widths=1), 'LMC')

            x = []
            for fixed_s in fixed_setting:
                x.append(acc_one4one[str(fixed_s)])
            add_label(ax.violinplot(x, [dimensions[-1] + 4], widths=1), "One4One")

            plt.legend(*zip(*labels), loc='lower center', prop={'size': 16})
            plt.xlabel('Number of dimensions D', fontsize=18)
            plt.ylabel('Test accuracy', fontsize=18)
            plt.title(f'{transform} - {arch} - pollen', fontsize=20)
            
            ax.tick_params(axis='x', which='major', labelsize=16)
            ax.tick_params(axis='y', which='major', labelsize=16)
            ax.set_xticks([1, 2, 3, 5, 8])
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

            ax.axvline(x=0, color='k', ls='--')
            ax.axvline(x=9, color='k', ls='--')

            ax.grid(True, linestyle=':')
            plt.savefig(f"{output}/figs/{transform}/d_{arch}_{l}_{w}.png", bbox_inches='tight', dpi=100)


if __name__ == '__main__':
    viz('mlpb_hydration', widths=[256], layers=[1], dimensions=[1, 2, 3, 5, 8])
    viz('sconvb_hydration', widths=[256], layers=[1], dimensions=[1, 2, 3, 5, 8])
    
    # dacc('mlpb', widths=[16], layers=[1])

    # viz('mlpb', widths=[32], layers=[1], dimensions=[1, 2, 3, 5, 8])
    # dacc('mlpb', widths=[32], layers=[1])

    # viz('mlpb', widths=[64], layers=[1], dimensions=[1, 2, 3, 5, 8])
    # dacc('mlpb', widths=[64], layers=[1])

    # viz('sconvb', widths=[32], layers=[2], dimensions=[1, 2, 3, 5, 8])
    # dacc('sconvb', widths=[32], layers=[2])
