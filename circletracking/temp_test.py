import circletracking as circ
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pims
import os

rootpath = os.path.abspath(r'F:\_data\tpm\TPM SEM pictures')

import trackpy as tp

def imread(filename, **kwargs):
    with pims.Bioformats(filename) as file:
        result = file[0]
    return result

def plot_circles(f, image, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image, origin='lower', interpolation='none', cmap='gray')
    for i in f.index:
        circle = plt.Circle((f.loc[i].x,
                             f.loc[i].y),
                            radius=f.loc[i].r,
                            fc='None', ec='r', ls='solid',
                            lw=1, label=i)
        ax.add_patch(circle)
    return ax

all_estimates = [1.35, 0.35, 1.8, 0.15, 1.9, 1.9]
all_sizing = ['TMP54', 'TMP51_EtOHwash', 'TMP55', 'TMP57', 'TMP58_oilbath', 'TMP58_oven']
Bioformats_Sequence = pims.image_sequence.customize_image_sequence(imread)

os.chdir(os.path.join(rootpath, 'sizing'))

min_diameter = 0.2
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

for estimate, path in zip(all_estimates[-1:], all_sizing[-1:]):
    images = Bioformats_Sequence(path + '_*.tif')
    f_accum = []
    for image in images:
        image = image[:850]
        mpp = image.metadata['mpp'] * 10000
        estimate_mpp = estimate / mpp / 2
        f = circ.locate_disks(image, (estimate_mpp / 2, estimate_mpp * 2),
                              threshold=0.2)

        ax.cla()
        ax = plot_circles(f, image, ax)
        ax.figure.savefig(images._filepaths[image.frame_no] + '_fitted.png', dpi=300)

        f['r_um'] = f['r'] * mpp
        f.to_csv(images._filepaths[image.frame_no] + '_fitted.txt')
        f_accum.append(f)
    pd.concat(f_accum).to_csv(path + '_accum.txt')
