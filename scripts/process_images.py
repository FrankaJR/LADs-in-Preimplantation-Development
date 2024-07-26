# import libraries
import os
import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scipy import ndimage as ndi
from scipy.signal import fftconvolve

import skimage
from skimage.measure import label, regionprops
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)

#################
### LOAD DATA ###
#################

# load arguments
INDIR = sys.argv[1] + '/'
BORDER_WIDTH_MICRON = 1
STEP_SIZE_MICRON = 0.5
N_STEPS = 10

# infer filenames
fnls = glob(INDIR + '*sample_info.tsv')
assert len(fnls) == 1
SAMPLE_INFO_FN = fnls[0]
fnls = glob(INDIR + '*channel_info.tsv')
assert len(fnls) == 1
CHANNEL_INFO_FN = fnls[0]

# generate output filenames
OUTFN_ALL = SAMPLE_INFO_FN.replace('sample_info.tsv', 'data_of_all_stacks.edge_%.2fum.tsv.gz' % BORDER_WIDTH_MICRON)
OUTFN_MP = SAMPLE_INFO_FN.replace('sample_info.tsv', 'data_of_maximum_projection.tsv.gz')
OUTFN_RINGS = SAMPLE_INFO_FN.replace('sample_info.tsv', 'data_of_concentric_rings_%.2fum.tsv.gz' % STEP_SIZE_MICRON)


# load sample metadata
sample_info = pd.read_table(SAMPLE_INFO_FN)
sample_info['sample_name'] = sample_info.apply(axis=1, func = lambda r: '%s_%s' % (r['condition'], r['embryo_id']))
assert sample_info['sample_name'].duplicated().sum() == 0
sample_info = sample_info.set_index('sample_name')
sample_info

# load channel metadata
channel_info = pd.read_table(CHANNEL_INFO_FN).sort_values(by=['channel_index']).reset_index(drop=True)
N_CHANNELS = len(channel_info)
assert all(channel_info['channel_index'].values == np.arange(N_CHANNELS))
CHANNEL_NAMES = channel_info['name'].values
DAPI_CHANNEL = channel_info.loc[channel_info['name']=='DAPI','channel_index'].values[0]

# load images
print('Loading images')
image_dict = dict()

for sample, row in tqdm(sample_info.iterrows(), total=len(sample_info)):
    im = skimage.io.imread(INDIR + row['filename'])

    s = np.array(im.shape)

    # Add an extra z-stack axis if necessary
    if len(s) < 4:
        # add z-stack axis
        im = im[np.newaxis,:,:,:]
        s = np.array(im.shape)
    assert len(s) == 4

    # infer which axis refers to which aspect of the data
    axis_ind = np.arange(len(s))
    frame_axes = axis_ind[s == max(s)]
    channel_axis = axis_ind[s == N_CHANNELS]
    assert len(channel_axis) == 1, (s, channel_axis)
    z_axis = axis_ind[~(np.isin(axis_ind, frame_axes) | np.isin(axis_ind, channel_axis))]
    assert len(z_axis) == 1

    # determine how new and old order relate (wanted order: (Z, X, Y, channel))
    ls = np.concatenate([z_axis, frame_axes, channel_axis])
    order_dict = {old: new for new, old in enumerate(ls)}
    new_order = [order_dict[i] for i in axis_ind]

    # enforce wanted order
    if not all(new_order == axis_ind):
        im = np.moveaxis(im, axis_ind, new_order)
        #print('Reordering axes: ', tuple(s), 'to', im.shape)

    # add data to dict
    sample_info.loc[sample,'n_zstacks'] = im.shape[0]
    image_dict[sample] = im

#################
### FUNCTIONS ###
#################

# Function to call objects in image
def get_objects(D, smooth_std=2, n_classes=3):

    # smooth
    Dsmooth = ndi.gaussian_filter(D, smooth_std)

    # select threshold
    thresholds = filters.threshold_multiotsu(Dsmooth, classes=n_classes)

    # test whether lowest threshold is appropriate:
    pass_lowest = (Dsmooth > thresholds[0]).mean()
    if (pass_lowest > 0.25) and n_classes > 1:
        thresholds = thresholds[1:]

    # define regions
    regions = np.digitize(Dsmooth, bins=thresholds)

    # fill holes
    if (len(regions.shape) == 3):
        objects = list()
        for i in range(regions.shape[0]):
            r = np.copy(regions[i,:,:])
            r = ndi.binary_fill_holes(r)
            objects.append(r)
        objects = np.array(objects)
    else:
        objects = ndi.binary_fill_holes(regions)

    # erode nuclear edge to account for expansion due to smoothing
    for _ in range(smooth_std):
        objects = morphology.binary_erosion(objects)

    # label objects
    objects = measure.label(objects)

    return thresholds, objects


# Function to call the best stack for each object
def determine_best_stack_per_object(object_calls):
    
    # determine number of z-slices
    Nz = object_calls.shape[0]
    all_objects = np.sort(np.unique(object_calls))
    all_objects = all_objects[all_objects != 0]
    
    # if only 1 z-slice is available, that is the best one for all objects
    if Nz == 1:
        best_stack_per_object = {obj: 0 for obj in all_objects}
        return best_stack_per_object
    
    ### Requirements for best stack
    # 1. Not in bottom or top 10 stacks. (if more than 20 stacks are provided)
    # 2. Biggest DAPI area after obeying requirements 1

    # determine DAPI area per stack
    object_areas = {i_object: list() for i_object in all_objects}
    for iz in range(Nz):
        for obj in all_objects:
            object_ind = (object_calls[iz,:,:] == obj)
            object_areas[obj].append(object_ind.sum())

    # determine best stack (== biggest DAPI area) for each object
    best_stack_per_object = dict()

    for obj in all_objects:

        areas = np.array(object_areas[obj])

        # disqualify top and bottom 10 stacks
        if len(areas) > 20:
            areas[:10] = 0
            areas[-10:] = 0

        max_area = max(areas)
        if max_area < 1000:
            continue

        # identify best stack
        best_iz = np.arange(len(areas))[areas == max_area]
        #if len(best_iz) > 1:
        #    print('Multiple best stacks, selecting first from: ', best_iz)
        best_iz = best_iz[0]

        best_stack_per_object[obj] = best_iz

    return best_stack_per_object

# Function to get the indices of nucleus edge
def get_edge_within_nucleus(Cloc, npix):

    # alternative
    Cfilt = ndi.uniform_filter(Cloc.astype(float), size=npix)
    Cfilt = np.round(Cfilt, 4)

    # find edge
    edge_ind = (Cfilt > 0) & (Cfilt < 1) & (Cloc == 1)
    interior_ind = (Cloc == 1) & (~edge_ind)

    return edge_ind, interior_ind

# Function to get location of concentric rings
def determine_concentric_rings(Cloc, step_size_npix, n_steps):

    rim_dict = dict()
    C_erode = np.copy(Cloc)

    for n in range(n_steps):
        Cfilt = ndi.uniform_filter(C_erode.astype(float), size=step_size_npix)
        Cfilt = np.round(Cfilt, 4)

        ind = (Cfilt > 0) & (Cfilt < 1) & (C_erode == 1)

        if ind.sum() == 0:
#             print('End reached: ', n)
            break

        rim_dict[n+1] = ind
        C_erode[ind] = 0

    return rim_dict

######################
### IMPLEMENTATION ###
######################


### Determine best z-stack for each object ###
print('Determining best stack for each nucleus')
best_stack_per_sample = dict()
all_stacks_per_sample = dict()

for condition in tqdm(image_dict.keys(), total=len(sample_info)):
    Dall = np.copy(image_dict[condition][:,:,:,DAPI_CHANNEL])
#     max_proj_objects = objects_in_max_projection[condition]
#     objects_in_max_proj = np.sort(np.unique(max_proj_objects))[1:]
    _, object_calls = get_objects(Dall, smooth_std=2, n_classes=3)
    
    # clean up object calls: per 3d object, retain biggest object per slice
    discard_label = np.max(object_calls)+1
    for iz in range(object_calls.shape[0]):
        
        # determine which objects are in this slice
        object_ls = np.sort(np.unique(object_calls[iz,:,:]))[1:]
        
        # find all separate pieces
        pieces = np.copy(object_calls[iz,:,:] > 0).astype(int)
        pieces = measure.label(pieces)
        pieces_ls = np.sort(np.unique(pieces))[1:]
        
        # if the number of pieces is the same as the number of objects, continue
        if len(object_ls) == len(pieces_ls):
            continue
        
        # find which object is fragmented
        for obj in object_ls:
            
            # which pieces match this object
            match_pieces = np.sort(np.unique(pieces[object_calls[iz,:,:]==obj]))
            if len(match_pieces) == 1:
                continue
                
            # identify which piece is the largest and discard the rest
            sizes = [(k, (pieces==k).sum()) for k in match_pieces]
            sizes = sorted(sizes, key = lambda k: -k[1])
            keep = sizes[0][0]
            remove_ind = (object_calls[iz,:,:]==obj) & (pieces != keep)
            
            # save result
            calls = object_calls[iz,:,:]
            calls[remove_ind] = discard_label
            object_calls[iz,:,:] = calls
            discard_label += 1
    
    # rename objects so they are ordered by size
    object_ls = np.sort(np.unique(object_calls))[1:]
    object_sizes = [(o, (object_calls==o).sum()) for o in object_ls]
    object_sizes = sorted(object_sizes, key = lambda k: -k[1])
    rename_dict = {k[0]: i+1 for i, k in enumerate(object_sizes)}
    new = np.zeros_like(object_calls)
    for obj in object_ls:
        new[object_calls==obj] = rename_dict[obj]
    object_calls = new

    # determine what the best stack is per object
    best_stack_per_object = determine_best_stack_per_object(object_calls)

    # store results
    best_stack_per_sample[condition] = best_stack_per_object
    all_stacks_per_sample[condition] = object_calls



### Determine location of nuclear edge ###
print('Determining nuclear edges')
edge_dict = {condition: dict() for condition in image_dict}
interior_dict = {condition: dict() for condition in image_dict}

for condition in tqdm(image_dict.keys(), total=len(sample_info)):

    Nz = image_dict[condition].shape[0]
    observed_objects = sorted(list(best_stack_per_sample[condition].keys()))
    nuclei = [1, 2] if len(observed_objects) > 1 else [1]

    # determine width of nuclear edge in number of pixels
    image_size = sample_info.loc[condition,'image_dimension']
    npix = image_dict[condition].shape[1]/image_size * BORDER_WIDTH_MICRON * 2
    npix = int(np.round(npix,0))

    for i_object in nuclei:

        best_iz = best_stack_per_sample[condition][i_object]
        edge_dict[condition][i_object] = dict()
        interior_dict[condition][i_object] = dict()

        for iz in range(Nz):
            object_ind = (all_stacks_per_sample[condition][iz,:,:] == i_object)

            C = np.copy(object_ind)
            edge_dict[condition][i_object][iz], interior_dict[condition][i_object][iz] = get_edge_within_nucleus(C, npix)

### Determine location of concentric rings ###
print('Determining concentric rings')
concentric_ring_dict = {condition: dict() for condition in image_dict}
for condition in tqdm(image_dict.keys(), total=len(sample_info)):

    Nz = image_dict[condition].shape[0]
    observed_objects = sorted(list(best_stack_per_sample[condition]))
    nuclei = [1, 2] if len(observed_objects) > 1 else [1]

    # determine number of pixels per step
    image_size = sample_info.loc[condition,'image_dimension']
    step_size_npix = image_dict[condition].shape[1]/image_size * STEP_SIZE_MICRON * 2
    step_size_npix = int(np.round(step_size_npix,0))

    for i_object in nuclei:

        concentric_ring_dict[condition][i_object] = dict()

        for iz in range(Nz):
            object_ind = (all_stacks_per_sample[condition][iz,:,:] == i_object)
            C = np.copy(object_ind).astype(int)
            concentric_ring_dict[condition][i_object][iz] = determine_concentric_rings(C, step_size_npix, N_STEPS)


### Collect all image data ###


### Get information on maximum projection ###
print('Collecting data for maximum projection')
output_mp = defaultdict(list)
for condition in tqdm(image_dict, total=len(image_dict.keys())):

    observed_objects = sorted(list(best_stack_per_sample[condition].keys()))
    nuclei = [1, 2] if len(observed_objects) > 1 else [1]

    # iterate over all nuclei
    for i_nucleus in nuclei:

        ### Collect data for maximum projection ###
        object_ind = (all_stacks_per_sample[condition] == i_nucleus).astype(int).max(axis=0).astype(bool)
        background_ind = (all_stacks_per_sample[condition] > 0).sum(axis=0) == 0
        output_mp['sample_name'].append(condition)
        output_mp['nucleus'].append(i_nucleus)
        output_mp['nucleus_area'].append(object_ind.sum())
        output_mp['background_area'].append(background_ind.sum())

        for ichannel, name in enumerate(CHANNEL_NAMES):
            D = np.copy(image_dict[condition][:,:,:,ichannel]).max(axis=0)

            tmp = {'background': background_ind, 'nucleus': object_ind}
            for area_name, ind in tmp.items():
                S = D[ind] if ind.sum() > 0 else np.array([np.nan])
                output_mp['%s.%s_mean' % (name,area_name)].append(S.mean())
                output_mp['%s.%s_std' % (name,area_name)].append(np.std(S))
                output_mp['%s.%s_p25' % (name,area_name)].append(np.percentile(S,25))
                output_mp['%s.%s_p50' % (name,area_name)].append(np.percentile(S,50))
                output_mp['%s.%s_p75' % (name,area_name)].append(np.percentile(S,75))

output_mp = pd.DataFrame(output_mp)
output_mp.to_csv(OUTFN_MP, sep='\t', header=True, index=False, na_rep='NA')


### Get statistics for all z-stacks ###
print('Collecting data for each stack')
output_all = defaultdict(list)
for condition in tqdm(image_dict, total=len(image_dict.keys())):

    observed_objects = sorted(list(best_stack_per_sample[condition].keys()))
    nuclei = [1, 2] if len(observed_objects) > 1 else [1]

    # iterate over all nuclei
    for i_nucleus in nuclei:

        # iterate over stacks and compute information
        Nz = image_dict[condition].shape[0]
        best_iz = best_stack_per_sample[condition][i_nucleus]

        for iz in range(Nz):
            output_all['sample_name'].append(condition)
            output_all['nucleus'].append(i_nucleus)
            output_all['zstack'].append(iz)
            output_all['is_best_stack'].append(iz==best_iz)
            is_ext = (iz<10) or (iz>(Nz-10))
            output_all['is_exterior_10_stack'].append(is_ext)

            object_ind = (all_stacks_per_sample[condition][iz,:,:] == i_nucleus)
            background_ind = (all_stacks_per_sample[condition][iz,:,:] == 0)
            edge_ind = edge_dict[condition][i_nucleus][iz]
            interior_ind = interior_dict[condition][i_nucleus][iz]

            pos = regionprops(object_ind.astype(int))[0].bbox if object_ind.sum() > 0 else ()
            output_all['nucleus_pos'].append(pos)
            output_all['nucleus_area'].append(object_ind.sum())
            output_all['background_area'].append(background_ind.sum())
            output_all['edge_area'].append(edge_ind.sum())
            output_all['interior_area'].append(interior_ind.sum())

            for ichannel, name in enumerate(CHANNEL_NAMES):

                # collect data from best slice
                M = np.copy(image_dict[condition][iz,:,:,ichannel])

                tmp = {'background': background_ind, 'nucleus': object_ind, 'edge': edge_ind, 'interior': interior_ind}
                for area_name, ind in tmp.items():
                    S = M[ind] if ind.sum() > 0 else np.array([np.nan])
                    output_all['%s.%s_mean' % (name,area_name)].append(S.mean())
                    output_all['%s.%s_std' % (name,area_name)].append(np.std(S))
                    output_all['%s.%s_p25' % (name,area_name)].append(np.percentile(S,25))
                    output_all['%s.%s_p50' % (name,area_name)].append(np.percentile(S,50))
                    output_all['%s.%s_p75' % (name,area_name)].append(np.percentile(S,75))


output_all = pd.DataFrame(output_all)
output_all.to_csv(OUTFN_ALL, sep='\t', header=True, index=False, na_rep='NA')


### Get statistics for concentric rings in all z-stacks ###
print('Collecting data for concentric rings')
output_rings = defaultdict(list)
for condition in tqdm(image_dict, total=len(image_dict.keys())):

    observed_objects = sorted(list(best_stack_per_sample[condition].keys()))
    nuclei = [1, 2] if len(observed_objects) > 1 else [1]

    # iterate over all nuclei
    for i_nucleus in nuclei:

        # iterate over stacks and compute information
        Nz = image_dict[condition].shape[0]
        best_iz = best_stack_per_sample[condition][i_nucleus]

        for iz in range(Nz):

            rim_dict = concentric_ring_dict[condition][i_nucleus][iz]
            object_ind = (all_stacks_per_sample[condition][iz,:,:] == i_nucleus)


            for nrim, rim_ind in rim_dict.items():

                output_rings['sample_name'].append(condition)
                output_rings['nucleus'].append(i_nucleus)
                output_rings['zstack'].append(iz)
                output_rings['is_best_stack'].append(iz==best_iz)
                is_ext = (iz<10) or (iz>(Nz-10))
                output_rings['is_exterior_10_stack'].append(is_ext)
                output_rings['ring'].append(nrim)
                output_rings['nucleus_area'].append(object_ind.sum())
                output_rings['ring_area'].append(rim_ind.sum())

                for ichannel, name in enumerate(CHANNEL_NAMES):

                    # collect data from best slice
                    M = np.copy(image_dict[condition][iz,:,:,ichannel])

                    tmp = {'nucleus': object_ind, 'ring': rim_ind}
                    for area_name, ind in tmp.items():
                        S = M[ind] if ind.sum() > 0 else np.array([np.nan])
                        output_rings['%s.%s_mean' % (name,area_name)].append(S.mean())
                        output_rings['%s.%s_std' % (name,area_name)].append(np.std(S))
                        output_rings['%s.%s_p25' % (name,area_name)].append(np.percentile(S,25))
                        output_rings['%s.%s_p50' % (name,area_name)].append(np.percentile(S,50))
                        output_rings['%s.%s_p75' % (name,area_name)].append(np.percentile(S,75))

output_rings = pd.DataFrame(output_rings)
output_rings.to_csv(OUTFN_RINGS, sep='\t', header=True, index=False, na_rep='NA')

#################################
### PLOT SEGMENTATION RESULTS ###
#################################

print('Plotting segmentation results')

for sample, row in tqdm(sample_info.iterrows(), total=len(sample_info)):

    observed_objects = sorted(list(best_stack_per_sample[sample].keys()))
    nuclei = [1, 2] if len(observed_objects) > 1 else [1]

    # set up figure
    ncol = N_CHANNELS + 1
    nrow = len(nuclei) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 5*nrow), sharex=True, sharey=True)

    for irow, i_nucleus in enumerate(nuclei):

        # recall best stack
        best_iz = best_stack_per_sample[sample][i_nucleus]

        ### plot segmentation results ###
        nucleus_loc = (all_stacks_per_sample[sample][best_iz,:,:] == i_nucleus)
        edge_loc = edge_dict[sample][i_nucleus][best_iz]
        interior_loc = interior_dict[sample][i_nucleus][best_iz]

        to_plot = np.copy(all_stacks_per_sample[sample][best_iz,:,:]).clip(0,1)
        to_plot[edge_loc] = 3
        to_plot[interior_loc] = 2
        to_plot = to_plot.astype(float)
        to_plot[to_plot==0] = np.nan

        ax = axes[irow+1,0]
        ax.set_title('Segmentation of best z-stack (#%d) for nucleus_%d' % (best_iz, i_nucleus))
        ax.imshow(to_plot, cmap='jet')

        minr, minc, maxr, maxc = regionprops(nucleus_loc.astype(int))[0].bbox
        rect = mpatches.Rectangle((minc-5, minr-5), maxc - minc + 10, maxr - minr + 10,
                                  fill=False, edgecolor='k', linewidth=2)
        ax.add_patch(rect)
        ax.text( np.mean([minc, maxc]), minr - 12, 'nucleus_%d' % i_nucleus, ha='center', va='bottom' , c='k')

        # plot segmentation maximum projection
        ax = axes[0,0]
        ax.set_title('Objects in maximum projection')

        to_plot = np.copy(all_stacks_per_sample[sample]).max(axis=0).astype(float)
#         to_plot = np.copy(objects_in_max_projection[sample]).astype(float)
        to_plot[to_plot == 0] = np.nan
        ax.imshow(to_plot, cmap='jet')

        ### Plot different channels ###
        for icol, channel in enumerate(CHANNEL_NAMES):

            ax = axes[irow+1,icol+1]
            ax.set_title(channel)

            D = image_dict[sample][best_iz,:,:,icol]
            vmx = np.percentile(D[nucleus_loc], 99.5)
            ax.imshow(D, vmin=0, vmax=vmx)

            # plot maximum projection
            ax = axes[0,icol+1]
            ax.set_title('%s - Maximum Projection' % channel)
            D = image_dict[sample][:,:,:,icol].max(axis=0)
            vmx = np.percentile(D, 99.9)
            ax.imshow(D, vmin=0, vmax=vmx)

    outfn = INDIR + '/' + sample + '.segmentation_results.edge_%.2fum.pdf' % BORDER_WIDTH_MICRON
    plt.savefig(outfn, bbox_inches='tight')
    plt.close()
