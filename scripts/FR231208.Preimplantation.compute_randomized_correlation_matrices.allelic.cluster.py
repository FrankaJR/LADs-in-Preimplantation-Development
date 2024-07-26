import numpy as np
import pandas as pd
import h5py
import sys
from glob import glob
from scipy import stats

rs = np.random.RandomState()
rs.seed(42)

STAGE = sys.argv[1]
GENOTYPE = sys.argv[2]
ALLELE = sys.argv[3]
TREATMENT = sys.argv[4]
CHROM = sys.argv[5]
BINSIZE = int(sys.argv[6])
N_ITER = int(sys.argv[7])
RESULTOUTFN=sys.argv[8]

ANNOFN = './metadata/FR230814.samplesheet.tsv'
BINARYFN = './data/damid_binary/FR230814.DamID.sample_smooth_binary.{construct}.all_stages.{genotype}.{allele}.damid_pass.binsize_{binsize:d}.hdf5'
CONSTRUCT = 'Dam-Lmnb1'
#RESULTOUTFN = './data/bin_coordination_matrices/FR220901.normalized_bin_coordination_results.smooth_binary_input.PearsonR.Dam-LMNB1.{stage}.{genotype}.{treatment}.damid_pass.binsize_{binsize:d}.hdf5'

#################
### Load anno ###
#################

anno = pd.read_csv(ANNOFN, sep='\t', dtype={'number_embryos': str})

## select samples passing quality filters
anno = anno[anno.DamID_PASS_allelic]

## select conditions of interest
ind = (anno.stage == STAGE) & (anno.genotype == GENOTYPE) & (anno.fusion_construct == CONSTRUCT) & (anno.treatment == TREATMENT) & (anno.cellcount == 1)
anno = anno[ind].reset_index(drop=True)
anno = anno.set_index(anno['damid_name'])

if ('male' in CHROM) or ('female' in CHROM):
    chromname = CHROM
    sex = CHROM.split('_')[1]
    CHROM = CHROM.split('_')[0]
    anno = anno[anno['sex'] == sex]
else:
    chromname = CHROM

###########################
### Loading binary data ###
###########################

damid_binary = dict()

fn = BINARYFN.format(construct=CONSTRUCT, genotype=GENOTYPE.replace('/',''), allele=ALLELE, binsize=BINSIZE)
assert len(glob(fn)) == 1, fn

with h5py.File(fn, 'r') as f:
    samples = [s.decode() for s in f['sample_names'][:]]
    #samples = f['sample_names'][:]
    all_data = f[CHROM][:]

    for i, sample in enumerate(samples):
        if sample in anno.index.values:
            damid_binary[sample] = all_data[:,i].astype(int)

##########################
### Required functions ###
##########################

# Function to randomize binary matrices
from numba import jit
@jit
def randomize_binary_map_numba(m, random_state=42):
    np.random.seed(random_state)
    n_iter = int(np.ceil(len(m) ** 2 / 2))
    mr = m.copy()
    for i in range(n_iter):
        cell_i = np.random.randint(mr.shape[0])
        cell_j = np.random.randint(mr.shape[0])
        if cell_i == cell_j:
            continue
        excl_i = np.where(mr[cell_i] & ~(mr[cell_j]))[0]
        excl_j = np.where(mr[cell_j] & ~(mr[cell_i]))[0]
        nmax_excl = min(excl_i.size, excl_j.size)
        if nmax_excl == 0:
            continue
        nswap = np.random.randint(nmax_excl)
        swap_i = np.random.permutation(excl_i.size)[:nswap]
        swap_j = np.random.permutation(excl_j.size)[:nswap]
        mr[cell_i][excl_i[swap_i]] = False
        mr[cell_j][excl_i[swap_i]] = True
        mr[cell_j][excl_j[swap_j]] = False
        mr[cell_i][excl_j[swap_j]] = True
    return mr

# Function to write arrays to HDF5 file
CHUNKSIZE_MAX = 256 * 1024
def write_counts(outfn, chrom, counts, mode='w'):
    itemsperrow = np.multiply.reduce(counts.shape[1:])
    itemsize = counts.itemsize
    chunkrows = max(1, np.floor(CHUNKSIZE_MAX / itemsperrow / itemsize))
    chunks = (min(counts.shape[0], chunkrows), ) + tuple(counts.shape[1:])

    with h5py.File(outfn, mode) as f:
        f.create_dataset(chrom, data=counts, chunks=chunks, compression=9)
        f.flush()
    return

#####################################
### Computing randomized matrices ###
#####################################

binary_result_matrices = dict()
samples = anno.index.values.astype(str)

# get original data
mat = np.array([damid_binary[s].astype(int) for s in samples])
corrmat = np.corrcoef(mat, rowvar=False)

# perform N_ITER iterations and save results
rand_ls = list()
for it in np.arange(N_ITER):
    rmat = randomize_binary_map_numba(mat, random_state=it)
    rcorr = np.corrcoef(rmat, rowvar=False)
    rand_ls.append(rcorr)
randcorr = np.array(rand_ls)
m_randcorrmat = np.mean(randcorr, axis=0)
s_randcorrmat = np.std(randcorr, axis=0)
randcorr = None

####################
### Save results ###
####################

# standardize original results based on random distribution
norm_corrmat = (corrmat.copy() - m_randcorrmat) / s_randcorrmat

corrmat[np.isnan(corrmat)] = 0
m_randcorrmat[np.isnan(m_randcorrmat)] = 0
s_randcorrmat[np.isnan(s_randcorrmat)] = 0
norm_corrmat[np.isnan(norm_corrmat)] = 0

# save results
#resultoutfn = RESULTOUTFN.format(stage=STAGE, genotype=GENOTYPE.replace('/', ''), treatment=TREATMENT, binsize=BINSIZE)
write_counts(RESULTOUTFN, '%s_orig' % chromname, corrmat.astype(np.float32), mode='a')
write_counts(RESULTOUTFN, '%s_norm' % chromname, norm_corrmat.astype(np.float32), mode='a')
write_counts(RESULTOUTFN, '%s_rand_mean' % chromname, m_randcorrmat.astype(np.float32), mode='a')
write_counts(RESULTOUTFN, '%s_rand_std' % chromname, s_randcorrmat.astype(np.float32), mode='a')
