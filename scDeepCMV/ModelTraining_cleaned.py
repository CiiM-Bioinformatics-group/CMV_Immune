# CMV analysis pipeline for data processing and downstream analysis.
# Copyright (C) 2026
# - Xun Jiang, Helmholtz Centre for Infection Research (HZI)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cleaned training script for DrugSti model development.

This version removes notebook artifacts, strips machine-specific absolute paths,
and adds section comments so the workflow is easier to follow and safer to share on GitHub.

Before running:
1. Place the required input files in your working directory or update the relative paths below.
2. Review dataset-specific column names and model checkpoint paths.
3. Install the Python packages imported in this script.
"""

# ==============================
# Imports
# ==============================

import scanpy as sc
import scipy.sparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
import gc
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import lion_pytorch
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch.nn.functional as F
# sc._settings.ScanpyConfig.n_jobs = -1

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import anndata as ad
# import rapids_singlecell as rsc
# import cupy as cp
# import rmm
# from rmm.allocators.cupy import rmm_cupy_allocator
# rmm.reinitialize(
#     managed_memory=True,  # Allows oversubscription
#     pool_allocator=False,  # default is False
#     devices=0,  # GPU device IDs to register. By default registers only GPU 0.
# )
# cp.cuda.set_allocator(rmm_cupy_allocator)
sc.settings.verbosity = 1  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=100, frameon=False, figsize=(4, 4), facecolor="white")

# ==============================
# Load datasets
# ==============================

adata_EUAS = sc.read_h5ad("EUAS.h5ad")

adata_BCG = sc.read_h5ad("BCG.h5ad")

for name, adata in zip(["adata_EUAS", "adata_BCG"], [adata_EUAS, adata_BCG]):
    print(f"{name}: max = {adata.X.max()}, min = {adata.X.min()}")

# print(adata.X.min(), adata.X.max())

# ------------------------------
# Align EUAS cell metadata
# ------------------------------

index_EUAS = pd.read_csv("cell_metadata.tsv", sep="\t")

index_EUAS["CellBarcode"] = index_EUAS.Barcode + "-" + index_EUAS.Library

# Extract the valid barcodes from index_EUAS
valid_barcodes = set(index_EUAS["CellBarcode"])
clean_index = adata_EUAS.obs.index.str.replace(" ", "")
# Create a boolean mask of matching cells
mask = clean_index.isin(valid_barcodes)

# Step 4: Subset the AnnData object
adata_EUAS = adata_EUAS[mask].copy()

# Ensure "CellBarcode" is the index for matching
index_EUAS_indexed = index_EUAS.set_index("CellBarcode")

# Reindex to match the order of adata_EUAS.obs
aligned_metadata = index_EUAS_indexed.loc[adata_EUAS.obs.index]

# Optional: check that alignment is correct
assert all(aligned_metadata.index == adata_EUAS.obs.index)

# Join using index alignment — allows for missing barcodes
adata_EUAS.obs = adata_EUAS.obs.join(index_EUAS_indexed)

# For adata_EUAS
sc.pp.normalize_total(adata_EUAS, target_sum=1e4)
sc.pp.log1p(adata_EUAS)

for name, adata in zip(["adata_EUAS", "adata_BCG"], [adata_EUAS, adata_BCG]):
    print(f"{name}: max = {adata.X.max()}, min = {adata.X.min()}")

# ------------------------------
# Harmonize gene identifiers
# ------------------------------

import mygene
mg = mygene.MyGeneInfo()

# Make sure you're using the AnnData object, not a set
genes_to_query = adata_EUAS.var_names.tolist()

# Query Ensembl IDs → gene symbols
query_results = mg.querymany(genes_to_query, scopes='ensembl.gene', fields='symbol', species='human')

# Create mapping from Ensembl ID to gene symbol
id_to_symbol = {item['query']: item.get('symbol', item['query']) for item in query_results}

# Rename the genes in adata_EUAS
adata_EUAS.var_names = adata_EUAS.var_names.map(id_to_symbol)

adata = sc.read_h5ad('CXCL9_TI_processed.h5ad')
# Make sure you're using the AnnData object, not a set
genes_to_query = adata.var_names.tolist()

# Query Ensembl IDs → gene symbols
query_results = mg.querymany(genes_to_query, scopes='ensembl.gene', fields='symbol', species='human')

# Create mapping from Ensembl ID to gene symbol
id_to_symbol = {item['query']: item.get('symbol', item['query']) for item in query_results}

# Rename the genes in adata_EUAS
adata.var_names = adata.var_names.map(id_to_symbol)

# ------------------------------
# Intersect shared genes across datasets
# ------------------------------

# Extract gene names from each dataset
genes = set(adata.var_names)
genes_EUAS = set(adata_EUAS.var_names)
genes_BCG = set(adata_BCG.var_names)

# Find shared genes
shared_genes = genes & genes_EUAS & genes_BCG

# Convert to sorted list if needed
shared_genes = sorted(shared_genes)
with open("shared_genes11038.txt", "w") as handle:
    for i in shared_genes:
        handle.write(f'{i}\n')
# del adata_aging, adata_EUAS, adata_BCG

# --- 0) Put your adatas in a tuple for easy looping ---
adatas = (adata, adata_EUAS, adata_BCG)

# --- 1) Standardize var_names (string, trimmed) and make unique ---
for i, ad in enumerate(adatas, start=1):
    ad.var_names = pd.Index(ad.var_names.astype(str).str.strip())
    if not ad.var_names.is_unique:
        # report duplicates before fixing
        dups = ad.var_names[ad.var_names.duplicated()].unique()
        print(f"[adata {i}] duplicates in var_names (n={len(dups)}). Examples:", dups[:10].tolist())
        ad.var_names_make_unique()  # appends .1, .2, ...

# --- 2) Clean your shared_genes and de-duplicate it ---
shared_genes = pd.Index(shared_genes).astype(str).str.strip()
shared_genes = shared_genes[~shared_genes.duplicated(keep="first")]  # ensure unique

# Optional: enforce that shared_genes actually exist in all adatas
common = adatas[0].var_names
for ad in adatas[1:]:
    common = common.intersection(ad.var_names)

# Tell you if your provided list had genes that don't exist everywhere
missing_anywhere = shared_genes.difference(common)
if len(missing_anywhere):
    print(f"{len(missing_anywhere)} genes in shared_genes are not present in ALL datasets; dropping them. "
          f"Examples: {missing_anywhere[:10].tolist()}")
# Use the intersection of your list and the true common set, preserving the order of the first AnnData
shared_clean = pd.Index([g for g in adata.var_names if (g in common) and (g in set(shared_genes))])

if len(shared_clean) == 0:
    raise ValueError("No overlapping genes after cleaning. Check naming/case and species.")

print(f"Subsetting to {len(shared_clean)} shared genes.")

# --- 3) Subset safely (order preserved by the first AnnData) ---
adata       = adata[:, shared_clean].copy()
adata_EUAS  = adata_EUAS[:, shared_clean].copy()
adata_BCG   = adata_BCG[:, shared_clean].copy()

# Pack your AnnData objects
adatas = (adata, adata_EUAS, adata_BCG)

# 1) Clean names (string/trim). Do NOT make unique; we want to DROP duplicates.
for i, ad in enumerate(adatas, start=1):
    ad.var_names = pd.Index(ad.var_names.astype(str).str.strip())
    # report duplicates
    dups = ad.var_names[ad.var_names.duplicated(keep=False)]
    if len(dups):
        print(f"[adata {i}] dropping duplicated genes (n={dups.nunique()}). Examples: {dups.unique()[:10].tolist()}")

# 2) Keep only genes that are unique within EACH dataset
uniq_sets = []
for ad in adatas:
    uniq = set(ad.var_names[~ad.var_names.duplicated(keep=False)])
    uniq_sets.append(uniq)

# 3) Intersect unique-only sets across all datasets
common_uniq = set.intersection(*uniq_sets)
print(f"Genes unique in every dataset: {len(common_uniq)}")

# 4) If you also have a user-specified shared_genes list, intersect with it (after de-duping/cleaning)
try:
    shared_genes  # if provided earlier
    shared_genes = pd.Index(shared_genes).astype(str).str.strip().drop_duplicates()
    common_uniq = common_uniq.intersection(set(shared_genes))
    print(f"After intersecting with provided shared_genes: {len(common_uniq)}")
except NameError:
    pass  # no user-provided list; skip

if not common_uniq:
    raise ValueError("No genes remain after removing duplicates and intersecting. Check gene naming/case.")

# 5) Preserve the order of the first AnnData and subset all
ordered = [g for g in adata.var_names if (g in common_uniq)]
print(f"Final gene count used for subsetting: {len(ordered)}")

adata       = adata[:, ordered].copy()
adata_EUAS  = adata_EUAS[:, ordered].copy()
adata_BCG   = adata_BCG[:, ordered].copy()

adata.obs["CMV"] = None

# ------------------------------
# Add donor-level metadata for EUAS
# ------------------------------

df_meta = pd.read_csv("sample_metadata.tsv", sep="\t")

# Step 1: Rename and deduplicate df_meta
df_meta = df_meta.rename(columns={"Age": "age"}).drop_duplicates(subset="Donor ID")

# Step 2: Set 'Donor ID' as index
df_meta_indexed = df_meta.set_index("Donor ID")

# Step 3: Map metadata into adata_EUAS.obs
adata_EUAS.obs["age"] = adata_EUAS.obs["Donor ID"].map(df_meta_indexed["age"])
adata_EUAS.obs["GenderF"] = adata_EUAS.obs["Donor ID"].map(df_meta_indexed["GenderF"])
adata_EUAS.obs["CMV"] = adata_EUAS.obs["Donor ID"].map(df_meta_indexed["CMV"])
# Keep only cells where 'CMV' is not NA
adata_EUAS = adata_EUAS[~adata_EUAS.obs["CMV"].isna()].copy()
# Convert to integer (removes .0) and then to categorical
adata_EUAS.obs["CMV"] = adata_EUAS.obs["CMV"].astype(int).astype("category")
adata_EUAS.obs["SampleID"] = (
    adata_EUAS.obs["Donor ID"].astype(str) + "_" + adata_EUAS.obs["Condition"].astype(str)
)

# Count how many SampleIDs fall into each CMV group
adata_EUAS.obs.groupby("CMV")["Donor ID"].nunique()

# ------------------------------
# Add donor-level metadata for BCG
# ------------------------------

df = pd.read_csv("donor_CMV_status.csv")

# Step 3: Set df index to donor ID (assumed to match adata_BCG.obs["Donor ID"])
df_indexed = df.set_index("ids")

# Step 4: Map CMV_PN and CMV_IgG_Index into adata_BCG.obs
adata_BCG.obs["CMV_PN"] = adata_BCG.obs["ids"].map(df_indexed["CMV_PN"])
adata_BCG.obs["CMV_IgG_Index"] = adata_BCG.obs["ids"].map(df_indexed["CMV_IgG_Index"])

# Map CMV_PN to binary CMV
adata_BCG.obs["CMV"] = adata_BCG.obs["CMV_PN"].map({"Negative": 0, "Positive": 1})
adata_BCG = adata_BCG[~adata_BCG.obs["CMV"].isna()].copy()
adata_BCG.obs["CMV"] = adata_BCG.obs["CMV"].astype(int).astype("category")

adata_BCG.obs["GenderF"] = adata_BCG.obs["gender"].map({"f": 1, "m": 0}).astype(int)
adata_BCG.obs["SampleID"] = adata_BCG.obs["ids"].astype(str) + "_" + adata_BCG.obs["stim"].astype(str)

adata.obs["SampleID"] = adata.obs["donor_id"].astype(str) + "_" + adata.obs["treatment"].astype(str)

adata_BCG.obs["SampleID"] 

# Concatenate datasets
# ------------------------------
# Merge datasets and compute PCA
# ------------------------------

adata_combined = ad.concat([adata, adata_EUAS, adata_BCG], join="inner", label="batch", keys=["CXCL", "EUAS", "BCG"])

sc.pp.pca(adata_combined, n_comps=100)

# 1. 统计每个 SampleID 的细胞数量
sample_counts = adata_combined.obs["SampleID"].value_counts()

# 2. 选出细胞数 >= 100 的样本
valid_sample_ids = sample_counts[sample_counts >= 200].index

# 3. 过滤数据，仅保留有效 SampleID 的细胞
adata_combined = adata_combined[adata_combined.obs["SampleID"].isin(valid_sample_ids)].copy()

def _coerce_obs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_object_dtype(s):
            # If entries are list/tuple/set/dict, serialize to JSON strings
            if s.map(lambda x: isinstance(x, (list, tuple, set, dict))).any():
                df[c] = s.map(lambda x: json.dumps(list(x) if isinstance(x, set) else x) if pd.notna(x) else pd.NA).astype("string")
            # If entries are pure strings/None, make categorical (compact & safe)
            elif s.map(lambda x: (x is None) or isinstance(x, (str, np.str_))).all():
                df[c] = s.astype("category")
            else:
                # Mixed scalars (ints/floats/bools/None): choose a stable dtype
                if s.map(lambda x: isinstance(x, (int, np.integer))).all():
                    df[c] = s.astype("Int64")  # nullable int
                elif s.map(lambda x: isinstance(x, (float, np.floating))).all():
                    df[c] = s.astype("float64")
                elif s.map(lambda x: isinstance(x, (bool, np.bool_))).all():
                    df[c] = s.astype("boolean")
                else:
                    df[c] = s.astype("string")
    return df

adata_combined.obs = _coerce_obs(adata_combined.obs)
ad.settings.allow_write_nullable_strings = True
adata_combined.write_h5ad("combinedDataCMV_Age.h5ad", compression="gzip")

# adata_combined.write_h5ad("combinedDataCMV_Age.h5ad", compression="gzip")
del adata, adata_BCG

# adata_combined = sc.read_h5ad("combinedDataCMV_Age.h5ad")

import pickle
# with open("adata_combined.pkl", "wb") as f:
#     pickle.dump(adata_combined, f)

# with open("adata_combined.pkl", "rb") as f:
#     adata_combined = pickle.load(f)

X_pca = adata_combined.obsm['X_pca_harmony'] if 'X_pca_harmony' in adata_combined.obsm else adata_combined.obsm['X_pca']

k = 32

import faiss
from collections import defaultdict
import os

X_pca = adata_combined.obsm['X_pca_harmony'] if 'X_pca_harmony' in adata_combined.obsm else adata_combined.obsm['X_pca']
donor_ids = adata_combined.obs["SampleID"].values
unique_donors = np.unique(donor_ids)
indices_all = np.zeros((adata_combined.n_obs, k), dtype=np.int32)

# 设置 FAISS 使用所有 CPU 线程
faiss.omp_set_num_threads(os.cpu_count())
print(f"FAISS using {faiss.omp_get_max_threads()} threads")

for donor in unique_donors:
    mask = donor_ids == donor
    X_donor = X_pca[mask].astype(np.float32)  # FAISS 要求 float32
    indices_in_adata = np.where(mask)[0]

    if len(X_donor) < k:
        print(f"Skipping donor {donor} (only {len(X_donor)} cells)")

    # 构建 CPU Index
    index = faiss.IndexFlatL2(X_donor.shape[1])
    index.add(X_donor)
    _, donor_indices = index.search(X_donor, k)

    # Map local indices back to global
    for i, row in enumerate(donor_indices):
        global_i = indices_in_adata[i]
        global_indices = indices_in_adata[row]
        indices_all[global_i] = global_indices

adata_combined.obsm["faiss_neighbors"] = indices_all

# ==============================
# Model definition
# ==============================

class MultiTaskContextModel(nn.Module):
    def __init__(self, num_genes, latent_dim=128, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()

        self.dropout = dropout

        # Gene encoder
        self.gene_encoder = nn.Sequential(
            nn.Linear(num_genes, 256),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, latent_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout)
        )

        # Positional + center cell embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, latent_dim))  # [1, cellNumbers, L]
        self.center_marker = nn.Parameter(torch.randn(latent_dim))  # [latent_dim]

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            activation='gelu',
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads
        self.cmv_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1)  # Binary classification
        )

        self.gender_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1)  # Binary classification (or regression logit)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, num_genes)
        )

    def forward(self, x, mask=None, mask_center_genes=False, mask_ratio=0.3):
        B, N, G = x.shape

        # Optionally mask center cell genes
        if mask_center_genes and self.training:
            gene_mask = (torch.rand(B, G, device=x.device) > mask_ratio).float()
            x[:, 0, :] *= gene_mask

        # Encode each cell independently
        x = x.view(B * N, G)
        x = self.gene_encoder(x)
        x = x.view(B, N, -1)

        # Add center marker and positional embedding
        x[:, 0, :] += self.center_marker
        x = x + self.pos_embedding[:, :N, :]

        # Transformer
        x_trans = self.transformer(x, src_key_padding_mask=mask)

        # Center latent
        center_latent = x_trans[:, 0, :]
        cmv = self.cmv_head(center_latent).squeeze(-1)
        gender = self.gender_head(center_latent).squeeze(-1)
        recon_center = self.decoder(center_latent)

        return recon_center, cmv, gender

from torch.utils.data import Dataset

# ==============================
# Dataset wrappers
# ==============================

class FaissKNNDataset(Dataset):
    def __init__(self, X_expr, indices_knn, cmv_status, sexes, donor_ids, donor_filter):
        """
        X_expr: ndarray (cells x genes)
        indices_knn: ndarray (cells x k), each row contains neighbor indices
        cmv_status: ndarray (cells,), binary label (0 or 1)
        sexes: ndarray (cells,), binary label (0 or 1)
        donor_ids: ndarray (cells,), donor identifier (string or int)
        donor_filter: list of donor ids to include
        """
        self.X_expr = X_expr
        self.indices_knn = indices_knn
        self.cmv_status = cmv_status
        self.sexes = sexes
        self.donor_ids = donor_ids

        # 筛选 donor 在 donor_filter 中的细胞索引
        self.valid_indices = np.where(np.isin(donor_ids, donor_filter))[0]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        idx = self.valid_indices[i]
        neighbor_ids = self.indices_knn[idx]  # shape: [k]

        return {
            'context': torch.tensor(self.X_expr[neighbor_ids], dtype=torch.float32),  # [k, genes]
            'center': torch.tensor(self.X_expr[idx], dtype=torch.float32),            # [genes]
            'cmv': torch.tensor(self.cmv_status[idx], dtype=torch.float32),           # scalar
            'sex': torch.tensor(self.sexes[idx], dtype=torch.float32),                # scalar
            'donor_id': self.donor_ids[idx]                                           # raw (str or int)
        }

class FaissKNNDataset_BalancedSampling(Dataset):
    def __init__(self, X_expr, indices_knn, cmv_status, sexes, donor_ids, donor_filter, max_cells_per_donor=None, seed=42):
        """
        Balanced dataset across donors. Only `max_cells_per_donor` cells per donor are included.
        If `max_cells_per_donor` is None, it defaults to the min cell count among filtered donors.

        Parameters:
            - X_expr: np.ndarray, shape (cells, genes)
            - indices_knn: np.ndarray, shape (cells, k)
            - cmv_status: np.ndarray, shape (cells,)
            - sexes: np.ndarray, shape (cells,)
            - donor_ids: np.ndarray, shape (cells,)
            - donor_filter: list or set of donor IDs to include
            - max_cells_per_donor: int or None
        """
        self.X_expr = X_expr
        self.indices_knn = indices_knn
        self.cmv_status = cmv_status
        self.sexes = sexes
        self.donor_ids = donor_ids

        # Find cells from donors in the filter
        np.random.seed(seed)
        filtered_indices = np.where(np.isin(donor_ids, donor_filter))[0]

        # Group indices by donor
        donor_to_indices = defaultdict(list)
        for idx in filtered_indices:
            donor_to_indices[donor_ids[idx]].append(idx)

        # Determine max cells per donor
        if max_cells_per_donor is None:
            max_cells_per_donor = min(len(indices) for indices in donor_to_indices.values())

        # Sample balanced indices
        balanced_indices = []
        for donor, indices in donor_to_indices.items():
            if len(indices) >= max_cells_per_donor:
                sampled = np.random.choice(indices, size=max_cells_per_donor, replace=False)
            else:
                sampled = np.random.choice(indices, size=max_cells_per_donor, replace=True)  # or skip if undesired
            balanced_indices.extend(sampled)

        self.valid_indices = np.array(balanced_indices)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        idx = self.valid_indices[i]
        neighbor_ids = self.indices_knn[idx]  # [k]

        return {
            'context': torch.tensor(self.X_expr[neighbor_ids], dtype=torch.float32),  # [k, genes]
            'center': torch.tensor(self.X_expr[idx], dtype=torch.float32),            # [genes]
            'cmv': torch.tensor(self.cmv_status[idx], dtype=torch.float32),           # scalar
            'sex': torch.tensor(self.sexes[idx], dtype=torch.float32),                # scalar
            'donor_id': self.donor_ids[idx]                                           # str or int
        }

adata_combined.obsm["faiss_neighbors"].shape

############ training with balanced sampling. 
adata_Train = adata_combined[adata_combined.obs["batch"] == "EUAS"].copy()
cmv = adata_Train.obs["CMV"].astype(np.float32).values

sc.pp.neighbors(adata_Train, n_neighbors=12, n_pcs=30)
sc.tl.umap(adata_Train)
# clustering
sc.tl.leiden(adata_Train, resolution=0.6)

adata_Train.obs["GenderF"] = adata_Train.obs_names.map(adata_EUAS.obs["GenderF"])
# adata_Train.write_h5ad("adata_Train.h5ad", compression="gzip")

sc.settings.set_figure_params(dpi=100, frameon=False, figsize=(4, 3), facecolor="white")
# adata_Train = sc.read_h5ad("adata_Train.h5ad")
sc.pl.umap(adata_Train, use_raw=False, color=["leiden", "CD8A", "CD8B", "GZMA", "GZMB", "NKG7", "PRF1", "CMV"], 
           cmap="turbo", legend_loc="on data")

def cd8_from_scores(adata, *, layer=None, organism="human",
                    include_cytotoxic=True, ctrl_size=56,
                    percentile=70, out_col="is_CD8T"):
    """
    Identify CD8 T cells using sc.tl.score_genes WITHOUT .raw.
    Assumes adata.X (or `layer`) is already normalized/log-transformed.
    """
    # markers
    if organism.lower().startswith("mouse"):
        CD8 = ["Cd8a","Cd8b1","Cd3d","Cd3e","Cd3g","Trac","Lck","Cd247"]
        CYT = ["Gzmb","Gzmk","Prf1","Nkg7","Gnly"]
        NEG = {
            "NK":   ["Ncr1","Klrd1","Klrk1","Klrc1","Fcgr3a","Nkg7"],
            "CD4":  ["Il7r","Ccr7"],
            "B":    ["Ms4a1","Cd79a","Cd79b"],
            "MONO": ["Lyz2","S100a8","S100a9","Lst1"],
        }
        gA,gB,gCD3 = "Cd8a","Cd8b1","Cd3d"
    else:
        CD8 = ["CD8A","CD8B","CD3D","CD3E","CD3G","CD247"]
        CYT = ["GZMB","GZMA","GZMK","PRF1","NKG7","GNLY"]
        NEG = {
            "NK":   ["NCR1","KLRD1","KLRK1","KLRC1","FCGR3A","NKG7"],
            "CD4":  ["IL7R","CCR7"],
            "B":    ["MS4A1","CD79A","CD79B"],
            "MONO": ["LYZ","S100A8","S100A9","LST1"],
        }
        gA,gB,gCD3,gGZMB = "CD8A","CD8B","CD3D","GZMB"

    if include_cytotoxic: CD8 = list(dict.fromkeys(CD8 + CYT))
    present = lambda gl: [g for g in gl if g in adata.var_names]

    pos = present(CD8)
    if len(pos) < 3:
        raise ValueError(f"Too few CD8 markers present: {pos}")

    # score on X/layer only (no raw)
    sc.tl.score_genes(adata, pos, score_name="score_CD8", use_raw=False, layer=layer, ctrl_size=ctrl_size)
    other_cols = []
    for nm, gl in NEG.items():
        gl = present(gl)
        if gl:
            sc.tl.score_genes(adata, gl, score_name=f"score_{nm}", use_raw=False, layer=layer, ctrl_size=ctrl_size)
            other_cols.append(f"score_{nm}")

    adata.obs["score_CD8_specific"] = adata.obs["score_CD8"] - (adata.obs[other_cols].max(axis=1) if other_cols else 0.0)

    # simple expression guard from same matrix/layer (still no raw)
    def expr(g):
        if g not in adata.var_names: return pd.Series(0, index=adata.obs_names, dtype=float)
        X = adata.layers[layer] if layer is not None else adata.X
        col = adata.var_names.get_loc(g)
        v = X[:, col]
        if hasattr(v, "toarray"): v = v.toarray()
        return pd.Series(np.asarray(v).ravel(), index=adata.obs_names)

    guard = (expr(gA) >= 0) & (expr(gB) < 1) & (expr(gGZMB) > 0)

    thr = np.percentile(adata.obs["score_CD8_specific"], percentile)
    adata.obs[out_col] = ((adata.obs["score_CD8_specific"] > thr) & guard).astype(bool)
    return adata.obs[out_col]

def cmono_from_scores(adata, *, layer=None, organism="human",
                      ctrl_size=56, percentile=70, out_col="is_cMono"):
    """
    Identify classical monocytes (CD14+ cMono) using sc.tl.score_genes WITHOUT .raw.
    Assumes adata.X (or `layer`) is already normalized/log-transformed.
    """

    if organism.lower().startswith("mouse"):
        # cMono (CD14hi-like) markers
        MONO_POS = ["Lyz2","S100a8","S100a9","S100a12","Ms4a7","Tyrobp",
                    "Lgals3","Vcan","Lst1","Clec12a","Fcn1","Cd14","Itgam","Ctss"]
        # Competing/non-target signatures to subtract
        COMP = {
            "nonclassical": ["Fcgr3","Itgal","Itgax","Lrrc25","Msr1"],   # CD16+ (approx.; Fcgr3a in human)
            "T_NK": ["Cd3d","Trac","Cd8a","Cd8b1","Nkg7","Gzmb","Prf1","Gnly"],
            "B": ["Ms4a1","Cd79a","Cd79b"],
            "DC": ["Fcer1a","Clec10a"],
            "PLT": ["Ppbp","Pf4"],
            "ERY": ["Hbb-bs","Hba-a1","Hba-a2"],
            "NEU": ["Mpo","Csf3r","Fcgr3","Elane"],
        }
        gCD14, gLYZ, gMS4A7 = "Cd14","Lyz2","Ms4a7"
    else:
        # Human cMono (CD14hi)
        MONO_POS = ["LYZ","S100A8","S100A9","S100A12","MS4A7","TYROBP","MACRO","CCR2",
                    "LGALS3","VCAN","LST1","CLEC12A","FCN1","CD14","ITGAM","CTSS"]
        COMP = {
            "nonclassical": ["FCGR3A","LILRB1","CX3CR1"],  # CD16+ signature
            "T_NK": ["CD3D","TRAC","CD8A","CD8B","NKG7","GZMB","PRF1","GNLY"],
            "B": ["MS4A1","CD79A","CD79B"],
            "DC": ["FCER1A","CLEC10A"],
            "PLT": ["PPBP","PF4"],
            "ERY": ["HBB","HBA1","HBA2"],
            "NEU": ["MPO","CSF3R","FCGR3B","ELANE"],
        }
        gCD14, gLYZ, gMS4A7 = "CD14","LYZ","MS4A7"

    present = lambda gl: [g for g in gl if g in adata.var_names]

    pos = present(MONO_POS)
    if len(pos) < 3:
        raise ValueError(f"Too few cMono markers present: {pos}")

    # --- scores on X / layer (no raw) ---
    sc.tl.score_genes(adata, pos, score_name="score_cMono", use_raw=False, layer=layer, ctrl_size=ctrl_size)

    other_cols = []
    for nm, gl in COMP.items():
        genes = present(gl)
        if genes:
            sc.tl.score_genes(adata, genes, score_name=f"score_{nm}", use_raw=False, layer=layer, ctrl_size=ctrl_size)
            other_cols.append(f"score_{nm}")

    # Specificity: cMono score minus the strongest competing lineage (esp. CD16+)
    adata.obs["score_cMono_specific"] = adata.obs["score_cMono"] - (adata.obs[other_cols].max(axis=1) if other_cols else 0.0)

    # Simple guard to avoid calling non-expressers as cMono (uses same matrix/layer)
    def expr(g):
        if g not in adata.var_names:
            return pd.Series(0, index=adata.obs_names, dtype=float)
        X = adata.layers[layer] if layer is not None else adata.X
        j = adata.var_names.get_loc(g)
        v = X[:, j]
        if hasattr(v, "toarray"): v = v.toarray()
        return pd.Series(np.asarray(v).ravel(), index=adata.obs_names)

    guard = (expr(gCD14) > 0) | (expr(gLYZ) > 0) | (expr(gMS4A7) > 0)

    # Threshold by percentile
    thr = np.percentile(adata.obs["score_cMono_specific"], percentile)
    adata.obs[out_col] = ((adata.obs["score_cMono_specific"] > thr) & guard).astype(bool)
    return adata.obs[out_col]

print(adata_Train.X.min(), adata_Train.X.max())

cd8_from_scores(adata_Train, layer=None, organism="human", percentile=75)

# 1) Derive donor_id from SampleID (e.g., "300EUAS012_RPMI" → "300EUAS012")
adata_Train.obs["donor_id"] = adata_Train.obs["SampleID"].astype(str).str.split("_").str[0]

# 2) Map CMV status to binary
pos = {"IgGpositive","PCRpositive","IgMpositive","Positive","Pos","Yes","1",1,True}
neg = {"Negative","Neg","No","0",0,False}

def to_bin(x):
    s = str(x).strip()
    if s in pos: return 1
    if s in neg: return 0
    return np.nan

adata_Train.obs["CMV_bin"] = adata_Train.obs["CMV"].map(to_bin)

# 3) Cell-level counts
total_cells   = adata_Train.n_obs
cells_pos     = int((adata_Train.obs["CMV_bin"] == 1).sum())
cells_neg     = int((adata_Train.obs["CMV_bin"] == 0).sum())
cells_unknown = int(adata_Train.obs["CMV_bin"].isna().sum())

# 4) Donor-level CMV status
g = adata_Train.obs.groupby("donor_id")["CMV_bin"]
donor_status = g.apply(lambda s: 1 if (s == 1).any()
                       else (0 if (s == 0).any() else np.nan))

total_donors   = donor_status.index.nunique()
donors_pos     = int((donor_status == 1).sum())
donors_neg     = int((donor_status == 0).sum())
donors_unknown = int(donor_status.isna().sum())

# 5) Summary table
summary = pd.DataFrame({
    "metric": [
        "Unique donors (total)",
        "CMV+ donors",
        "CMV- donors",
        "Unknown donors",
        "Cells (total)",
        "CMV+ cells",
        "CMV- cells",
        "Unknown cells",
    ],
    "count": [
        total_donors,
        donors_pos,
        donors_neg,
        donors_unknown,
        total_cells,
        cells_pos,
        cells_neg,
        cells_unknown,
    ],
})

print(summary.to_string(index=False))

sc.pl.umap(adata_Train, use_raw=False, color=["leiden","CD8A","CD8B","CD3D","CD3E","CD3G","TRAC","LCK","CD247", "GZMA", "GZMB", "NKG7", "PRF1", "CMV", "is_CD8T"], 
           cmap="turbo")#, legend_loc="on data")

# sc.tl.rank_genes_groups(
#     adata_Train,
#     groupby="leiden",
#     groups=["1","3"],       # only compute for 1 and 3
#     reference="rest",
#     method="wilcoxon",
#     use_raw=False,           # set False if your counts are in .X/layer and not .raw
#     tie_correct=True,
#     pts=True,               # proportion expressing in group/rest
# )

# adata_euas = sc.read_h5ad("Train_CD8T.h5ad") 
# cd8_from_scores(adata_euas, layer=None, organism="human", percentile=60)
adata_euas = adata_Train[adata_Train.obs["is_CD8T"] == True].copy()
# adata_euas = adata_Train[adata_Train.obs["leiden"].isin(["1", "3", "0"])].copy()

sc.pl.umap(adata_euas, use_raw=False, color=["leiden","CD8A","CD8B","CD3D","CD3E","CD3G","TRAC","LCK","CD247", "GZMA", "GZMB", "NKG7", "PRF1", "CMV", "is_CD8T"], 
           cmap="turbo")#, legend_loc="on data")

# Map GenderF values from adata_EUAS.obs to adata_euas.obs based on matching obs_names

adata_euas.obs#.batch.value_counts()

# 1) Derive donor_id from SampleID (e.g., "300EUAS012_RPMI" → "300EUAS012")
adata_euas.obs["donor_id"] = adata_euas.obs["SampleID"].astype(str).str.split("_").str[0]

# 2) Map CMV status to binary
pos = {"IgGpositive","PCRpositive","IgMpositive","Positive","Pos","Yes","1",1,True}
neg = {"Negative","Neg","No","0",0,False}

def to_bin(x):
    s = str(x).strip()
    if s in pos: return 1
    if s in neg: return 0
    return np.nan

adata_euas.obs["CMV_bin"] = adata_euas.obs["CMV"].map(to_bin)

# 3) Cell-level counts
total_cells   = adata_euas.n_obs
cells_pos     = int((adata_euas.obs["CMV_bin"] == 1).sum())
cells_neg     = int((adata_euas.obs["CMV_bin"] == 0).sum())
cells_unknown = int(adata_euas.obs["CMV_bin"].isna().sum())

# 4) Donor-level CMV status
g = adata_euas.obs.groupby("donor_id")["CMV_bin"]
donor_status = g.apply(lambda s: 1 if (s == 1).any()
                       else (0 if (s == 0).any() else np.nan))

total_donors   = donor_status.index.nunique()
donors_pos     = int((donor_status == 1).sum())
donors_neg     = int((donor_status == 0).sum())
donors_unknown = int(donor_status.isna().sum())

# 5) Summary table
summary = pd.DataFrame({
    "metric CD8T": [
        "Unique donors (total)",
        "CMV+ donors",
        "CMV- donors",
        "Unknown donors",
        "Cells (total)",
        "CMV+ cells",
        "CMV- cells",
        "Unknown cells",
    ],
    "count": [
        total_donors,
        donors_pos,
        donors_neg,
        donors_unknown,
        total_cells,
        cells_pos,
        cells_neg,
        cells_unknown,
    ],
})

print(summary.to_string(index=False))

adata_euas.obs[adata_euas.obs.SampleID=="206_IAV"].GenderF.value_counts()

# sc.pp.pca(adata_euas, n_comps=100)

k = 20 #32
X_pca = adata_euas.obsm['X_pca_harmony'] if 'X_pca_harmony' in adata_euas.obsm else adata_euas.obsm['X_pca']
donor_ids = adata_euas.obs["SampleID"].values
unique_donors = np.unique(donor_ids)
indices_all = np.zeros((adata_euas.n_obs, k), dtype=np.int32)

# 设置 FAISS 使用所有 CPU 线程
faiss.omp_set_num_threads(os.cpu_count())
print(f"FAISS using {faiss.omp_get_max_threads()} threads")

for donor in unique_donors:
    mask = donor_ids == donor
    X_donor = X_pca[mask].astype(np.float32)  # FAISS 要求 float32
    indices_in_adata = np.where(mask)[0]

    if len(X_donor) < k:
        print(f"Skipping donor {donor} (only {len(X_donor)} cells)")

    # 构建 CPU Index
    index = faiss.IndexFlatL2(X_donor.shape[1])
    index.add(X_donor)
    _, donor_indices = index.search(X_donor, k)

    # Map local indices back to global
    for i, row in enumerate(donor_indices):
        global_i = indices_in_adata[i]
        global_indices = indices_in_adata[row]
        indices_all[global_i] = global_indices

adata_euas.obsm["faiss_neighbors"] = indices_all
# Step 2: 获取 donor 列表，并随机划分 80% 训练、20% 验证
donors_all = adata_euas.obs["SampleID"].unique()
donors_train, donors_val = train_test_split(donors_all, test_size=0.02, random_state=42)

# Step 3: 提取表达矩阵
X_expr = adata_euas.X.toarray() if not isinstance(adata_euas.X, np.ndarray) else adata_euas.X
X_expr = X_expr.astype(np.float32)

# Step 4: CMV 标签
cmv = adata_euas.obs["CMV"].astype(np.float32).values

# Step 5: Sex
sexes = adata_euas.obs["GenderF"].astype(np.float32).values

# Step 6: donor_ids
donor_ids = adata_euas.obs["SampleID"].values

# Step 7: FAISS 邻居索引
indices_knn = adata_euas.obsm["faiss_neighbors"]  # shape: [n_cells, k]

# del adata_euas
# ✅ 输出检查
print("EUAS donors total:", len(donors_all))
print("Train donors:", len(donors_train))
print("Validation donors:", len(donors_val))
print("Train cells (to be filtered later by donor):", X_expr.shape[0])

# adata_euas.write_h5ad("Train_CD8T.h5ad", compression="gzip")

print(X_expr.shape)         # should be [n_cells, n_genes]
print(cmv.shape)            # should be [n_cells]
print(sexes.shape)          # should be [n_cells]
print(donor_ids.shape)      # should be [n_cells]
print(indices_knn.shape)    # should be [n_cells, k]
print(k)

# Get all donor_ids
donor_ids = adata_euas.obs['SampleID'].values

# Optional: check distribution
from collections import Counter
print(Counter(donor_ids))

# Create dataset
train_dataset = FaissKNNDataset_BalancedSampling(
    X_expr=X_expr,
    indices_knn=indices_knn,
    cmv_status=cmv,
    sexes=sexes,
    donor_ids=donor_ids,
    donor_filter=donor_ids.unique(),
    max_cells_per_donor=90  # or None to auto-balance
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
# Losses
loss_recon = nn.MSELoss()
loss_gender = nn.BCEWithLogitsLoss()
loss_cmv = nn.BCEWithLogitsLoss()

# model = MultiTaskContextModel(num_genes=9901, latent_dim=128)
model = MultiTaskContextModel(num_genes=11030, latent_dim=128)
model.load_state_dict(torch.load("TrainedModel/multitask_model_mask_centerMarker_balancedTrained_11030.pt", map_location="cuda", weights_only=False))
model = model.cuda()  # if using GPU

# Optimizer
lr = 4e-6
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

# ==============================
# Model training
# ==============================

# Balanced training
from tqdm.notebook import tqdm

all_batch_losses = []  # store loss per batch for all epochs
num_epochs = 3

for epoch in range(num_epochs):
    # Create dataset
    train_dataset = FaissKNNDataset_BalancedSampling(
        X_expr=X_expr,
        indices_knn=indices_knn,
        cmv_status=cmv,
        sexes=sexes,
        donor_ids=donor_ids,
        donor_filter=donor_ids.unique(),
        max_cells_per_donor=120  # or None to auto-balance
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    model.train()
    total_loss = 0.0
    epoch_losses = []

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, batch in enumerate(loop):
        context = batch['context'].cuda()
        center = batch['center'].cuda()
        cmv_label = batch['cmv'].cuda()
        sex = batch['sex'].cuda()

        # Forward pass
        recon, pred_cmv, pred_sex = model(context, mask_center_genes=True)

        # Compute losses
        loss_r = loss_recon(recon, center)
        loss_cmv_val = loss_cmv(pred_cmv, cmv_label)
        loss_sex_val = loss_gender(pred_sex, sex)

        # Combine losses with optional weights
        loss = loss_r * 10 + loss_cmv_val * 1 + loss_sex_val * 1

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        loss_val = loss.item()
        epoch_losses.append(loss_val)
        all_batch_losses.append(loss_val)
        total_loss += loss_val

        loop.set_postfix(
                        loss=f"{loss_val:.3f}",
                        cmv=f"{loss_cmv_val.item():.4f}",
                        recon=f"{loss_r.item():.4f}",
                        sex=f"{loss_sex_val.item():.4f}"
                    )

    print(f"Epoch {epoch+1} complete — avg loss: {total_loss / len(train_loader):.4f}", end="\r")

plt.plot(all_batch_losses)
plt.xlabel("Batch")
plt.ylabel("Total Loss")
plt.title("Training Loss over Time")
plt.show()

print("X_expr.shape:", X_expr.shape)
print("cmv.shape:", cmv.shape)
print("donor_ids.shape:", donor_ids.shape)

# embeddingBaseStrategy_multitask_model_mask_centerMarker_CD8T_balancedTrained_finetuned.pt
torch.save(model.state_dict(), "TrainedModel/multitask_model_mask_centerMarker_CD8T_balancedTrained_11030_k20_01.pt")

adata_bcg = adata_combined[adata_combined.obs["batch"] == "BCG"].copy()
cd8_from_scores(adata_bcg, layer=None, organism="human", percentile=75)

adata_bcg = adata_bcg[adata_bcg.obs.is_CD8T==True]
print(adata_bcg.X.max(), adata_bcg.X.min())
# gc.collect()

# # --- 1) read genes from file (one gene symbol per line) ---
# adata_bcg = adata_BCG.copy()
# gene_file = "shared_gens9901.txt"
# with open(gene_file, "r") as f:
#     genes_in_file = [
#         line.strip().split()[0]
#         for line in f
#         if line.strip() and not line.lstrip().startswith(("#", ""))
#     ]

# # keep original order, deduplicate while preserving order
# genes_in_file = list(dict.fromkeys(genes_in_file))

# # --- 2) drop duplicated genes from adata_bcg to avoid issues ---
# dup_mask = pd.Index(adata_bcg.var_names).duplicated(keep=False)
# if dup_mask.any():
#     print(f"Removing {dup_mask.sum()} duplicated gene entries from adata_bcg.var_names")
# adata_bcg_nodup = adata_bcg[:, ~dup_mask].copy()

# # --- 3) intersect and preserve the order from the file ---
# var_set = set(map(str, adata_bcg_nodup.var_names))
# genes_present = [g for g in genes_in_file if g in var_set]
# genes_missing = [g for g in genes_in_file if g not in var_set]

# print(f"Requested genes: {len(genes_in_file)}")
# print(f"Found in adata_bcg: {len(genes_present)}")
# print(f"Missing from adata_bcg: {len(genes_missing)}")

# # Optional: inspect a few missing
# print("First 20 missing:", genes_missing[:20])

# # --- 4) subset adata_bcg to the found genes (in file order) ---
# adata_bcg = adata_bcg_nodup[:, genes_present].copy()
# del adata_bcg_nodup
# gc.collect()
# # adata_bcg_sel is your subset AnnData
# adata_bcg

# adata_bcg = adata_bcg[adata_bcg.obs["clusters1"]=="CD8+ T"]

# sc.pp.neighbors(adata_bcg, n_neighbors=12, n_pcs=30)
# sc.tl.umap(adata_bcg)
# gc.collect()
# torch.cuda.empty_cache()
# # clustering
# sc.tl.leiden(adata_bcg, resolution=0.6)
# sc.pl.umap(adata_bcg, use_raw=False, color=["leiden", "CD8A", "CD8B", "GZMA", "GZMB", "NKG7", "PRF1", "CMV", "is_CD8T"], 
#            cmap="turbo", legend_loc="on data")

# sc.pp.pca(adata_bcg, n_comps=100)

# adata_bcg.write_h5ad("test_bcg.h5ad", compression="gzip")

# adata_bcg = adata_bcg[adata_bcg.obs["Leiden"] == True]

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Step 1: 提取 BCG 样本

X_pca = adata_bcg.obsm['X_pca_harmony'] if 'X_pca_harmony' in adata_bcg.obsm else adata_bcg.obsm['X_pca']

donor_ids = adata_bcg.obs["SampleID"].values
cmv = adata_bcg.obs["CMV"].astype(np.float32).values
unique_donors = np.unique(donor_ids)
indices_all = np.zeros((adata_bcg.n_obs, k), dtype=np.int32)

# 设置 FAISS 使用所有 CPU 线程
faiss.omp_set_num_threads(os.cpu_count())
print(f"FAISS using {faiss.omp_get_max_threads()} threads")

for donor in unique_donors:
    mask = donor_ids == donor
    X_donor = X_pca[mask].astype(np.float32)  # FAISS 要求 float32
    indices_in_adata = np.where(mask)[0]

    if len(X_donor) < k:
        print(f"Skipping donor {donor} (only {len(X_donor)} cells)")

    # 构建 CPU Index
    index = faiss.IndexFlatL2(X_donor.shape[1])
    index.add(X_donor)
    _, donor_indices = index.search(X_donor, k)

    # Map local indices back to global
    for i, row in enumerate(donor_indices):
        global_i = indices_in_adata[i]
        global_indices = indices_in_adata[row]
        indices_all[global_i] = global_indices
adata_bcg.obsm["faiss_neighbors"] = indices_all

X_expr = adata_bcg.X.toarray() if not isinstance(adata_bcg.X, np.ndarray) else adata_bcg.X
X_expr = X_expr.astype(np.float32)

# Step 4: CMV 标签
cmv = adata_bcg.obs["CMV"].astype(np.float32).values

# Step 5: Sex
adata_bcg.obs["GenderF"] = "0"
sexes = adata_bcg.obs["GenderF"].astype(np.float32).values

# Step 7: FAISS 邻居索引
indices_knn = adata_bcg.obsm["faiss_neighbors"]  # shape: [n_cells, k]

# # Step 8: donor_filter（仅包含训练用 donor）
# donor_filter = adata_bcg.obs["SampleID"].unique()  # 用于 FaissKNNDataset

# del adata_bcg
# Step 3: 构建 BCG Dataset（所有 donor 都包含）
bcg_dataset = FaissKNNDataset(
    X_expr=X_expr,
    indices_knn=indices_knn,
    cmv_status=cmv,
    sexes=sexes,
    donor_ids=donor_ids,
    donor_filter=np.unique(donor_ids)
)

bcg_loader = DataLoader(bcg_dataset, batch_size=128, shuffle=False, num_workers=0)
print(X_expr.shape, indices_knn.shape)
# model = MultiTaskContextModel(num_genes=9901, latent_dim=128)
# # model = MultiTaskContextModel(num_genes=11030, latent_dim=128)
# model.load_state_dict(torch.load("TrainedModel/embeddingBaseStrategy_multitask_model_mask_centerMarker_CD8T_balancedTrained_finetuned.pt", map_location="cuda", weights_only=False))
# model = model.cuda()  # if using GPU

# Step 4: 模型预测
model.eval()
all_cmv_preds, all_cmv_labels = [], []
all_cmv_prob = []
all_pred_cmv = []
all_sex_preds, all_sex_labels = [], []
Donors = []

with torch.no_grad():
    for batch in tqdm(bcg_loader, desc="Evaluating on BCG set"):
        context = batch['context'].cuda()
        cmv_true = batch['cmv'].cpu().numpy()
        sex_true = batch['sex'].cpu().numpy()
        sampleID = batch['donor_id']

        # 预测
        _, pred_cmv, pred_sex = model(context, mask_center_genes=False)

        # 转为概率
        pred_cmv_prob = torch.sigmoid(pred_cmv).cpu().numpy()
        pred_sex_prob = torch.sigmoid(pred_sex).cpu().numpy()

        all_cmv_prob.extend(pred_cmv_prob)
        all_pred_cmv.extend(pred_cmv.cpu().numpy())

        # 二值化
        pred_cmv_bin = (pred_cmv_prob > 0.5).astype(int)
        pred_sex_bin = (pred_sex_prob > 0.5).astype(int)

        # 收集预测结果
        all_cmv_preds.extend(pred_cmv_bin)
        all_cmv_labels.extend(cmv_true)

        all_sex_preds.extend(pred_sex_bin)
        all_sex_labels.extend(sex_true)
        Donors.extend(sampleID)

# Step 5: 准确率
cmv_acc = accuracy_score(all_cmv_labels, all_cmv_preds)
# sex_acc = accuracy_score(all_sex_labels, all_sex_preds)

print("\n✅ Evaluation on BCG set:")
print(f"CMV Accuracy:  {cmv_acc:.4f}")
# print(f"Sex Accuracy:  {sex_acc:.4f}")

print(len(all_cmv_labels), adata_bcg.obs["CMV"].shape)

148825 + 42095

df_eval = pd.DataFrame({
    "SampleID": Donors,
    "CMV_true": np.array(all_cmv_labels).flatten(),
    "CMV_pred": np.array(all_cmv_preds).flatten(),
    "CMV_score": np.array(all_cmv_prob).flatten(),
    "CMV_score_raw": np.array(all_pred_cmv).flatten(),
    # "Sex_true": np.array(all_sex_labels).flatten(),
    # "Sex_pred": np.array(all_sex_preds).flatten()
})

# df_eval.CMV_true = 0
# if len(df_eval) == adata_bcg.n_obs:
#     df_eval["is_CD8T"] = adata_bcg.obs["is_CD8T"].to_numpy()

print(df_eval.shape, adata_bcg.obs.shape)

df_eval["CMV_true"] = df_eval["CMV_true"].astype(int)

# df_eval = df_eval[df_eval.is_CD8T==True].copy()
cmv_acc = accuracy_score(df_eval.CMV_true, df_eval.CMV_pred)
print("\n✅ Evaluation on BCG set:")
print(f"CMV Accuracy:  {cmv_acc:.4f}")
# print(f"Sex Accuracy:  {sex_acc:.4f}")

# Create per-donor summary: CMV+ prediction proportion and true CMV label
# # Set colors by true CMV status colors = plot_data["CMV"].map({0: "#5DADE2", 1: "#E74C3C"})  # blue for CMV-, red for CMV+
df_grouped = df_eval.groupby("SampleID").agg(
    cmv_prop=("CMV_score", "mean"),
    cmv_true=("CMV_true", "first")
).reset_index()

# Sort by predicted CMV+ proportion
df_grouped_sorted = df_grouped.sort_values("cmv_prop")

# Define bar colors: red for CMV+ donor, light blue for CMV- donor
bar_colors = df_grouped_sorted["cmv_true"].map({1: "#E74C3C", 0: "#5DADE2"})

# Plot the sorted per-donor CMV+ cell proportion
plt.figure(figsize=(8, 6))
plt.bar(
    df_grouped_sorted["SampleID"],
    df_grouped_sorted["cmv_prop"],
    color=bar_colors
)

plt.title("Per-Donor Predicted CMV+ Score", fontsize=14)
plt.ylabel("Predicted CMV+ Score")
plt.xticks(rotation=90)
# plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)

plt.tight_layout()
plt.show()

df_grouped = df_eval.groupby("SampleID").agg(
    cmv_prop=("CMV_pred", "mean"),
    cmv_true=("CMV_true", "first")
).reset_index()

df_grouped_sorted = df_grouped.sort_values("cmv_prop")
bar_colors = df_grouped_sorted["cmv_true"].map({1: "#E74C3C", 0: "#5DADE2"})

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_grouped_sorted["SampleID"], df_grouped_sorted["cmv_prop"], color=bar_colors)

ax.set_title("Per-Donor Predicted CMV+ Score", fontsize=14)
ax.set_ylabel("Predicted CMV+ Score")

# smaller x labels + rotate
ax.tick_params(axis="x", labelsize=7)   # <- smaller font
plt.xticks(rotation=90)

# remove any grid that might be active
ax.grid(False)

plt.tight_layout()
plt.show()

from sklearn.metrics import roc_auc_score, roc_curve, auc

# --- 1) Prepare y_true and scores (use probabilities) ---
y_true  = pd.to_numeric(df_eval["CMV_true"], errors="coerce")
y_score = pd.to_numeric(df_eval["CMV_score"], errors="coerce")  # or use "CMV_score_raw" if you prefer logits

# Drop rows with missing/invalid values
mask = y_true.notna() & y_score.notna() & np.isfinite(y_score)
y_true  = y_true[mask].astype(int).values
y_score = y_score[mask].astype(float).values

# Sanity check: need both classes to compute ROC–AUC
if len(np.unique(y_true)) < 2:
    raise ValueError("ROC–AUC undefined: y_true has fewer than 2 classes after filtering.")

# --- 2) Compute AUC ---
auc_roc = roc_auc_score(y_true, y_score)
print(f"Overall ROC–AUC: {auc_roc:.4f}")

# --- 3) ROC curve plot ---
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC (all cells)")
plt.legend(loc="lower right")
plt.tight_layout()
# plt.savefig("roc_overall.png", dpi=300, bbox_inches="tight")  # optional
plt.show()

from sklearn.metrics import average_precision_score, precision_recall_curve

# --- 1) Prepare y_true and scores (use probabilities) ---
y_true  = pd.to_numeric(df_eval["CMV_true"], errors="coerce")
y_score = pd.to_numeric(df_eval["CMV_score"], errors="coerce")  # or use "CMV_score_raw" (logits)

# Drop rows with missing/invalid values
mask = y_true.notna() & y_score.notna() & np.isfinite(y_score)
y_true  = y_true[mask].astype(int).values
y_score = y_score[mask].astype(float).values

# Sanity check: need both classes
if len(np.unique(y_true)) < 2:
    raise ValueError("PR–AUC undefined: y_true has fewer than 2 classes after filtering.")

# --- 2) Compute PR–AUC (Average Precision) ---
ap = average_precision_score(y_true, y_score)
print(f"Overall PR–AUC (AP): {ap:.4f}")

# --- 3) Precision–Recall curve plot ---
precision, recall, _ = precision_recall_curve(y_true, y_score)
pos_rate = y_true.mean()  # baseline precision

plt.figure(figsize=(5, 4))
plt.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
plt.hlines(pos_rate, xmin=0, xmax=1, linestyles="--", linewidth=1, label=f"Baseline = {pos_rate:.3f}")
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall (all cells)")
plt.legend(loc="lower left")
plt.tight_layout()
# plt.savefig("pr_overall.png", dpi=300, bbox_inches="tight")  # optional
plt.show()

# 1) Derive donor_id from SampleID (e.g., "300EUAS012_RPMI" → "300EUAS012")
adata_bcg.obs["donor_id"] = adata_bcg.obs["SampleID"]#.astype(str).str.split("_").str[0]

# 2) Map CMV status to binary
pos = {"IgGpositive","PCRpositive","IgMpositive","Positive","Pos","Yes","1",1,True}
neg = {"Negative","Neg","No","0",0,False}

def to_bin(x):
    s = str(x).strip()
    if s in pos: return 1
    if s in neg: return 0
    return np.nan

adata_bcg.obs["CMV_bin"] = adata_bcg.obs["CMV"].map(to_bin)

# 3) Cell-level counts
total_cells   = adata_bcg.n_obs
cells_pos     = int((adata_bcg.obs["CMV_bin"] == 1).sum())
cells_neg     = int((adata_bcg.obs["CMV_bin"] == 0).sum())
cells_unknown = int(adata_bcg.obs["CMV_bin"].isna().sum())

# 4) Donor-level CMV status
g = adata_bcg.obs.groupby("donor_id")["CMV_bin"]
donor_status = g.apply(lambda s: 1 if (s == 1).any()
                       else (0 if (s == 0).any() else np.nan))

total_donors   = donor_status.index.nunique()
donors_pos     = int((donor_status == 1).sum())
donors_neg     = int((donor_status == 0).sum())
donors_unknown = int(donor_status.isna().sum())

# 5) Summary table
summary = pd.DataFrame({
    "metric CD8T test": [
        "Unique donors (total)",
        "CMV+ donors",
        "CMV- donors",
        "Unknown donors",
        "Cells (total)",
        "CMV+ cells",
        "CMV- cells",
        "Unknown cells",
    ],
    "count": [
        total_donors,
        donors_pos,
        donors_neg,
        donors_unknown,
        total_cells,
        cells_pos,
        cells_neg,
        cells_unknown,
    ],
})

print(summary.to_string(index=False))

adata_bcg.obs["predicted_CMV"] = all_cmv_preds
adata_bcg.obs["predicted_SEX"] = all_sex_preds

sc.pp.neighbors(adata_bcg, n_neighbors=12, n_pcs=30)
sc.tl.umap(adata_bcg)

# clustering
sc.tl.leiden(adata_bcg, resolution=0.6)
# sc.settings.set_figure_params(dpi=100, frameon=False, figsize=(4, 3), facecolor="white")
sc.pl.umap(adata_bcg, color=["leiden", "batch"], legend_loc="on data")

sc.pl.umap(adata_bcg, use_raw=False, color=["leiden", "CD8A", "CD8B", "GZMA", "GZMB", "NKG7", "PRF1", "CMV", "predicted_CMV"],
           legend_loc="on data", cmap="turbo")

# 9, 8, 14, 11, 10, 16, 1, 15
target_clusters = ["5", "7", "3","1", "0"] #["9", "8", "14", "11", "10", "16", "1", "15"]  # 注意是字符串
adata_bcg_sel = adata_bcg#[adata_bcg.obs["leiden"].isin(target_clusters)].copy()

# adata_bcg_sel = adata_bcg_sel[(adata_bcg_sel.obs["predicted_CMV_prob"]>0.5) &(adata_bcg_sel.obs["predicted_CMV_prob"]>adata_bcg_sel.obs["predicted_CMV_prob"].max())]
# adata_bcg_sel.obs["predicted_CMV"].mean()

# 计算 cutoff
cutoff = adata_bcg_sel.obs["predicted_CMV"].mean()
print(f"Mean predicted_CMV_prob cutoff: {cutoff:.4f}")

# 聚合每个 SampleID 的 predicted CMV 状态（基于均值比较）
df = adata_bcg_sel.obs[["SampleID", "predicted_CMV", "CMV"]].copy()
donor_stats = df.groupby("SampleID").agg({
    "predicted_CMV": "mean",
    "CMV": "first"  # 每个 donor 的真实标签一样
}).rename(columns={"predicted_CMV": "mean_prob", "CMV": "true_cmv"})

# 使用 cutoff 判断是否为 CMV+
donor_stats["pred_cmv"] = (donor_stats["mean_prob"] > cutoff).astype(int)

# 总体准确率
acc_total = (donor_stats["pred_cmv"] == donor_stats["true_cmv"]).mean()

# CMV+ 准确率
cmv_pos = donor_stats[donor_stats["true_cmv"] == 1]
acc_pos = (cmv_pos["pred_cmv"] == cmv_pos["true_cmv"]).mean()

# CMV- 准确率
cmv_neg = donor_stats[donor_stats["true_cmv"] == 0]
acc_neg = (cmv_neg["pred_cmv"] == cmv_neg["true_cmv"]).mean()

print(f"✅ Accuracy summary:")
print(f"  Total accuracy       = {acc_total:.3f}")
print(f"  Accuracy for CMV+    = {acc_pos:.3f}  (n={len(cmv_pos)})")
print(f"  Accuracy for CMV−    = {acc_neg:.3f}  (n={len(cmv_neg)})")

# Ensure predicted binary labels exist
df["predicted_binary"] = (df["predicted_CMV"] > 0.5).astype(int)

# Compute % of predicted CMV+ per donor
plot_data = df.groupby(["SampleID", "CMV"])["predicted_binary"].mean().reset_index()
plot_data = plot_data.rename(columns={"predicted_binary": "predicted_CMVplus_ratio"})

# Sort by predicted CMV+ percentage
plot_data = plot_data.sort_values(by="predicted_CMVplus_ratio")

# Set colors by true CMV status
colors = plot_data["CMV"].map({0: "#5DADE2", 1: "#E74C3C"})  # blue for CMV-, red for CMV+

# Set font size globally (optional)
plt.rcParams.update({
    'font.size': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 16
})

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(plot_data["SampleID"], plot_data["predicted_CMVplus_ratio"], color=colors)

# Formatting
# plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
plt.ylabel("Predicted CMV+ Cell Ratio")
plt.xticks(rotation=90)
plt.title("Per-Donor Predicted CMV+ Cell Proportion (Sorted)")
plt.tight_layout()
plt.show()

# Ensure predicted binary labels exist
df["predicted_binary"] = (df["predicted_CMV"] > 0.5).astype(int)

# Compute % of predicted CMV+ per donor
plot_data = df.groupby(["SampleID", "CMV"])["predicted_binary"].mean().reset_index()
plot_data = plot_data.rename(columns={"predicted_binary": "predicted_CMVplus_ratio"})

# Sort by predicted CMV+ percentage
plot_data = plot_data.sort_values(by="predicted_CMVplus_ratio")

# Set colors by true CMV status
bar_colors = plot_data["CMV"].map({0: "#5DADE2", 1: "#E74C3C"})  # blue for CMV-, red for CMV+

# Set font size globally (optional)
plt.rcParams.update({
    'font.size': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 16
})

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(plot_data["SampleID"], plot_data["predicted_CMVplus_ratio"], color=bar_colors)

# Formatting
plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
plt.ylabel("Predicted CMV+ Cell Ratio")
plt.title("Per-Donor Predicted CMV+ Cell Proportion (Sorted)")
plt.xticks(rotation=90)

# Change xtick label color for CMV+ donors
ax = plt.gca()
xtick_labels = ax.get_xticklabels()
for label, is_cmv_plus in zip(xtick_labels, plot_data["CMV"]):
    if is_cmv_plus == 1:
        label.set_color("#E74C3C")  # red
    else:
        label.set_color("#5DADE2")  # blue

plt.tight_layout()
plt.show()

# 获取每个 sample 的真实 CMV 状态
sample_cmv_truth = adata_bcg_sel.obs.groupby("SampleID")["CMV"].first()
sample_order = sample_cmv_truth.sort_values().index.tolist()

# 计算每个样本的预测阳性比例
sample_pos_rate = adata_bcg_sel.obs.groupby("SampleID")["predicted_CMV_prob"].mean()

# 图像设置
n = len(sample_order)
ncols = min(n, 5)
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3), sharex=True, sharey=True)
axes = axes.flatten()

for i, sample in enumerate(sample_order):
    ax = axes[i]
    data = adata_bcg_sel.obs[adata_bcg_sel.obs["SampleID"] == sample]
    color = "#B03A2E" if sample_cmv_truth[sample] == 1 else "#379392"

    # 绘制直方图 + KDE
    sns.histplot(data["predicted_CMV_prob"], kde=True, color=color, ax=ax, bins=20, edgecolor=None, stat='density')

    # 添加 0.5 阈值线
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)

    # 标题
    ax.set_title(
        f"{sample}\nCMV={int(sample_cmv_truth[sample])}, "
        f"Pred+={sample_pos_rate[sample]:.2f}",
        fontsize=9
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("")

# 删除多余子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Predicted CMV Probability Distributions per SampleID", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Create per-sample demographic summary
sample_summary = adata_combined.obs[adata_combined.obs["batch"].isin(["EUAS", "BCG"])].groupby("SampleID").agg({
    "CMV": "first",
    "GenderF": "first",
    "age": "first",
    "batch": "first"
}).reset_index()

# Set up 1x3 subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

# CMV+ / CMV- distribution
sns.countplot(data=sample_summary, x="CMV", hue="batch", ax=axes[0])
axes[0].set_title("CMV Status by Batch")
axes[0].set_xlabel("CMV Status")
axes[0].set_ylabel("Number of Samples")

# Gender distribution
sns.countplot(data=sample_summary, x="GenderF", hue="batch", ax=axes[1])
axes[1].set_title("Gender (F=1, M=0) by Batch")
axes[1].set_xlabel("GenderF")
axes[1].set_ylabel("Number of Samples")

# Age distribution
sns.histplot(data=sample_summary, x="age", hue="batch", ax=axes[2], kde=True, bins=50)
axes[2].set_title("Age Distribution by Batch")
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Number of Samples")

plt.suptitle("Demographic Overview of EUSA and BCG Samples", fontsize=16)
plt.tight_layout()
plt.show()

adata_combined.obs["batch"].value_counts

adata_combined.obs["donor"] = adata_combined.obs["SampleID"].astype(str).str.split("_").str[0]

# Create per-sample demographic summary
sample_summary = adata_combined.obs[adata_combined.obs["batch"].isin(["EUAS", "BCG"])].groupby("donor").agg({
    "CMV": "first",
    "GenderF": "first",
    "age": "first",
    "batch": "first"
}).reset_index()

# Set up 1x3 subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

# CMV+ / CMV- distribution
sns.countplot(data=sample_summary, x="CMV", hue="batch", ax=axes[0])
axes[0].set_title("CMV Status by Batch")
axes[0].set_xlabel("CMV Status")
axes[0].set_ylabel("Number of Samples")

# Gender distribution
sns.countplot(data=sample_summary, x="GenderF", hue="batch", ax=axes[1])
axes[1].set_title("Gender (F=1, M=0) by Batch")
axes[1].set_xlabel("GenderF")
axes[1].set_ylabel("Number of Samples")

# Age distribution
sns.histplot(data=sample_summary, x="age", hue="batch", ax=axes[2], kde=True, bins=50)
axes[2].set_title("Age Distribution by Batch")
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Number of Samples")

plt.suptitle("Demographic Overview of EUSA and BCG Donors", fontsize=16)
plt.tight_layout()
plt.show()

del adata_combined

del adata_bcg, adata_bcg_sel

# start to impute the single cell data. 
with open("shared_gens9901.txt", "r") as f:
    shared_genes = [line.strip() for line in f if line.strip()]

# adata_aging = sc.read_h5ad("scData_4m_hUSI.h5ad")

adata_aging = sc.read_h5ad("scData_4m_hUSI.h5ad")

adata_aging.layers.clear()
adata_aging = adata_aging[:, shared_genes]

##################################################################################################################################################################################################################

del adata_bcg, adata_bcg_sel, adata_combined, adata_euas

adata_aging = sc.read_h5ad("4m_scData_CMV_imputed.h5ad")

print(adata_aging.X.max(), adata_aging.X.min())

sc.pl.umap(adata_aging, use_raw=False, color=["leiden_res0_5", "CD8A", "CD8B", "GZMA", "GZMB", "NKG7", "PRF1"], legend_loc="on data", cmap="turbo")

target_clusters = ["3", "8", "4", "16"]
adata_aging = adata_aging[adata_aging.obs["leiden_res0_5"].isin(target_clusters)]

# Count number of cells per donor
cell_counts = adata_aging.obs['donor_id'].value_counts()
print(cell_counts)
# Filter donors with more than 128 cells
valid_donors = cell_counts[(cell_counts > 200) & (cell_counts < 2000)].index
# Subset the AnnData object
adata_aging = adata_aging[adata_aging.obs['donor_id'].isin(valid_donors)].copy()

k = 32
X_pca = adata_aging.obsm['X_pca_harmony'] if 'X_pca_harmony' in adata_aging.obsm else adata_aging.obsm['X_pca']
donor_ids = adata_aging.obs["donor_id"].values
unique_donors = np.unique(donor_ids)
indices_all = np.zeros((adata_aging.n_obs, k), dtype=np.int32)

# 设置 FAISS 使用所有 CPU 线程
faiss.omp_set_num_threads(os.cpu_count())
print(f"FAISS using {faiss.omp_get_max_threads()} threads")

for donor in unique_donors:
    mask = donor_ids == donor
    X_donor = X_pca[mask].astype(np.float32)  # FAISS 要求 float32
    indices_in_adata = np.where(mask)[0]

    if len(X_donor) < k:
        print(f"Skipping donor {donor} (only {len(X_donor)} cells)")

    # 构建 CPU Index
    index = faiss.IndexFlatL2(X_donor.shape[1])
    index.add(X_donor)
    _, donor_indices = index.search(X_donor, k)

    # Map local indices back to global
    for i, row in enumerate(donor_indices):
        global_i = indices_in_adata[i]
        global_indices = indices_in_adata[row]
        indices_all[global_i] = global_indices

adata_aging.obsm["faiss_neighbors"] = indices_all

class FaissKNNDatasetPredictSparse(Dataset):
    def __init__(self, adata, indices_knn):
        self.adata = adata
        self.indices_knn = indices_knn

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        neighbor_ids = self.indices_knn[idx]

        context = self.adata.X[neighbor_ids].toarray().astype(np.float32)  # [k, genes]

        return {
            "context": torch.tensor(context, dtype=torch.float32),
            "cell_index": idx
        }

# Prepare data
indices_knn = adata_aging.obsm["faiss_neighbors"]  # reused from earlier step
dataset = FaissKNNDatasetPredictSparse(adata_aging, indices_knn)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

# model = MultiTaskContextModel(num_genes=9901, latent_dim=128)
# model.load_state_dict(torch.load("TrainedModel/embeddingBaseStrategy_multitask_model_mask_centerMarker_finetuned.pt", map_location="cuda", weights_only=False))
# model = model.cuda()  # if using GPU

from tqdm import tqdm
predicted_CMV, predicted_Sex = [],[]
predicted_CMV_score, predicted_Sex_score = [],[]
with torch.no_grad():
    for batch in tqdm(loader, desc="Predicting CMV and Sex"):
        context = batch["context"].cuda()   # [1+k, genes]

        # Forward pass
        recon, cmv_prob, sex_prob = model(context, mask_center_genes=False)  # Add batch dim

        pred_cmv_prob = torch.sigmoid(cmv_prob).cpu().numpy()
        pred_sex_prob = torch.sigmoid(sex_prob).cpu().numpy()

        # 二值化
        pred_cmv_bin = (pred_cmv_prob > 0.5).astype(int)
        pred_sex_bin = (pred_sex_prob > 0.5).astype(int)

        predicted_CMV.append(pred_cmv_bin)
        predicted_Sex.append(pred_sex_bin)
        predicted_CMV_score.append(cmv_prob)
        predicted_Sex_score.append(sex_prob)

# predicted_CMV

adata_aging.obs["predicted_CMV"] = np.concatenate(predicted_CMV)
adata_aging.obs["predicted_Sex"] = np.concatenate(predicted_Sex)
adata_aging.obs["predicted_CMV_score"] = np.concatenate([t.cpu().numpy() for t in predicted_CMV_score])
adata_aging.obs["predicted_Sex_score"] = np.concatenate([t.cpu().numpy() for t in predicted_Sex_score])

df = adata_aging.obs
df.to_csv("CMV_inputed_cell.csv")

# adata_aging.write_h5ad("4m_scData_CMV_imputed.h5ad", compression="gzip")
# adata_aging = sc.read_h5ad("4m_scData_CMV_imputed.h5ad")

sc.pl.umap(adata_aging, use_raw=False, color=["leiden_res0_5", "predicted_CMV_score"], legend_loc="on data", cmap="turbo")

sc.pl.umap(adata_aging, use_raw=False, color=["leiden_res0_5", "CD8A", "CD8B", "GZMA", "GZMB", "NKG7", "PRF1", "predicted_CMV"], legend_loc="on data", cmap="turbo")

# target_clusters = ["3", "8", "4"]
# gc.collect()
# # selected_cells = adata_aging[:, "PRF1"].X > 1.2
# adata_sel = adata_aging[adata_aging.obs["leiden_res0_5"].isin(target_clusters)].copy() # adata_aging[selected_cells, :]#
# adata_sel

# Step 1: Compute cutoff
cutoff = adata_aging.obs["predicted_CMV"].mean()
print(f"Mean predicted_CMV_prob cutoff: {cutoff:.4f}")

# Step 2: Aggregate predicted CMV probabilities per donor
df = adata_aging.obs[["donor_id", "predicted_CMV", "sex"]].copy()

# Step 3: Aggregate mean predicted CMV per donor
donor_stats = df.groupby("donor_id").agg({
    "predicted_CMV": "mean"
}).rename(columns={"predicted_CMV": "mean_prob"})

# Step 4: Add binary CMV prediction based on cutoff
donor_stats["pred_cmv"] = (donor_stats["mean_prob"] > cutoff).astype(int)

# Step 5: Merge donor-level sex info (assuming one sex per donor)
donor_sex = df[["donor_id", "sex"]].drop_duplicates(subset="donor_id")
donor_stats = donor_stats.merge(donor_sex, on="donor_id", how="left")

# Step 6: Check counts
print(donor_stats["pred_cmv"].value_counts())

donor_stats.to_csv("CMV2Donor.csv")

# Step 1: Compute cutoff
cutoff = adata_aging.obs["predicted_CMV_score"].mean()
print(f"Mean predicted_CMV_prob cutoff: {cutoff:.4f}")

# Step 2: Create DataFrame with relevant donor info
df = adata_aging.obs[["donor_id", "predicted_CMV_score", "sex"]].copy()

# Step 3: Aggregate mean predicted CMV score per donor
donor_stats = df.groupby("donor_id").agg({
    "predicted_CMV_score": "mean"
}).rename(columns={"predicted_CMV_score": "mean_prob"})

# Step 4: Binary CMV+ prediction based on cutoff
donor_stats["pred_cmv"] = (donor_stats["mean_prob"] > cutoff).astype(int)

# Step 5: Add sex (one per donor)
donor_sex = df[["donor_id", "sex"]].drop_duplicates(subset="donor_id")
donor_stats = donor_stats.merge(donor_sex, on="donor_id", how="left")

# Step 6: Save results
donor_stats.to_csv("CMV2Donor_scoreBase.csv")

donor_stats = pd.read_csv("CMV2Donor.csv")

# # Sort by mean_prob
# donor_stats_sorted = donor_stats.sort_values(by='mean_prob', ascending=True)
# order = donor_stats_sorted['donor_id'].tolist()

# # Plot
# # Create color palette based on pred_cmv
# color_map = {1: 'red', 0: 'lightblue'}
# palette = donor_stats_sorted['pred_cmv'].map(color_map)

# # Plot
# plt.figure(figsize=(10, 4))
# sns.barplot(
#     x='donor_id',
#     y='mean_prob',
#     data=donor_stats_sorted,
#     palette=palette.tolist(),
#     order=order
# )
# plt.xticks([], [])   # Hide x-axis labels
# plt.xlabel('')
# plt.ylabel('Mean CMV Probability')
# plt.title('Mean Predicted CMV Probability per Donor (Sorted)')
# plt.tight_layout()
# plt.show()

# # Step 1: Group by donor
# grouped = adata_sel.obs.groupby("donor_id")

# # Step 2: Calculate proportion of predicted CMV+ (==1) cells per donor
# donor_stats = grouped["predicted_CMV"].mean().reset_index()
# donor_stats.columns = ["donor_id", "prop_predicted_CMV"]

# # Step 3 (Optional): Add cell count per donor
# donor_stats["n_cells"] = grouped.size().values

# global_cmv_rate = (adata_aging.obs["predicted_CMV"] == 1).mean()
# global_cmv_rate

# #Create new column CMV_Donor
donor_stats["CMV_Donor"] = donor_stats.pred_cmv
donor_stats["prop_predicted_CMV"] = donor_stats.mean_prob

# donor_stats_sorted = donor_stats.sort_values(by='prop_predicted_CMV', ascending=True)
# order = donor_stats_sorted['donor_id'].tolist()

# # Plot
# # Create color palette based on pred_cmv
# color_map = {1: 'red', 0: 'lightblue'}
# palette = donor_stats_sorted['CMV_Donor'].map(color_map)

# # Plot
# plt.figure(figsize=(10, 4))
# sns.barplot(
#     x='donor_id',
#     y='prop_predicted_CMV',
#     data=donor_stats_sorted,
#     palette=palette.tolist(),
#     order=order
# )
# plt.xticks([], [])   # Hide x-axis labels
# plt.xlabel('')
# plt.ylabel('CMV+ cell ratio')
# plt.title('CMV+ cell ratio per Donor (Sorted)')
# plt.tight_layout()
# plt.show()

donor_stats_sorted = donor_stats.sort_values(by='mean_prob', ascending=True)

adata_aging[adata_aging.obs["orig.ident"]=="Data5"].obs

adata_aging.obs["CMV"] = "Unknown"

df = pd.read_csv("donor_CMV_status.csv")

df["donor_id"] = "Data5_" + df.ids

df["CMV"] = df["CMV_PN"].map({"Positive": 1, "Negative": 0})

# Step 1: Ensure donor_id is the index in df or use merge
df_subset = df[["donor_id", "CMV"]]

# Step 2: Map CMV values to adata_sel.obs using donor_id
adata_aging.obs["CMV"] = adata_aging.obs["donor_id"].map(
    dict(zip(df_subset["donor_id"], df_subset["CMV"]))
)

donor_stats_sorted["Batch"] = donor_stats_sorted.donor_id.str.split("_").str[0]

# 1. Initialize with "Unknown"
donor_stats_sorted["CMV_true"] = "Unknown"

# 2. Map Positive/Negative to "+" / "-"
df["CMV"] = df["CMV_PN"].map({"Positive": "+", "Negative": "-"})

# 3. Create a mapping from donor_id to CMV
cmv_map = dict(zip(df["donor_id"], df["CMV"]))

# 4. Overwrite only where donor_id matches
donor_stats_sorted["CMV_true"] = donor_stats_sorted["donor_id"].map(cmv_map).fillna("Unknown")

temp = donor_stats_sorted[donor_stats_sorted.sex=="M"]
order = temp['donor_id'].tolist()

# Add hue column to control bar color
color_map = {1: 'red', 0: 'lightblue'}
temp["color"] = temp["CMV_Donor"].map(color_map)

# Plot
plt.figure(figsize=(12, 4))
ax = sns.barplot(
    x='donor_id',
    y='prop_predicted_CMV',
    hue='CMV_Donor',                   # Set hue
    data=temp,
    order=order,
    palette=color_map,                 # Pass color map to hue
    dodge=False,                       # Avoid side-by-side bars
    legend=False                       # No legend needed
)

# Hide x-axis labels
plt.xticks([], [])
plt.xlabel('')
plt.ylabel('CMV+ cell ratio (Male)')
plt.title('CMV+ cell ratio per Donor (Male Sorted)')

# Annotate only "+" and "-" from CMV_true
for i, label in enumerate(temp["CMV_true"]):
    if label == "+":
        ax.text(i, -0.02, "+", color="red", ha="center", va="top", fontsize=9)
    elif label == "-":
        ax.text(i, -0.02, "-", color="blue", ha="center", va="top", fontsize=9)

plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.show()

temp = donor_stats_sorted[donor_stats_sorted.sex=="F"]
order = temp['donor_id'].tolist()

# Add hue column to control bar color
color_map = {1: 'red', 0: 'lightblue'}
temp["color"] = temp["CMV_Donor"].map(color_map)

# Plot
plt.figure(figsize=(12, 4))
ax = sns.barplot(
    x='donor_id',
    y='prop_predicted_CMV',
    hue='CMV_Donor',                   # Set hue
    data=temp,
    order=order,
    palette=color_map,                 # Pass color map to hue
    dodge=False,                       # Avoid side-by-side bars
    legend=False                       # No legend needed
)

# Hide x-axis labels
plt.xticks([], [])
plt.xlabel('')
plt.ylabel('CMV+ cell ratio (Female)')
plt.title('CMV+ cell ratio per Donor (Female Sorted)')

# Annotate only "+" and "-" from CMV_true
for i, label in enumerate(temp["CMV_true"]):
    if label == "+":
        ax.text(i, -0.02, "+", color="red", ha="center", va="top", fontsize=9)
    elif label == "-":
        ax.text(i, -0.02, "-", color="blue", ha="center", va="top", fontsize=9)

plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.show()

# Use donor_stats_sorted or temp as needed
temp = donor_stats_sorted.copy()
order = temp['donor_id'].tolist()

# Map 'M' → 0 (lightblue), 'F' → 1 (red)
sex_map = {'M': 0, 'F': 1}
temp['sex_num'] = temp['sex'].map(sex_map)

# Define palette
sex_color_map = {0: 'royalblue', 1: 'tomato'}

# Plot
plt.figure(figsize=(12, 3.5))
ax = sns.barplot(
    x='donor_id',
    y='prop_predicted_CMV',
    data=temp,
    order=order,
    hue='sex_num',
    palette=sex_color_map,
    dodge=False,
    alpha=1, linewidth=0.001
)

# Hide x-axis labels
plt.xticks([], [])
plt.xlabel('')
plt.ylabel('CMV+ cell ratio')
plt.title('CMV+ cell ratio per Donor (Colored by Sex)')

# Annotate "+" and "-" from CMV_true
for i, label in enumerate(temp["CMV_true"]):
    if label == "+":
        ax.text(i, -0.01, "*", color="black", ha="center", va="top", fontsize=9)
    elif label == "-":
        ax.text(i, 0.015, ".", color="black", ha="center", va="top", fontsize=9)

plt.ylim(-0.05, 1.05)
plt.tight_layout()
# Save figure in high resolution
plt.savefig("cmv_ratio_per_donor.png", dpi=300)
plt.show()

# Copy data
temp = donor_stats_sorted.copy()
order = temp['donor_id'].tolist()

# Map sex to colors
sex_color_map = {'M': 'royalblue', 'F': 'tomato'}
bar_colors = temp['sex'].map(sex_color_map).tolist()

# Plot
plt.figure(figsize=(16, 3.5))
ax = sns.barplot(
    x='donor_id',
    y='prop_predicted_CMV',
    data=temp,
    order=order,
    palette=bar_colors
)

# Hide x-axis labels
plt.xticks([], [])
plt.xlabel('')
plt.ylabel('CMV+ cell ratio')
plt.title('CMV+ cell ratio per Donor (Colored by Sex)')

# Annotate "+" and "-" from CMV_true
for i, label in enumerate(temp["CMV_true"]):
    if label == "+":
        ax.text(i, -0.01, "*", color="black", ha="center", va="top", fontsize=9)
    elif label == "-":
        ax.text(i, 0.015, ".", color="black", ha="center", va="top", fontsize=9)

plt.ylim(-0.05, 1.05)
plt.tight_layout()

# Save high-resolution figure
plt.savefig("cmv_ratio_per_donor_clean.png", dpi=600)
plt.show()

#0.017544 ~ 0.893720
select_donors = donor_stats_sorted[(donor_stats_sorted.prop_predicted_CMV >= 0.827751) | (donor_stats_sorted.prop_predicted_CMV <= 0.103734)]

select_donors.to_csv("selectedDonors_CMVPredicted.csv")

donors = select_donors.donor_id

adata_sel = adata_aging[adata_aging.obs["donor_id"].isin(donors)]

# Step 2: Map CMV values to adata_sel.obs using donor_id
adata_sel.obs["CMV_Donor"] = adata_sel.obs["donor_id"].map(
    dict(zip(select_donors["donor_id"], select_donors["CMV_Donor"]))
)

# Make sure CMV_Donor is treated as string (optional but clean)
donor_df = adata_sel.obs[["donor_id", "age", "CMV_Donor"]].drop_duplicates()
donor_df["CMV_Donor"] = donor_df["CMV_Donor"].astype(str)

# Define palette with string keys
palette = {'0': "blue", '1': "red"}

# Plot
plt.figure(figsize=(5, 4))
sns.stripplot(
    data=donor_df,
    x="CMV_Donor",
    y="age",
    hue="CMV_Donor",
    # palette=palette,
    jitter=True,
    dodge=False,
    legend=False, alpha=0.4)

plt.xlabel("CMV_Donor (0 = Negative, 1 = Positive)")
plt.ylabel("Age")
plt.title("Age Distribution by CMV_Donor Status")
plt.tight_layout()
plt.show()

# | Age Group   | Estimated CMV Seroprevalence |
# | ----------- | ---------------------------- |
# | 20–40 years | 30–50%                       |
# | 40–60 years | 50–70%                       |
# | ≥65 years   | **60–90%**                   |

# Step 1: Create donor-level metadata (one row per donor)
donor_df = adata_sel.obs[["donor_id", "sex", "age", "CMV_Donor"]].drop_duplicates()

# Optional: convert CMV_Donor and sex to string if needed for hue/palette
donor_df["CMV_Donor"] = donor_df["CMV_Donor"].astype(str)
donor_df["sex"] = donor_df["sex"].astype(str)

# Step 2: Create subplot layout
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot 1: Age distribution (stripplot)
sns.stripplot(data=donor_df, x="CMV_Donor", y="age", hue="sex",
              palette="tab10", jitter=True, dodge=True, ax=axes[0], alpha=0.6)
axes[0].set_title("Age by CMV_Donor and Sex")
axes[0].set_xlabel("CMV_Donor")
axes[0].set_ylabel("Age")

# Plot 2: Sex distribution (countplot)
sns.countplot(data=donor_df, x="sex", hue="CMV_Donor",ax=axes[1], alpha=0.7)
axes[1].set_title("Sex Distribution by CMV_Donor")
axes[1].set_xlabel("Sex")
axes[1].set_ylabel("Number of Donors")

# # Plot 3: Age distribution
# sns.histplot(data=donor_df, x="age", kde=True, bins=20, hue="sex", ax=axes[2])
# axes[2].set_title("Age Distribution")
# axes[2].set_xlabel("Age")
# axes[2].set_ylabel("Number of Donors")

# Plot 3: Age distribution
sns.histplot(data=donor_df, x="age", kde=True, bins=20, hue="CMV_Donor", ax=axes[2])
axes[2].set_title("Age Distribution")
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Number of Donors")

plt.tight_layout()
plt.show()

donor_df = adata_aging.obs[["donor_id", "sex", "age"]].drop_duplicates()
# Optional: convert CMV_Donor and sex to string if needed for hue/palette
donor_df["sex"] = donor_df["sex"].astype(str)

donor_df[donor_df.age>65].sex.value_counts()
