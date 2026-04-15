# ModelTraining_for_DrugSti

Python workflow for preparing multi-cohort single-cell RNA-seq data and training a context-aware multitask neural network to predict **CMV status** and **sex** from CD8 T-cell expression profiles.

The script integrates multiple `AnnData` datasets, harmonizes gene identifiers, constructs within-sample FAISS nearest-neighbor cell neighborhoods, and trains a Transformer-based model on local cellular context.

## What this script does

1. Loads three single-cell datasets:
   - `EUAS.h5ad`
   - `BCG.h5ad`
   - `CXCL9_TI_processed.h5ad`
2. Aligns external sample and donor metadata.
3. Harmonizes gene names across datasets using `mygene`.
4. Intersects shared genes and removes duplicated features.
5. Merges datasets into one combined `AnnData` object.
6. Computes PCA and builds per-sample FAISS neighbor graphs.
7. Selects and annotates CD8 T-cell populations.
8. Trains a multitask model to:
   - reconstruct the center cell expression profile,
   - classify **CMV status**,
   - classify **sex**.
9. Saves the trained model weights.

## Repository contents

- `ModelTraining_for_DrugSti_cleaned.py` — cleaned training script
- `README.md` — project description and usage notes

## Expected input files

Place the following files in the working directory unless you modify the paths in the script:

### AnnData objects
- `EUAS.h5ad`
- `BCG.h5ad`
- `CXCL9_TI_processed.h5ad`

### Metadata tables
- `cell_metadata.tsv`
- `sample_metadata.tsv`
- `donor_CMV_status.csv`

## Main dependencies

This script uses the following Python packages:

- `scanpy`
- `anndata`
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `torch`
- `matplotlib`
- `seaborn`
- `faiss`
- `mygene`
- `lion-pytorch`

## Installation

A typical environment can be created with:

```bash
pip install scanpy anndata pandas numpy scipy scikit-learn torch matplotlib seaborn mygene lion-pytorch faiss-cpu
```

If you want GPU-enabled FAISS or CUDA-enabled PyTorch, install the versions compatible with your system instead.

## Usage

Run the script from the project directory:

```bash
python ModelTraining_for_DrugSti_cleaned.py
```

## Output files

The script writes several intermediate or final outputs, including:

- `shared_genes11038.txt` — shared gene list across datasets
- `combinedDataCMV_Age.h5ad` — merged dataset after metadata integration and filtering
- `TrainedModel/multitask_model_mask_centerMarker_CD8T_balancedTrained_11030_k20_01.pt` — trained model weights

Make sure the `TrainedModel/` directory exists before saving the final checkpoint.

## Model overview

The core model is `MultiTaskContextModel`, which contains:

- a gene-expression encoder,
- a Transformer encoder over local cell neighborhoods,
- a CMV prediction head,
- a sex prediction head,
- a decoder for center-cell reconstruction.

Each training example consists of a center cell and its FAISS-derived neighboring cells from the same sample.

## Important assumptions

This script assumes that:

- all datasets are human scRNA-seq data,
- metadata column names match those used in the script,
- PCA coordinates are stored in `X_pca` or `X_pca_harmony`,
- donor/sample identifiers are already present and consistent across files,
- the user will adapt filtering thresholds and marker-based cell selection as needed.

## Notes before sharing or reusing

- This script is still highly dataset-specific.
- Some sections are exploratory and reflect a research workflow rather than a fully packaged pipeline.
- Before public release, consider moving hard-coded thresholds and filenames into a config file or command-line arguments.
- A `requirements.txt` or `environment.yml` file would improve reproducibility.

## Suggested next improvements

For a cleaner GitHub repository, you may also want to add:

- `requirements.txt`
- `.gitignore`
- `environment.yml`
- a small example dataset or schema description
- argument parsing with `argparse`

## Citation / acknowledgement

If you reuse this workflow in a manuscript or another repository, please cite the relevant dataset sources and acknowledge the original study context.
