# DIETNETWORK
Pytorch implementation of DietNetwork
## Scripts
### Main scripts
1. **create_dataset.py** : Create dataset and partition data into folds. The script takes snps.txt and labels.txt files as input to create dataset.npz and folds_indexes.npz
2. **generate_embedding.py** : Takes dataset.npz and folds_indexes.npz files created in the previous step and computes the embedding (genotypic frequency) of every fold. Embedding of each fold is saved in embedding.npz
3. **train.py** : For now, this script preprocesses data for discrim net.
  (i) Feature mean is computed on the training set
  (ii) Missing values are replaced in train and valid sets with the feature mean
  (iii) Features are normalized using the feature mean computed in (i) and a sd computed on the training set
### Helper scripts
- **dataset_utils.py** : Data related functions (shuffle, partition, split, get_fold_data, replace_missing_values, normalize, ...)
- **model.py** : Model definition of feature embedding (auxiliary) and discriminative (main) networks.
- **preprocess_data.py** : This script will be removed eventually. Shuffles data, partition data into folds, creates datasets by fold (train, valid, test) and replace missing values with feature means. Dataset for each fold are saved in dataset_by_fold.npz
## Data preprocessing
### Auxiliary net
- Missing values are -1 and are not included in the computation of genotypic frequencies
- Embedding values are computed on train and valid sets
### Main net
- Missing values are replaced in train, valid and test sets by the feature mean of the training set
- Features of train, valid and test sets are normalized using mean and sd computed on training set
## To do
- [x] Embedding
- [x] Data preprocessing : Missing values
- [x] Data preprocessing : Data normalization
- [x] Dataset class (for dataloader)
- [x] Auxiliary and Main networks models
- [x] Training loop
- [ ] Loss/Accuracy monitoring of train and valid
- [ ] Early stopping
- [ ] Test for in-sample data
- [ ] Test for out-of-sample data
## Packages
- Python 3.6
- Pytorch installation on kepler :
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
