# DIETNETWORK
Pytorch implementation of DietNetwork
## Scripts
1. **create_dataset.py** : Create dataset and partition data into folds. The script takes snps.txt and labels.txt files as input to create dataset.npz and folds_indexes.npz
2. **generate_embedding.py** : Takes dataset.npz and folds_indexes.npz files created in the previous step and computes the embedding (genotypic frequency) of every fold. Embedding of each fold is saved in embedding.npz
3. **model.py** : Model definition of feature embedding (auxiliary) and discriminative (main) networks.
4. **preprocess_data.py** : This script will be removed eventually. Shuffles data, partition data into folds, creates datasets by fold (train, valid, test) and replace missing values with feature means. Dataset for each fold are saved in dataset_by_fold.npz
## Missing values
- Not considered in embedding computation
- Feature mean computed over training and validation sets
## To do
- Missing values should be handle after embedding computation
- Data loader, inlcuding genotypes normalization
- Training loop structure
- Accuracy monitoring and early stopping
- Test for in-sample data
- Test for out-of-sample data
## Packages
- Python 3.6
- Pytorch installation on kepler :
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
