# DIETNETWORK
Pytorch implementation of DietNetwork
## Scripts
1. **create_dataset.py** : takes snps.txt and labels.txt files and creates dataset.npz
2. **preprocess_data.py** : Shuffles data, partition data into folds, creates datasets by fold (train, valid, test) and replace missing values with feature means. Dataset for each fold are saved in dataset_by_fold.npz
3. **generate_embedding.py** : Takes dataset_by_fold.npz and computes the embedding (genotype freq by genotype by class) of every fold. Embedding of each fold is saved in embedding.npz
4. **model.py** : Model definition of feature embedding (auxiliary) and discriminative (main) networks.
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
