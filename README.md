# DIETNETWORK
Pytorch implementation of DietNetwork
## Scripts
1. **create_dataset.py** : takes snps.txt and labels.txt files and creates dataset.npz
2. **preprocess_data.py** : Shuffles data, partition data into folds, creates datasets (train, valid, test) and replace missing values with feature means

## Questions
- Missing values : features means computed on training and valid sets together
