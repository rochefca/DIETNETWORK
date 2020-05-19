import numpy as np

def main():
    data = np.load('../dataset_by_fold.npz',allow_pickle=True)

    for fold in range(len(data['data_by_fold'])):
        filename = 'pretty_print_fold' + str(fold) + '.txt'
        f = open(filename, 'w')

        # First line (samples + snp names)
        f.write('samples')
        for snp in data['snp_names']:
            f.write('\t'+snp)
        f.write('\n')

        # Training set (one ind and it's genotypes per line)
        f.write('Training')
        for i in range(len(data['snp_names'])):
            f.write('\t*')
        f.write('\n')

        for ind,x in zip(data['data_by_fold'][fold][0][2],
                data['data_by_fold'][fold][0][0]):
            f.write(ind)
            for genotype in x:
                f.write('\t'+str(round(genotype,2)))
            f.write('\n')

        # Validations set
        f.write('Validation')
        for i in range(len(data['snp_names'])):
            f.write('\t*')
        f.write('\n')

        for ind,x in zip(data['data_by_fold'][fold][1][2],
                data['data_by_fold'][fold][1][0]):
            f.write(ind)
            for genotype in x:
                f.write('\t'+str(round(genotype,2)))
            f.write('\n')

        # Test set
        f.write('Test')
        for i in range(len(data['snp_names'])):
            f.write('\t*')
        f.write('\n')

        for ind,x in zip(data['data_by_fold'][fold][2][2],
                data['data_by_fold'][fold][2][0]):
            f.write(ind)
            for genotype in x:
                f.write('\t'+str(genotype))
            f.write('\n')

        f.close()

if __name__ == '__main__':
    main()
