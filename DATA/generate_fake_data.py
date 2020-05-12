"""
Generate individuals and genotypes for testing
This is fictional data
"""
from random import randrange

# Make individuals
individuals = ['ind'+str(i) for i in range(1,51)]

# Populations
pop = ['POP'+'1' for i in range(10)]
pop += ['POP'+'2' for i in range(10)]
pop += ['POP'+'3' for i in range(10)]
pop += ['POP'+'4' for i in range(10)]
pop += ['POP'+'5' for i in range(10)]

print(individuals)
print(pop)
print(len(individuals), len(pop))

# Genotypes
pop1_gen = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
pop2_gen = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
pop3_gen = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
pop4_gen = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
pop5_gen = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

genotypes = [pop1_gen.copy() for i in range(10)]
genotypes += [pop2_gen.copy() for i in range(10)]
genotypes += [pop3_gen.copy() for i in range(10)]
genotypes += [pop4_gen.copy() for i in range(10)]
genotypes += [pop5_gen.copy() for i in range(10)]

# Randomly add missing values
nb_missing_values = 50
for i in range(nb_missing_values):
    row = randrange(len(genotypes))
    col = randrange(len(genotypes[0]))
    genotypes[row][col] = 'NA'

# Write data to file
with open('labels.txt', 'w') as f:
    f.write('samples\tpop\n')
    for i,l in zip(individuals, pop):
        f.write(i+'\t'+l+'\n')
with open('snps.txt', 'w') as f:
    f.write('samples')
    for i in range(1,len(genotypes[0])+1):
        f.write('\tsnp_'+str(i))
    f.write('\n')
    for i,gen in zip(individuals, genotypes):
        f.write(i)
        for g in gen:
            f.write('\t'+str(g))
        f.write('\n')
