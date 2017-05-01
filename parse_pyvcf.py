import sys,argparse
import numpy as np
import pickle
import vcf
from pandas import *
from sklearn.decomposition import PCA
from pprint import pprint

parser = argparse.ArgumentParser(
                    description="parse input VCF",
                    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-i', '--input_vcf', help='Input VCF', default=None)
parser.add_argument('--pheno_file', help='phenotypes file', default=None)

def convert_to_matrix(infile):
    """
        parse input VCF file into a 2D matrix
        of users x genotypes. Each entry will
        have value (0,1,2) for the number of 
        minor alleles. Additional info is stored
        in a separate dictionary.
    """
    if infile is None:
        genotypes = pickle.load(open("genotypes.p", "rb"))
        snp_info = pickle.load(open("snp_info.p", "rb"))
        return genotypes, snp_info

    genotypes = {}
    snp_info = {}

    vcf_reader = vcf.Reader(open(infile, 'r'))
    count = 0
    # for now, just take SNPs from chr 7
    for record in vcf_reader.fetch('7',20000000,29000000):
        count += 1
        if count % 1000 == 0:
            print count
        snp_info[record.ID] = {"chrom":record.CHROM,
                               "pos":record.POS,
                               "ref":record.REF,
                               "alt":record.ALT}
        user_genos = {}
        for call in record.samples:
            if call.called:
                user_id = call.sample[4:]
                user_id = user_id.split("_")[0]
                user_id = float(user_id)
                genos = call.data.GT
                geno_score = float(genos[0]) + float(genos[-1])
                user_genos[user_id] = geno_score
        genotypes[record.ID] = user_genos

    genotypes = DataFrame(genotypes)#.fillna(0)
    snp_info = DataFrame(snp_info)

    pickle.dump(genotypes, open("genotypes.p","wb"))
    pickle.dump(snp_info,open("snp_info.p","wb"))

    return genotypes, snp_info

def process_phenotypes(pheno_file):
    phenotypes = pandas.read_csv(pheno_file, sep=';', index_col=0)
    phenotypes = phenotypes[['Height', 'Eye color']]
    for i, row in phenotypes.iterrows():
        height_val = phenotypes.ix[i,'Height']
        if isinstance(height_val, str):
            height_val = height_val.replace(" ","")
            if "cm" in height_val:
                try:
                    hval = float(''.join([c for c in height_val if c.isdigit()]))
                    phenotypes.set_value(i,'Height',hval)
                    continue
                except:
                    pass
            elif "'" in height_val:
                try:
                    hval = height_val.strip("'").strip('"')
                    ft_ins = hval.split("'")
                    if len(ft_ins) == 1:
                        hval = float(ft_ins[0]) * 12 * 2.54
                    elif len(ft_ins) == 2:
                        hval = (float(ft_ins[0])*12 + float(ft_ins[1])) * 2.54
                    phenotypes.set_value(i,'Height',hval)
                    continue
                except:
                    pass
        phenotypes.set_value(i,'Height',float('NaN'))

    for i, row in phenotypes.iterrows():
        eye_val = phenotypes.ix[i,'Eye color']
        if isinstance(eye_val, str):
            eye_val = eye_val.lower()
            if 'brown' in eye_val:
                eye_score = 2.0
            elif 'green' in eye_val:
                eye_score = 1.0
            elif 'blue' in eye_val:
                eye_score = 0.0
            else:
                eye_score = float('NaN')
        else:
            eye_score = float('NaN')
        phenotypes.set_value(i,'Eye color',eye_score)

    pickle.dump(phenotypes, open("phenotypes.p","wb"))
    return phenotypes

# currently available phenotypes: Height, Eye color
def get_XY(genotypes, phenotypes, pheno_name='Height'):
    # remove column if more than 80% of data is NaN
    genotypes = genotypes.loc[:, genotypes.isnull().mean() < 0.8]
    pheno_filter = phenotypes[phenotypes[pheno_name].notnull()][[pheno_name]]
    pheno_filter = pheno_filter[(pheno_filter['Height']<=200) & (pheno_filter['Height']>=100)]
    pheno_indices = list(pheno_filter.index)
    geno_indices = set(list(genotypes.index)).intersection(pheno_indices)

    X = genotypes.ix[geno_indices] # n x p
    Y = phenotypes.ix[geno_indices] # n x 1


    # perform PCA
    #genotypes_pca = genotypes.loc[:, genotypes.isnull().mean() < 0.2]
    genotypes_pca = X
    genotypes_pca = genotypes_pca.fillna(0)
    gpca = PCA(n_components=10).fit(genotypes_pca).transform(genotypes_pca)
    U = DataFrame(gpca)
    G = DataFrame(np.cov(U.transpose()))
    #R = DataFrame(np.eye(U.shape[0]))
    #V = DataFrame(U.dot(G.dot(U.transpose())) + R)

    pickle.dump(X, open("X.p","wb"))
    pickle.dump(Y, open("Y.p","wb"))
    pickle.dump(U, open("U.p","wb"))
    pickle.dump(G, open("G.p","wb"))
    return X, Y, U, G

def get_matrices(pheno_name='Height'):
    genotypes = DataFrame(pickle.load(open("genotypes.p", "rb")))
    phenotypes = pickle.load(open("phenotypes.p", "rb"))
    X, Y, U, G = get_XY(genotypes, phenotypes, pheno_name)
    return X, Y, U, G

def test_matching_indices(genotypes, snp_info, phenotypes):
    from sklearn.decomposition import PCA
    genotypes = genotypes.fillna(0)
    a = PCA(n_components=4).fit(genotypes).transform(genotypes)
    pprint(a)
    print genotypes.shape
    print a.shape

    pprint(genotypes)
    pprint(snp_info)
    pprint(phenotypes)
    #combined = pandas.merge(genotypes, phenotypes, how='left', left_index=True, right_index=True)
    #pprint(combined)
    pheno_filter = phenotypes[phenotypes['Height'].notnull()]
    pprint(pheno_filter)
    pheno_filter2 = phenotypes[phenotypes['Eye color'].notnull()][['Eye color']]
    pprint(pheno_filter2)
    pheno_indices = pheno_filter2.index

    geno_filt2 = genotypes.ix[pheno_indices]
    pprint(geno_filt2)

    #print genotypes.ix['1013','rs999407']
    #print type(genotypes.ix['1013','rs999407'])

def main(args):
    args = parser.parse_args(args)

    if args.pheno_file is None:
        phenotypes = pickle.load(open("phenotypes.p", "rb"))
    else:
        phenotypes = process_phenotypes(args.pheno_file)

    if args.input_vcf is None:
        genotypes = DataFrame(pickle.load(open("genotypes.p", "rb")))
        snp_info = DataFrame(pickle.load(open("snp_info.p", "rb")))
    else:
        genotypes, snp_info = convert_to_matrix(args.input_vcf)

    #test_matching_indices(genotypes, snp_info, phenotypes)
    get_XY(genotypes, phenotypes, 'Height')

if __name__ == '__main__':
    main(sys.argv[1:])
