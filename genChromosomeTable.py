import os
from subprocess import call
import glob
import argparse
import pandas as pd

def parse_args():
    """
    Handles input of chromosome number and path to search for on command line
    """
    parser = argparse.ArgumentParser(description="Chromosome # and file path to look in")
    parser.add_argument('chrom', nargs='+', help="Chromosome #")
    parser.add_argument('path',  nargs='+', help='file path')
    return parser.parse_args()


def generateChromosomeFile(chromNum, path):
    """
    Runs egrep command in path dir for all 23andme files returning all
    lines with chromNum

    chromNum: Chromosome number
    path: filePath to directory with 23andme files

    Example of run: python genChromosomeTable.py 21 ./opensnp_txt_data/
    """
    #call(["convert","-coalesce",gif_path[idx],"gifs/"+tag+"/"+str(idx)+"/%05d.png"])

    with open(path+chromNum+".txt","w") as f:
        f.write("rsID\tchromNum\tSNP\tAllele\n")
    params = ["egrep","-sh","\"rs\d.*\t"+chromNum+"\t\""]
    params.extend(glob.glob(path+'*23andme.txt'))
    params.extend(">>")
    params.extend([path+chromNum+".txt"])
    call(params)

    #call(["egrep","-sh","\"rs\d.*\t"+chromNum+"\t\"","*23andme.txt",">",path+"/"+chromNum+".txt"])
    #egrep -sh "rs\d.*\t22\t" *23andme.txt > chrom22.txt

def generateChromosomeDF(chromNum, path):
    """
    Returns pandas dataframe with all chromosome file info

    chromNum: Chromosome number
    path: filePath to directory with 23andme files

    Example of run: python genChromosomeTable.py 21 ./opensnp_txt_data/
    """
    df = pd.read_csv(filepath_or_buffer=path+chromNum+".txt",sep='\t', dtype=np.str)
    return df

if __name__ == '__main__':
    args = parse_args()
    chromNum = args.chrom[0]
    path = args.path[0]
    generateChromosomeFile(chromNum, path)
    df = generateChromosomeDF(chromNum, path)
