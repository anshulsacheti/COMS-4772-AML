import os
import subprocess
import glob
import argparse
import pandas as pd
import numpy as np
import re

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
    lines with chromNum in new file at path/chromNum.txt
    Removes some lines that cause problems (some files have lines with 5 values)

    chromNum: Chromosome number
    path: filePath to directory with 23andme files

    Example of run: python genChromosomeTable.py 21 ./opensnp_txt_data/
    """
    #call(["convert","-coalesce",gif_path[idx],"gifs/"+tag+"/"+str(idx)+"/%05d.png"])

    #Add columns
    with open(path+"chrom"+chromNum+".txt","w") as f:
        f.write("rsID\tchromNum\tSNP\tAllele\n")

    #Generate file with info from each other file
    grep=subprocess.Popen("egrep -sh \"rs\d.*\t%s\t\" %s*23andme.txt >> ./testSet/chrom%s.txt" %
                        (chromNum, path, chromNum),shell=True, stdout=subprocess.PIPE)
    grep.communicate()

    #Clean generate file of misc bad data
    #   1. Some lines have 5 values instead of 4
    with open(path+"chrom"+chromNum+".txt","r") as f:
        lines = f.readlines()

    with open(path+"chrom"+chromNum+".txt","w") as f:
        pattern = re.compile('\S.*\t\S.*\t\S.*\t\S.*\t\S')
        for l in lines:
            if not(pattern.match(l)):
                f.write(l)

def generateChromosomeDF(chromNum, path):
    """
    Returns pandas dataframe with all chromosome file info

    chromNum: Chromosome number
    path: filePath to directory with 23andme files

    Example of run: python genChromosomeTable.py 21 ./opensnp_txt_data/
    """
    df = pd.read_csv(filepath_or_buffer=path+"chrom"+chromNum+".txt",sep='\t', dtype=np.str)
    return df

if __name__ == '__main__':
    args = parse_args()
    chromNum = args.chrom[0]
    path = args.path[0]
    generateChromosomeFile(chromNum, path)
    df = generateChromosomeDF(chromNum, path)
    return df
