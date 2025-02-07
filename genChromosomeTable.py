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

    chromNum=str(chromNum)
    #Add columns
    with open(path+"chrom"+chromNum+".txt","w") as f:
        f.write("user\trsID\tchromNum\tSNP\tAllele\n")

    #Generate file with info from each other file
    print("Running grep")
    grep=subprocess.Popen("egrep -s \"rs\d.*\t%s\t\" %s*23andme.txt >> %schrom%s.txt" %
                        (chromNum, path, path, chromNum),shell=True, stdout=subprocess.PIPE)
    grep.communicate()

    #Clean generate file of misc bad data
    #   1. Some lines have 5 values instead of 4
    print("Creating file header")
    with open(path+"chrom"+chromNum+".txt","r") as f:
        lines = f.readlines()

    print("Writing lines to file")
    with open(path+"chrom"+chromNum+".txt","w") as f:
        pattern = re.compile('\S.*\t\S.*\t\S.*\t\S.*\t\S.*\t\S.*')
        for l in lines:
            if not(pattern.match(l)):
                splitLine=l.split("\t")
                if splitLine[1]=="rsID":
                    f.write(l)
                    continue
                userFile = splitLine[0]; userFile=userFile.split("/")[-1]
                userFile = [userFile.split("_")[0]]
                rsID = [splitLine[0].split(":")[-1]]
                userFile.extend(rsID)
                userFile.extend(splitLine[1:])
                l = "\t".join(userFile)
                f.write(l)

def generateChromosomeDF(chromNum, path):
    """
    Returns pandas dataframe with all chromosome file info

    chromNum: Chromosome number
    path: filePath to directory with 23andme files

    Example of run: python genChromosomeTable.py 21 ./opensnp_txt_data
    """
    df = pd.read_csv(filepath_or_buffer=path+"chrom"+str(chromNum)+".txt",delim_whitespace=True, dtype=np.str)
    return df

if __name__ == '__main__':
    args = parse_args()
    chromNum = args.chrom[0]
    path = args.path[0]
    generateChromosomeFile(chromNum, path)
    #df = generateChromosomeDF(chromNum, path)
