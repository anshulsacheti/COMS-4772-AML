import os
from subprocess import call
import glob
import argparse

def parse_args():
    """
    Handles input of chromosome number and path to search for on command line
    """
    parser = argparse.ArgumentParser(description="Chromosome # and file path to look in")
    parser.add_argument('chrom', nargs='+', help="Chromosome #")
    parser.add_argument('path',  nargs='+', help='file path')
    return parser.parse_args()


def generateChromosomeTable(chromNum, path):
    """
    Runs egrep command in path dir for all 23andme files returning all
    lines with chromNum

    chromNum: Chromosome number
    path: filePath to directory with 23andme files
    """
    #call(["convert","-coalesce",gif_path[idx],"gifs/"+tag+"/"+str(idx)+"/%05d.png"])
    params = ["egrep","-sh","\"rs\d.*\t"+chromNum+"\t\""]
    params.extend(glob.glob(path+'*23andme.txt'))
    params.extend(">")
    params.extend([path+chromNum+".txt"])
    call(params)

    #call(["egrep","-sh","\"rs\d.*\t"+chromNum+"\t\"","*23andme.txt",">",path+"/"+chromNum+".txt"])
    #egrep -sh "rs\d.*\t22\t" *23andme.txt > chrom22.txt


if __name__ == '__main__':
    args = parse_args()
    chromNum = args.chrom[0]
    path = args.path[0]
    generateChromosomeTable(chromNum, path)
