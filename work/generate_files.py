import os
import pandas as pd
import glob
from utils import *
import argparse

def generate_train(dirRoot):
    files = sorted(glob.glob(os.path.join(dirRoot, '*.jpg')))
    tot = len(files)
    trainFileList = []
    testFileList = []

    trainFileList = files[:int(tot * 0.8)]
    testFileList = files[int(tot * 0.8):]

    fileCSV = pd.DataFrame(trainFileList)
    # print(fileCSV)
    fileCSV.to_csv(dirRoot.replace('images', 'train.csv'), sep=',', index=False, header=False)

    fileCSV = pd.DataFrame(testFileList)
    # print(fileCSV)
    fileCSV.to_csv(dirRoot.replace('images', 'test.csv'), sep=',', index=False, header=False)

def generate_test(dirRoot):
    files = sorted(glob.glob(os.path.join(dirRoot, '*.jpg')))
    testFileList = []
    for fileName in files:
        testFileList.append(fileName)

    fileCSV = pd.DataFrame(testFileList)
    # print(fileCSV)
    fileCSV.to_csv(dirRoot.replace('images', 'test.csv'), sep=',', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainData', type=int, default=1,
                        help='1 shows to generate train data')
    parser.add_argument('--root', type=str, default='data/dehw_train_dataset/images',
                        help='1 shows to generate train data')                    
    args = parser.parse_args()
    
    if args.trainData:
        generate_train(args.root)
    else:
        generate_train(args.root)
