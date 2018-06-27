from numpy import *


def loadDataSet(fileName):
    fr = open(fileName,mode='r')  # open file
    lines = fr.readlines()
    # numLine = len(lines)  # lines.__len__()
    # print(numLine)
    # print(type(lines))
    dataMat = []
    labelMat = []
    # print(type(dataMat))
    for xline in lines:
        currLine = xline.strip().split(sep='\t')
        lineArr = []
        # print(type(currLine))
        for i in range(len(currLine)-1):  # len -1 is left properties part
            lineArr.append(float(currLine[i]))  # convert string to float values
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))  # last value is label data, convert to float
    return dataMat, labelMat




fileName = "./data/ch08_line_reg_example_data01.txt"
loadDataSet(fileName)
