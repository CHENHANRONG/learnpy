from numpy import *


def loadDataSet(fileName):
    fr = open(fileName, mode='r')  # open file
    valuesLen = len(fr.readline().strip().split(sep='\t'))
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
        for i in range(valuesLen-1):  # len -1 is left properties part
            lineArr.append(float(currLine[i]))  # convert string to float values
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))  # last value is label data, convert to float
    return dataMat, labelMat


def standardLinearRegression(xArr,yArr):
    xMat = mat(xArr)
    trans_xMat = xMat.T  # xMat transpose
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    inverse_xTx = xTx.I
    if linalg.det(xMat) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws1 = inverse_xTx * (trans_xMat*yMat)  # ws = inverse(xTx)*(transpose(xMat)*yMat)
    ws = linalg.solve(xMat, yMat)
    return ws



fileName = "./data/ch08_line_reg_example_data01.txt"
dataMat, labelMat = loadDataSet(fileName)
ws = standardLinearRegression(dataMat, labelMat)
