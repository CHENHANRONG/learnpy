import numpy as np


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
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    if np.linalg.det(np.transpose(xMat) * xMat) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # xTx = np.transpose(xMat) * xMat  # xT * x
    # solve x * w = y => xT*x * w = xTy => solve w = inverse(transpose(x) * x) * transpose(x) * y
    # ws1 = np.linalg.inv(np.transpose(xMat) * xMat) * ( np.transpose(xMat) * yMat)  # ws = inverse(xTx)*(transpose(xMat)*yMat)
    # solve x * w= y => xT*x * w = xTy => solve w
    ws = np.linalg.solve(np.transpose(xMat) * xMat, np.transpose(xMat) * yMat)
    return ws



fileName = "./data/ch08_line_reg_example_data01.txt"
dataMat, labelMat = loadDataSet(fileName)
ws = standardLinearRegression(dataMat, labelMat)
