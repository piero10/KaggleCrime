import numpy as np

class Grid:
    maxX = 0
    minX = 0
    maxY = 0
    minY = 0

    stepX = 0
    stepY = 0

    stepX2 = 0
    stepX2 = 0

    stepsInEachAxis = 1

    grid = 0


    def CalcMaxMins(self, coordinates):
        maxX = coordinates[0,0]
        minX = coordinates[0,0]
        maxY = coordinates[0,1]
        minY = coordinates[0,1]

        for i in range(0, len(coordinates)):
            if coordinates[i, 0] > maxX:
                maxX = coordinates[i, 0]
            
            if coordinates[i, 0] < minX:
                minX = coordinates[i, 0]
            
            if coordinates[i, 1] > maxY:
                maxY = coordinates[i, 1]

            if coordinates[i, 1] < minY:
                minY = coordinates[i, 1]

        return (maxX, minX,  maxY, minY)



    def GetIndexesForCoord(self, coordinates):
        x = (coordinates[i, 0] - minX) / stepX2
        y = (coordinates[i, 1] - minY) / stepY2

        return(x, y)  



    def GetDataForCoord(self, coordinates):
        x, y = GetIndexesForCoord(coordinates)



    def SplitDataToGridCells(self, trainData, labels, firstCoordinateColumnTrainData):
        data = np.column_stack((trainData, labels))
        res = SplitDataToGridCells(data, firstCoordinateColumnTrainData)

        end = len(trainData) + len(labels)
        r1 = res[:,0:len(trainData)]
        r2 = res[:,len(trainData):end]

        return (r1, r2)



    def SplitDataToGridCells(self, trainData, firstCoordinateColumnTrainData):
        trainDataByCells = np.zeros_like(grid)
        indexes = np.zeros_like(grid)
        n = int(grid.shape[0]*grid.shape[1])

        for i in range(0, trainDataByCells.shape[0]):
            for j in range(0, trainDataByCells.shape[1]):
                trainDataByCells[i, j] = np.zeros(int(len(trainData) * 2/ n))
                indexes[i, j] = 0

        for i in range(0, len(trainData)):
            x, y = GetIndexesForCoord(trainData[firstCoordinateColumnTrainData], trainData[firstCoordinateColumnTrainData+1])
            curIndex = indexes[i, j]
            trainDataByCells[x, y][curIndex] = trainData[i]
            indexes[i, j] = indexes[i, j] + 1

        for i in range(0, trainDataByCells.shape[0]):
            for j in range(0, trainDataByCells.shape[1]):
                curIndex = indexes[i, j]
                trainDataByCells[i, j] = trainDataByCells[i, j][0:curIndex]

        return trainDataByCells



    def Create(self, coordinates, cellsNum = 3):
        maxX, minX, maxY, minY = CalcMaxMins(coordinates)

        stepX = (maxX - minX) / cellsNum
        stepY = (maxY - minY) / cellsNum

        stepX2 = stepX / 2
        stepX2 = stepX / 2

        stepsInEachAxis = cellsNum * 2 - 1

        grid = np.empty((stepsInEachAxis, stepsInEachAxis))
        coordInEachCell = np.empty((stepsInEachAxis, stepsInEachAxis), dtype = int)

        for i in range(0, stepsInEachAxis):
            for j in range(0, stepsInEachAxis):
                grid[i, j] = np.empty((len(coordinates / 3), 2))

        for i in range(0, len(coordinates)):
            x, y = GetIndexesForCoord(coordinates[i])

            coordInEachCell[x, y] = coordInEachCell[x, y] + 1
            grid[x,y][coordInEachCell[x, y]] = coordinates[i]

        for i in range(0, stepsInEachAxis):
            for j in range(0, stepsInEachAxis):
                grid[i, j] = grid[i, j][0:coordInEachCell[i, j]]
