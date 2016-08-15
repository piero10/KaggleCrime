allPossibleAlgorithms2 = {1 : [3, 2],
                          2 : [8, 2],
                          3:  [2, 2],
                          4 : [4, 2],
                          5 : [10, 2],
                          6:  [2, 3]}



allPossibleAlgorithms ={1 : ['rf', 20, 100, 3],
                        2 : ['rf', 30, 100, 5],
                        3 : ['rf', 20, 100, 5],
                        4 : ['rf', 20, 50, 5],
                        5 : ['gb', 200],
                        6 : ['gb', 1000],
                        7 : ['gb', 50],
                        8 : ['rf', 30, 100, 3],
                        9 : ['rf', 10, 100, 3],
                        10 : ['rf', 20, 100, 2],
                        11 : ['rf', 20, 200, 3],
                        12 : ['gb', 50],
                        13 : ['gb', 100],
                        14 : ['gb', 2000],
                        15 : ['gb', 4000],}

featureAlgos = [    [0,  1, False],
                    [1,  2, False],
                    [2,  3, True], # very small
                    [3,  3, True], # very small
                    [4,  2, False],
                    [5,  4, False],
                    [6,  1, False],
                    [7,  5, False],
                    [8,  1, False],
                    [9,  1, False],
                    [10, 3, True], # very small
                    [11, 3, True], # very small
                    [12, 6, False],
                    [13, 4, False],
                    [14, 4, True], #fixed. very small
                    [15, 3, False],
                    [16, 5, False],
                    [17, 4, False],
                    [18, 1, False],
                    [19, 5, False],
                    [20, 2, False],
                    [21, 2, False],
                    [22, 3, True], # fixed. very small
                    [23, 2, False],
                    [24, 1, False],
                    [25, 1, False],
                    [26, 3, False],
                    [27, 4, False],
                    [28, 4, False],
                    [29, 3, True], # very small
                    [30, 3, False],
                    [31, 3, False],
                    [32, 4, False],
                    [33, 3, True], # fixed. very small
                    [34, 4, False],
                    [35, 2, False],
                    [36, 5, False],
                    [37, 2, False],
                    [38, 4, False]]
# [model, algo, fixed]
"""featureAlgos = [    [0, 1, False],
                    [1, 2, False],
                    [2, 1, True], # very small
                    [3, 1, True], # very small
                    [4, 3, False],
                    [5, 6, False],
                    [6, 6, False],
                    [7, 2, False],
                    [8, 6, False],
                    [9, 1, False],
                    [10, 1, True], # very small
                    [11, 1, True], # very small
                    [12, 1, False],
                    [13, 6, False],
                    [14, 1, True], #fixed. very small
                    [15, 5, False],
                    [16, 2, False],
                    [17, 6, False],
                    [18, 2, False],
                    [19, 2, False],
                    [20, 2, False],
                    [21, 3, False],
                    [22, 1, True], # fixed. very small
                    [23, 2, False],
                    [24, 1, False],
                    [25, 1, False],
                    [26, 4, False],
                    [27, 5, False],
                    [28, 1, False],
                    [29, 1, True], # very small
                    [30, 5, False],
                    [31, 1, False],
                    [32, 2, False],
                    [33, 1, True], # fixed. very small
                    [34, 6, False],
                    [35, 1, False],
                    [36, 2, False],
                    [37, 6, False],
                    [38, 1, False]]"""


