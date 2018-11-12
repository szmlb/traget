# -*- coding: utf-8 -*-
import numpy as np

def calcEuclidianDistanceList(c):
    """
    点列cvから各点列間のユークリッド距離を計算し, リストに格納して返す
    c = np.array of 3d control vertices
    """

    N = len(c)
    T = []
    for i in range(N-1):
        T.append(np.linalg.norm(c[i+1]-c[i]))

    return T
