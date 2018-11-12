# -*- coding: utf-8 -*
import cubicSpline
import matplotlib.pyplot as plt
import numpy as np
import util as utl

def pieceWiseCubicSplineGamma(s, n_initial, n_length, coeff):
    """
    各制御点毎のスプライン軌跡を計算

    input
    s:   s
    n_initial: 区分区間の開始インデックス
    n_length: 区分区間の点数
    coeff: 多項式係数

    output
    gamma(s): 位置の点列 list(n)
    """
    gamma = []
    dgamma = []
    ddgamma = []

    for i in range(n_length):

        gammatmp = coeff[0] + coeff[1] * (s[i+n_initial] - s[n_initial]) + coeff[2] * (s[i+n_initial] - s[n_initial])**2 + coeff[3] * (s[i+n_initial] - s[n_initial])**3
        dgammatmp = coeff[1] + 2.0 * coeff[2] * (s[i+n_initial] - s[n_initial]) + 3.0 * coeff[3] * (s[i+n_initial] - s[n_initial])**2
        ddgammatmp = 2.0 * coeff[2] + 6.0 * coeff[3] * (s[i+n_initial] - s[n_initial])

        gamma.append(gammatmp)
        dgamma.append(dgammatmp)
        ddgamma.append(ddgammatmp)

    return gamma, dgamma, ddgamma

if __name__ == "__main__":

    q_via_points = []

    q0 = [-110.0*np.pi/180.0, -40*np.pi/180.0, 210*np.pi/180.0, -35*np.pi/180.0, 20*np.pi/180.0, 30*np.pi/180.0]
    q1 = [-57.0*np.pi/180.0, -57.0*np.pi/180.0, 218.0*np.pi/180.0, -25.0*np.pi/180.0, 17.0*np.pi/180.0, 30.0*np.pi/180.0]
    q2 = [-14.0*np.pi/180.0, -85.0*np.pi/180.0, 238.0*np.pi/180.0, -11.0*np.pi/180.0, 12.0*np.pi/180.0, 30.0*np.pi/180.0]
    q3 = [28.0*np.pi/180.0, -69.0*np.pi/180.0, 229.0*np.pi/180.0, 2.38*np.pi/180.0, 8.3*np.pi/180.0, 30.0*np.pi/180.0]
    q4 = [115.0*np.pi/180.0, -35.0*np.pi/180.0, 210.0*np.pi/180.0, 15.0*np.pi/180.0, 20.0*np.pi/180.0, 30.0*np.pi/180.0]
    q_via_points.append(q0)
    q_via_points.append(q1)
    q_via_points.append(q2)
    q_via_points.append(q3)
    q_via_points.append(q4)
    cv = np.array(q_via_points)

    dof = cv.shape[1]
    v0 = np.zeros((1, dof))
    vn = np.zeros((1, dof))

    # 角関節の最大速度/加速度
    dq_limit = [328*np.pi/180.0, 300*np.pi/180.0, 375*np.pi/180.0, 375*np.pi/180.0, 375*np.pi/180.0, 600*np.pi/180.0]
    ddq_limit = [1400*np.pi/180.0, 900*np.pi/180.0, 1300*np.pi/180.0, 1800*np.pi/180.0, 1600*np.pi/180.0, 5000*np.pi/180.0]

    # 各区間毎のユークリッド距離を計算: S = [S0, S1, S2, ..., SN-1]
    S_euclid = utl.calcEuclidianDistanceList(cv)

    # 各区間ごとの s(t) in [S0, Se]におけるS0を格納したリストを作成: s_initial = [0, S0, S1, ..., SN-1]
    s_initial = [0.0]
    for i in range(len(S_euclid)):
        tmp = 0
        for j in range(i+1):
            tmp = tmp + S_euclid[j]
        s_initial.append(tmp)

    cs = cubicSpline.CubicSpline()

    # 補完計算のための各区間初期速度ベクトルを計算 (各区間の区分補完曲線が速度連続になるように速度を計算)
    vv = cs.calcVelocity(cv, v0, vn, S_euclid)

    # 各区間ごとの補完係数を計算
    coeffs = cs.calcCoeffsWithVelocity(cv, vv, S_euclid)

    # s空間の長さを計算
    s_end = np.sum(S_euclid)

    # 最適化計算のためにsを離散化
    n_section = 20 * len(S_euclid) # sを離散化するステップ数の総数 (区間数で割り切れる数を選ぶ) nは離散化した時の離散化区間の数に等しい
    n_point = n_section + 1
    n_per = int(n_section / len(S_euclid))

    # 補完区間ごとに離散化 (各区間ごとに n_per 分割 -> 全体の区間数は n で, 離散点数の総数は n+1になる)
    s_list = []
    for i in range(len(S_euclid)):
        if i == len(S_euclid)-1:
            for j in range(n_per):
                #s_list.append( s_initial[i] + j * S_euclid[i] / (n_per-1) )
                s_list.append( s_initial[i] + j * S_euclid[i] / (n_per) )
        else:
            for j in range(n_per):
                s_list.append( s_initial[i] + j * S_euclid[i] / (n_per) )
    s_list.append(s_end) # n_point

    N_list = [0]
    for i in range(len(S_euclid)):
        N_list.append(n_per * (i+1))
    if N_list[-1] != n_point-1:
        N_list.append(n_point-1) # 最後の要素を追加する処理を追加

    # サンプル点間ごとの差分を計算
    N_list_diff = []
    for i in range(len(N_list)-1):
        if i == len(N_list)-1-1:
            N_list_diff.append(n_section-N_list[i])
        else:
            N_list_diff.append(N_list[i+1]-N_list[i])

    # 各区間ごとの関節角度, 角速度, 各加速度を求める
    gamma = [[] for i in range(dof)]
    dgamma = [[] for i in range(dof)]
    ddgamma = [[] for i in range(dof)]
    gammatmp = [[] for i in range(dof)]
    dgammatmp = [[] for i in range(dof)]
    ddgammatmp = [[] for i in range(dof)]

    for i in range(len(S_euclid)):
        if i == len(S_euclid)-1:
            for j in range(dof):
                gammatmp[j], dgammatmp[j], ddgammatmp[j] = pieceWiseCubicSplineGamma(s_list, N_list[i], N_list_diff[i]+1, coeffs[i, j, :])
                gamma[j].extend(gammatmp[j])
                dgamma[j].extend(dgammatmp[j])
                ddgamma[j].extend(ddgammatmp[j])
        else:
            for j in range(dof):
                gammatmp[j], dgammatmp[j], ddgammatmp[j] = pieceWiseCubicSplineGamma(s_list, N_list[i], N_list_diff[i]+1, coeffs[i, j, :])
                gamma[j].extend(gammatmp[j][:-1])
                dgamma[j].extend(dgammatmp[j][:-1])
                ddgamma[j].extend(ddgammatmp[j][:-1])

    # numpyのarrayにしておく　(速いから)
    gamma = np.array(gamma)
    dgamma = np.array(dgamma)
    ddgamma = np.array(ddgamma)
    print(gamma.shape)
    print(dgamma.shape)
    print(ddgamma.shape)

    # 離散化間隔
    #h = s_list[1] - s_list[0] # 固定
    h = []
    for i in range(len(s_list)-1):
        h.append(s_list[i+1] - s_list[i])
    print(len(h))
    print(h[0])

    # 評価関数
    def func(x):
        tmp = 0.0
        for i in range(n_point-1):
            tmp = tmp + 2.0 * h[i] / (x[i]**(1/2) + x[i+1]**(1/2))

        print(tmp)

        return tmp

    # 初期解の設定
    import random
    import math
    xx = np.zeros(n_point) #bi, i=0,2,...n-1を説明変数とした最適化問題として解く
    #説明変数の物理的な意味は　v(s)**2　なので,初期解をそれっぽく
    initial_guess = [random.uniform(1.0*math.pi/180.0, 10.0*math.pi/180.0) for i in range(len(xx))]

    # 制約を追加
    d = []
    # Inequality means that it is to be non-negative. Note that COBYLA only supports inequality constraints.
    # リストで制約をどんどん追加していって最後にタプルにしても解ける

    #  速度制約を追加
    for i in range(dof):
        for j in range(n_point):
            d.append({'type': 'ineq', 'fun' : lambda x: np.array(-1.0 * (dgamma[i, j]**2 * x[j]) + dq_limit[i]**2) })
            d.append({'type': 'ineq', 'fun' : lambda x: np.array(1.0 * (dgamma[i, j]**2 * x[j]) ) })

    # 加速度制約を追加
    for i in range(dof):
        for j in range(n_point-1):
            d.append({'type': 'ineq', 'fun' : lambda x: np.array(-1.0 * (dgamma[i, j] * (x[j+1] - x[j]) / (2.0 * h[j]) + ddgamma[i, j] * x[j]) + ddq_limit[i] ) })
            d.append({'type': 'ineq', 'fun' : lambda x: np.array(1.0 * (dgamma[i, j] * (x[j+1] - x[j]) / (2.0 * h[j]) + ddgamma[i, j] * x[j]) + ddq_limit[i] ) })

    # 解は全部正
    for i in range(dof):
        for j in range(n_point):
            d.append({'type': 'ineq', 'fun' : lambda x: np.array(x[j]) })

    # 初期/終端条件を追加
    d.append({'type': 'eq', 'fun' : lambda x: np.array(x[0]) })
    d.append({'type': 'eq', 'fun' : lambda x: np.array(x[n_point-1]) })

    cons = tuple(d)

    # 制約あり
    import time
    start = time.time()
    from scipy.optimize import minimize
    res = minimize(func, initial_guess, constraints=cons, method='SLSQP')
    elapsed_time = time.time() - start
    print('Calculation time is: ' + str(elapsed_time))
    print("最適化完了")
    print(res)
    print("最適解:")
    print(res.x)
