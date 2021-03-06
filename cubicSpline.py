# -*- coding: utf-8 -*-
import numpy as np

class CubicSpline:

    def __init__(self):
        pass

    def calcEuclidianDistanceList(self, c):
        """
        点列cvから各点列間のユークリッド距離を計算し, リストに格納して返す
        c = np.array of 3d control vertices
        """

        N = len(c)
        T = []
        for i in range(N-1):
            T.append(np.linalg.norm(c[i+1]-c[i]))

        return T


    def calcCoeffsWithVelocity(self, c, v, S):
        """
        位置と速度の点列が与えられた時の各係数を計算
        c: 位置の点列 (n, dim)
        v: 速度の点列 (n, dim)
        """

        coeffs = np.zeros((c.shape[0]-1, c.shape[1], 3+1))
        for i in range(c.shape[0]-1):
            for j in range(c.shape[1]):
                    coeffs[i, j, 0] = c[i, j]
                    coeffs[i, j, 1] = v[i, j]
                    coeffs[i, j, 2] = 1 / S[i] * ( 3 * (c[i+1, j] - c[i, j]) / S[i] - 2 * v[i, j] - v[i+1, j] )
                    coeffs[i, j, 3] = 1 / S[i]**2 * ( 2 * (c[i, j] - c[i+1, j]) / S[i] + v[i, j] + v[i+1, j] )

        return coeffs

    def calcVelocity(self, c, v0, vn, S):
        """
        input
        位置と速度の点列が与えられた時の各係数を計算
        c: 位置の点列 (n, dim)
        v0: 位置の点列 (1, dim)
        vn: 位置の点列 (1, dim)

        output
        v: 速度の点列 (n, dim)

        """

        dims = c.shape[1]
        v = np.zeros((len(S)-1, dims))

        for i in range(dims):

            A = np.zeros((len(S)-1, len(S)-1))
            b = np.zeros(len(S)-1)

            for j in range(len(S)-1):
                A[j, j] = 2.0 * (S[j] + S[j+1] )

            for j in range(len(S)-2):
                A[j, j+1] = S[j]
                A[j+1, j] = S[j+2]

            for j in range(len(S)-1):
                if j == 0:
                    b[j] = 3.0 / (S[j] * S[j+1] ) * (S[j]**2 * (c[j+2, i] - c[j+1, i]) + S[j+1]**2 * (c[j+1, i] - c[j, i])) - S[j+1] * v0[0, i]
                elif j == len(S)-2:
                    b[j] = 3.0 / (S[j] * S[j+1] ) * (S[j]**2 * (c[j+2, i] - c[j+1, i]) + S[j+1]**2 * (c[j+1, i] - c[j, i])) - S[j+1] * vn[0, i]
                else:
                    b[j] = 3.0 / (S[j] * S[j+1] ) * (S[j]**2 * (c[j+2, i] - c[j+1, i]) + S[j+1]**2 * (c[j+1, i] - c[j, i]))

            vtmp = np.linalg.solve(A, b)
            vtmp = vtmp.reshape(vtmp.shape[0], 1)
            v[:, i] = vtmp[:, 0]

        v = np.vstack((v0, v))
        v = np.vstack((v, vn))

        return v

    def pieceWiseCubicSpline(self, sk, Tk, coeff, n):
        """
        各制御点毎のスプライン軌跡を計算

        input
        sk: 軌道開始時の独立変数の値 (時間 or ユークリッド距離)
        Tk: 独立変数の長さ (時間 or ユークリッド距離)
        vn: 位置の点列 (1, dim)

        output
        q: 位置の点列 list(n)
        s: 独立変数の点列 list(n)
        """
        q = []
        dq = []
        ddq = []
        s = []

        for i in range(n):

            s.append(sk + Tk / n * i) #　sを等間隔で分割

            qtmp = coeff[0] + coeff[1] * (s[i] - sk) + coeff[2] * (s[i] - sk)**2 + coeff[3] * (s[i] - sk)**3
            dqtmp = coeff[1] + 2.0 * coeff[2] * (s[i] - sk) + 3.0 * coeff[3] * (s[i] - sk)**2
            ddqtmp = 2.0 * coeff[2] + 6.0 * coeff[3] * (s[i] - sk)

            q.append(qtmp)
            dq.append(dqtmp)
            ddq.append(ddqtmp)

        return q, dq, ddq, s

    def cubicSpline1dTimeSeries(self, cv, v0, vn, n):
        """
        スプライン曲線の全体の軌道を計算

        input
        c: 位置の点列 (n, dim)
        v0: 位置の点列 (1, dim)
        vn: 位置の点列 (1, dim)
        n: 制御点間の点数

        output
        y:   位置
        dy:  速度
        ddy: 加速度
        """
        t = cv[:, 0].tolist()
        T = []
        for i in range(len(t)-1):
            T.append(t[i+1] - t[i])

        vv = self.calcVelocity(cv, v0, vn, T)
        coeffs = self.calcCoeffsWithVelocity(cv, vv, T)

        for i in range(len(T)):
            if i == 0:
                y, dy, ddy, tt = self.pieceWiseCubicSpline(t[i], T[i], coeffs[i, 1, :], n)

            else:
                ytmp, dytmp, ddytmp, tttmp = self.pieceWiseCubicSpline(t[i], T[i], coeffs[i, 1, :], n)

                y.extend(ytmp)
                dy.extend(dytmp)
                ddy.extend(ddytmp)

                tt.extend(tttmp)

        return y, dy, ddy, tt, coeffs

    def cubicSplineANYd(self, cv, v0, vn, n, closed):
        """
        スプライン曲線の全体の軌道を計算

        input
        c: 位置の点列 (n, dim)
        v0: 位置の点列 (1, dim)
        vn: 位置の点列 (1, dim)
        n: 制御点間の点数

        output
        x, dx, ddx, ss
        """

        S = self.calcEuclidianDistanceList(cv)
        s = [0.0]
        for i in range(len(S)):
            tmp = 0
            for j in range(i+1):
                tmp = tmp + S[j]
            s.append(tmp)

        vv = self.calcVelocity(cv, v0, vn, S)
        coeffs = self.calcCoeffsWithVelocity(cv, vv, S)

        x = [[] for i in range(cv.shape[1])]
        dx = [[] for i in range(cv.shape[1])]
        ddx = [[] for i in range(cv.shape[1])]
        for i in range(len(S)):
            if i == 0:
                for j in range(len(x)):
                        x[j], dx[j], ddx[j], ss = self.pieceWiseCubicSpline(s[i], S[i], coeffs[i, j, :], n)

            else:

                for j in range(len(x)):
                    xtmp, dxtmp, ddxtmp, sstmp = self.pieceWiseCubicSpline(s[i], S[i], coeffs[i, j, :], n)

                    x[j].extend(xtmp)
                    dx[j].extend(dxtmp)
                    ddx[j].extend(ddxtmp)
                ss.extend(sstmp)

        return x, dx, ddx, ss, coeffs

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def test1dTimeSeries():
        """
        y=f(t)のスプライン補完を試す関数
        """
        cv = np.array([[ 0.0,  3.0],
                       [ 0.1,  1.0],
                       [ 0.2,  0.5],
                       [ 0.3,  1.5],
                       [ 0.4,  0.5],
                       [ 0.5,  3.0],
                       [ 0.6,  1.5],
                       [ 0.7,  0.5],
                       [ 0.8,  3.0],
                       [ 0.9,  0.5],
                       [ 1.0,  3.0]
                       ])

        v0 = np.zeros((1, 2))
        vn = np.zeros((1, 2))

        cs = CubicSpline()
        y, dy, ddy, t, coeffs = cs.cubicSpline1dTimeSeries(cv, v0, vn, n=100)

        import matplotlib.pyplot as plt
        plt.subplot(311)
        plt.plot(cv[:, 0], cv[:, 1], 'o-', label='Control Points')
        plt.plot(t, y)
        plt.grid()
        plt.subplot(312)
        plt.plot(t, dy)
        plt.grid()
        plt.subplot(313)
        plt.plot(t, ddy)
        plt.grid()

        plt.show()

    def testANYdGeometricPath():
        """
        3次元のスプライン補完を試す関数 f(x,y,z) = 0
        """
        cv = np.array([[ 50.,  25.,  -1.],
                       [ 59.,  12.,  -1.],
                       [ 50.,  10.,   1.],
                       [ 57.,   2.,   1.],
                       [ 40.,   4.,   1.],
                       [ 40.,   14.,  -1.]
                       ])

        v0 = np.zeros((1, 3))
        vn = np.zeros((1, 3))

        # 軌道を計算
        npoints = 100
        cs = CubicSpline()
        x, dx, ddx, ss, coeffs = cs.cubicSplineANYd(cv, v0, vn, npoints, closed=False)

        #
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        fig = plt.figure(0)
        ax = fig.gca(projection='3d')
        ax.plot(cv[:, 0], cv[:, 1], cv[:, 2], 'o-', label='Control Points')
        ax.plot(x[0], x[1], x[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(35, 70)
        ax.set_ylim(0, 30)
        ax.set_zlim(-1.5, 2.0)

        plt.show()

    #
    test1dTimeSeries()
    #testANYdGeometricPath()

