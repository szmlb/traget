"""
B-spline
textbook: Trajectory Planning for Automatic Machines and Robots, pp. 194-208
"""

import numpy as np
import math

class BSpline:

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

    # Simple Cox - DeBoor recursion
    def coxDeBoor(self, u, knots, j, p):

        # Test for end conditions
        if (p == 0):
            if (knots[j] <= u and u < knots[j+1]):
                return 1
            return 0

        Den1 = knots[j+p] - knots[j]
        Den2 = knots[j+p+1] - knots[j+1]
        Eq1 = 0.0
        Eq2 = 0.0

        if Den1 > 0:
            Eq1 = ((u-knots[j]) / Den1) * self.coxDeBoor(u, knots, j, p-1)
        if Den2 > 0:
            Eq2 = ((knots[j+p+1]-u) / Den2) * self.coxDeBoor(u, knots, j+1, p-1)

        return Eq1 + Eq2

    # Simple Cox - DeBoor recursion for derivative of basis function
    def coxDeBoorDerivative(self, u, knots, j, p, der):

        a = np.zeros([der+1, der+1])
        a[0, 0] = 1.0
        for k in range(1, der+1):
            Den = knots[j+p-k+1] - knots[j]
            if Den > 0.0:
                a[k, 0] = a[k-1, 0] / Den
        for k in range(1, der+1):
            for i in range(1, der):
                Den = knots[j+p+i-k+1] - knots[j+i]
                if Den > 0.0:
                    a[k, i] = (a[k-1, i] - a[k-1, i-1]) / Den
        Den = knots[j+p+1] - knots[j+der]
        if Den > 0.0:
            a[der, der] = - a[der-1, der-1] / Den

        tmp = 0
        for i in range(der+1):
            tmp = tmp + a[der, i] * self.coxDeBoor(u, knots, j+i, p-der)
        tmp = tmp * math.factorial(p) / math.factorial(p-der)

        return tmp

    def calcCoeffBspline(self, s, n, u, p, p0, pn, v0, vn, a0, an, cv, dim):

        # 係数行列を生成
        m = n + p
        A = np.zeros([3*2+n-1, 3*2+n-1])
        c = np.zeros([3*2+n-1])

        # Aに代入
        for i in range(m+1):
            A[0, i] = self.coxDeBoor(s[0], u, i, p)
            A[1, i] = self.coxDeBoorDerivative(s[0], u, i, p, der=1)
            A[2, i] = self.coxDeBoorDerivative(s[0], u, i, p, der=2)

            #u=umaxの時は特別な対処が必要 (textbook p.470参照)
            eps = 0.001
            A[-3, i] = self.coxDeBoor(s[-1]-eps, u, i, p)
            A[-2, i] = self.coxDeBoorDerivative(s[-1]-eps, u, i, p, der=1)
            A[-1, i] = self.coxDeBoorDerivative(s[-1]-eps, u, i, p, der=2)
        for i in range(n-1):
            for j in range(m+1):
                A[i+3, j] = self.coxDeBoor(s[i+1], u, j, p)

        # cに代入
        c[0] = p0[dim]
        c[1] = v0[dim]
        c[2] = a0[dim]
        c[-3] = pn[dim]
        c[-2] = vn[dim]
        c[-1] = an[dim]
        for i in range(n-1):
            c[i+3] = cv[i+1, dim]

        # 係数を計算
        pcoeff = np.linalg.solve(A, c)

        return pcoeff

    def bspline1dTimeSeries(self, cv, v0, a0, vn, an, n=100):
        """
        cv = np.array of control vertices
        v0 = np.array of initial velocities
        a0 = np.array of initial acceleration
        vn = np.array of final velocities
        an = np.array of final acceleration
        n = number of samples (default: 100)
        p = curve degree (default: 4)
        """

        ### pが偶数(p=4)の時に特化して実装　 (mの値,  A, cのサイズあたりをpに合わせて宣言するようにすれば対応できる)
        p = 4
        ### pが奇数の時はknotsベクトルの表現方法を変える. この時mの値も変わる (textbook: p.195参照)
        count = len(cv) - 1 # = n in textbook

        u = np.zeros(2*p+count+2)
        for i in range(p+1):
            u[i] = cv[0, 0]
        for i in range(count):
            u[i+p+1] = (cv[i+1, 0] + cv[i, 0]) / 2.0
        for i in range(p+1):
            u[i+p+1+count] = cv[-1, 0]

        # 係数行列を生成
        t = cv[:, 0]
        m = count + p
        A = np.zeros([3*2+count-1, 3*2+count-1])
        c = np.zeros([3*2+count-1])

        # Aに代入
        for i in range(m+1):
            A[0, i] = self.coxDeBoor(t[0], u, i, p)
            A[1, i] = self.coxDeBoorDerivative(t[0], u, i, p, der=1)
            A[2, i] = self.coxDeBoorDerivative(t[0], u, i, p, der=2)

            #u=umaxの時は特別な対処が必要 (textbook p.470参照)
            eps = 0.001
            A[-3, i] = self.coxDeBoor(t[-1]-eps, u, i, p)
            A[-2, i] = self.coxDeBoorDerivative(t[-1]-eps, u, i, p, der=1)
            A[-1, i] = self.coxDeBoorDerivative(t[-1]-eps, u, i, p, der=2)
        for i in range(count-1):
            for j in range(m+1):
                A[i+3, j] = self.coxDeBoor(t[i+1], u, j, p)

        # cに代入
        c[0] = cv[0, 1]
        c[1] = v0[0]
        c[2] = a0[0]
        c[-3] = cv[-1, 1]
        c[-2] = vn[0]
        c[-1] = an[0]
        for i in range(count-1):
            c[i+3] = cv[i+1, 1]

        # 係数を計算
        pcoeff = np.linalg.solve(A, c)

        # スプライン軌道を計算
        y = []
        dy = []
        ddy = []
        tsample = np.linspace(0,  t[-1],  n+1)
        for i in range(len(tsample)):
            tmp = 0
            tmp_der = 0
            tmp_der2 = 0
            for j in range(m+1):
                if i == len(tsample)-1:
                    eps = 0.001
                    tmp = tmp + pcoeff[j] * self.coxDeBoor(tsample[i]-eps, u, j, p)
                    tmp_der = tmp_der + pcoeff[j] * self.coxDeBoorDerivative(tsample[i]-eps, u, j, p, der=1)
                    tmp_der2 = tmp_der2 + pcoeff[j] * self.coxDeBoorDerivative(tsample[i]-eps, u, j, p, der=2)
                else:
                    tmp = tmp + pcoeff[j] * self.coxDeBoor(tsample[i], u, j, p)
                    tmp_der = tmp_der + pcoeff[j] * self.coxDeBoorDerivative(tsample[i], u, j, p, der=1)
                    tmp_der2 = tmp_der2 + pcoeff[j] * self.coxDeBoorDerivative(tsample[i], u, j, p, der=2)
            y.append(tmp)
            dy.append(tmp_der)
            ddy.append(tmp_der2)

        return y,  dy,  ddy, tsample,  pcoeff

    def calcBspline(self, s, sample_num, pcoeff, u, n, p):

        m = n + p

        # s(t)の計画
        ssample = np.linspace(0,  s[-1],  sample_num+1) # 等分割で決定

        # スプライン軌道を計算
        y = []
        dy = []
        ddy = []
        for i in range(len(ssample)):
            tmp = 0
            tmp_der = 0
            tmp_der2 = 0
            for j in range(m+1):
                if i == len(ssample)-1:
                    eps = 0.001
                    tmp = tmp + pcoeff[j] * self.coxDeBoor(ssample[i]-eps, u, j, p)
                    tmp_der = tmp_der + pcoeff[j] * self.coxDeBoorDerivative(ssample[i]-eps, u, j, p, der=1)
                    tmp_der2 = tmp_der2 + pcoeff[j] * self.coxDeBoorDerivative(ssample[i]-eps, u, j, p, der=2)
                else:
                    tmp = tmp + pcoeff[j] * self.coxDeBoor(ssample[i], u, j, p)
                    tmp_der = tmp_der + pcoeff[j] * self.coxDeBoorDerivative(ssample[i], u, j, p, der=1)
                    tmp_der2 = tmp_der2 + pcoeff[j] * self.coxDeBoorDerivative(ssample[i], u, j, p, der=2)
            y.append(tmp)
            dy.append(tmp_der)
            ddy.append(tmp_der2)

        return y, dy, ddy, ssample

    def bspline2d(self, cv, v0, a0, vn, an, sample_num=100, p=4):
        """
        cv = np.array of 3d control vertices
        v0 = np.array of initial velocities
        a0 = np.array of initial acceleration
        vn = np.array of final velocities
        an = np.array of final acceleration
        sample_num = number of samples (default: 100)
        p = curve degree (default: 4)
        """

        ### pが偶数(p=4)の時に特化して実装　 (mの値,  A, cのサイズあたりをpに合わせて宣言するようにすれば対応できる)
        ### pが奇数の時はknotsベクトルの表現方法を変える. この時mの値も変わる (textbook: p.195参照)
        n = len(cv) - 1 # = n in textbook

        S = self.calcEuclidianDistanceList(cv)
        s = [0.0]
        for i in range(len(S)):
            tmp = 0
            for j in range(i+1):
                tmp = tmp + S[j]
            s.append(tmp)

        u = np.zeros(2*p+n+2)
        for i in range(p+1):
            u[i] = s[0]
        for i in range(n):
            u[i+p+1] = (s[i+1] + s[i]) / 2.0
        for i in range(p+1):
            u[i+p+1+n] = s[-1]

        pcoeffx = self.calcCoeffBspline(s, n, u, p, cv[0, :], cv[-1, :], v0, vn, a0, an, cv, 0)
        pcoeffy = self.calcCoeffBspline(s, n, u, p, cv[0, :], cv[-1, :], v0, vn, a0, an, cv, 1)

        x, dx, ddx, s = self.calcBspline(s, sample_num, pcoeffx, u, n, p)
        y, dy, ddy, s = self.calcBspline(s, sample_num, pcoeffy, u, n, p)

        return x, y, dx, dy, ddx, ddy, s

    def bspline3d(self, cv, v0, a0, vn, an, sample_num=100, p=4):
        """
        cv = np.array of 3d control vertices
        v0 = np.array of initial velocities
        a0 = np.array of initial acceleration
        vn = np.array of final velocities
        an = np.array of final acceleration
        sample_num = number of samples (default: 100)
        p = curve degree (default: 4)
        """

        ### pが偶数(p=4)の時に特化して実装　 (mの値,  A, cのサイズあたりをpに合わせて宣言するようにすれば対応できる)
        ### pが奇数の時はknotsベクトルの表現方法を変える. この時mの値も変わる (textbook: p.195参照)
        n = len(cv) - 1 # = n in textbook

        S = self.calcEuclidianDistanceList(cv)
        s = [0.0]
        for i in range(len(S)):
            tmp = 0
            for j in range(i+1):
                tmp = tmp + S[j]
            s.append(tmp)

        u = np.zeros(2*p+n+2)
        for i in range(p+1):
            u[i] = s[0]
        for i in range(n):
            u[i+p+1] = (s[i+1] + s[i]) / 2.0
        for i in range(p+1):
            u[i+p+1+n] = s[-1]

        pcoeffx = self.calcCoeffBspline(s, n, u, p, cv[0, :], cv[-1, :], v0, vn, a0, an, cv, 0)
        pcoeffy = self.calcCoeffBspline(s, n, u, p, cv[0, :], cv[-1, :], v0, vn, a0, an, cv, 1)
        pcoeffz = self.calcCoeffBspline(s, n, u, p, cv[0, :], cv[-1, :], v0, vn, a0, an, cv, 2)

        x, dx, ddx, s = self.calcBspline(s, sample_num, pcoeffx, u, n, p)
        y, dy, ddy, s = self.calcBspline(s, sample_num, pcoeffy, u, n, p)
        z, dz, ddz, s = self.calcBspline(s, sample_num, pcoeffz, u, n, p)

        return x, y, z, dx, dy, dz, ddx, ddy, ddz, s

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def test1dTimeSeries():

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

        v0 = np.zeros(1)
        vn = np.zeros(1)
        a0 = np.zeros(1)
        an = np.zeros(1)

        v0[0] = 0.0
        vn[0] = 0.0
        a0[0] = 0.0
        vn[0] = 0.0

        bs = BSpline()
        y,  dy,  ddy, t, coeff = bs.bspline1dTimeSeries(cv, v0, a0, vn, an, n=100)
        cv = cv.T
        plt.subplot(311)
        plt.plot(cv[0], cv[1], 'o-', label='Control Points')
        plt.plot(t, y)
        plt.grid()
        plt.subplot(312)
        plt.plot(t, dy)
        plt.grid()
        plt.subplot(313)
        plt.plot(t, ddy)
        plt.grid()

        plt.show()

    def test2d():
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

        v0 = np.zeros([2, 1])
        vn = np.zeros([2, 1])
        a0 = np.zeros([2, 1])
        an = np.zeros([2, 1])

        bs = BSpline()
        x, y, dx, dy, ddx, ddy, s = bs.bspline2d(cv, v0, a0, vn, an, sample_num=100, p=4) #速度の条件が全く設定できない

        import matplotlib.pyplot as plt
        plt.figure(0)
        plt.plot(cv[:, 0], cv[:, 1], 'o-', label='Control Points')
        plt.plot(x, y)
        plt.grid()

        plt.show()

    def test3d():
        cv = np.array([[ 50.,  25.,  -1.],
                       [ 59.,  12.,  -1.],
                       [ 50.,  10.,   1.],
                       [ 57.,   2.,   1.],
                       [ 40.,   4.,   1.],
                       [ 40.,   14.,  -1.]])


        v0 = np.zeros([3, 1])
        vn = np.zeros([3, 1])
        a0 = np.zeros([3, 1])
        an = np.zeros([3, 1])

        #p = bspline(cv,n=100,d=3,closed=closed)
        bs = BSpline()
        x, y, z, dx, dy, dz, ddx, ddy, ddz, s = bs.bspline3d(cv, v0, a0, vn, an, sample_num=100, p=4) #速度の条件が全く設定できない

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        fig = plt.figure(0)
        ax = fig.gca(projection='3d')
        ax.plot(cv[:, 0], cv[:, 1], cv[:, 2], 'o-', label='Control Points')
        ax.plot(x, y, z)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(35, 70)
        ax.set_ylim(0, 30)
        ax.set_zlim(-1.5, 2.0)
        plt.show()

    test1dTimeSeries()
    #test2d()
    #test3d()

