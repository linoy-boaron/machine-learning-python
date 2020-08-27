import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr


data_r = np.random.binomial(1, 0.25, (100000, 1000))
epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
colors = []
div = np.arange(1, 1001)


mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([x_y[0]], [x_y[1]], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def part23():
    '''
    Generation random points
    '''
    plot_3d(x_y_z)
    plt.title("Q23: Generation random points ")
    plt.show()

def part24():
    '''
    Transform the data with the scaling matrix
    '''
    matrix = np.matrix([[0.1,0,0],[0,0.5,0],[0,0,2] ])
    data = matrix*x_y_z
    plot_3d(data)
    plt.title("Q24:Transform the data with the scaling matrix")
    plt.show()
    new_cov = np.cov(data)
    print(new_cov)


def part25():
    '''
     Multiply the scaled data by random orthogonal matrix
    '''
    matrix = np.matrix([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])
    random = get_orthogonal_matrix(3)
    data = random * matrix * x_y_z
    plot_3d(data)
    plt.title("Q25:Multiply the scaled data by random orthogonal matrix")
    plt.show()
    new_cov = np.cov(data)
    print(new_cov)


def part26():
    '''
    projection of the data to the x, y axes
    '''
    matrix = np.matrix([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])
    data = matrix * x_y_z
    plot_2d(data)
    plt.title("Q26 :projection of the data to the x, y axes")
    plt.show()

def part27():
    '''
    Projection of the data to the x,y axes for -0.4 < z < 0.1
    '''
    filtered_points = x_y_z[0:2, np.where(((-0.4 < x_y_z[2]) & (x_y_z[2] < 0.1)))]
    plot_2d(filtered_points)
    plt.title("Q27: Projection of the data to the x,y axes for -0.4 < z < 0.1")
    plt.show()

def part29_a():
    '''
    estimation of Xm as a function of m for the first 5 sequences of 1000 tosses
    '''
    a = np.cumsum(data_r[0]) / div
    b = np.cumsum(data_r[1]) / div
    c = np.cumsum(data_r[2]) / div
    d = np.cumsum(data_r[3]) / div
    e = np.cumsum(data_r[4]) / div
    plt.plot(a)
    plt.plot(b)
    plt.plot(c)
    plt.plot(d)
    plt.plot(e)
    plt.title("Q29a: estimation of Xm as a function of m for the first 5 sequences of 1000 tosses")
    plt.show()


def cher(x, ep):
    '''
    Chevishev bound
    '''
    y = (float(1) / (4 * (ep * ep * x)))
    return y


def hofd(x, ep):
    '''
    Hoeffding bound
    '''
    y = (2.0 * np.exp(-2 * ep * ep * x))
    return y


def part29b():
    '''
    For each bound (Chebyshev and Hoeffding) and for each epsilon-plot the upper bound on
    probability of inequality in the ex as a function of m
     '''
    m = np.arange(1, 1001)
    for i in epsilon:
        plt.title("Q29b: Upper bound for epsilon = $ %s $" % str(i))
        plt.ylabel("Upper Bound")
        plt.xlabel("Num of tosses, m")
        che = [(1 if cher(m, i) > 1 else cher(m, i)) for m in m]
        hof = [(1 if hofd(m, i) > 1 else hofd(m, i)) for m in m]
        plt.plot(m, che, label='Chevishev')
        plt.plot(m, hof, label='Hoeffding')
        plt.legend()
        plt.show()


def part29c():
    '''
    percentage of sequences that satisfy the inequality in the ex as a function of m when  the expected_value = 0.25
    '''
    m = np.arange(1, 1001)
    expected_value = 0.25
    for i in epsilon:
        m_data = np.cumsum(data_r, axis=1)/m
        p = np.where(np.abs(m_data - expected_value) >= i, 1, 0)
        plt.plot(np.sum(p, axis=0)/100000)
        chevishev = [(1 if cher(m, i) > 1 else cher(m, i)) for m in m]
        hoeffding = [(1 if hofd(m, i) > 1 else hofd(m, i)) for m in m]
        plt.plot(m, hoeffding, label='Hoeffding')
        plt.plot(m, chevishev, label='Chevishev')
        plt.title("Q29c:percentage of sequences with p=0.25, epsilon=$ %s $ " % str(i))
        plt.ylabel("percentage of the tosses")
        plt.xlabel("Num of tosses, m")
        plt.legend()
        plt.show()




