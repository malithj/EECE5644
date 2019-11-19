import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def linear_interpolation(in_time_seq, out_time_seq, seq_in):
    seq_out = np.zeros((out_time_seq.shape[0], 2))
    for i in range(int(out_time_seq.shape[0])):
        for j in range(int(in_time_seq.shape[0])):
            if out_time_seq[i] < in_time_seq[j]:
                seq_out[i] = seq_in[j - 1] + ((seq_in[j] - seq_in[j - 1]) / (in_time_seq[j] - in_time_seq[j - 1])) * (out_time_seq[i] - in_time_seq[j - 1])
                break

    # An alternate way of computing; one liner ;)
    # delta_t = 2
    # seq_out = np.add(seq_in[:-1], np.divide(np.diff(seq_in, axis=0), delta_t))
    return seq_out


def cross_validate(true_y, predicted_y):
    return np.linalg.norm(predicted_y - true_y)

def get_kalman_sequence(measurements, K, S):
    """
    Computes Kalman Filter sequence
    :return: Kalman sequence as list
    """
    A = np.array([[1, 2, 2, 0, 0, 0], [0, 1, 2, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 2, 2], [0, 0, 0, 0, 1, 2], [0, 0, 0, 0, 0, 1]])
    C = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

    x0 = np.array([0, 0, 0, 0, 0, 0])
    var_w = K * np.identity(A.shape[0])
    var_m = S * np.identity(C.shape[0])
    seq = kalman_filter(A, C, x0, var_w, var_m, measurements)
    # return h & b
    return seq[:, [0, 3]]


def kalman_filter(A, C, x0, var_w, var_m, measurements):
    """
    Kalman Filtering
    A: Numpy array of coefficients
    C: Numpy array of coefficients
    x0: Initial estimate
    var_w: Variance of w
    var_w: Variance of m
    :return: sequence
    """
    sequence = np.array([])
    x_t_arr = np.array([np.array(x0)])
    # calculate the measurement mean and variance t = t
    x_t = np.mean(x_t_arr, axis=0)
    p_t = np.var(x_t_arr, axis=0)

    for idx, y_t_1 in enumerate(measurements.values):
        # update the measurement using prior measurement t = t + 1
        x_t_1_t = np.matmul(A, x_t)
        p_t_1_t = np.matmul(np.matmul(A, p_t), np.transpose(A)) + var_w

        # kalman gain matrix
        K_t_1 = np.matmul(np.matmul(p_t_1_t, np.transpose(C)), np.linalg.inv(np.matmul(np.matmul(C, p_t_1_t), np.transpose(C)) + var_m))

        # update x_t_1 given t_1 measurement
        x_t_1_t_1 = x_t_1_t + np.matmul(K_t_1, (y_t_1 - np.matmul(C, x_t_1_t)))
        p_t_1_t_1 = p_t_1_t - np.matmul(np.matmul(K_t_1, C), p_t_1_t)

        # append the new measurements
        x_t = x_t_1_t_1
        p_t = p_t_1_t_1
        sequence = np.append(sequence, x_t_1_t_1).reshape((-1, 6))
    return sequence


def main():
    """
    Initialize Program
    :return: 0
    """
    # plot the data
    train_data = pd.read_csv('Q2train.csv', header=None)
    train_data.columns = ['t', 'h', 'b']
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(train_data.h, train_data.b, c='red', marker='o', facecolor=None)
    plt.plot(train_data.h, train_data.b, c='k', linestyle='--')
    ax.grid(axis='both')
    ax.set_xlabel('h')
    ax.set_ylabel('b')
    ax.set_title('Object Position')
    plt.savefig('obj_position.png')

    # read test data
    test_data = pd.read_csv('Q2test.csv', header=None)
    test_data.columns = ['t', 'h', 'b']

    # Kalman filter
    result_matrix = np.zeros((np.arange(-2, 2, 0.5).shape[0], np.arange(-1.5, 1.5, 0.5).shape[0]))
    for i, K in enumerate(np.arange(-2, 2, 0.5)):
        for j, S in enumerate(np.arange(-1.5, 1.5, 0.5)):
            if [K, S] in [[-1, 1], [-0.5, 0.5], [0, 0], [0.5, -0.5], [1, -1], [1.5, -1.5]]:
                continue
            seq = get_kalman_sequence(train_data.iloc[:, 1:], K, S)

            # create time sequence
            in_time_seq = train_data.t
            time_seq = test_data.t

            # interpolation
            seq_out = linear_interpolation(in_time_seq, time_seq, seq)
            result_matrix[i][j] = cross_validate(test_data.iloc[:, 1:], seq_out)

    # contour plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    x = np.arange(-2.0, 2.0, 0.5)
    y = np.arange(-1.5, 1.5, 0.5)
    X, Y = np.meshgrid(x, y)
    levels = np.arange(10, 50, 5)
    cs = ax.contour(X, Y, np.transpose(result_matrix), levels=levels)
    idx = np.unravel_index(np.argmin(result_matrix[np.nonzero(result_matrix)], axis=None), result_matrix.shape)
    ax.annotate('Minimum: {0:.2f}'.format(np.amin(result_matrix[np.nonzero(result_matrix)])), xy=(x[idx[0]], y[idx[1]]), xytext=(-1.8, 0.46),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1))
    ax.clabel(cs, inline=1, fontsize=10)
    ax.set_xlabel("K")
    ax.set_ylabel("S")
    ax.set_title("Contour Plot : Cross Validation as K & S Vary")
    plt.savefig('contour-plot.png')

    # plot all data
    K = -2
    S = 0.9
    seq = get_kalman_sequence(train_data.iloc[:, 1:], K, S)
    # create time sequence
    in_time_seq = train_data.t
    time_seq = test_data.t
    # interpolation
    seq_out = linear_interpolation(in_time_seq, time_seq, seq)
    result_matrix[i][j] = cross_validate(test_data.iloc[:, 1:], seq_out)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(train_data.h, train_data.b, c='red', marker='o', facecolor=None, label='Train Data')
    ax.scatter(test_data.h, test_data.b, c='blue', marker='x', facecolor=None, label='Test Data')
    plt.plot(seq_out[:, 0], seq_out[:, 1], c='k', linestyle='--', label='Kalman Filter Output')
    ax.grid(axis='both')
    ax.set_xlabel('h')
    ax.set_ylabel('b')
    ax.set_title('Train & Test Object Position')
    plt.legend()
    plt.savefig('train_test_obj_position.png')
    plt.show()


if __name__ == '__main__':
    main()