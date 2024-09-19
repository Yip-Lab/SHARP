import numpy as np
import argparse
import os


def Preprocessing(path2matrix):
    matrix = np.load(path2matrix, allow_pickle=True)["matrix"]
    arrayy = matrix.reshape(matrix.shape[0] * matrix.shape[0])
    arrayy = arrayy.astype('int64')
    number_array = np.bincount(arrayy)
    zero_count = 0
    max_memory = 0
    for i in range(number_array.shape[0]):
        if number_array[i] != 0:
            zero_count = 0
            max_memory = i
        else:
            zero_count += 1
        if zero_count >= 10:
            matrix[np.where(matrix > max_memory)] = max_memory
            matrix = matrix.astype('float64')
            np.savez_compressed(path2matrix, matrix=matrix)
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing script for chromosome matrices.')
    parser.add_argument('--path2matrix', type=str, help='Path to the matrix to be preprocessed.', default='../Data/chr20_high_matrix.npz')

    args = parser.parse_args()
    Preprocessing(args.path2matrix)

    ####Usage:
    # python Preprocessing.py --path2matrix '../Data/chr20_high_matrix.npz' --chr_list 20 21 --rate_list 150 100
