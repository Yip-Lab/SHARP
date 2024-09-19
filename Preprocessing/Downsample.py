import numpy as np
import argparse
import os


def Downsample(chr_list, rate_list):
    for rate_index in rate_list:
        rate = 1/rate_index
        os.mkdir('../Data/1_' + str(rate_index))
        for ii_index in chr_list:
            matrix = np.load('../Data/chr' + str(ii_index) + '_high_matrix.npz', allow_pickle=True)["matrix"]
            matrix = matrix.astype(int)
            result = np.random.binomial(matrix, rate)
            low_tril = np.triu(result, 1)
            low_triu = np.triu(result, 0)
            result = low_triu + low_tril.T
            result = result.astype(int)

            #Normalization for the low-resolution matrices
            arrayy = result.reshape(result.shape[0]*result.shape[0])
            arrayy = arrayy.astype('int64')
            number_array = np.bincount(arrayy)
            zero_count = 0
            max_memory = 0
            for i in range(number_array.shape[0]):
                if number_array[i] != 0:
                    zero_count = 0
                    max_memory = i
                else:
                    zero_count = zero_count + 1
                if zero_count >= 10:
                    result[np.where(result>max_memory)] = max_memory
                    break
            result = result.astype('float64')
            np.savez_compressed('../Data/1_' + str(rate_index) + '/chr' + str(ii_index) + '_low_matrix.npz', matrix=result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing script for chromosome matrices.')
    parser.add_argument('--chr_list', nargs='+', type=int, help='List of chromosome numbers')
    parser.add_argument('--rate_list', nargs='+', type=int, help='List of downsample rate')

    args = parser.parse_args()
    Downsample(args.chr_list, args.rate_list)