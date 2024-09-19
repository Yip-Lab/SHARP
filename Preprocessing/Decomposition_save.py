import numpy as np
import argparse


def Decomposition_save(chr_in, k_para, matrix, record, output_path, use_blacklist, black_list1, black_list2):
    high = np.load(matrix, allow_pickle=True)["matrix"]
    high = high.astype('float64')
    resolution = 5000
    size = high.shape[0]


    chr_in = int(chr_in)
    for i in range(size):
        for j in range(size):
            index = np.abs(i - j) + 1
            normal_value = k_para * np.power(index * resolution, -3 / 2) * np.exp(-1400 / (index * resolution * index * resolution))  #3000000
            high[i][j] = high[i][j] - normal_value

    if use_blacklist:
        listt = []
        for i in range(len(black_list1)):
            for j in range(black_list1[i], black_list2[i]+1):
                listt.append(j)
        all = list(np.arange(0, size, 1))
        compact_index_list = list(set(all).difference(set(listt)))
        compact_index = np.array(compact_index_list)

        size2 = compact_index.shape[0]
        compacted_matrix = np.zeros((size2, size2))
        for i, idx in enumerate(compact_index):
            compacted_matrix[i, :] = high[idx][compact_index]
    else:
        compacted_matrix = high


    info = []
    with open(record, 'r') as f:
        for line in f.readlines():
            info.append(line)
    for i in range(len(info)):
        type = info[i].split(" ")[1]
        out_mean = float(info[i].split(" ")[3])
        in_mean = float(info[i].split(" ")[5])
        out_mean2 = float(info[i].split(" ")[7])
        in_mean2 = float(info[i].split(" ")[9])
        start_x = int(info[i].split(" ")[11])
        start_y = int(info[i].split(" ")[13])
        end_x = int(info[i].split(" ")[15])
        end_y = int(info[i].split(" ")[17])
        if type == 'F':
            compacted_matrix[start_x: end_x, start_y: end_y] = compacted_matrix[start_x: end_x, start_y: end_y] * out_mean / in_mean
        elif type == 'O':
            compacted_matrix[start_x: end_x, start_y: end_y] = compacted_matrix[start_x: end_x, start_y: end_y] * out_mean / in_mean
        elif type == 'U':
            compacted_matrix[start_x: end_x, start_y: end_y] = compacted_matrix[start_x: end_x, start_y: end_y] * out_mean2 / in_mean2
            diff = in_mean - out_mean
            upper_tri = np.flip(np.triu(np.flip(compacted_matrix[start_x: end_x, start_y: end_y], 1), 1), 1) * diff / in_mean
            compacted_matrix[start_x: end_x, start_y: end_y] = compacted_matrix[start_x: end_x, start_y: end_y] - upper_tri
        elif type == 'L':
            compacted_matrix[start_x: end_x, start_y: end_y] = compacted_matrix[start_x: end_x, start_y: end_y] * out_mean2 / in_mean2
            diff = in_mean - out_mean
            lower_tri = np.flip(np.tril(np.flip(compacted_matrix[start_x: end_x, start_y: end_y], 1), -1), 1) * diff / in_mean
            compacted_matrix[start_x: end_x, start_y: end_y] = compacted_matrix[start_x: end_x, start_y: end_y] - lower_tri

    if use_blacklist:
        for i, s_idx in enumerate(compact_index):
            high[s_idx, compact_index] = compacted_matrix[i]
    else:
        high = compacted_matrix


    block_size = 64
    size_parameter = 5
    low_resolution = []
    size = high.shape[0]
    for ii in range(0, size, block_size):
        for jj in range(ii, size, block_size):
            if ii + block_size < size and jj + block_size < size and np.abs(jj - ii) <= size_parameter*block_size:
                low_resolution.append(high[ii: ii + block_size, jj: jj + block_size])

    np.savez_compressed(output_path + str(chr_in) + '_Type3_patches.npz', matrix=low_resolution)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decomposition script for chromosome matrices.')
    parser.add_argument('--chr', type=int, help='chromosome', default=20)
    parser.add_argument('--k_para', type=int, help='K paratemer used for signal decomposition', default=3500000)
    parser.add_argument('--matrix', type=str, help='Path to the matrix', default="../Data/1_150/chr20_low_matrix.npz")
    parser.add_argument('--output_path', type=str, help='Output path', default="../Data/1_150/")
    parser.add_argument('--record_file', type=str, help='Record file', default="../Data/1_150/chr20_record.txt")
    parser.add_argument('--use_blacklist', type=bool, help='Whether to use the blacklist', default='True')
    parser.add_argument('--black_list1', nargs='+', type=int, help='Blacklist array1')
    parser.add_argument('--black_list2', nargs='+', type=int, help='Blacklist array2')

    args = parser.parse_args()
    Decomposition_save(args.chr, args.k_para, args.matrix, args.record_file, args.output_path, args.use_blacklist, args.black_list1, args.black_list2)
