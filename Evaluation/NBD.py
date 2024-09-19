import numpy as np
import argparse


def NBD(matrix):
    file_to_detect = np.load(matrix, allow_pickle=True)["matrix"]
    size = file_to_detect.shape[0]

    stride = 64
    bound = 321
    chunk_size = 64
    diff_sum_a = 0
    diff_sum_b = 0
    diff_sum_c = 0
    diff_sum_d = 0
    diff_mid_aa = 0
    diff_mid_bb = 0
    diff_mid_cc = 0
    diff_mid_dd = 0
    for i in range(0, size, stride):
        for j in range(0, size, stride):
            if abs(i - j) <= bound and i + chunk_size < size and j + chunk_size < size:
                boundary1 = file_to_detect[i, j:j + chunk_size]
                boundary2 = file_to_detect[i + chunk_size - 1, j:j + chunk_size]
                boundary3 = file_to_detect[i:i + chunk_size, j]
                boundary4 = file_to_detect[i:i + chunk_size, j + chunk_size - 1]
                if i > 0:
                    outside1 = file_to_detect[i - 1, j:j + chunk_size]
                    diff1 = np.sum(np.abs((outside1 - boundary1))) / chunk_size
                    mid = file_to_detect[i + 1, j:j + chunk_size]
                    diff_mid = np.sum(np.abs((mid - boundary1))) / chunk_size
                    diff_sum_a = diff_sum_a + diff1
                    diff_mid_aa = diff_mid_aa + diff_mid
                if i + chunk_size + 1 < size:
                    outside2 = file_to_detect[i + chunk_size, j:j + chunk_size]
                    diff2 = np.sum(np.abs((outside2 - boundary2))) / chunk_size
                    diff_sum_b = diff_sum_b + diff2
                    mid = file_to_detect[i + chunk_size - 2, j:j + chunk_size]
                    diff_mid = np.sum(np.abs((boundary2 - mid))) / chunk_size
                    diff_mid_bb = diff_mid_bb + diff_mid
                if j > 0:
                    outside3 = file_to_detect[i:i + chunk_size, j - 1]
                    diff3 = np.sum(np.abs((outside3 - boundary3))) / chunk_size
                    diff_sum_c = diff_sum_c + diff3
                    mid = file_to_detect[i:i + chunk_size, j + 1]
                    diff_mid = np.sum(np.abs((mid - boundary3))) / chunk_size
                    diff_mid_cc = diff_mid_cc + diff_mid
                if j + chunk_size + 1 < size:
                    outside4 = file_to_detect[i:i + chunk_size, j + chunk_size]
                    diff4 = np.sum(np.abs((outside4 - boundary4))) / chunk_size
                    diff_sum_d = diff_sum_d + diff4
                    mid = file_to_detect[i:i + chunk_size, j + chunk_size - 2]
                    diff_mid = np.sum(np.abs((mid - boundary4))) / chunk_size
                    diff_mid_dd = diff_mid_dd + diff_mid

    aa = (diff_sum_a - diff_mid_aa) / diff_mid_aa
    bb = (diff_sum_b - diff_mid_bb) / diff_mid_bb
    cc = (diff_sum_c - diff_mid_cc) / diff_mid_cc
    dd = (diff_sum_d - diff_mid_dd) / diff_mid_dd

    NBD = (aa+bb+cc+dd)/4
    print(NBD)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NBD.")
    parser.add_argument('--matrix', type=str, help='Path to the matrix', default="../Data/Output/SHARP_20_reconstructed.npz")

    args = parser.parse_args()
    NBD(args.matrix)


