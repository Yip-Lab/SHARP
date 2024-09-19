import numpy as np
import argparse


def TRS(refer_TAD_list, detect_TAD_list, thres=2):
    f_refer = np.loadtxt(refer_TAD_list, dtype='str', delimiter='\t')
    f_detect = np.loadtxt(detect_TAD_list, dtype='str', delimiter='\t')

    overlap_array = np.zeros((len(f_refer), len(f_detect)))
    for j in range(0, len(f_refer)):
        start_point = int(f_refer[j][1]) / 5000
        end_point = int(f_refer[j][2]) / 5000
        for i in range(0, len(f_detect)):
            start_point2 = int(f_detect[i][1]) / 5000
            end_point2 = int(f_detect[i][2]) / 5000
            start_overlap_min = max(start_point, start_point2)
            start_overlap_max = min(end_point, end_point2)
            overlap_length = max(0, start_overlap_max - start_overlap_min)
            overlap_ratio = (overlap_length / (end_point - start_point) + overlap_length / (end_point2 - start_point2)) / 2
            overlap_array[j][i] = overlap_ratio

    correlation_matrix = overlap_array
    len_a, len_b = correlation_matrix.shape
    c = [-1] * len_a
    used_b = [False] * len_b

    all_correlations = []
    for i in range(0, len_a):
        for j in range(0, len_b):
            all_correlations.append((correlation_matrix[i][j], i, j))

    all_correlations.sort(reverse=True, key=lambda x: x[0])

    for value, i, j in all_correlations:
        if c[i] == -1 and not used_b[j]:
            c[i] = j
            used_b[j] = True

    close_to_boundary_overlap = 0
    block_size = 64
    a_countt = 0
    b_countt = 0
    for iii in range(0, len(f_refer)):
        start_point = int(f_refer[iii][1]) / 5000
        end_point = int(f_refer[iii][2]) / 5000
        if start_point % block_size >= (
                block_size - thres) or start_point % block_size <= thres or end_point % block_size >= (
                block_size - thres) or end_point % block_size <= thres:
            a_countt = a_countt + 1
            if c[iii] != -1:
                if correlation_matrix[iii][c[iii]] != 0:
                    close_to_boundary_overlap = close_to_boundary_overlap + correlation_matrix[iii][c[iii]]

    for jjj in range(0, len(f_detect)):
        start_point2 = int(f_detect[jjj][1]) / 5000
        end_point2 = int(f_detect[jjj][2]) / 5000
        if start_point2 % block_size >= (
                block_size - thres) or start_point2 % block_size <= thres or end_point2 % block_size >= (
                block_size - thres) or end_point2 % block_size <= thres:
            b_countt = b_countt + 1

    TRS = close_to_boundary_overlap / a_countt - b_countt / len(f_detect)
    print(TRS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TRS.")
    parser.add_argument('--refer_TAD_list', type=str, help='TAD detected on the actual high-resolution maps', default="/Users/qinyao/Desktop/Norm_results/DI_result/raw_GM12878/raw_chr20_new.txt")
    parser.add_argument('--detect_TAD_list', type=str, help='TAD detected on the reconstructed maps', default="/Users/qinyao/Desktop/Norm_results/DI_result/1_150/Single_chr20.txt")
    parser.add_argument('--thres', type=int, help='Threshold to judge whether a TAD close to a patch boundary', default=2)

    args = parser.parse_args()
    TRS(args.refer_TAD_list, args.detect_TAD_list, args.thres)