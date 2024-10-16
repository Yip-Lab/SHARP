import numpy as np
import argparse


def Decomposition(chr_in, k_para, matrix, output_path, use_blacklist, black_list1, black_list2):
    high = np.load(matrix, allow_pickle=True)["matrix"]
    high = high.astype('float64')
    ######fist_stage parameters
    biass = 10
    cpoint = 0
    basic_size = 10
    add_bias_maxsize = 400
    step = 10
    resolution = 5000
    size = high.shape[0]

    log_filename_record = output_path + 'chr' + str(chr_in) + '_record.txt'
    log_fp_record = open(log_filename_record, 'w')

    for i in range(size):
        for j in range(size):
            index = np.abs(i - j) + 1
            normal_value = k_para * np.power(index * resolution, -3 / 2) * np.exp(-1400 / (index * resolution * index * resolution))  #3000000
            high[i][j] = high[i][j] - normal_value
            if high[i][j] < 0:
                high[i][j] = 0

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

    size2 = compacted_matrix.shape[0]

    list_block = []
    while cpoint < size2 - basic_size:
        basic_size = 10
        cpoint_start_flag = 0
        if cpoint > 10:
            basic_left_line = compacted_matrix[cpoint: cpoint + 3, cpoint: cpoint + basic_size].mean()
            compare_left_line = compacted_matrix[cpoint - 10: cpoint, cpoint: cpoint + basic_size].mean()
            if basic_left_line - compare_left_line > 0:
                cpoint_start_flag = 1
            else:
                basic_left_line2 = compacted_matrix[cpoint: cpoint + 3, cpoint: cpoint + 2 * basic_size].mean()
                compare_left_line2 = compacted_matrix[cpoint - 10: cpoint, cpoint: cpoint + 2 * basic_size].mean()
                if basic_left_line2 - compare_left_line2 > 0:
                    cpoint_start_flag = 1
                    basic_size = 20
                else:
                    basic_left_line3 = compacted_matrix[cpoint: cpoint + 3, cpoint: cpoint + 3 * basic_size].mean()
                    compare_left_line3 = compacted_matrix[cpoint - 10: cpoint, cpoint: cpoint + 3 * basic_size].mean()
                    if basic_left_line3 - compare_left_line3 > 0:
                        cpoint_start_flag = 1
                        basic_size = 40
        else:
            if compacted_matrix[cpoint: cpoint + 10, cpoint: cpoint + 10].mean() > 0.05:
                cpoint_start_flag = 1
                basic_size = 10
        if cpoint_start_flag == 1:  # start now!
            change_flag = 0
            for add_bias in range(10, add_bias_maxsize, step):
                if cpoint + basic_size + add_bias + step < size2 and cpoint - 15 >= 0:
                    add_candidate_lower = compacted_matrix[cpoint: cpoint + basic_size + add_bias - step, cpoint + basic_size + add_bias - step: cpoint + basic_size + add_bias].mean()
                    # compare_candidate_lower = compacted_matrix[cpoint: cpoint + basic_size + add_bias - step, cpoint - 10: cpoint].mean()
                    compare_candidate_lower = compacted_matrix[cpoint: cpoint + basic_size + add_bias - step, cpoint - 15: cpoint-2].mean()
                    candidate_mean = compacted_matrix[cpoint: cpoint + basic_size + add_bias - step, cpoint: cpoint + basic_size + add_bias - step].mean()
                    out_candidate_lower = compacted_matrix[cpoint: cpoint + basic_size + add_bias - step, cpoint + basic_size + add_bias: cpoint + basic_size + add_bias + step].mean()
                    if add_candidate_lower < compare_candidate_lower:
                        in_mean = np.mean(compacted_matrix[cpoint: cpoint + basic_size + add_bias, cpoint: cpoint + basic_size + add_bias])
                        if in_mean > 0:
                            list_block.append({'start_point': cpoint, 'end_point': cpoint + basic_size + add_bias})
                            change_flag = 1
                        break
                    elif candidate_mean < out_candidate_lower * 1.3:
                        in_mean = np.mean(compacted_matrix[cpoint: cpoint + basic_size + add_bias, cpoint: cpoint + basic_size + add_bias])
                        if in_mean > 0:
                            list_block.append({'start_point': cpoint, 'end_point': cpoint + basic_size + add_bias})
                            change_flag = 1
                        # break
            if change_flag == 0:
                list_block.append({'start_point': cpoint, 'end_point': cpoint + basic_size})
        cpoint = cpoint + 1


    list_array = np.zeros((len(list_block), 2))
    for kkk in range(len(list_block)):
        list_array[kkk][0] = list_block[kkk]['start_point']
        list_array[kkk][1] = list_block[kkk]['end_point']

    list_array = np.unique(list_array, axis=0)

    list_array2 = np.zeros((len(list_array), 3))
    for kkk in range(len(list_array2)):
        list_array2[kkk][0] = list_array[kkk][0]
        list_array2[kkk][1] = list_array[kkk][1]
        start_point = int(list_array2[kkk][0])
        end_point = int(list_array2[kkk][1])
        list_array2[kkk][2] = end_point - start_point


    list_sorted = sorted(list_array2, key=lambda x: x[2], reverse=False)

    for removal_index in range(len(list_sorted)):
        start_point = int(list_sorted[removal_index][0])
        end_point = int(list_sorted[removal_index][1])
        in_mean = np.mean(compacted_matrix[start_point: end_point, start_point: end_point])
        if in_mean > 0.05:
            block_length = end_point - start_point
            upper_half = np.sum(np.triu(np.flip(compacted_matrix[start_point: end_point, start_point: end_point], 1), 1)) / (block_length * (block_length - 1) / 2)
            lower_half = np.sum(np.tril(np.flip(compacted_matrix[start_point: end_point, start_point: end_point], 1), -1)) / (block_length * (block_length - 1) / 2)
            if upper_half > lower_half * 1.5:
                part1_mean = np.sum(np.tril(np.flip(compacted_matrix[start_point: end_point, start_point: end_point], 1), -1)) / (block_length * (block_length - 1) / 2)
                part2_mean = np.sum(np.triu(np.flip(compacted_matrix[start_point: end_point, start_point: end_point], 1), 1)) / (block_length * (block_length - 1) / 2)
                if part1_mean > 0:
                    diff = part2_mean - part1_mean
                    upper_tri = np.flip(np.triu(np.flip(compacted_matrix[start_point: end_point, start_point: end_point], 1), 1), 1) * diff / part2_mean
                    compacted_matrix[start_point: end_point, start_point: end_point] = compacted_matrix[start_point: end_point, start_point: end_point] - upper_tri
                    in_mean2 = np.mean(compacted_matrix[start_point: end_point, start_point: end_point])
                    if in_mean2 > 0:
                        if end_point + biass < size2:
                            out_block1_mean = np.mean(compacted_matrix[start_point: end_point, end_point: end_point + biass])
                        else:
                            out_block1_mean = np.mean(compacted_matrix[start_point: end_point, start_point - biass: start_point])
                        if start_point - biass > 0:
                            out_block2_mean = np.mean(compacted_matrix[start_point - biass: start_point, start_point: end_point])
                        else:
                            out_block2_mean = np.mean(compacted_matrix[end_point: end_point + biass, start_point: end_point])
                        out_mean2 = np.mean([out_block1_mean, out_block2_mean])
                        compacted_matrix[start_point: end_point, start_point: end_point] = compacted_matrix[start_point: end_point, start_point: end_point] * (out_mean2 / in_mean2)
                    else:
                        in_mean2 = out_mean2 = 1
                    log_fp_record.write(
                        "type: U " + "out_mean: " + str(part1_mean) + " " + "in_mean: " + str(
                            part2_mean) + " " + "out_mean2: " + str(out_mean2) + " " + "in_mean2: " + str(
                            in_mean2) + " " + "start_x: " + str(start_point) + " " + "start_y: " + str(start_point) + " " + "end_x: " + str(end_point)
                        + " " + "end_y: " + str(end_point) + '\n')
                    log_fp_record.flush()


            elif lower_half > upper_half * 1.5:
                part1_mean = np.sum(np.triu(np.flip(compacted_matrix[start_point: end_point, start_point: end_point], 1), 1)) / (block_length * (block_length - 1) / 2)
                part2_mean = np.sum(np.tril(np.flip(compacted_matrix[start_point: end_point, start_point: end_point], 1), -1)) / ((block_length) * (block_length - 1) / 2)
                if part2_mean > 0:
                    diff = part2_mean - part1_mean
                    lower_tri = np.flip(np.tril(np.flip(compacted_matrix[start_point: end_point, start_point: end_point], 1), -1), 1) * diff / part2_mean
                    compacted_matrix[start_point: end_point, start_point: end_point] = compacted_matrix[start_point: end_point, start_point: end_point] - lower_tri
                    in_mean2 = np.mean(compacted_matrix[start_point: end_point, start_point: end_point])
                    if in_mean2 > 0:
                        if end_point + biass < size2:
                            out_block1_mean = np.mean(compacted_matrix[start_point: end_point, end_point: end_point + biass])
                        else:
                            out_block1_mean = np.mean(compacted_matrix[start_point: end_point, start_point - biass: start_point])
                        if start_point - biass > 0:
                            out_block2_mean = np.mean(compacted_matrix[start_point - biass: start_point, start_point: end_point])
                        else:
                            out_block2_mean = np.mean(compacted_matrix[end_point: end_point + biass, start_point: end_point])
                        out_mean2 = np.mean([out_block1_mean, out_block2_mean])
                        compacted_matrix[start_point: end_point, start_point: end_point] = compacted_matrix[start_point: end_point,start_point: end_point] * (out_mean2 / in_mean2)
                    else:
                        in_mean2 = out_mean2 = 1
                    log_fp_record.write(
                        "type: L " + "out_mean: " + str(part1_mean) + " " + "in_mean: " + str(
                            part2_mean) + " " + "out_mean2: " + str(out_mean2) + " " + "in_mean2: " + str(
                            in_mean2) + " " + "start_x: " + str(start_point) + " " + "start_y: " + str(start_point) + " " +
                        "end_x: " + str(end_point) + " " + "end_y: " + str(end_point) + '\n')
                    log_fp_record.flush()
                log_fp_record.flush()
            else:
                if end_point + biass < size2:
                    out_block1_mean = np.mean(compacted_matrix[start_point: end_point, end_point: end_point + biass])
                else:
                    out_block1_mean = np.mean(compacted_matrix[start_point: end_point, start_point - biass: start_point])
                if start_point - biass > 0:
                    out_block2_mean = np.mean(compacted_matrix[start_point - biass: start_point, start_point: end_point])
                else:
                    out_block2_mean = np.mean(compacted_matrix[end_point: end_point + biass, start_point: end_point])
                out_mean = np.mean([out_block1_mean, out_block2_mean])
                if out_mean > 0:
                    compacted_matrix[start_point: end_point, start_point: end_point] = compacted_matrix[start_point: end_point, start_point: end_point] * (out_mean / in_mean)
                    log_fp_record.write(
                        "type: F " + "out_mean: " + str(out_mean) + " " + "in_mean: " + str(
                            in_mean) + " " + "out_mean2: " + str(1) + " " + "in_mean2: " + str(
                            1) + " " + "start_x: " + str(
                            start_point) + " " + "start_y: " + str(start_point) + " " + "end_x: " + str(
                            end_point) + " " + "end_y: " + str(end_point) + '\n')
                    log_fp_record.flush()


    # second_stage parameters
    block_size = 5
    second_step = 1
    add_bias_maxsize2 = 300
    step2 = 5
    biass = 20

    i_index = 0
    while i_index < size2-block_size:
        j_index = i_index + 1
        while j_index < size2-block_size:
            if i_index >= block_size and j_index >= block_size:
                block = compacted_matrix[i_index: i_index + block_size, j_index: j_index + block_size].mean() #up, left
                compare_block_1 = compacted_matrix[i_index - block_size: i_index, j_index: j_index + block_size].mean() #up
                compare_block_2 = compacted_matrix[i_index: i_index + block_size, j_index - block_size: j_index].mean() #left
                if block > compare_block_1 and block > compare_block_2 and compare_block_1 > 0 and compare_block_2 > 0 and block > 0.05:
                    for add_bias_i in range(5, add_bias_maxsize2, step2):
                        add_i_block = compacted_matrix[i_index + add_bias_i + block_size - step2: i_index + add_bias_i + block_size, j_index: j_index + block_size].mean()
                        i_block_mean = compacted_matrix[i_index: i_index + add_bias_i + block_size - step2, j_index: j_index + block_size].mean()
                        i_length = i_index + add_bias_i + block_size - step2
                        if add_i_block > 0.05 and add_i_block > compare_block_1 and add_i_block > compare_block_2 and i_index + add_bias_i + block_size < j_index and i_block_mean <= add_i_block:
                            i_length = i_index + add_bias_i + block_size
                        else:
                            break
                    for add_bias_j in range(5, add_bias_maxsize2, step2):
                        add_j_block = compacted_matrix[i_index: i_length, j_index + add_bias_j + block_size - step2: j_index + add_bias_j + block_size].mean()
                        j_block_mean = compacted_matrix[i_index: i_length, j_index: j_index + add_bias_j + block_size - step2].mean()
                        j_length = j_index + add_bias_j + block_size - step2
                        if add_j_block > 0.05 and add_j_block > compare_block_2 and add_j_block > compare_block_2 and j_block_mean <= add_j_block:
                            j_length = j_index + add_bias_j + block_size
                        else:
                            break
                    if i_length + biass < size and j_length + biass < size:
                        out_block1_mean = compacted_matrix[i_index: i_length, j_length: j_length + biass].mean()
                        out_block2_mean = compacted_matrix[i_length: i_length + biass, j_index: j_length].mean()
                    else:
                        out_block1_mean = compacted_matrix[i_index: i_length, j_index - biass: j_index].mean()
                        out_block2_mean = compacted_matrix[i_index - biass: i_index, j_index: j_length].mean()
                    if i_index - biass >= 0 and j_index - biass >= 0:
                        out_block3_mean = compacted_matrix[i_index: i_length, j_index - biass: j_index].mean()
                        out_block4_mean = compacted_matrix[i_index - biass: i_index, j_index: j_length].mean()
                    else:
                        out_block3_mean = compacted_matrix[i_index: i_length, j_length: j_length + biass].mean()
                        out_block4_mean = compacted_matrix[i_length: i_length + biass, j_index: j_length].mean()

                    out_mean = np.mean([out_block1_mean, out_block2_mean, out_block3_mean, out_block4_mean])
                    in_mean = np.mean(compacted_matrix[i_index: i_length, j_index: j_length])
                    if out_mean < in_mean and out_mean > 0:
                        compacted_matrix[i_index: i_length, j_index: j_length] = compacted_matrix[i_index: i_length, j_index: j_length] * out_mean / in_mean
                        log_fp_record.write(
                            "type: O " + "out_mean: " + str(out_mean) + " " + "in_mean: " + str(
                                in_mean) + " " + "out_mean2: " + str(1) + " " + "in_mean1: " + str(
                                1) + " " + "start_x: " + str(i_index) + " " + "start_y: " + str(j_index)
                            + " " + "end_x: " + str(i_length) + " " + "end_y: " + str(j_length) + '\n')
                        log_fp_record.flush()
            j_index = j_index + second_step
        i_index = i_index + second_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decomposition script for chromosome matrices.')
    parser.add_argument('--chr', type=int, help='chromosome', default=20)
    parser.add_argument('--k_para', type=int, help='K paratemer used for signal decomposition', default=3500000)
    parser.add_argument('--matrix', type=str, help='Path to the matrix', default='../Data/1_150/chr20_low_matrix.npz')
    parser.add_argument('--output_path', type=str, help='Output path', default='../Data/1_150/')
    parser.add_argument('--use_blacklist', type=bool, help='Whether to use the blacklist', default='True')
    parser.add_argument('--black_list1', nargs='+', type=int, help='Blacklist array1')
    parser.add_argument('--black_list2', nargs='+', type=int, help='Blacklist array2')

    args = parser.parse_args()
    Decomposition(args.chr, args.k_para, args.matrix, args.output_path, args.use_blacklist, args.black_list1, args.black_list2)
    # python Decomposition.py --chr 20 --k_para 3500000 --matrix "../Data/1_150/chr20_low_matrix.npz" --output_path "../Data/1_150/" --use_blacklist True --black_list1 0 5273 5788 6199 9579 --black_list2 14 5783 5853 6249 9580



