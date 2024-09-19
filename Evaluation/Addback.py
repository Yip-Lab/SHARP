import numpy as np
import torch
import argparse
from torch.utils.data import Dataset


class MAXIM_Dataset_multiplechr_test(Dataset):
    def __init__(self, low_matrix, patch):
        super(MAXIM_Dataset_multiplechr_test, self).__init__()
        block_size = 64
        size_parameter = 5
        self.start_point = []
        self.end_point = []
        self.low_resolution = np.load(patch, allow_pickle=True)["matrix"]
        size = low_matrix.shape[0]
        for ii in range(0, size, block_size):
            for jj in range(ii, size, block_size):
                if ii + block_size < size and jj + block_size < size and np.abs(jj - ii) <= size_parameter*block_size:
                    self.start_point.append(ii)
                    self.end_point.append(jj)

    def __len__(self):
        return self.low_resolution.shape[0]

    def __getitem__(self, idx):
        low_matrix = torch.FloatTensor(self.low_resolution[idx]).unsqueeze(0)
        start_point = int(self.start_point[idx])
        end_point = int(self.end_point[idx])
        return {'low': low_matrix, 'start': start_point, 'end': end_point}


def Addback(chrrr, matrix_path, multi, model, patch, record, k_para, output_path, use_blacklist, black_list1, black_list2):

    low_matrix = np.load(matrix_path, allow_pickle=True)["matrix"]
    low_matrix = low_matrix.astype('float64')
    thres = np.max(low_matrix) * multi

    low_removed = np.load(matrix_path, allow_pickle=True)["matrix"]
    low_removed = low_removed.astype('float64')
    device = torch.device('cpu')
    model = torch.load(model, map_location='cpu')
    model.eval()
    test_dataset = MAXIM_Dataset_multiplechr_test(low_matrix=low_matrix, patch=patch)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=50)



    with torch.no_grad():
        for i, data in enumerate(test_loader):
            low = data['low'].to(device).float()
            start_point = data['start']
            end_point = data['end']
            sr = model(low).squeeze(1).squeeze(0).detach().cpu().numpy()
            for batch_idx in range(low.shape[0]):
                output_batch = sr[batch_idx]
                start_point_tem = start_point[batch_idx]
                end_point_tem = end_point[batch_idx]
                low_removed[start_point_tem: start_point_tem + 64, end_point_tem: end_point_tem + 64] = output_batch

    if use_blacklist:
        size = low_matrix.shape[0]
        listt = []
        for i in range(len(black_list1)):
            for j in range(black_list1[i], black_list2[i]+1):
                listt.append(j)
        all = list(np.arange(0, size, 1))
        compact_index_list = list(set(all).difference(set(listt)))
        compact_index = np.array(compact_index_list)

        compacted_low = np.zeros((compact_index.shape[0], compact_index.shape[0]))
        for k, idx in enumerate(compact_index):
            compacted_low[k, :] = low_removed[idx][compact_index]
    else:
        compacted_low = low_removed

    info = []
    with open(record, 'r') as f:
        for line in f.readlines():
            info.append(line)
    for ii in range(len(info)):
        i = len(info) - ii - 1
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
            if out_mean > 0:
                compacted_low[start_x: end_x, start_y: end_y] = compacted_low[start_x: end_x, start_y: end_y] * in_mean / out_mean
        if type == 'O':
            if out_mean > 0:
                compacted_low[start_x: end_x, start_y: end_y] = compacted_low[start_x: end_x, start_y: end_y] * in_mean / out_mean
        if type == 'U':
            if out_mean > 0:
                compacted_low[start_x: end_x, start_y: end_y] = compacted_low[start_x: end_x, start_y: end_y] * in_mean2 / out_mean2
                upper_tri = np.flip(np.triu(np.flip(compacted_low[start_x: end_x, start_y: end_y], 1), 1), 1) * in_mean / out_mean
                lower_tri = np.flip(np.tril(np.flip(compacted_low[start_x: end_x, start_y: end_y], 1)), 1)
                compacted_low[start_x: end_x, start_y: end_y] = lower_tri + upper_tri
        if type == 'L':
            if out_mean > 0:
                compacted_low[start_x: end_x, start_y: end_y] = compacted_low[start_x: end_x, start_y: end_y] * in_mean2 / out_mean2
                lower_tri = np.flip(np.tril(np.flip(compacted_low[start_x: end_x, start_y: end_y], 1), -1), 1) * in_mean / out_mean
                upper_tri = np.flip(np.triu(np.flip(compacted_low[start_x: end_x, start_y: end_y], 1)), 1)
                compacted_low[start_x: end_x, start_y: end_y] = lower_tri + upper_tri

    if use_blacklist:
        for i, s_idx in enumerate(compact_index):
            low_removed[s_idx, compact_index] = compacted_low[i]
    else:
        low_removed = compacted_low

    size = low_removed.shape[0]
    resolution = 5000
    for i in range(size):
        for j in range(i, size):
            index = np.abs(i - j) + 1
            normal_value = k_para * np.power(index * resolution, -3 / 2) * np.exp(-1400 / (index * resolution * index * resolution))  # 3000000
            low_removed[i][j] = low_removed[i][j] + normal_value

    block_size = 64
    size_parameter = 5
    for ii in range(0, size, block_size):
        for jj in range(ii, size, block_size):
            if ii + block_size < size and jj + block_size < size and np.abs(jj - ii) <= size_parameter * block_size:
                low_matrix[ii: ii+block_size, jj: jj+block_size] = low_removed[ii: ii+block_size, jj: jj+block_size]



    low_tril = np.triu(low_matrix, 1)
    low_triu = np.triu(low_matrix, 0)
    low_matrix = low_triu + low_tril.T

    reconstructed_2 = low_matrix
    reconstructed_2[np.where(reconstructed_2 < 0.5)] = 0

    x_exceed, y_exceed = np.where(reconstructed_2 > thres)
    for i_index in range(len(x_exceed)):
        tempory_new = np.median(reconstructed_2[x_exceed[i_index] - 1: x_exceed[i_index] + 1, y_exceed[i_index] - 1: y_exceed[i_index] + 1])
        reconstructed_2[x_exceed[i_index]][y_exceed[i_index]] = tempory_new


    low_tril = np.triu(reconstructed_2, 1)
    low_triu = np.triu(reconstructed_2, 0)
    reconstructed_2 = low_triu + low_tril.T

    np.savez_compressed(output_path + 'SHARP_' + str(chrrr) + '_reconstructed.npz', matrix=reconstructed_2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Decomposition script for chromosome matrices.")
    parser.add_argument('--chr', type=int, help='chromosome', default=20)
    parser.add_argument('--k_para', type=int, help='K paratemer used for signal decomposition', default=160000000)
    parser.add_argument('--matrix', type=str, help='Path to the matrix', default="../Data/1_150/chr20_low_matrix.npz")
    parser.add_argument('--model', type=str, help='Trained model', default="../Model/SHARP_best_model.pt")
    parser.add_argument('--multi', type=int, help='Enhance rate', default="150")
    parser.add_argument('--output_path', type=str, help='Output path', default="../Data/Output/")
    parser.add_argument('--patches', type=str, help='Type 3 patches', default="../Data/1_150/20_Type3_patches.npz")
    parser.add_argument('--record_file', type=str, help='Record file', default="../Data/1_150/chr20_record.txt")

    parser.add_argument('--use_blacklist', type=bool, help='Whether to use the blacklist', default="True")
    parser.add_argument('--black_list1', nargs='+', type=int, help='Blacklist array1')
    parser.add_argument('--black_list2', nargs='+', type=int, help='Blacklist array2')

    args = parser.parse_args()

    Addback(args.chr, args.matrix, args.multi, args.model, args.patches, args.record_file, args.k_para, args.output_path, args.use_blacklist, args.black_list1, args.black_list2)
    #--black_list1 0 5273 5788 6199 9579 --black_list2 14 5783 5853 6249 9580


