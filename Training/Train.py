import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn as nn
import numpy as np
import random
from scipy import stats
import datetime
from Dataloader import MAXIM_Dataset_multiplechr_tri_patches as Dataset
from Maxim_model import MAXIM_dns_3s_hr8_lr4 as Net
import time
from Configure import args
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


torch.utils.backcompat.broadcast_warning.enabled = True
torch.multiprocessing.set_sharing_strategy('file_system')

# log txt
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)


distin = args.distin
model_dir = args.model_dir


# time
start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

writer_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

log_filename = start_time + '_SHARP.txt'
log_full_name = os.path.join(args.log_dir, log_filename)
os.mknod(log_full_name)
log_fp = open(log_full_name, 'w')


# gpu
if args.gpu:
    device = torch.device(args.gpu_device)
else:
    device = 'cpu'


torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)


valid_dataset = Dataset(low_data_dir=args.low_data_dir, high_data_dir=args.high_data_dir, chr=[16, 17, 21, 22])
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)
train_dataset = Dataset(low_data_dir=args.low_data_dir, high_data_dir=args.high_data_dir, chr=[1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 18, 19])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers)

if args.gpu:
    model = Net().to(device)
else:
    model = Net()


criterion = nn.L1Loss()
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-7)



loss_sum = 0

for epoch in range(args.epochs):
    model.train()
    train_countt = 0
    train_PLCC_ave = 0
    train_SROCC_ave = 0
    train_PSNR_ave = 0
    train_SSIM_ave = 0
    loss_sum = 0
    for i, data in enumerate(train_loader):
        if args.gpu:
            low = data['low'].to(device).float()
            high = data['high'].to(device).float()
        else:
            low = data['low'].float()
            high = data['high'].float()
        optimizer.zero_grad()
        outputs = model(low)
        loss = criterion(outputs, high)
        high_train = high.squeeze(1).squeeze(0).detach().cpu().numpy()
        outputs_cpu_train = outputs.squeeze(1).squeeze(0).detach().cpu().numpy()
        for batch_idx in range(outputs_cpu_train.shape[0]):
            output_batch = outputs_cpu_train[batch_idx]
            high_batch = high_train[batch_idx]
            loww2 = np.reshape(output_batch, (output_batch.shape[0] * output_batch.shape[0]))
            highh2 = np.reshape(high_batch, (high_batch.shape[0] * high_batch.shape[0]))
            rangee = max(highh2)
            train_PSNR_block = compare_psnr(high_batch, output_batch, data_range=rangee)
            train_SSIM_block = compare_ssim(high_batch, output_batch, win_size=3)
            train_SROCC_block = stats.spearmanr(loww2, highh2)[0]
            train_PLCC_block = stats.pearsonr(loww2, highh2)[0]
            if str(train_PLCC_block) != 'nan' and str(train_SROCC_block) != 'nan':
                train_PLCC_ave = train_PLCC_ave + np.abs(train_PLCC_block)
                train_SROCC_ave = train_SROCC_ave + np.abs(train_SROCC_block)
                train_PSNR_ave = train_PSNR_ave + np.abs(train_PSNR_block)
                train_SSIM_ave = train_SSIM_ave + np.abs(train_SSIM_block)
                train_countt = train_countt + 1
        print('train: ', i, loss.item())
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss.item()
    train_PLCC_ave = train_PLCC_ave / train_countt
    train_SROCC_ave = train_SROCC_ave / train_countt
    train_PSNR_ave = train_PSNR_ave / train_countt
    train_SSIM_ave = train_SSIM_ave / train_countt

    train_loss = loss_sum / (i + 1)

    model.eval()

    valid_PLCC_ave = 0
    valid_SROCC_ave = 0
    valid_PSNR_ave = 0
    valid_SSIM_ave = 0
    countt = 0
    L = 0
    valid_loss_sum = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            if args.gpu:
                low = data['low'].to(device).float()
            else:
                low = data['low'].float()
            high = data['high'].to(device).float()
            outputs = model(low)
            loss = criterion(outputs, high)
            print('valid: ', i, loss.item())
            high = high.squeeze(1).squeeze(0).detach().cpu().numpy()
            outputs_cpu = outputs.squeeze(1).squeeze(0).detach().cpu().numpy()
            for batch_idx in range(outputs_cpu.shape[0]):
                output_batch = outputs_cpu[batch_idx]
                high_batch = high[batch_idx]
                loww2 = np.reshape(output_batch, (output_batch.shape[0] * output_batch.shape[0]))
                highh2 = np.reshape(high_batch, (high_batch.shape[0] * high_batch.shape[0]))
                rangee = max(highh2)
                valid_PSNR_block = compare_psnr(high_batch, output_batch, data_range=rangee)
                valid_SSIM_block = compare_ssim(high_batch, output_batch, win_size=3)
                valid_SROCC_block = stats.spearmanr(loww2, highh2)[0]
                valid_PLCC_block = stats.pearsonr(loww2, highh2)[0]
                if str(valid_PLCC_block) != 'nan' and str(valid_SROCC_block) != 'nan':
                    valid_PLCC_ave = valid_PLCC_ave + np.abs(valid_PLCC_block)
                    valid_SROCC_ave = valid_SROCC_ave + np.abs(valid_SROCC_block)
                    valid_PSNR_ave = valid_PSNR_ave + np.abs(valid_PSNR_block)
                    valid_SSIM_ave = valid_SSIM_ave + np.abs(valid_SSIM_block)
                    countt = countt + 1
            valid_loss_sum = valid_loss_sum + loss.item()
        valid_loss = valid_loss_sum / (i + 1)
        valid_PLCC_ave = valid_PLCC_ave / countt
        valid_SROCC_ave = valid_SROCC_ave / countt
        valid_PSNR_ave = valid_PSNR_ave / countt
        valid_SSIM_ave = valid_SSIM_ave / countt
        scheduler.step()

    # test_PLCC_ave = 0
    # test_SROCC_ave = 0
    # test_PSNR_ave = 0
    # test_SSIM_ave = 0
    # countt = 0
    # L = 0
    # min_SROCC = 100
    # min_PLCC = 100
    # max_SROCC = 0
    # max_PLCC = 0
    # test_logg = []
    # test_loss_sum = 0
    # with torch.no_grad():
    #     for i, data in enumerate(test_loader):
    #         if args.gpu:
    #             low = data['low'].to(device).float()
    #         else:
    #             low = data['low'].float()
    #         high = data['high'].to(device).float()
    #         outputs = model(low)
    #         loss = criterion(outputs, high)
    #         print('test: ', i, loss.item())
    #         high = high.squeeze(1).squeeze(0).detach().cpu().numpy()
    #         outputs_cpu = outputs.squeeze(1).squeeze(0).detach().cpu().numpy()
    #         for batch_idx in range(outputs_cpu.shape[0]):
    #             output_batch = outputs_cpu[batch_idx]
    #             high_batch = high[batch_idx]
    #             # output_batch[np.where(output_batch < 0.005)] = 0
    #             loww2 = np.reshape(output_batch, (output_batch.shape[0] * output_batch.shape[0]))
    #             highh2 = np.reshape(high_batch, (high_batch.shape[0] * high_batch.shape[0]))
    #             rangee = max(highh2)
    #             test_PSNR_block = compare_psnr(high_batch, output_batch, data_range=rangee)
    #             test_SSIM_block = compare_ssim(high_batch, output_batch, win_size=3)
    #             test_SROCC_block = stats.spearmanr(loww2, highh2)[0]
    #             test_PLCC_block = stats.pearsonr(loww2, highh2)[0]
    #             start_point = data['start'][batch_idx]
    #             end_point = data['end'][batch_idx]
    #             chrr = data['chr'][batch_idx]
    #             # print(output_batch)
    #             if str(test_PLCC_block) != 'nan' and str(test_SROCC_block) != 'nan':
    #                 test_PLCC_ave = test_PLCC_ave + np.abs(test_PLCC_block)
    #                 test_SROCC_ave = test_SROCC_ave + np.abs(test_SROCC_block)
    #                 test_PSNR_ave = test_PSNR_ave + np.abs(test_PSNR_block)
    #                 test_SSIM_ave = test_SSIM_ave + np.abs(test_SSIM_block)
    #                 countt = countt + 1
    #         test_loss_sum = test_loss_sum + loss.item()
    #     test_loss = test_loss_sum / (i + 1)
    #     test_PLCC_ave = test_PLCC_ave / countt
    #     test_SROCC_ave = test_SROCC_ave / countt
    #     test_PSNR_ave = test_PSNR_ave / countt
    #     test_SSIM_ave = test_SSIM_ave / countt


    if epoch == 0:
        best_PLCC = valid_PLCC_ave
        best_SROCC = valid_SROCC_ave
        best_PSNR = valid_PSNR_ave
        best_SSIM = valid_SSIM_ave
    elif best_SROCC * best_PLCC < valid_PLCC_ave * valid_SROCC_ave:
        best_PLCC = valid_PLCC_ave
        best_SROCC = valid_SROCC_ave
        best_PSNR = valid_PSNR_ave
        best_SSIM = valid_SSIM_ave
        best_trained_model_file = 'SHARP_best_model.pt'
        torch.save(model, model_dir + best_trained_model_file)


    log_fp.write(
        ' epoch: ' + str(epoch) + ' train_loss: ' + str(train_loss) + ' valid_loss: ' + str(
            valid_loss) +
        ' train_PLCC_block ' + str(train_PLCC_ave) + ' train_SROCC_block: ' + str(train_SROCC_ave) +
        ' train_PSNR_block: ' + str(train_PSNR_ave) + ' train_SSIM_block: ' + str(train_SSIM_ave) +
        ' valid_PLCC_block ' + str(valid_PLCC_ave) + ' valid_SROCC_block: ' + str(valid_SROCC_ave) +
        ' valid_PSNR_block: ' + str(valid_PSNR_ave) + ' valid_SSIM_block: ' + str(valid_SSIM_ave) +
        ' best_PLCC_block: ' + str(best_PLCC) +
        ' best_SROCC_block: ' + str(best_SROCC) + ' best_PSNR_block: ' + str(best_PSNR) +
        ' best_SSIM_block: ' + str(best_SSIM) + '\n')
    log_fp.flush()
