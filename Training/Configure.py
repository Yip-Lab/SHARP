from argparse import ArgumentParser

parser = ArgumentParser(description='Training for the SHARP mpde;')
parser.add_argument('--gpu_device', type=str, default='cuda:0')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--train_batch_size', type=int, default=40)
parser.add_argument('--test_batch_size', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument("--log_dir", type=str, default='log')
parser.add_argument("--model_dir", type=str, default='../Model/')
parser.add_argument("--low_data_dir", type=str, default='../Data/1_150/')
parser.add_argument("--high_data_dir", type=str, default='../Data/')


args = parser.parse_args()