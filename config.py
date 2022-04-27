import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str,
                        default="Ours")
    parser.add_argument("--data_path", type=str,
                        default="./Dataset/")
    parser.add_argument("--pre_trained_path", type=str,
                        default=None)
    parser.add_argument("--dataset", type=str, default="GTSRB")
    parser.add_argument("--model_type", type=str, default='resnet18')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_class", type=int, default=43)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=31)
    parser.add_argument("--lr_optimizer_for_c", type=float, default=1e-3)
    parser.add_argument("--lr_optimizer_for_t", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--a", type=float, default=0.3)
    parser.add_argument("--b", type=float, default=0.1)
    parser.add_argument("--to_save", type=str, default='True')
    parser.add_argument("--to_print", type=str, default='True')
    return parser


opt = get_arguments().parse_args()
