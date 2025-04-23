import argparse


def boolean_string(s):
    if s not in {'True', 'False'}:
        raise ValueError("%s: Not a valid Boolean argument string" % s)
    return s == 'True'


def get_args():
    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument('--save_dir', default="checkpoints", type=str, help="saving directory")
    parser.add_argument('--save_config', default="CNN-", type=str, help="saving config")
    parser.add_argument('--model_config', default=None, type=str, help="model config")
    parser.add_argument('--batch_size',
                       type=int,
                       help="batch size train",
                       default=32)
    parser.add_argument('--class_num',
                       type=int,
                       help="class num",
                       default=3)  # Changed to 3 for our three-class classification
    parser.add_argument('--epoch_num',
                       type=int,
                       help="epoch num",
                       default=30)
    parser.add_argument('--lr',
                       type=float,
                       help="learning rate",
                       default=1e-4)
    parser.add_argument('--cuda',
                       type=int,
                       help="cuda idx",
                       default=0)
    args = parser.parse_args()
    return args


def print_args(args):
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')


if __name__ == "__main__":
    args = get_args()
    print_args(args)