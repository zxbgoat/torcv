import argparse as ap


def parsargs():
    parser = ap.ArgumentParser(description='Parse args for the detectors')
    # args about program mode
    parser.add_argument('--tra', dest='train', action='store_true',
                        help='Train a detector')
    parser.add_argument('--tes', dest='test', action='store_true',
                        help='Test a detector')
    parser.add_argument('--dtc', dest='detect', action='store_true',
                        help='Perform detection')
    # args about model
    parser.add_argument('--det', dest='detname', type=str, default=None,
                        help='The detector to be used')
    parser.add_argument('--bkb', dest='backbone', type=str, default=None,
                        help='The name of the backbone model')
    # args about data
    parser.add_argument('--dat', dest='dataset', type=str, default=None,
                        help='The dataset to be used')
    parser.add_argument('--dir', dest='datadir', type=str, default=None,
                        help='The dir of the dataset')
    parser.add_argument('--bcs', dest='batchsz', type=int, default=None,
                        help='the batch size used during training')
    # args about train
    parser.add_argument('--sav', dest='savedir', type=str, default=None,
                        help='The dir of the weights')
    parser.add_argument('--opt', dest='optim', type=str, default=None,
                        help='The optimizer used to train the model')
    parser.add_argument('--epo', dest='epochs', type=int, default=None,
                        help='number of epochs to train')
    parser.add_argument('--pre', dest='pretain', type=bool, default=None,
                        help='if load the pretrained weights of the backbone')
    parser.add_argument('--sch', dest='schdl', type=str, default=None,
                        help='the learning rate scheduler')
    parser.add_argument('--ims', dest='imgsz', type=int, default=None,
                        help='the image size during training')
    return parser.parse_args()
