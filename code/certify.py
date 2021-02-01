# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
import os
from architectures import get_architecture

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=5000, help="stop after this many examples")
parser.add_argument("--start", type=int, default=0, help="start at this point of dataset")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=200000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    save_directory = os.path.dirname(args.outfile)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # prepare output file
    f = open(args.outfile, 'a')
    print("idx\tlabel\tcount\tpredict\tradiusR\tradiusG\tradiusB\tradius_L1\tradius_LInf\tradius_L2\tradius_cohen\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(args.start, args.start + (args.skip*args.max), args.skip):

        (x, label) = dataset[i]
        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        count, maxC, meanC, meanR, meanG, meanB, prediction, radiusR, radiusG, radiusB, radiusL1, radiusLInf, radiusL2, radiusOp, radiusCohen = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch, label)
        after_time = time()
        correct = int(prediction == label)
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}".format(
            i, label, count, prediction, radiusR, radiusG, radiusB, radiusL1, radiusLInf, radiusL2, radiusCohen, correct, time_elapsed), file=f, flush=True)

    f.close()
