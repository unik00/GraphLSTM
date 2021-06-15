import random
import argparse

import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score

from cdr_data import CDRData
from graph_lstm import GraphLSTM
from train import make_golden, make_tensor_from_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--limit", help="limit on the length of evaluation set", type=int, default=10 ** 9)
    parser.add_argument("-d", "--data_type", help="one of train/dev/test", type=str, default="dev")
    args = parser.parse_args()

    dataset = CDRData()

    data_path = dataset.DEV_DATA_PATH
    if args.data_type == "dev":
        data_path = dataset.DEV_DATA_PATH
    elif args.data_type == "train":
        data_path = dataset.TRAIN_DATA_PATH
    elif args.data_type == "test":
        data_path = dataset.TEST_DATA_PATH

    model = GraphLSTM(dataset)

    model.load_weights("saved_weights/saved")

    dev_data = dataset.build_data_from_file(data_path, mode='intra', limit=args.limit)
    random.shuffle(dev_data)

    all_pred = list()
    all_golden = list()

    batch_size = 1

    for step, x_dev in enumerate(dev_data):
        if len(x_dev['Chemical']) == 0 or len(x_dev['Disease']) == 0:
            continue

        y_dev = make_golden(x_dev)

        logits = model(x_dev)
        logits = make_tensor_from_dict(logits)

        all_pred += [int(tf.math.argmax(logit[0])) for logit in logits]
        # all_pred += [1] * len(y_dev)
        all_golden += [int(tf.math.argmax(golden[0])) for golden in y_dev]

        if step % 10 == 0:
            print("Seen so far: {}/{} samples".format((step + 1) * batch_size, len(dev_data)))

    print("Finished inference.")

    # model.summary()
    print(all_golden)
    print(all_pred)
    print("micro f1: ", f1_score(all_golden, all_pred, average='micro'))
    print("macro f1: ", f1_score(all_golden, all_pred, average='macro'))
    print("acc: ", accuracy_score(all_golden, all_pred))
    print("binary f1: ", f1_score(all_golden, all_pred, average='binary'))

