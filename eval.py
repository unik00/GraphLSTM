import random

import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score

from cdr_data import CDRData
from graph_lstm import GraphLSTM
from train import make_golden, make_tensor_from_dict


if __name__ == "__main__":
    dataset = CDRData()
    model = GraphLSTM(dataset)

    model.load_weights("saved_weights/saved")
    dev_data = dataset.build_data_from_file(dataset.TRAIN_DATA_PATH, mode='intra', limit=None)
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

    model.summary()
    print(all_golden)
    print(all_pred)
    print("micro f1: ", f1_score(all_golden, all_pred, average='micro'))
    print("macro f1: ", f1_score(all_golden, all_pred, average='macro'))
    print("binary f1: ", f1_score(all_golden, all_pred, average='binary'))
    print("acc: ", accuracy_score(all_golden, all_pred))

