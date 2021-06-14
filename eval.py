import tensorflow as tf
from sklearn.metrics import f1_score

from cdr_data import CDRData
from graph_lstm import GraphLSTM
from train import make_golden, make_tensor_from_dict


if __name__ == "__main__":
    dataset = CDRData()
    model = GraphLSTM(dataset)

    model.load_weights("saved_weights/saved")
    dev_data = dataset.build_data_from_file(dataset.DEV_DATA_PATH, mode='intra', limit=200)

    all_pred = list()
    all_golden = list()

    for step, x_dev in enumerate(dev_data):
        if len(x_dev['Chemical']) == 0 or len(x_dev['Disease']) == 0:
            continue

        y_dev = make_golden(x_dev)

        with tf.GradientTape() as tape:
            logits = model(x_dev)
            logits = make_tensor_from_dict(logits)

        all_pred += [tf.math.argmax(logit[0]) for logit in logits]
        all_golden += [tf.math.argmax(golden[0]) for golden in y_dev]

    model.summary()
    print(all_golden)
    print(all_pred)
    print("micro f1: ", f1_score(all_golden, all_pred, average='micro'))
    print("macro f1: ", f1_score(all_golden, all_pred, average='macro'))

