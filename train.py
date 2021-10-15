import argparse
import time
import tensorflow as tf
from sklearn.metrics import f1_score

from cdr_data import CDRData
from graph_lstm import GraphLSTM


def make_golden(input_dict):
    golden = list()

    for c in input_dict['Chemical']:
        for d in input_dict['Disease']:
            if (c, d) in input_dict['relation']:
                golden.append([tf.constant([0., 1.])])
            else:
                golden.append([tf.constant([1., 0.])])
    return tf.convert_to_tensor(golden)


def make_tensor_from_dict(output_dict):
    output = list()
    for key in output_dict:
        output.append(output_dict[key])
    return tf.convert_to_tensor(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="number of epoch", type=int, default=20)
    parser.add_argument("-l", "--limit", help="limit on the length of train set", type=int, default=10 ** 9)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=1)
    parser.add_argument("-f", "--from_pretrained", help="load pretrained weights", type=bool, default=False)
    train_args = parser.parse_args()

    print("Arguments: ", train_args)

    dataset = CDRData()
    model = GraphLSTM(dataset)

    if train_args.from_pretrained:
        model.load_weights("saved_weights/saved")

    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.1)
    loss_fn = tf.keras.losses.categorical_crossentropy

    start_time = time.time()
    train_data = dataset.build_data_from_file(dataset.TRAIN_DATA_PATH, mode='inter', limit=train_args.limit)
    print("Length train data: ", len(train_data))

    print("Load data time: ", time.time() - start_time)

    for epoch in range(train_args.epochs):
        print("\nStart of epoch %d" % (epoch,))

        all_pred = list()
        all_golden = list()

        for step, x_train in enumerate(train_data):
            if len(x_train['Chemical']) == 0 or len(x_train['Disease']) == 0:
                continue

            y_train = make_golden(x_train)

            with tf.GradientTape() as tape:
                logits = model(x_train)  # Logits for this minibatch
                logits = make_tensor_from_dict(logits)

                # print("logits: ", logits, y_train)
                loss_value = loss_fn(y_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            # print(model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 10 == 9 or step == len(train_data) - 1:
                all_pred += [int(tf.math.argmax(logit[0])) for logit in logits]
                all_golden += [int(tf.math.argmax(golden[0])) for golden in y_train]
                print(
                    "Training loss (for one batch) at step {}: {}".format(step, tf.norm(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * train_args.batch_size))

                print("binary f1: ", f1_score(all_golden, all_pred, average='binary'))

            if step % 100 == 99 or step == train_args.epochs - 1:
                model.save_weights("saved_weights/saved")

