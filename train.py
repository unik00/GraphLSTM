import time
import tensorflow as tf

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
    dataset = CDRData()
    model = GraphLSTM(dataset)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    loss_fn = tf.keras.losses.categorical_crossentropy

    start_time = time.time()
    train_data = dataset.build_data_from_file(dataset.TRAIN_DATA_PATH, mode='intra')
    print("Length train data: ", len(train_data))

    print("Load data time: ", time.time() - start_time)

    epochs = 5
    batch_size = 1

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

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

            if step % 10 == 0:
                print(
                    "Training loss (for one batch) at step {}: {}".format(step, tf.norm(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))
        model.save_weights("saved_weights/saved")

