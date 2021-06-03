import tensorflow as tf

from cdr_data import CDRData
from graph_lstm import GraphLSTM

if __name__ == "__main__":
    s = "A bilateral retrobulbar neuropathy with an unusual central bitemporal hemianopic scotoma was found"

    data = CDRData()

    model = GraphLSTM(data)

    optimizer = tf.keras.optimizers.SGD(learning_rate=1)
    loss_fn = tf.keras.losses.MSE

    epochs = 1
    batch_size = 1
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate([(s, tf.zeros(shape=(13, 150)))]):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train)  # Logits for this minibatch
                print("logits: ", logits, y_batch_train)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 1 == 0:
                print(
                    "Training loss (for one batch) at step {}: {}".format(step, loss_value)
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))

    model.save_weights("saved_weights/saved")

    new_model = GraphLSTM(dataset=data)

    print("before: ")
    print(new_model(s))

    new_model.load_weights("saved_weights/saved")
    print("after: ")
    print(new_model(s))

    new_model.summary()