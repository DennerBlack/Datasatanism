import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # на винде можно убрать

#
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#
x_train, x_test = x_train / 255.0, x_test / 255.0

#
image_size = x_train[0].shape[:2]

##
x_train = x_train.reshape(x_train.shape[0], *image_size, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], *image_size, 1).astype('float32')

#
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(38, (1, 1), input_shape=(*image_size, 1), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

#
optimizer = tf.keras.optimizers.Adam(0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

#
n_epochs = 2
batch_size = 16

new_model = 1

if new_model:
    #
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    #
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=n_epochs,
        validation_data=(x_test, y_test),
        verbose=True
    )

    i = 0
    while os.path.exists(f"weights/simple_conv_E{n_epochs}_v{i}.h5"):
        i += 1
    #
    model.save(f'weights/simple_conv_E{n_epochs}_v{i}.h5')

    #
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{f"simple_conv_E{n_epochs}_v{i}"} fitting history')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

else:
    i = 0
    while os.path.exists(f"weights/simple_conv_E{n_epochs}_v{i + 1}.h5"):
        i += 1
    print(f'load model: simple_conv_E{n_epochs}_v{i}.h5')
    ##
    model = tf.keras.models.load_model(f'weights/simple_conv_E{n_epochs}_v{i}.h5')

#
test_image = x_test[0].reshape(1, *image_size, 1)

#
prediction = model.predict(test_image).argmax()

##
plt.imshow(test_image[0])
plt.title(f'Prediction: {prediction}, label: {y_test[0]}')
plt.show()
