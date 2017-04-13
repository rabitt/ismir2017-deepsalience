from __future__ import print_function
import medleydb as mdb
import matplotlib.pyplot as plt
import os

import core

DATA_PATH = "/scratch/rmb456/multif0_ismir2017/training_data_with_blur/multif0_complete/"
MTRACK_LIST = mdb.TRACK_LIST_V1 + mdb.TRACK_LIST_V2 + mdb.TRACK_LIST_EXTRA
INPUT_PATCH_SIZE = (360, 50)
OUTPUT_PATH_SIZE = (360, 50)

SAMPLES_PER_EPOCH = 512
NB_EPOCHS = 100
NB_VAL_SAMPLES = 1000


SAVE_PATH = "/scratch/rmb456/multif0_ismir/saved_models"
SAVE_KEY = os.path.basename(__file__).split('.')[0]
MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "{}.pkl".format(SAVE_KEY))
PLOT_SAVE_PATH = os.path.join(SAVE_PATH, "{}_loss.pdf".format(SAVE_KEY))
TEST_EXAMPLE_PATH = os.path.join(SAVE_PATH, SAVE_KEY)


def main():

    ### DATA SETUP ###
	dat = core.Data(
        MTRACK_LIST, DATA_PATH, input_patch_size=INPUT_PATCH_SIZE,
        output_patch_size=OUTPUT_PATH_SIZE, batch_size=10
	)
    train_generator = dat.get_train_generator()
    validation_generator = dat.get_validation_generator()
    test_generator = dat.get_test_generator()

    ### DEFINE MODEL ###
    input_shape = (None, None, 6)
    inputs = Input(shape=input_shape)

    y1 = Conv2D(64, (5, 5), padding='same', activation='relu', name='bendy1')(inputs)
    y2 = Conv2D(64, (5, 5), padding='same', activation='relu', name='bendy2')(y1)
    y3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='smoothy1')(y2)
    y4 = Conv2D(64, (3, 3), padding='same', activation='relu', name='smoothy2')(y3)
    y5 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y4)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y5)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=bkld, metrics=['mse', soft_binary_accuracy], optimizer='adam')

    print(model.summary(line_length=80))

    ### FIT MODEL ###
    history = model.fit_generator(
        train_generator, SAMPLES_PER_EPOCH, epochs=NB_EPOCHS, verbose=1,
        validation_data=validation_generator, validation_steps=NB_VAL_SAMPLES,
        callbacks=[
            keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
            keras.callbacks.EarlyStopping(patience=15, verbose=0)
        ]
    )

    ### load best weights ###
    model.load_weights(MODEL_SAVE_PATH)

    ### Results plots ###
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 1, 1)
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('mean squared error')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    plt.subplot(3, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    plt.subplot(3, 1, 3)
    plt.plot(history.history['soft_binary_accuracy'])
    plt.plot(history.history['val_soft_binary_accuracy'])
    plt.title('soft_binary_accuracy')
    plt.ylabel('soft_binary_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    plt.savefig(PLOT_SAVE_PATH, format='pdf')

    ### Evaluate ###
    test_eval = best_model.evaluate_generator(test_generator, 5000, max_q_size=10)
    if not os.path.exists(TEST_EXAMPLE_PATH):
        os.mkdir(TEST_EXAMPLE_PATH)

    for test_pair in dat.test_files:
        save_path = os.path.join(TEST_EXAMPLE_PATH, os.path.basename(test_pair[0]).split('.')[0])
        predicted_output, true_output = generate_prediction(test_pair, model, save_path=save_path)

        scores = core.compute_metrics(predicted_output, true_output)
        # TODO add to pandas dataframe

    # TODO save pandas dataframe yay


if __name__ == '__main__':
	main()
