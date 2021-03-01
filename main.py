from mhadatareader import MhaDataReader
from classes import ParticipantsData, Scan, ProficiencyLabel, FoldSplit
import utils as ut
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

DIR_NAME = './data_bckp'


def sanity_check(data: ParticipantsData):
    for part in data:
        assert len(part.get_transforms(Scan.ALL)) == \
               len(part.get_transforms(Scan.LUQ)) + len(part.get_transforms(Scan.RUQ)) + \
               len(part.get_transforms(Scan.PERICARD)) + len(part.get_transforms(Scan.PELVIC))

        assert part.get_time() == part.get_reg_time(Scan.RUQ) + part.get_reg_time(Scan.LUQ) + \
               part.get_reg_time(Scan.PERICARD) + part.get_reg_time(Scan.PELVIC)


def load_data():
    parser = MhaDataReader()
    intermediates = parser.read_data(f'{DIR_NAME}/Intermediates/')
    sanity_check(intermediates)

    ut.add_path_len(intermediates)
    ut.add_linear_speed(intermediates)
    ut.add_angular_speed(intermediates)

    x_intermed = ut.prepare_data_all_reg(intermediates)

    experts = parser.read_data(f'{DIR_NAME}/Experts/')
    sanity_check(experts)

    ut.add_path_len(experts)
    ut.add_linear_speed(experts)
    ut.add_angular_speed(experts)
    x_expert = ut.prepare_data_all_reg(experts)

    novices = parser.read_data(f'{DIR_NAME}/Novices/')
    sanity_check(novices)

    ut.add_path_len(novices)
    ut.add_linear_speed(novices)
    ut.add_angular_speed(novices)
    x_novice = ut.prepare_data_all_reg(novices)

    return x_novice, x_intermed, x_expert


def build_model(input_shape, num_classes, filters, kernel_size, dropout_rate):
    input_layer = keras.layers.Input(shape=input_shape)

    conv1 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', kernel_regularizer='l2')(
        input_layer)
    conv1 = keras.layers.Dropout(dropout_rate)(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', kernel_regularizer='l2')(
        conv1)
    conv2 = keras.layers.Dropout(dropout_rate)(conv2)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', kernel_regularizer='l2')(
        conv2)
    conv3 = keras.layers.Dropout(dropout_rate)(conv3)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


if __name__ == '__main__':
    novices, intermediates, experts = load_data()
    folds = ut.form_folds(novices, intermediates, experts)

    folds_stats = []
    n_classes = len(ProficiencyLabel)

    # hyper-parameters
    KERNEL_SIZE = (3)
    FILTERS = 16
    EPOCHS = 10
    BATCH_SIZE = 16
    DROPOUT_RATE = 0.8

    # for x_novice, x_intermed, x_expert in FoldSplit(folds):
    x_novice, x_intermed, x_expert = folds[0]

    val_novice, val_intermed, val_expert = folds[1]
    x_val = np.hstack((np.hstack((val_novice, val_intermed)), val_expert))
    y_val = np.append(np.append(np.full((len(val_novice),), ProficiencyLabel.Novice),
                                np.full((len(val_intermed),), ProficiencyLabel.Intermediate)),
                                np.full((len(val_expert),), ProficiencyLabel.Expert))

    test_novice, test_intermed, test_expert = folds[2]
    x_test = np.hstack((np.hstack((test_novice, test_intermed)), test_expert))
    y_test = np.append(np.append(np.full((len(test_novice),), ProficiencyLabel.Novice),
                                np.full((len(test_intermed),), ProficiencyLabel.Intermediate)),
                                np.full((len(test_expert),), ProficiencyLabel.Expert))

    x_train, y_train = ut.augment_and_split2(x_novice, x_intermed, x_expert)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_val = keras.utils.to_categorical(y_val, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    x_val, y_val = ut.shuffle(x_val, y_val)
    x_test, y_test = ut.shuffle(x_test, y_test)

    # callbacks = [
    #     keras.callbacks.ModelCheckpoint(
    #         'best_model.h5', save_best_only=True, monitor='val_loss'
    #     ),
    #     keras.callbacks.ReduceLROnPlateau(
    #         monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001
    #     ),
    #     keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1),
    # ]
    model = build_model(x_train.shape[1:], n_classes, kernel_size=KERNEL_SIZE, filters=FILTERS,
                        dropout_rate=DROPOUT_RATE)
    model.summary()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', 'mse'],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        # callbacks=callbacks,
        validation_data=(x_val, y_val),
        # shuffle=True,
        verbose=1,
    )
    # model = keras.models.load_model('best_model.h5')

    test_loss, test_acc, mse = model.evaluate(x_test, y_test)

    folds_stats.append((test_loss, test_acc))

    print('Test accuracy', test_acc)
    print('Test loss', test_loss)

    # metric = 'categorical_accuracy'
    # plt.figure()
    # plt.plot(history.history[metric])
    # plt.plot(history.history['val_' + metric])
    # plt.title('model ' + metric)
    # plt.ylabel(metric, fontsize='large')
    # plt.xlabel('epoch', fontsize='large')
    # plt.legend(['train', 'val'], loc='best')
    # plt.show()
    # plt.close()

    avg_acc = 0
    for i in range(len(folds_stats)):
        avg_acc = avg_acc + folds_stats[i][1]
        if i == len(folds_stats) - 1:
            avg_acc = avg_acc / len(folds_stats)

    avg_loss = 0
    for i in range(len(folds_stats)):
        avg_loss = avg_loss + folds_stats[i][0]
        if i == len(folds_stats) - 1:
            avg_loss = avg_loss / len(folds_stats)

    print(f'AVG acc: {avg_acc}')
    print(f'AVG loss: {avg_loss}')
    print(folds_stats)
    print('DOne')
