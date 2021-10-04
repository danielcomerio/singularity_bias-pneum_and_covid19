from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import argparse
import os


def build_train_val_datasets(dataset_dir, batch_size, img_size):
    train_dir = os.path.join(dataset_dir, "train")

    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 image_size=img_size)

    # build validation dataset from the train dataset.
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset


def build_model(input_size, data_augmentation, args):
    IMG_SHAPE = input_size + (3,)

    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    base_model = tf.keras.applications.resnet50.ResNet50(input_shape=IMG_SHAPE,
                                                         include_top=False,
                                                         weights='imagenet')

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(3)

    inputs = tf.keras.Input(shape=(250, 250, 3))
    if data_augmentation is not None:
        x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    if args.add_dense:
        x = tf.keras.layers.Dense(256)
        x = tf.keras.layers.Dense(256)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def build_data_augmentation(do_augmentation):
    if not do_augmentation:
        return None

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip(
            mode="horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(
            (-0.1, 0.1), (-0.1, 0.1)),
        tf.keras.layers.experimental.preprocessing.RandomZoom(
            (-0.2, 0.2), (-0.2, 0.2))
    ])

    return data_augmentation


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to the dataset", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-l", "--lr", help="learning rate",
                        type=float, default=1e-4)
    parser.add_argument("-n", "--n_epochs", type=int, default=40)
    parser.add_argument("-a", "--augment",
                        help="perform data augmentation",
                        type=int, choices=[0, 1], default=1)
    parser.add_argument("-d", "--add_dense",
                        help="wether to add dense layers before the output",
                        type=int, choices=[0, 1], default=0)
    parser.add_argument("-t", "--tag", type=str, default="",
                        help="add the text to the output dir name")

    args = parser.parse_args()

    return args


def main():
    args = parse_command_line_args()

    DATASETS = [
        ("Figure1-COVID-chestxray-dataset-master", "model_figure1"),
        ("covid-chestxray-dataset-master", "model_covid_chestxray"),
        ("COVID-19 Radiography Database", "model_covid19_radiography"),
        ("Actualmed-COVID-chestxray-dataset-master", "model_actualmed"),
        ("rsna-pneumonia-detection-challenge", "model_rsna")
    ]

    for dataset in DATASETS:
        dataset_path = os.path.join(args.dataset, dataset[0])
        output_dir = os.path.join("../trained_models")  # , dataset[1]

        IMG_SIZE = (250, 250)
        train_ds = build_train_val_datasets(
            dataset_path, args.batch_size, IMG_SIZE)

        data_augmentation = build_data_augmentation(args.augment)
        model = build_model(IMG_SIZE, data_augmentation, args)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy']
        )
        model.summary()
        model.fit(train_ds, epochs=args.n_epochs)
        model.save(os.path.join(output_dir, dataset[1] + '.h5'))

        print(dataset[1])
        print("Ok.")


if __name__ == "__main__":
    main()
