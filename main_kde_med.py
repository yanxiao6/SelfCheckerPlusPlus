# %%
import argparse
import os

# from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
# from keras.models import load_model
import tensorflow as tf
HeNormal = tf.keras.initializers.he_normal()
from tensorflow.keras.models import load_model

# from kdes_generation import fetch_kdes
from kdes_generation import fetch_kdes_gen
from utils import *
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set GPU Limits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="pathmnist")
    parser.add_argument("--m", "-m", help="Model", type=str, default="vgg16")
    parser.add_argument("--save_path", "-save_path", help="Save path", type=str, default="./tmp/")
    parser.add_argument("--batch_size", "-batch_size", help="Batch size", type=int, default=128)
    parser.add_argument("--var_threshold", "-var_threshold", help="Variance threshold", type=float, default=1e-5)
    parser.add_argument("--num_classes", "-num_classes", help="The number of classes", type=int, default=10)

    args = parser.parse_args()
    args.save_path = args.save_path + args.d + "/" + args.m + "/"
    dir = os.path.dirname(args.save_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    print(args)

    # load dataset
    dataset = np.load(f'{args.d}.npz')
    dataset = dict(dataset)

    # add axis if greyscale
    # define input_shape
    if len(dataset['train_images'].shape) == 3:
        input_shape = (28, 28, 1)
        dataset['train_images'] = dataset['train_images'][..., np.newaxis]
        dataset['val_images'] = dataset['val_images'][..., np.newaxis]
        dataset['test_images'] = dataset['test_images'][..., np.newaxis]
    else:
        input_shape = (28, 28, 3)

    # define classes
    if args.d == 'pathmnist':
        classes = 9
    elif args.d == 'octmnist':
        classes = 4
    elif args.d == 'tissuemnist':
        classes = 8

    x_train = dataset['train_images']/255.0
    x_valid = dataset['val_images']/255.0
    x_test = dataset['test_images']/255.0
    y_train = dataset['train_labels']
    y_valid = dataset['val_labels']
    y_test = dataset['test_labels']

    # Load pre-trained model.
    model = load_model("./medmnist_models/model_" + args.d + "_" + args.m + ".h5", custom_objects={'HeNormal': HeNormal}, compile=False)
    model.summary()

    score = model.evaluate(
        x_test,
        y_test,
        batch_size=128,
        verbose=1,
    )
    print(f"Test loss: {score[0]} - Test acc: {score[1]}")


    # fetch_kdes_gen(model, x_train, x_gen, x_valid, x_test, y_train, y_gen, y_valid, y_test, layer_names, args)
