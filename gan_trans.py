# imports
import os
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

from time import *


def repair(adv):
    def reconstruct(images, temp_model=None, iv3_model=None, batch_size=None, back_prop=True,
                    reconstructor_id=0, z_init_val=None):
        """Creates the reconstruction op for Defense-GAN.

        Args:
            X: Input tensor

        Returns:
            The `tf.Tensor` of the reconstructed input.
        """

        rec_rr = trans_config['rec_rr']
        rec_lr = trans_config['rec_lr']
        rec_iters = trans_config['rec_iters']
        generator = tf.keras.models.load_model(trans_config['generator'])

        # Batch size is needed because the latent codes are `tf.Variable`s and
        # need to be built into TF's static graph beforehand.

        batch_size = batch_size if batch_size else trans_config['batch_size']

        x_shape = images.get_shape().as_list()
        x_shape[0] = batch_size

        # Repeat images rec_rr times to handle random restarts in parallel.
        images_tiled_rr = tf.reshape(
            images, [x_shape[0], np.prod(x_shape[1:])])
        images_tiled_rr = tf.tile(images_tiled_rr, [1, rec_rr])
        images_tiled_rr = tf.reshape(
            images_tiled_rr, [x_shape[0] * rec_rr] + x_shape[1:])

        # Number of reconstruction iterations.
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            rec_iter_const = tf.get_variable(
                'rec_iter_{}'.format(reconstructor_id),
                initializer=tf.constant(0),
                trainable=False, dtype=tf.int32,
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
            )
            # The latent variables.
            z_hat = tf.get_variable(
                'z_hat_rec_{}'.format(reconstructor_id),
                shape=[batch_size * rec_rr, trans_config['latent_dim']],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(1.0 / trans_config['latent_dim'])),
                collections=[tf.GraphKeys.LOCAL_VARIABLES]
            )

        def get_learning_rate(
                init_lr=None, decay_epoch=None,
                decay_mult=None, iters_per_epoch=None,
                decay_iter=None,
                global_step=None, decay_lr=True):
            """Prepares the learning rate.

            Args:
                init_lr: The initial learning rate
                decay_epoch: The epoch of decay
                decay_mult: The decay factor
                iters_per_epoch: Number of iterations per epoch
                decay_iter: The iteration of decay [either this or decay_epoch
                should be set]
                global_step:
                decay_lr:

            Returns:
                `tf.Tensor` of the learning rate.
            """

            if decay_lr:
                if decay_epoch:
                    decay_iter = decay_epoch * iters_per_epoch
                return tf.train.exponential_decay(init_lr,
                                                  global_step,
                                                  decay_iter,
                                                  decay_mult,
                                                  staircase=True)

        # Learning rate for reconstruction.
        rec_lr_op_from_const = get_learning_rate(init_lr=rec_lr,
                                                 global_step=rec_iter_const,
                                                 decay_mult=0.1,
                                                 decay_iter=np.ceil(
                                                     rec_iters *
                                                     0.8).astype(
                                                     np.int32))

        # The optimizer.
        rec_online_optimizer = tf.train.MomentumOptimizer(
            learning_rate=rec_lr_op_from_const, momentum=0.7,
            name='rec_optimizer')

        init_z = tf.no_op()
        if z_init_val is not None:
            init_z = tf.assign(z_hat, z_init_val)

        z_hats_recs = generator(z_hat)
        num_dim = len(z_hats_recs.get_shape())
        axes = list(range(1, num_dim))  # modified

        if trans_config['reduce_type'] == 'pixel':
            image_rec_loss_pixel = tf.reduce_mean(
                tf.square(z_hats_recs - images_tiled_rr),
                axis=axes)
            image_rec_loss = image_rec_loss_pixel
        elif trans_config['reduce_type'] == 'classifier':
            inter_layer_recs = temp_model(z_hats_recs / 2.0)
            inter_layer_orig = temp_model(images_tiled_rr / 2.0)
            image_rec_loss_inter = tf.reduce_mean(
                tf.square(inter_layer_recs - inter_layer_orig),
                axis=axes)
            image_rec_loss = image_rec_loss_inter

        rec_loss = tf.reduce_sum(image_rec_loss)
        rec_online_optimizer.minimize(rec_loss, var_list=[z_hat])

        def rec_body(i, *args):
            z_hats_recs = generator(z_hat)
            if trans_config['reduce_type'] == 'pixel':
                image_rec_loss_pixel = tf.reduce_mean(
                    tf.square(z_hats_recs - images_tiled_rr),
                    axis=axes)
                image_rec_loss = image_rec_loss_pixel
            elif trans_config['reduce_type'] == 'classifier':
                inter_layer_recs = temp_model(z_hats_recs / 2.0)
                image_rec_loss_inter = tf.reduce_mean(
                    tf.square(inter_layer_recs - inter_layer_orig),
                    axis=axes)
                image_rec_loss = image_rec_loss_inter

            rec_loss = tf.reduce_sum(image_rec_loss)

            train_op = rec_online_optimizer.minimize(rec_loss,
                                                     var_list=[z_hat])

            return tf.tuple(
                [tf.add(i, 1), rec_loss, image_rec_loss, z_hats_recs],
                control_inputs=[train_op])

        rec_iter_condition = lambda i, *args: tf.less(i, rec_iters)
        for opt_var in rec_online_optimizer.variables():
            tf.add_to_collection(
                tf.GraphKeys.LOCAL_VARIABLES,
                opt_var,
            )

        with tf.control_dependencies([init_z]):
            online_rec_iter, online_rec_loss, online_image_rec_loss, \
            all_z_recs = tf.while_loop(
                rec_iter_condition,
                rec_body,
                [rec_iter_const, rec_loss, image_rec_loss, z_hats_recs]
                , parallel_iterations=1, back_prop=back_prop,
                swap_memory=False)
            final_recs = []
            for i in range(batch_size):
                ind = i * rec_rr + tf.argmin(
                    online_image_rec_loss[
                    i * rec_rr:(i + 1) * rec_rr
                    ],
                    axis=0)
                final_recs.append(all_z_recs[tf.cast(ind, tf.int32)])

            online_rec = tf.stack(final_recs)

            return tf.reshape(online_rec, x_shape)

    def reconstruct_dataset():
        """Reconstructs the images of the config's dataset with the generator.
        """

        set_session(session)
        orig_model = tf.keras.models.load_model(orig_model_path)
        orig_model.summary()
        if trans_config['arch'] == 'conv':
            if trans_config['data_type'] == None:
                outputs = orig_model.get_layer("activation_6").output
            else:
                outputs = orig_model.get_layer("flatten").output
        elif trans_config['arch'] == 'resnet':
            if trans_config['data_type'] == None:
                outputs = orig_model.get_layer("activation_15").output
            else:
                outputs = orig_model.get_layer("activation_35").output
        orig_model = Model(inputs=orig_model.input, outputs=outputs)
        orig_model.trainable = False
        iv3_model = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))

        if trans_config['dataset'] == 'cifar10':
            images_pl = tf.placeholder(tf.float32, shape=[trans_config['batch_size'], 32, 32, 3])  # modified
        elif trans_config['dataset'] == 'mnist':
            images_pl = tf.placeholder(tf.float32, shape=[trans_config['batch_size'], 28, 28, 1])  # modified
        elif trans_config['dataset'] == 'fmnist':
            images_pl = tf.placeholder(tf.float32, shape=[trans_config['batch_size'], 28, 28, 1])  # modified

        rec = reconstruct(images_pl, temp_model=orig_model, iv3_model=iv3_model)

        session.run(tf.local_variables_initializer())
        rets = {}

        all_recs = []
        all_targets = []
        orig_imgs = []
        if trans_config['dataset'] == 'cifar10':
            (_, _), (X_test, y_test) = cifar10.load_data()
            X_test = (X_test.astype('float32') - 127.5) / 127.5
            y_test = y_test.reshape([y_test.shape[0]])
        elif trans_config['dataset'] == 'mnist':
            (_, _), (X_test, y_test) = mnist.load_data()
            X_test = np.expand_dims(X_test, axis=-1)
            X_test = (X_test.astype('float32') - 127.5) / 127.5
        elif trans_config['dataset'] == 'fmnist':
            (_, _), (X_test, y_test) = fashion_mnist.load_data()
            X_test = np.expand_dims(X_test, axis=-1)
            X_test = (X_test.astype('float32') - 127.5) / 127.5

        if trans_config['data_type'] != None:
            adv_x = np.load(
                "./data/" + trans_config['dataset'] + "/adv/" + trans_config['data_type'] + "_x.npy")
            adv_y = np.load(
                "./data/" + trans_config['dataset'] + "/adv/" + trans_config['data_type'] + "_y.npy")
            adv_x = (adv_x * 255.0 - 127.5) / 127.5
            X_test = np.concatenate((X_test, adv_x), axis=0)
            y_test = np.concatenate((y_test, adv_y), axis=0)

        # yields batches from X, y dataset strictly of batch_size
        def data_generator(X, y, batch_size):
            assert X.shape[0] == y.shape[0]
            batches = X.shape[0] // batch_size
            for i in range(batches):
                yield X[batch_size * i:batch_size * (i + 1)], y[batch_size * i:batch_size * (i + 1)]

        loop = 0
        for images, targets in data_generator(X_test, y_test, trans_config['batch_size']):

            # added
            session.run(tf.local_variables_initializer())
            recs = session.run(
                rec, feed_dict={images_pl: images},
            )
            all_recs.append(recs)

            all_targets.append(np.squeeze(targets))
            orig_imgs.append(images)  # modified
            loop += 1
            print('batch:', loop)
        all_recs = np.concatenate(all_recs)
        if trans_config['dataset'] == 'cifar10':
            all_recs = all_recs.reshape([-1] + [32, 32, 3])
            orig_imgs = np.concatenate(orig_imgs).reshape(
                [-1] + [32, 32, 3])
        elif trans_config['dataset'] == 'mnist':
            all_recs = all_recs.reshape([-1] + [28, 28, 1])
            orig_imgs = np.concatenate(orig_imgs).reshape(
                [-1] + [28, 28, 1])
        elif trans_config['dataset'] == 'fmnist':
            all_recs = all_recs.reshape([-1] + [28, 28, 1])
            orig_imgs = np.concatenate(orig_imgs).reshape(
                [-1] + [28, 28, 1])
        all_targets = np.concatenate(all_targets)

        rets['test'] = [all_recs, all_targets, orig_imgs]

        return rets['test']

    # config
    trans_config = {
        'run_name': 'repair_conv_cifar10',  # unique name for each repair.py running
        'arch': 'resnet',  # conv or resnet
        'dataset': 'cifar10',  # cifar10, mnist, fmnist
        'data_type': adv,
        'reduce_type': 'pixel',  # pixel, classifier or fid
        'rec_rr': 2,
        'rec_lr': 10.0,
        'rec_iters': 200,
        'latent_dim': 128,
        'batch_size': 20,  # dataset_size must be divisible by batch_size
        'visible_gpus': '0',
        'generator': 'generators/resnet_cifar10_epoch090.h5'
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = trans_config['visible_gpus']
    gpu_config = ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    session = InteractiveSession(config=gpu_config)
    path = 'classifier/' + trans_config['dataset'] + '/' + trans_config['arch'] + '/'
    orig_model_path = path + 'model_' + trans_config['dataset'] + '_' + trans_config['arch'] + '.h5'
    if trans_config['data_type'] == None:
        tp_path = path + 'TP_idx.npy'
        fp_path = path + 'FP_idx.npy'
    else:
        tp_path = path + trans_config['data_type'] + '_TP_idx.npy'
        fp_path = path + trans_config['data_type'] + '_FP_idx.npy'

    begin_time = time()
    # repair
    data_dict = reconstruct_dataset()
    print('data_dict:', data_dict[0].shape, data_dict[1].shape, data_dict[2].shape)
    rec_images_in = data_dict[0]
    labels_in = data_dict[1]
    orig_images_in = data_dict[2]

    # convert range from (-1, 1) to (-0.5, 0.5) that conforms to the classifier preprocessing
    # test_rec_images = rec_images_in / 2.0
    # test_orig_images = orig_images_in / 2.0
    test_rec_images = (rec_images_in * 127.5 + 127.5) / 255.0
    test_orig_images = (orig_images_in * 127.5 + 127.5) / 255.0
    test_labels = labels_in.astype("uint8")

    orig_model = tf.keras.models.load_model(orig_model_path)
    orig_preds = np.argmax(orig_model.predict(test_orig_images), axis=1)
    print("model origin accuracy: {}".format(np.mean(orig_preds == test_labels)))
    end_time = time()
    run_time = end_time - begin_time
    print("run_time(ms): {}".format(run_time * 1000 / test_labels.shape[0]))
    print(test_labels.shape[0])

    new_preds = np.argmax(orig_model.predict(test_rec_images), axis=1)
    print("model rec accuracy: {}".format(np.mean(new_preds == test_labels)))

    # generate idx of incorrect predictions
    TP_idx = np.load(tp_path)
    FP_idx = np.load(fp_path)
    wrong_idx = np.where(orig_preds != test_labels)[0]

    print("number of correct predictions: {} in TP: {}".format(np.sum(new_preds[TP_idx] == test_labels[TP_idx]),
                                                               TP_idx.shape[0]))
    print("number of correct predictions: {} in FP: {}".format(np.sum(new_preds[FP_idx] == test_labels[FP_idx]),
                                                               FP_idx.shape[0]))
    print(
        "number of correct predictions: {} in wrong: {}".format(np.sum(new_preds[wrong_idx] == test_labels[wrong_idx]),
                                                                wrong_idx.shape[0]))
    print("original accuracy: {} vs modified accuracy: {}".format(FP_idx.shape[0] / (FP_idx.shape[0] + TP_idx.shape[0]),
                                                                  (np.sum(new_preds[TP_idx] == test_labels[
                                                                      TP_idx]) + np.sum(
                                                                      new_preds[FP_idx] == test_labels[FP_idx])) / (
                                                                          FP_idx.shape[0] + TP_idx.shape[0])))

    print("original accuracy: {} vs modified accuracy: {}".format(
        (test_labels.shape[0] - wrong_idx.shape[0]) * 100 / test_labels.shape[0], (
                np.sum(new_preds[TP_idx] == test_labels[TP_idx]) + np.sum(
            new_preds[FP_idx] == test_labels[FP_idx]) + test_labels.shape[0] - wrong_idx.shape[0] - FP_idx.shape[
                    0]) * 100 / test_labels.shape[0]))


advs = ["robot", "adapt", "fgsm", "pgd"]
for adv in advs:
    print("\r\nadv: {}".format(adv))
    repair(adv)