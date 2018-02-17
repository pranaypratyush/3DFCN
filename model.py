from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant, glorot_normal
from keras.layers import *
from keras.models import Model
import os
from input import *

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# tf=K.get_session()
pc_path = "./kitti/training/velodyne/*.bin"
label_path = "./kitti/training/label_2/*.txt"
calib_path = "./kitti/training/calib/*.txt"

# input = Input(shape=(300,300,300,1),dtype='float32',name='input')
input = Input(shape=(400, 400, 20, 1), dtype='float32', name='input')

layer1 = Conv3D(10, [5, 5, 5], strides=(2, 2, 2), padding='same', use_bias=True, activation='relu', name='C1'
                , kernel_initializer=glorot_normal(), bias_initializer=Constant(value=0))(input)
batch_norm = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones',
                                moving_mean_initializer='zeros', moving_variance_initializer='ones')(layer1)
layer2 = Conv3D(20, [5, 5, 5], strides=(2, 2, 2), padding='same', use_bias=True, activation='relu', name='C2'
                , kernel_initializer=glorot_normal(), bias_initializer=Constant(value=0))(batch_norm)
batch_norm = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones',
                                moving_mean_initializer='zeros', moving_variance_initializer='ones')(layer2)
layer3 = Conv3D(30, [3, 3, 3], strides=(2, 2, 2), padding='same', use_bias=True, activation='relu', name='C3'
                , kernel_initializer=glorot_normal(), bias_initializer=Constant(value=0))(batch_norm)
batch_norm = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones',
                                moving_mean_initializer='zeros', moving_variance_initializer='ones')(layer3)

# layer4 = Conv3D(64,[5,5,5],(1,1,1),padding='same',use_bias=True,activation='relu',name='C1')(layer3)
objectness = Conv3DTranspose(2, [3, 3, 3], (2, 2, 2), padding='same', kernel_initializer=Constant(value=0.01)
                             , activation='softmax', name='objectness')(batch_norm)
# objectness = Reshape([-1,100,100,6],name='objectness')(objectness)
bound_box = Conv3DTranspose(24, [3, 3, 3], (2, 2, 2), padding='same', kernel_initializer=Constant(value=0.01)
                            , name='bound_box')(batch_norm)

model = Model(inputs=input, outputs=[objectness, bound_box])


def obj_loss(model):
    def loss(y_true, y_pred):
        elosion = 0.00001
        sh = K.int_shape(y_pred)
        print("y_true " + str(K.int_shape(y_true)))
        print("y_pred " + str(K.int_shape(y_pred)))
        # print('\n')

        # ones = K.ones(shape=(2,100,100,6,2), dtype='float32')
        # non_gmap = -y_true + ones
        non_gmap = y_true[:, :, :, :, :] - 1
        non_gmap = -non_gmap

        is_obj_loss = - K.sum(y_true[:, :, :, :, 1] * K.log(y_pred[:, :, :, :, 1] + elosion))
        log = K.log(y_pred[:, :, :, :, 1] + elosion)
        # not_obj_loss = K.dot(- K.sum(K.dot(non_gmap,log)), 0.0008)
        not_obj_loss = - K.sum(non_gmap[:, :, :, :, 1] * log) * 0.0008
        obj_loss = K.sum(is_obj_loss) + K.sum(not_obj_loss)
        # print("obj_loss " + str(K.int_shape(obj_loss)))
        # sys.stdout.flush()

        return obj_loss

    return loss


def bbox_loss(model):
    def loss1(y_true, y_pred):
        g_map = model.outputs[0]
        # cord_diff = K.dot(g_map, K.sum(K.square(y_pred - y_true), axis=4))
        cord_diff = g_map * K.sum(K.square(y_pred - y_true), axis=4, keepdims=True)
        # cord_loss = K.dot(K.sum(cord_diff),0.02)
        cord_loss = K.sum(cord_diff) * 0.02
        return cord_loss

    return loss1


print(model.outputs)
sys.stdout.flush()

adam = optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss={'objectness': obj_loss(model=model), 'bound_box': bbox_loss(model=model)},
              metrics={'objectness': 'accuracy', 'bound_box': 'accuracy'})
print(model.summary())
# print(model.get_config())
# config = model.get_config()
# l = layers.deserialize({'class_name': model.__class__.__name__,
#                             'config': config})
# print(l.output_shape)
# data_ratio = 0.001
# part_ratio = 0.8
data_ratio = float(sys.argv[1])
part_ratio = float(sys.argv[2])
batch_size = float(sys.argv[3])
checkpoint = [ModelCheckpoint(filepath='checkpoint.hdf5', monitor='val_loss', verbose=1, save_best_only=True)]
history = model.fit_generator(
    data_generator(batch_size, pc_path=pc_path, label_path=label_path, calib_path=calib_path,
                   start=0, end=int(7480 * part_ratio * data_ratio), testing=False),
    epochs=101, verbose=2, callbacks=checkpoint, steps_per_epoch=7482 / 5 + 1)
model.save_weights('final.hdf5')

final_loss = model.evaluate_generator(
    data_generator(batch_size, pc_path=pc_path, label_path=label_path, calib_path=calib_path,
                   start=int(7480 * part_ratio * data_ratio + 1), end=int(7480 * data_ratio), testing=False))

print(final_loss)
