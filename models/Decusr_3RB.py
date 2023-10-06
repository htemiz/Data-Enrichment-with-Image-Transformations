"""
DECUSR model with 3 Repeating Blocks. This is a DeepSR file.

To run this model, e.g., training:

--python.exe -m DeepSR.DeepSR --modelfile Decusr_4RB.py --train

Please refer to the documentation of DeepSR from the following address for other command instructions and setting:
    https://github.com/htemiz/DeepSR
"""

from keras import metrics
from keras import losses
from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D,  LocallyConnected2D, Conv2DTranspose, concatenate, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from os.path import  dirname, abspath, basename
from keras.optimizers import Adam, SGD, RMSprop


eps = 1.1e-6

# PARAMETER INSTRUCTION WRT SCALE FACTOR #
"""

SCALE 2:
        stride=10, inputsize=16

SCALE 3:
        stride=8, inputsize=12

SCALE 4:
        stride=6, inputsize=8
"""

# parameters
settings = \
{
"activation": "relu",
'augment':[], # any combination of [90,180,270, 'flipud', 'fliplr', 'flipudlr' ]
'backend': 'tensorflow',
'batchsize':256,
'channels':3,
'colormode':'RGB', #
'crop': 0,
'crop_test': 6,
'decay':1e-6,
'dilation_rate':(1,1),
'decimation': 'bicubic',
'espatience' : 25,
'epoch':10,
'inputsize':16, #
'interp_compare': 'lanczos',
'interp_up': 'bicubic',
'kernel_initializer': 'he_normal',
'lrate':1e-3,
'lrpatience': 10,
'lrfactor' : 0.5,
'metrics': ["PSNR"],
'minimumlrate' : 1e-7,
'modelname':basename(__file__).split('.')[0],
'noise':'',
'normalization':['divide', '255.0'], # ['standard', "53.28741141", "40.73203139"],
'normalizeback': False,
'normalizeground':False,
'outputdir':'',
'scale':2,
'seed': 19,
'shuffle' : True,
'stride':10, #
'target_channels': 3,
'target_cmode' : 'RGB',
'testpath': [r'D:\calisma\projeler\augmentation\dataset\test'],# replace with path to your test files
'traindir': r"D:\calisma\projeler\augmentation\dataset\train", # replace with path to your training files
'upscaleimage': False,
'valdir': r'D:\calisma\projeler\augmentation\dataset\val', # replace with path to your validation files
'weightpath':'',
'workingdir': '',
}


def build_model(self, testmode=False):
    if testmode:
        input_size = None
    else:
        input_size = self.inputsize

    input_shape = (input_size, input_size, self.channels)

    main_input = Input(shape=input_shape, name='main_input')
    
    pre_block = Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(main_input)
    pre_block = Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(pre_block)
    pre_block = Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(pre_block)
    pre_block = Conv2D(16, (1, 1), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(pre_block)

    upsampler_LC = UpSampling2D(self.scale, name='upsampler_locally_connected')(pre_block)
    upsampler_direct = UpSampling2D(self.scale)(main_input)

    # REPEATING BLOCKS #
    block1 = concatenate([upsampler_LC, upsampler_direct])
    block1 = Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(block1)
    block1 = Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(block1)
    block1 = Conv2D(16, (1, 1), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(block1)

    block2 = concatenate([upsampler_LC, upsampler_direct, block1])
    block2 = Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(block2)
    block2 = Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(block2)
    block2 = Conv2D(16, (1, 1), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(block2)

    block3 = concatenate([upsampler_LC, upsampler_direct, block1, block2])
    block3 = Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(block3)
    block3 = Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(block3)
    block3 = Conv2D(16, (1, 1), activation=self.activation, kernel_initializer=self.kernel_initializer,  padding='same')(block3)

    nihai = Conv2D(self.channels, (3, 3), activation=self.lactivation, kernel_initializer=self.kernel_initializer,  padding='same')(block3)

    model = Model(main_input, outputs=nihai)
    model.compile(Adam(self.lrate, self.decay), loss=losses.mean_squared_error)

    # model.summary()

    return model

