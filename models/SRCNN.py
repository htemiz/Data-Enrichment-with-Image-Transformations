"""
SRCNN model tailored for processing 3-channel images. This is a DeepSR file.

To run this model, e.g., training:

--python.exe -m DeepSR.DeepSR --modelfile SRCNN.py --train

Please refer to the documentation of DeepSR from the following address for other command instructions and setting:
    https://github.com/htemiz/DeepSR
"""

from keras import metrics
from keras import losses
from keras.layers import Input, Dense
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from os.path import  dirname, abspath, basename

eps = 1.1e-6

settings = \
{
'augment':[], # any combination of [90,180,270, 'flipud', 'fliplr', 'flipudlr' ]
'backend': 'tensorflow',
'batchsize':512,
'channels':3,
'colormode':'RGB', # 'YCbCr' or 'RGB'
'crop': 6,
'crop_test': 0,
'decay':1e-6,
'decimation': 'bicubic',
'dilation_rate':(1,1),
'espatience' : 25,
'epoch':10,
'inputsize':33, #
'interp_compare': '',
'interp_up': 'bicubic',
'kernel_initialer':'glorot_uniform',
'lrate':0.001,
'lrpatience': 25,
'lrfactor' : 0.5,
'metrics': 'PSNR',
'minimumlrate' : 1e-7,
'modelname':basename(__file__).split('.')[0],
'noise':'',
'normalization':['divide', 255.0], # ['standard', "53.28741141", "40.73203139"],
'normalizeback': False,
'normalizeground':False,
'outputdir':'',
'scale':2,
'seed': 19,
'shuffle' : False,
'stride':11, #
'target_channels': 3,
'target_cmode' : 'RGB',
'testpath': [r'D:\calisma\projeler\augmentation\dataset\test'], # replace with path to your test files
'traindir': r"D:\calisma\projeler\augmentation\dataset\train", # replace with path to your training files
'upscaleimage': True,
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

    SRCNN_INPUT = Input(shape=input_shape)
    SRCNN = Conv2D(64,(9,9), kernel_initializer=self.kernel_initialer, dilation_rate=self.dilation_rate,
                   padding='valid', input_shape=input_shape)(SRCNN_INPUT)
    SRCNN = self.apply_activation(SRCNN, self.activation, self.activation + "_01")
    SRCNN = Conv2D(32, (1,1), kernel_initializer=self.kernel_initialer, dilation_rate=self.dilation_rate, padding='valid')(SRCNN)
    SRCNN = self.apply_activation(SRCNN, self.activation, self.activation + "_02")
    SRCNN = Conv2D(self.target_channels,(5,5), kernel_initializer=self.kernel_initialer, dilation_rate=self.dilation_rate, padding='valid')(SRCNN)
    SRCNN = self.apply_activation(SRCNN, self.lactivation, self.lactivation + "_03")

    SRCNN = Model(SRCNN_INPUT, outputs=SRCNN)

    SRCNN.compile(Adam(self.lrate, self.decay), loss=losses.mean_squared_error)

    # SRCNN.summary()
    return SRCNN


