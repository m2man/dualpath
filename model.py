import tensorflow as tf
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from tensorflow import keras
from tensorflow.keras.regularizers import l2
import numpy as np
import pickle
import config_dualpath as config

cosine_loss = keras.losses.CosineSimilarity()

# ===== LOAD DATA =====
with open(config.dictionary_path, 'rb') as f:
  my_dictionary = pickle.load(f)

with open(config.dataset_path, 'rb') as f:
  dataset_flickr30k = pickle.load(f)

# ===== LOAD KERNEL INIT =====
from gensim.models import KeyedVectors
model_word = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

NUMB_WORDS = len(my_dictionary)
NUMB_FT = 300
kernel_init = np.zeros((NUMB_WORDS, NUMB_FT))
for idx, word in enumerate(my_dictionary):
  try:
    word_2_vec_ft = model_word.word_vec(word)
    word_2_vec_ft = np.reshape(word_2_vec_ft, (1, NUMB_FT))
  except KeyError:
    word_2_vec_ft = np.random.rand(1, NUMB_FT)
  kernel_init[idx,:] = word_2_vec_ft

def kernel_initialization(shape, dtype=None):
    kernel = np.zeros(shape)
    kernel[0,0,:,:] = kernel_init
    return kernel

# ===== DEFINE BRANCHES AND BLOCKS =====    
class BottleNeckResidualBlock(keras.layers.Layer):
  def __init__(self, n_bottleneckfilters, n_filters, kernel_regular = None, downsampling = None):
    super(BottleNeckResidualBlock, self).__init__()
    self.n_filters = n_filters
    self.n_bottleneckfilters = n_bottleneckfilters
    self.kernel_regular = kernel_regular
    self.downsampling = downsampling

  def build(self, input_shape):
    self.projection_or_not = (int(input_shape[-1]) != self.n_filters) or self.downsampling
    
    first_strides = [1, 1] 
    if self.downsampling:
      first_strides = [1, 2]

    self.main_conv1 = keras.layers.Convolution2D(filters=self.n_bottleneckfilters,
                                    kernel_size=[1, 1],
                                    strides=first_strides,
                                    padding='same',
                                    kernel_regularizer=self.kernel_regular,
                                    use_bias=False,
                                    activation=None)
    self.batch1 = keras.layers.BatchNormalization()
    self.relu = keras.layers.ReLU()
    self.main_conv2 = keras.layers.Convolution2D(filters=self.n_bottleneckfilters,
                                    kernel_size=[1, 2],
                                    strides=[1, 1],
                                    padding='same',
                                    kernel_regularizer=self.kernel_regular,
                                    use_bias=False,
                                    activation=None)
    self.batch2 = keras.layers.BatchNormalization()    
    self.main_conv3 = keras.layers.Convolution2D(filters=self.n_filters,
                                    kernel_size=[1, 1],
                                    strides=[1, 1],
                                    padding='same',
                                    kernel_regularizer=self.kernel_regular,
                                    use_bias=False,
                                    activation=None)
    self.batch3 = keras.layers.BatchNormalization()
    
    if self.projection_or_not == True:
      self.project_conv = keras.layers.Convolution2D(filters=self.n_filters,
                                    kernel_size=[1, 1],
                                    strides=first_strides,
                                    padding='same',
                                    kernel_regularizer=self.kernel_regular,
                                    use_bias=False,
                                    activation=None)
      self.project_batch = keras.layers.BatchNormalization()

  def call(self, inputs):
    main_path = self.main_conv1(inputs)
    main_path = self.batch1(main_path)
    main_path = self.relu(main_path)
    main_path = self.main_conv2(main_path)
    main_path = self.batch2(main_path)
    main_path = self.relu(main_path)
    main_path = self.main_conv3(main_path)
    main_path = self.batch3(main_path)
    if self.projection_or_not == True:
      shorcut = self.project_batch(self.project_conv(inputs))
    else:
      shorcut = inputs
    final = main_path + shorcut
    final = self.relu(final)
    return final

class Deep_CNN_Text_Model(keras.Model):
  def __init__(self, l2_rate, kernel_init=True):
    super(Deep_CNN_Text_Model, self).__init__()
    self.l2_rate = l2_rate
    if kernel_init:
      self.block1 = keras.Sequential([
                          keras.layers.Convolution2D(filters=300,
                                              kernel_size=[1,1],
                                              kernel_initializer=kernel_initialization,
                                              strides=[1,1],
                                              padding='same',
                                              kernel_regularizer=l2(self.l2_rate),
                                              use_bias=False,
                                              activation=None),
                          keras.layers.BatchNormalization(),
                          keras.layers.ReLU()            
      ])
    else:
      self.block1 = keras.Sequential([
                          keras.layers.Convolution2D(filters=300,
                                              kernel_size=[1,1],
                                              strides=[1,1],
                                              padding='same',
                                              kernel_regularizer=l2(self.l2_rate),
                                              use_bias=False,
                                              activation=None),
                          keras.layers.BatchNormalization(),
                          keras.layers.ReLU()            
      ])
    self.block2 = keras.Sequential([
                        #keras.layers.MaxPool2D(pool_size=[1,2], strides=[1,2]),
                        BottleNeckResidualBlock(64, 256, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(64, 256, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(64, 256, kernel_regular=l2(self.l2_rate), downsampling=False),      
    ])
    self.block3 = keras.Sequential([
                        BottleNeckResidualBlock(128, 512, kernel_regular=l2(self.l2_rate), downsampling=True),
                        BottleNeckResidualBlock(128, 512, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(128, 512, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(128, 512, kernel_regular=l2(self.l2_rate), downsampling=False),
    ])
    self.block4 = keras.Sequential([
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=True),
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=False),
    ])
    self.block5 = keras.Sequential([
                        BottleNeckResidualBlock(512, 2048, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(512, 2048, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(512, 2048, kernel_regular=l2(self.l2_rate), downsampling=False),
    ])
    self.block6 = keras.Sequential([
                        keras.layers.GlobalAveragePooling2D(),
                        keras.layers.Dense(2048),
                        keras.layers.BatchNormalization(),
                        keras.layers.ReLU(),
                        keras.layers.Dropout(0.5),
                        keras.layers.Dense(2048),   
    ])

  def call(self, inputs):
    x = self.block1(inputs)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    return x

# ===== DEFINE LOSS =====
def total_loss(model, input_x, target_y, alpha=1, lamb_0=1, lamb_1=1, lamb_2=1, training=True):
  image_y, text_y, f_image_y, f_text_y = model(input_x, training=training)
  true_class = np.argmax(target_y, axis=1)
  batch_size = true_class.shape[0]
  #batch_size = tf.dtypes.cast(batch_size, tf.int32)

  p_visual = tf.math.reduce_sum(tf.math.multiply(image_y[0:batch_size], target_y), axis = 1)
  L_visual = -tf.math.reduce_mean(tf.math.log(p_visual + 1e-20))

  p_text = tf.math.reduce_sum(tf.math.multiply(text_y[0:batch_size], target_y), axis = 1)
  L_text = -tf.math.reduce_mean(tf.math.log(p_text + 1e-20))
  #print("P_Vis {:.6f}: P_Txt {:.6f}: L_Vis {:.6f}: L_Txt {:.6f}".format(tf.math.reduce_min(p_visual), 
  #                                                                      tf.math.reduce_min(p_text),
  #                                                                      L_visual, L_text))
  instance_loss = tf.math.add(lamb_1*L_visual, lamb_2*L_text)

  ranking_loss = 0
  for i in range(batch_size):
    Ia = f_image_y[i]
    Ta = f_text_y[i]
    In = f_image_y[batch_size + i]
    Tn = f_text_y[batch_size + i]
    ranking_loss += tf.math.add(tf.math.maximum(0.0, alpha - (cosine_loss(Ia, Ta) - cosine_loss(Ia, Tn))),
                                tf.math.maximum(0.0, alpha - (cosine_loss(Ta, Ia) - cosine_loss(Ta, In))))
  ranking_loss = ranking_loss / batch_size
  #print("instance loss: {} --- ranking loss: {}".format(instance_loss, ranking_loss))
  loss = tf.math.add(lamb_0 * ranking_loss, instance_loss)
  return loss, L_visual, L_text, ranking_loss


# ===== DEFINE GRAD =====
def grad(model, input_x, target_y, alpha=1, lamb_0=1, lamb_1=1, lamb_2=1):
  with tf.GradientTape() as tape:
    loss, loss_vis, loss_text, ranking_loss = total_loss(model, input_x, target_y, alpha, lamb_0, lamb_1, lamb_2)
  return loss, loss_vis, loss_text, ranking_loss, tape.gradient(loss, model.trainable_variables)

# ===== DEFINE ENTIRE MODEL =====
def create_model(nclass, nword=len(my_dictionary), pretrained_model='', ft_resnet=False):
  # ft_resnet: fine tunning resnet or not (stage 1: False, stage 2: should True)
  INPUT_SIZE = (224, 224, 3)
  input_test = keras.layers.Input(shape=INPUT_SIZE)
  resnet = keras.applications.ResNet50(input_shape=INPUT_SIZE,
                                      weights='imagenet',
                                      include_top=False)
  resnet.trainable = ft_resnet
  average_pooling = keras.layers.GlobalAveragePooling2D()
  fc_1 = keras.layers.Dense(2048)
  bn_1 = keras.layers.BatchNormalization()
  relu_1 = keras.layers.ReLU()
  do_1 = keras.layers.Dropout(0.5)
  fc_2 = keras.layers.Dense(2048)
  deep_cnn_branch = keras.Sequential([resnet, average_pooling, fc_1, bn_1, relu_1,
                                      do_1, fc_2])

  deep_text_branch = Deep_CNN_Text_Model(l2_rate = 0.001, kernel_init=config.kernel_init)

  image_input = keras.Input(shape=(224, 224, 3), name='image')  # Variable-length sequence of ints
  text_input = keras.Input(shape=(1, 32, nword), name='text') 
  image_f = deep_cnn_branch(image_input)
  text_f = deep_text_branch(text_input)
  share_weights = keras.layers.Dense(nclass, activation='softmax')
  image_final_class = share_weights(image_f)
  text_final_class = share_weights(text_f)

  dualpath_model = keras.Model(inputs=[image_input, text_input], outputs=[image_final_class, text_final_class, image_f, text_f])
  dualpath_model.summary()

  return dualpath_model
