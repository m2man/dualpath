import tensorflow as tf

print("Require Tensorflow >= 2.0.0 and Current TF version is " + str(tf.__version__))
#tf.enable_eager_execution() # Enable interactive tensorflow
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import nltk
nltk.download('stopwords')
nltk.download('punkt')

import time
import random
import datetime

import numpy as np
import pickle
import json
import mylibrary as mylib
from tqdm import tqdm
import os
import config_dualpath as config
from model import *

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # CPU

# ===== MAIN FUNCTION =====
def main():
  save_path = 'checkpoints/' # directory path to save checkpoint
  if config.stage_2:
    log_filename = os.path.join('report/dualpath/','dualpath_stage_2.log')
  else:
    log_filename = os.path.join('report/dualpath/','dualpath_stage_1.log')

  if not os.path.exists('report/dualpath/'):
    os.mkdir('report/')
    os.mkdir('report/dualpath/')
 
  images_names = list(dataset_flickr30k.keys())

  index_flickr30k = [i for i in range(len(dataset_flickr30k))]
  index_train, index_validate, index_test = mylib.generate_3_subsets(index_flickr30k, ratio = [0.93, 0.035, 0.035])
  print("Number of samples in 3 subsets are {}, {} and {}". format(len(index_train), len(index_validate), len(index_test)))

  images_names_train = [images_names[i] for i in index_train]
  images_names_val = [images_names[i] for i in index_validate]
  images_names_test = [images_names[i] for i in index_test]

  list_dataset = []
  all_class = []
  for idx, img_name in enumerate(images_names_train):
    img_class = img_name[0:-4] # remove '.jpg'
    all_class += [img_class]
    for des in dataset_flickr30k[img_name]:
      temp = [img_name, des, img_class]
      list_dataset.append(temp)
  label_encoder, onehot_all_class = mylib.create_onehot_all_label(all_class)

  seeds = [x for x in range(config.numb_epochs)]

  if config.stage_2: # parameter for ranking loss (stage 2)
    lamb_0 = config.lamb_0
  else:
    lamb_0 = 0

  model = create_model(nclass=len(images_names_train), nword=len(my_dictionary), ft_resnet=False)
  
  optimizer = keras.optimizers.Adadelta()
  ckpt = tf.train.Checkpoint(iters=tf.Variable(0), 
                            epoch=tf.Variable(0), 
                            optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=1)
  
  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")
  last_epoch = int(ckpt.epoch)
  last_index = int(ckpt.iters)
    
  for current_epoch in range(last_epoch, config.numb_epochs):
    epoch_loss_total_avg = tf.keras.metrics.Mean() # track mean loss in current epoch
    epoch_loss_visual_avg = tf.keras.metrics.Mean() # track mean loss in current epoch
    epoch_loss_text_avg = tf.keras.metrics.Mean() # track mean loss in current epoch
    if config.stage_2:
      epoch_loss_ranking_avg = tf.keras.metrics.Mean() # track mean loss in current epoch
    
    print("Generate Batch")
    batch_dataset = mylib.generate_batch_dataset(list_dataset, config.batch_size, seed=seeds[current_epoch])
    
    print("Start Training")
    for index in tqdm(range(last_index, len(batch_dataset))):
      batch_data = batch_dataset[index]
      img_ft, txt_ft, lbl = mylib.get_feature_from_batch(batch_data, 
                                                        image_folders=config.image_folders,
                                                        dictionary=my_dictionary,
                                                        list_label=all_class,
                                                        onehot_all_label=onehot_all_class,
                                                        resize_img=224, max_len=32)
      inputs = [tf.convert_to_tensor(img_ft, dtype=tf.float32), 
                tf.convert_to_tensor(txt_ft, dtype=tf.float32)]

      loss_value, loss_vis, loss_txt, loss_rank, grads = grad(model, inputs, lbl, 
                                                              alpha=config.alpha, lamb_0=lamb_0, 
                                                              lamb_1=config.lamb_1, lamb_2=config.lamb_2)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      # Track mean loss in current epoch    
      epoch_loss_total_avg(loss_value)
      epoch_loss_visual_avg(loss_vis)
      epoch_loss_text_avg(loss_txt)
      if config.stage_2:
        epoch_loss_ranking_avg(loss_rank)
        info = "Epoch {}/{}: Iter batch {}/{}\nLoss_Visual: {:.6f}\nLoss_Text: {:.6f}\nLoss_Ranking: {:.6f}\nLoss_Total: {:.6f}\n-----".format(current_epoch+1, config.numb_epochs, 
                                                                                                                                                index+1, len(batch_dataset),
                                                                                                                                                epoch_loss_visual_avg.result(),
                                                                                                                                                epoch_loss_text_avg.result(),
                                                                                                                                                epoch_loss_ranking_avg.result(),
                                                                                                                                                epoch_loss_total_avg.result())
      else:
        info = "Epoch {}/{}: Iter batch {}/{}\nLoss_Visual: {:.6f}\nLoss_Text: {:.6f}\nLoss_Total: {:.6f}\n-----".format(current_epoch+1, config.numb_epochs, 
                                                                                                                          index+1, len(batch_dataset),
                                                                                                                          epoch_loss_visual_avg.result(),
                                                                                                                          epoch_loss_text_avg.result(),
                                                                                                                          epoch_loss_total_avg.result())
      ckpt.iters.assign_add(1)                                                            
      
      if (index+1) % 20 == 0 or index <= 9:
        print(info)
      
      if (index+1) % 20 == 0:
        save_path = manager.save()
        print("Saved checkpoint for epoch {} - iter {}: {}".format(int(ckpt.epoch)+1, int(ckpt.iters), save_path))
        with open(log_filename, 'a') as f:
          f.write('{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
          f.write(info+'\n')
    
    print(info)
    save_path = manager.save()
    print("Saved checkpoint for epoch {} - iter {}: {}".format(int(ckpt.epoch), int(ckpt.iters), save_path)) 
    
    last_index = 0
    ckpt.iters = tf.Variable(0)
    ckpt.epoch.assign_add(1)
     

if __name__ == "__main__":
    main()
