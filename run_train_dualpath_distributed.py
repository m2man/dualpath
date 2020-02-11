import tensorflow as tf

print("Require Tensorflow >= 2.0.0 and Current TF version is " + str(tf.__version__))
#tf.enable_eager_execution() # Enable interactive tensorflow
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

import time
import random
import datetime

#from tensorflow import keras
#from tensorflow.keras.regularizers import l2
import numpy as np
import pickle
import json
import mylibrary as mylib
from tqdm import tqdm
import os
import config_dualpath as config
from model import *
import argparse
import shutil

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #CPU
# export CUDA_VISIBLE_DEVICES="" 
# unset CUDA_VISIBLE_DEVICES
tf.keras.backend.set_floatx('float32')
mirrored_strategy = tf.distribute.MirroredStrategy()

def parse_arguments():
  parser = argparse.ArgumentParser(
      description="Run dualpath training"
  )

  parser.add_argument(
        "-fs",
        "--from_scratch",
        action="store_true",
        default=False,
        help="Run from scratch, remove all saved checkpoints",
  )

  return parser.parse_args()

# ===== MAIN FUNCTION =====
def main():
	
	num_gpu = int(mirrored_strategy.num_replicas_in_sync)
	global_batch_size = config.batch_size * num_gpu
	
	@tf.function
	def train_step(dist_img, dist_txt, dist_lbl):
		def step_fn(img, txt, lbl):
			inputs = [tf.dtypes.cast(img, tf.float32), tf.dtypes.cast(txt, tf.float32)]
			with tf.GradientTape() as tape:
				loss, loss_vis, loss_text, ranking_loss = total_loss_distributed(model, inputs, lbl, config.alpha, config.lamb_0, config.lamb_1, config.lamb_2)
				loss = loss*(1.0/global_batch_size)
				loss_vis = loss_vis*(1.0/global_batch_size)
				loss_text = loss_text*(1.0/global_batch_size)
				ranking_loss = ranking_loss*(1.0/global_batch_size)
			grads = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
			return loss, loss_vis, loss_text, ranking_loss
			
		per_loss, per_loss_vis, per_loss_text, per_ranking_loss = mirrored_strategy.experimental_run_v2(step_fn, args=(dist_img, dist_txt, dist_lbl,))
		
		print(per_loss)
		
		mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss, axis=0) / global_batch_size
		mean_loss_vis = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss_vis, axis=0) / global_batch_size
		mean_loss_text = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss_text, axis=0) / global_batch_size
		mean_ranking_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_ranking_loss, axis=0) / global_batch_size
		
		return mean_loss, mean_loss_vis, mean_loss_text, mean_ranking_loss
		
	args = parse_arguments()

	save_path = 'checkpoints_distributed/' # directory path to save checkpoint
	if config.stage_2:
		log_filename = os.path.join('report_distributed/','dualpath_stage_2.log')
	else:
		log_filename = os.path.join('report_distributed','dualpath_stage_1.log')

	if args.from_scratch:
		try:
			shutil.rmtree('report_distributed')
			shutil.rmtree('checkpoints_distributed')
		except:
			pass

	if not os.path.exists(save_path):
		os.mkdir(save_path)

	if not os.path.exists('report_distributed/'):
		os.makedirs('report_distributed/')

	images_names = list(dataset_flickr30k.keys())

	index_flickr30k = [i for i in range(len(dataset_flickr30k))]
	index_train, index_validate, index_test = mylib.generate_3_subsets(index_flickr30k, ratio = [0.93, 0.035, 0.035])
	print("Number of samples in 3 subsets are {}, {} and {}". format(len(index_train), len(index_validate), len(index_test)))

	images_names_train = [images_names[i] for i in index_train]
	images_names_val = [images_names[i] for i in index_validate]
	images_names_test = [images_names[i] for i in index_test]

	list_dataset = []
	all_class = [x[0:-4] for x in images_names_val]
	all_class_sort = sorted(all_class)
	for idx, img_name in enumerate(images_names_val):
		img_class = all_class_sort.index(img_name[0:-4])
		for des in dataset_flickr30k[img_name]:
			temp = [img_name, des, img_class]
			list_dataset.append(temp)

	seeds = [x for x in range(config.numb_epochs)]
	
	#print("Running on " + config_gpu)

	with mirrored_strategy.scope():
		model = create_model(nclass=len(images_names_val), nword=len(my_dictionary), ft_resnet=False)
	optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

	ckpt = tf.train.Checkpoint(iters=tf.Variable(0), 
				                    epoch=tf.Variable(0), 
				                    optimizer=optimizer, model=model)
	manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=1)
	
	if args.from_scratch == False:
		ckpt.restore(manager.latest_checkpoint)
		if manager.latest_checkpoint:
			print("Restored from {}".format(manager.latest_checkpoint))
		else:
			print("Initializing from scratch.")
	else:
		print("Force to initializing from scratch.")
			
	last_epoch = int(ckpt.epoch)
	last_index = int(ckpt.iters)
		
	for current_epoch in range(last_epoch, config.numb_epochs):
		with tf.device('/device:CPU:0'):
			epoch_loss_total_avg = tf.keras.metrics.Mean() # track mean loss in current epoch
			epoch_loss_visual_avg = tf.keras.metrics.Mean() # track mean loss in current epoch
			epoch_loss_text_avg = tf.keras.metrics.Mean() # track mean loss in current epoch
			if config.stage_2:
				epoch_loss_ranking_avg = tf.keras.metrics.Mean() # track mean loss in current epoch
		
		print("Generate Batch")
		batch_dataset = mylib.generate_batch_dataset(list_dataset, config.batch_size, seed=seeds[current_epoch])
		n_samples = len(batch_dataset)
		
		print("Start Training")
		index = 0
		last_index=0
		
		for index in range(last_index, n_samples, num_gpu):
			with tf.device('/device:CPU:0'):
				for idx_gpu in range(num_gpu):
					batch_data = batch_dataset[index + idx_gpu]
					img_ft, txt_ft, lbl = mylib.get_feature_from_batch(batch_data, 
		                                                         image_folders=config.image_folders,
		                                                         dictionary=my_dictionary,
		                                                         resize_img=224, max_len=32)
					if idx_gpu == 0:
						total_img_ft = img_ft
						total_txt_ft = txt_ft
						total_lbl = lbl
					else:
						total_img_ft = np.concatenate((total_img_ft, img_ft))
						total_txt_ft = np.concatenate((total_txt_ft, txt_ft))
						total_lbl = np.concatenate((total_lbl, lbl))
				
				print(f"{total_img_ft.shape} --- {total_txt_ft.shape} --- {total_lbl.shape}")
				
				n_img = total_img_ft.shape[0]
				n_txt = total_txt_ft.shape[0]
				n_lbl = total_lbl.shape[0]
				
				total_img_ft = tf.data.Dataset.from_tensor_slices(total_img_ft).batch(n_img)
				total_txt_ft = tf.data.Dataset.from_tensor_slices(total_txt_ft).batch(n_txt)
				total_lbl = tf.data.Dataset.from_tensor_slices(total_lbl).batch(n_lbl)
				
				total_img_ft = mirrored_strategy.experimental_distribute_dataset(total_img_ft)
				total_txt_ft = mirrored_strategy.experimental_distribute_dataset(total_txt_ft)
				total_lbl = mirrored_strategy.experimental_distribute_dataset(total_lbl)	
		
			with mirrored_strategy.scope():
				for img_ft, txt_ft, lbl in zip(total_img_ft, total_txt_ft, total_lbl):
					loss_value, loss_vis, loss_txt, loss_rank = train_step(img_ft, txt_ft, lbl)
					
			# Track mean loss in current epoch    
			epoch_loss_total_avg(loss_value)
			epoch_loss_visual_avg(loss_vis)
			epoch_loss_text_avg(loss_txt)
			if config.stage_2:
				epoch_loss_ranking_avg(loss_rank)
				info = "Epoch {}/{}: Iter batch {}/{}\nLoss_Visual: {:.6f}\nLoss_Text: {:.6f}\nLoss_Ranking: {:.6f}\nLoss_Total: {:.6f}\n-----".format(current_epoch+1, config.numb_epochs, 
										                                                                                                index+1, n_samples,
										                                                                                                epoch_loss_visual_avg.result(),
										                                                                                                epoch_loss_text_avg.result(),
										                                                                                                epoch_loss_ranking_avg.result(),
										                                                                                                epoch_loss_total_avg.result())
			else:
				info = "Epoch {}/{}: Iter batch {}/{}\nLoss_Visual: {:.6f}\nLoss_Text: {:.6f}\nLoss_Total: {:.6f}\n-----".format(current_epoch+1, config.numb_epochs, 
										                                                                          index+1, n_samples/num_gpu,
										                                                                          epoch_loss_visual_avg.result(),
										                                                                          epoch_loss_text_avg.result(),
										                                                                          epoch_loss_total_avg.result())
			ckpt.iters.assign_add(2)                                                            
				
			if (index+1) % 20 == 0:
				save_path = manager.save()
				print("Saved checkpoint for epoch {} - iter {}: {}".format(int(ckpt.epoch)+1, int(ckpt.iters), save_path))
				with open(log_filename, 'a') as f:
					f.write('{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
					f.write(info+'\n')
				print(info)
				
			 
		print(info)
		save_path = manager.save()
		print("Saved checkpoint for epoch {} - iter {}: {}".format(int(ckpt.epoch), int(ckpt.iters), save_path)) 
		
		last_index = 0
		ckpt.iters = tf.Variable(0)
		ckpt.epoch.assign_add(1)


if __name__ == "__main__":
    main()
