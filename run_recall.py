'''
Run evaluation by recall@k in validate set
'''
from model import *
import mylibrary as mylib
import random
import tqdm
import numpy as np
import pickle
import config_dualpath as config
import os
import argparse
import tensorflow.keras as keras
from scipy import spatial 

def parse_arguments():
  parser = argparse.ArgumentParser(
      description="Run evaluation on recall@k"
  )

  parser.add_argument(
        "-se",
        "--start_epoch",
        type=int,
        nargs="?",
        help="Choose trained model from specific epoch to run evaluation",
  )

  parser.add_argument(
        "-ee",
        "--end_epoch",
        type=int,
        nargs="?",
        help="Choose trained model to specific epoch to run evaluation",
        default=0
  )

  parser.add_argument(
        "-k",
        "--top_k",
        type=str,
        nargs="?",
        help="Choose top k (Can be multiple value separated by , ",
        default='5'
  )

  return parser.parse_args()

def get_branches_from_model(model):
  '''
  Get image and text branches from the general model
  Return 2 branches (keras sequential layers)
  '''
  image_layers_index = [0, 2]
  text_layers_index = [1, 3]

  image_branch = keras.Sequential()
  for idx in image_layers_index:
    image_branch.add(model.get_layer(index=idx))

  text_branch = keras.Sequential()
  for idx in text_layers_index:
    text_branch.add(model.get_layer(index=idx))

  return image_branch, text_branch


def get_img_ft_from_branch(image_branch, list_img_names, batch=1):
	'''
	Get embedded feature from image branch from list img names
	Return numpy array feature
	'''
	resize_img = 224
	print("Processing Images Branch ...")
	images_out = np.zeros([len(list_img_names), 2048])
	for idx in range(0, len(list_img_names), batch):
		batch_img = np.zeros([min(batch, len(list_img_names)-idx), resize_img, resize_img, 3])
		for idx_batch in range(batch_img.shape[0]):
			img_name = list_img_names[idx:idx+idx_batch]
			try:
				for image_folder in config.image_folders:
					try:
						image_input_x = mylib.embedding_image(link_to_image=image_folder+img_name, resize=224, prob_aug=0)
						break
					except FileNotFoundError:
						continue
			except:
				print("Error happen in {}!".format(img_name))
			batch_img[idx_batch] = image_input_x
		image_input_x = tf.convert_to_tensor(batch_img, dtype=tf.float32)
		images_out[idx:(idx+idx_batch+1)] = image_branch(image_input_x, training=False).numpy()
		
	'''	
	for idx, img_name in enumerate(list_img_names):
		try:
			for image_folder in config.image_folders:
				try:
					image_input_x = mylib.embedding_image(link_to_image=image_folder+img_name, resize=224, prob_aug=0)
					break
				except FileNotFoundError:
					continue
		except:
			print("Error happen in {}!".format(img_name))

	image_input_x = tf.convert_to_tensor(image_input_x, dtype=tf.float32)
	images_out[idx] = image_branch(image_input_x, training=False).numpy()
	'''
	print("Finished processing Images Branch ...")
	return images_out

def get_txt_ft_from_branch(text_branch, list_text, batch=1):
  '''
  Get embedded feature from text branch from list text
  Return numpy array feature
  '''
	print("Processing Text Branch ...")
	max_len = 32
	text_out = np.zeros([len(list_text), 2048])
	for idx in range(0, len(list_text), batch):
		batch_txt = np.zeros([min(batch, len(list_text)-idx), 1, max_len, len(my_dictionary)])
		for idx_batch in range(batch_txt.shape[0]):
			text_x = list_text[idx+idx_batch]
			batch_txt[idx_batch] = mylib.embedding_sentence(text_x, my_dictionary, max_len=max_len)
		text_input_x = tf.convert_to_tensor(batch_txt, dtype=tf.float32)
		text_out[idx:(idx+idx_batch+1)] = text_branch(text_input_x, training=False).numpy()
		
	'''
	for idx, text_x in enumerate(list_text):
		text_input_x = mylib.embedding_sentence(text_x, my_dictionary, max_len=max_len)
		text_input_x = tf.convert_to_tensor(text_input_x, dtype=tf.float32)
		text_out[idx] = text_branch(text_input_x, training=False).numpy()
	'''
	print("Finished processing Text Branch ...")
	return text_out

def recall_at_k_faster(images_out_array, images_lbl_array, text_out_array, text_lbl_array, k_values_list = [5]):
  # Calculate recall@k in 2 cases: image query and text query
  # image query is to give an image --> find relevant text descriptions of that image in top k_value
  # text query is to give a text --> find a relevant image of that text in top k_value
  # Input:
  #   + images_out_array, text_out_array: result of the 2 branches in dualpath model
  #   + images_lbl_array, text_lbl_array: list of classes of each elements (need to be corresponding to the order of out_array)
  #   + k_values_list (list --> run for multiple value of k): top k_value that we calculate for the recall
  # Output:
  #   + recall_image_query (vector of n_images), recall_text_query (vector of n_text) is the recall of each case that have run for all element in the list images/text
  #   + Note that the final result should be the average of the output

  total_text = text_out_array.shape[0]
  total_images = images_out_array.shape[0]
  k_values_list_sorted = sorted(k_values_list, reverse=True) # descending order [1, 10, 5] --> [10, 5, 1]

  print("Processing recall IMAGE query")
  recall_image_query = np.zeros([len(k_values_list), total_images])
  cosine_distance = spatial.distance.cdist(images_out_array, text_out_array, 'cosine')
  sorted_index = np.argsort(cosine_distance, axis=1)
  sorted_index = sorted_index[:, 0:k_values_list_sorted[0]]
  for img_idx in range(total_images):
    ranked_text_lbl = [text_lbl_array[idx] for idx in sorted_index[img_idx]]
    for k_idx, k_value in enumerate(k_values_list):
      ranked_text_lbl_k_value = ranked_text_lbl[0:k_value]
      correct = [1 for x in ranked_text_lbl_k_value if x == images_lbl_array[img_idx]]
      #rc = sum(correct)/min(5, k_value) # should be 5, each of images have 5 relevant text
      #rc = sum(correct)/5
      if sum(correct) > 0:
        rc = 1
      else:
        rc = 0
      recall_image_query[k_idx, img_idx] = rc
  print("Finished recall IMAGE query")
  
  print("Processing recall TEXT query")
  recall_text_query = np.zeros([len(k_values_list), total_text])
  cosine_distance = spatial.distance.cdist(text_out_array, images_out_array, 'cosine')
  sorted_index = np.argsort(cosine_distance, axis=1)
  sorted_index = sorted_index[:, 0:k_values_list_sorted[0]]
  for txt_idx in range(total_text):
    ranked_images_lbl = [images_lbl_array[idx] for idx in sorted_index[txt_idx]]
    for k_idx, k_value in enumerate(k_values_list):
      ranked_images_lbl_k_value = ranked_images_lbl[0:k_value]
      correct = [1 for x in ranked_images_lbl_k_value if x == text_lbl_array[txt_idx]]
      #rc = sum(correct)/1 # each text only have 1 relevant image
      if sum(correct) > 0:
        rc = 1
      else:
        rc = 0
      recall_text_query[k_idx, txt_idx] = rc
  print("Finished recall TEXT query")

  return recall_image_query, recall_text_query

def main():
  # Argument parsing
	args = parse_arguments()

	top_k_str = args.top_k
	top_k_str = top_k_str.replace(" ", "")
	top_k_str = top_k_str.split(",")
	top_k = [int(x) for x in top_k_str]

	start_epoch = args.start_epoch
	if args.end_epoch == 0 or args.end_epoch < start_epoch:
		end_epoch = start_epoch
	else:
		end_epoch = args.end_epoch

	info = "Evaluation Recall from epoch {} to {} with top_k = [".format(start_epoch, end_epoch)
	for k in top_k:
		info += "{},".format(k)
	info = info[0:-1] + "]"
	print("---------------\n\n" + info + "\n\n---------------")    

	log_filename = os.path.join('report_sgd/','stage_1_recall_train.log')
	if not os.path.exists('report_sgd'):
		os.mkdir('report_sgd')

	images_names = list(dataset_flickr30k.keys())

	index_flickr30k = [i for i in range(len(dataset_flickr30k))]
	index_train, index_validate, index_test = mylib.generate_3_subsets(index_flickr30k, ratio = [0.93, 0.035, 0.035])
	print("Number of samples in 3 subsets are {}, {} and {}". format(len(index_train), len(index_validate), len(index_test)))

	images_names_train = [images_names[i] for i in index_train]
	images_names_val = [images_names[i] for i in index_validate]
	images_names_test = [images_names[i] for i in index_test]

	list_dataset = []
	all_class = [x[0:-4] for x in images_names_train]
	all_class_sort = sorted(all_class)
	for idx, img_name in enumerate(images_names_train):
		img_class = all_class_sort.index(img_name[0:-4])
		for des in dataset_flickr30k[img_name]:
			temp = [img_name, des, img_class]
			list_dataset.append(temp)
	
	list_img_names = images_names_val
	images_class = [x[2] for x in list_dataset]
	list_text = [x[1] for x in list_dataset]
	text_class = images_class
	
	for epoch in range(start_epoch, end_epoch+1):
		print("Calculating Recall in epoch {}". format(epoch))

		pretrained_model = 'weights/checkpoints_' + str(epoch) + '/my_checkpoint'
		model = create_model(nclass=len(images_names_train), nword=len(my_dictionary), ft_resnet=False)
		model.load_weights(pretrained_model)

		img_branch, txt_branch = get_branches_from_model(model)

		images_out = get_img_ft_from_branch(img_branch, list_img_names)
		text_out = get_txt_ft_from_branch(txt_branch, list_text)

		a, b = recall_at_k_faster(images_out, images_class, text_out, text_class, top_k)

		info = 'Epoch {}\n'.format(epoch)
		for k_idx, k_value in enumerate(top_k):
			info += "Recall@{}_Image_Query: {:6f}\nRecall@{}_Text_Query: {:6f}\n".format(k_value, np.mean(a[k_idx]), k_value, np.mean(b[k_idx]))
		info += '---------------\n'
		print("Writing to report file")
		with open(log_filename, 'a') as f:
			f.write(info)

if __name__ == '__main__':
  main()
