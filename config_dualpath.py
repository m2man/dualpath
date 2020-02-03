dictionary_path = 'Flickr30k_dictionary.pickle'
dataset_path = 'Flickr30k_dataset.pickle'
numb_epochs = 11
seeds = [x for x in range(numb_epochs)]
batch_size = 32
#learn_rate = 0.01
#moment_val = 0.9

image_folders = ['flickr30k_images/']

# Stage 2 or stage 1
stage_2 = False

# Loss parameter
alpha = 1
lamb_0 = 0
lamb_1 = 1
lamb_2 = 1

kernel_init = True # Only set true if run from sratch (if load pretrained --> should make false for faster loading)
