dictionary_path = 'Flickr30k_dictionary.pickle'
dataset_path = 'Flickr30k_dataset.pickle'
numb_epochs = 11
seeds = [x for x in range(numb_epochs)]
last_epoch = 0
last_index = 0
batch_size = 32
learn_rate = 0.01
moment_val = 0.7

image_folders = ['flickr30k_images/']

# Apply decay lr or not
decay_lr = False
decay_lr_portion = 0.3 # portion of decreasing lr --> lr[t] = (1-portion)*lr[t-1]
decay_lr_min = 0.001 # minimum of lr

# Stage 2 or stage 1
stage_2 = False

# Loss parameter
alpha = 1
lamb_0 = 0
lamb_1 = 1
lamb_2 = 1

kernel_init = True # Only set true if run from sratch (if load pretrained --> should make false for faster loading)