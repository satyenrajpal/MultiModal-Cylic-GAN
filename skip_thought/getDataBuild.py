import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import pickle

data = dset.CocoCaptions(root = '<images path>',
                        annFile = '<annontations path>',
                        transform=transforms.ToTensor())

with open('/home/ubuntu/vocab_file.pkl', 'rb') as handle:
    indx_to_word_dict = pickle.load(handle)
    
word_to_indx_dict = dict (zip(indx_to_word_dict.values(),indx_to_word_dict.keys()))

train_proc_data_w = []
for i in np.arange(0, len(data)):
    img, captions = data[i]
    curr_image_captions = []
    for caption in captions:
        words_curr = caption.replace(".","").replace(",","").lower().split()
        caps_ind = []
        for word in words_curr:
            if word in word_to_indx_dict.keys():
                caps_ind.append(word_to_indx_dict[word])
            else:
                continue;
        #print(len(caps_ind))       
        #caps_ind = [word_to_indx_dict[word] for word in caption.replace(".","").replace(",","")lower().split()]
        curr_image_captions.append(np.array(caps_ind))
    train_proc_data_w.append(np.array(curr_image_captions))
print(len(train_proc_data_w))
print(train_proc_data_w[1000])