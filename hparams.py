import cv2
import os
import editdistance
from os.path import join
import numpy as np
from collections import Counter
from tqdm import tqdm


class Hparams:
    def __init__(self):

        # Working directory
        self.dir = "log/"

        # Paths to checkpoints.
        self.chk = ''  # Training
        # Inference
        self.weights1 = "weights1.pt"
        self.weights2 = "weights2.pt"
        self.weights3 = "weights3.pt"
        
        # Path to folder with target texts
        self.trans_dir = 'train/words'
        
        # Path to folder with images
        self.image_dir = 'train/images'
        
        # Path to folder with test data
        self.test_dir = 'data/'
        
        # These characters will be deleted
        self.del_sym = ['b', 'd', 'a', 'c', '×', '⊕', ')', '|', 'n', 'm', 'g', 'ǂ', '/', 'k', 'o', '–', '⊗', 'l', '…', 'u','h','і', 'f','t','p', 'r', 'e','s']
        
        # Learning rate
        self.lr = 1e-4
        
        # BatchSize
        self.batch_size = 16 
        
        # Hidden layer size
        self.hidden = 512
        
        # Number of encoder layers in Transformer
        self.enc_layers = 1
        
        # Number of decoder layers in Transformer
        self.dec_layers = 1
        
        # Number of attention heads in Transformer
        self.nhead = 4
        
        # Dropout
        self.resnextdropout = 0.16
        self.densenetdropout = 0.2
 
        # Image size
        self.width = 1024
        self.height = 128

        # Stretch value for stretching / squeezing augmentation
        self.stretch = (1, 1)

        # Hyperparameters for ExtraLines augmentation
        self.number_of_lines = 1
        self.width_of_lines = 8


# Getting hyperparameters.
hp = Hparams()


def process_texts(image_dir,trans_dir):
    """The function ignores samples containing characters from del_sym."""
    lens,lines,names = [],[],[]
    letters = ''
    all_word = {}
    all_files = os.listdir(trans_dir)
    for filename in os.listdir(image_dir):
        if filename[:-3]+'txt' in all_files:
            name, _ = os.path.splitext(filename)
            txt_filepath = join(trans_dir, name + '.txt')
            with open(txt_filepath, 'r', encoding="utf-8") as file:
                data = file.read()
                if len(data)==0:
                    continue
                if len(set(data).intersection(hp.del_sym))>0:
                    continue
                lines.append(data)
                names.append(filename)
                lens.append(len(data))
                letters = letters + ' ' + data
    words = letters.split()
    for word in words:
        if not word in all_word:
            all_word[word] = 0
        else:
            all_word[word] += 1

    cnt = Counter(letters)

    print('Max string length:', max(Counter(lens).keys()))
    return names,lines,cnt,all_word


def text_to_labels(s, p2idx):
    """Translates text to an array of indexes."""
    return [p2idx['SOS']] + [p2idx[i] for i in s if i in p2idx.keys()] + [p2idx['EOS']]


def labels_to_text(s, idx2p):
    """Translates indexes to text."""
    string = "".join([idx2p[i] for i in s])
    if string.find('EOS') == -1:
        return string
    else:
        return string[:string.find('EOS')]


def phoneme_error_rate(p_seq1, p_seq2):
    """CER count."""
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return editdistance.eval(''.join(c_seq1),
                                ''.join(c_seq2)) / len(c_seq2)


def process_image(img):
    """
    The following function loads images, changes them to the required size,
    and normalizes them.
    """
    h, w = img.shape
    if h > w * 1.25:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  
    img  = np.stack([img, img, img], axis=-1)
    w, h,_ = img.shape
    
    new_w = hp.height
    new_h = int(h * (new_w / w)) 
    img = cv2.resize(img, (new_h, new_w))
    w, h,_ = img.shape
    
    img = img.astype('float32')
    
    new_h = hp.width
    if h < new_h:
        add_zeros = np.full((w, new_h-h,3), 255)
        img = np.concatenate((img, add_zeros), axis=1)
    
    if h > new_h:
        img = cv2.resize(img, (new_h,new_w))
    
    return img


def generate_data(names,image_dir='train1/images'):
    data_images = []
    for name in tqdm(names):
        img = cv2.imread(image_dir+'/'+name,cv2.IMREAD_GRAYSCALE)
        img = process_image(img)
        data_images.append(img.astype('uint8'))
    return data_images


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
