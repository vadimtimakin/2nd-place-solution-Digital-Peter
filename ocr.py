import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hparams import Hparams, process_texts, text_to_labels, labels_to_text, phoneme_error_rate, process_image, generate_data, count_parameters
import cv2, os, argparse, time, random, math
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json
from custom_functions import TripleEnsemble, postprocess


# Сделать текст в батче одной длины
class TextCollate():
    def __call__(self, batch):
        x_padded = []
        max_y_len = max([i[1].size(0) for i in batch])
        y_padded = torch.LongTensor(max_y_len, len(batch))
        y_padded.zero_()

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))
            y = batch[i][1]
            y_padded[:y.size(0), i] = y
        
        x_padded = torch.cat(x_padded)
        return x_padded, y_padded   


# Датасет загрузки изображений и тексты
class TextLoader(torch.utils.data.Dataset):
    def __init__(self,name_image,label,image_dir,eval=False):
        self.name_image = name_image
        self.label = label
        self.image_dir = image_dir
        self.eval = eval
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(hp.height*1.05), int(hp.width*1.05))),
            transforms.RandomCrop((hp.height, hp.width)),
            transforms.RandomRotation(degrees=(-2, 2),fill=255),
            transforms.ToTensor()
            ])
        self.transformblur = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(hp.height*1.05), int(hp.width*1.05))),
            transforms.RandomCrop((hp.height, hp.width)),
            transforms.RandomRotation(degrees=(-2, 2),fill=255),
            transforms.GaussianBlur(7, sigma=(1.0 , 2.0)),
            transforms.ToTensor()
            ])
     
    def __getitem__(self, index):
        img = self.name_image[index]
        if not self.eval:
            if random.random() < 0.25:               
                img = self.transformblur(img)
            else:
                img = self.transform(img)
            img = img / img.max()
            img = img**(random.random()*0.7 + 0.6)
        else:
            img = np.transpose(img,(2,0,1))
            img = img / img.max()
        c, h, w = img.shape
        if (h > w * 1.25):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  

        label = text_to_labels(self.label[index], p2idx)
        return (torch.FloatTensor(img), torch.LongTensor(label))

    def __len__(self):
        return len(self.label)


# Кодирование позиции символа
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, name, outtoken, hidden = 128, enc_layers=1, dec_layers=1, nhead = 1, dropout=0.1,pretrained=False):
        super(TransformerModel, self).__init__()
        self.backbone = models.__getattribute__(name)(pretrained=pretrained)
        #self.backbone.avgpool = nn.MaxPool2d((4, 1))
        self.backbone.fc = nn.Conv2d(2048, hidden//4, 1)
        
        self.pos_encoder = PositionalEncoding(hidden, dropout)
        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=hidden*4, dropout=dropout, activation='relu')
        
        self.fc_out = nn.Linear(hidden, outtoken)
        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        x = self.backbone.conv1(src)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        #x = self.backbone.avgpool(x)
        
        x = self.backbone.fc(x)
        x = x.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)
        src_pad_mask = self.make_len_mask(x[:,:,0])
        src = self.pos_encoder(x)

        trg_pad_mask = self.make_len_mask(trg)
        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)

        return output


class TransformerModelDnet(nn.Module):
    def __init__(self, name, outtoken, hidden = 128, enc_layers=1, dec_layers=1, nhead = 1, dropout=0.1,pretrained=False):
        super(TransformerModelDnet, self).__init__()
        self.backbone = models.__getattribute__(name)(pretrained=pretrained)
        # self.backbone.avgpool = nn.MaxPool2d((4, 1))
        self.backbone.fc = nn.Conv2d(2048, hidden // 4, 1)
 
        self.backbone.features.denseblock4.denselayer1.norm1 = nn.BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer1.conv1 = nn.Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer1.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer2.norm1 = nn.BatchNorm2d(1097, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer2.conv1 = nn.Conv2d(1097, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer2.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer3.norm1 = nn.BatchNorm2d(1138, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer3.conv1 = nn.Conv2d(1138, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer3.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer4.norm1 = nn.BatchNorm2d(1179, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer4.conv1 = nn.Conv2d(1179, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer4.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer5.norm1 = nn.BatchNorm2d(1220, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer5.conv1 = nn.Conv2d(1220, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer5.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer6.norm1 = nn.BatchNorm2d(1261, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer6.conv1 = nn.Conv2d(1261, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer6.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer7.norm1 = nn.BatchNorm2d(1302, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer7.conv1 = nn.Conv2d(1302, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer7.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer8.norm1 = nn.BatchNorm2d(1343, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer8.conv1 = nn.Conv2d(1343, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer8.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer9.norm1 = nn.BatchNorm2d(1384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer9.conv1 = nn.Conv2d(1384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer9.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer10.norm1 = nn.BatchNorm2d(1425, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer10.conv1 = nn.Conv2d(1425, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer10.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer11.norm1 = nn.BatchNorm2d(1466, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer11.conv1 = nn.Conv2d(1466, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer11.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer12.norm1 = nn.BatchNorm2d(1507, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer12.conv1 = nn.Conv2d(1507, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer12.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer13.norm1 = nn.BatchNorm2d(1548, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer13.conv1 = nn.Conv2d(1548, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer13.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer14.norm1 = nn.BatchNorm2d(1589, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer14.conv1 = nn.Conv2d(1589, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer14.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer15.norm1 = nn.BatchNorm2d(1630, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer15.conv1 = nn.Conv2d(1630, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer15.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer16.norm1 = nn.BatchNorm2d(1671, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer16.conv1 = nn.Conv2d(1671, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer16.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer17.norm1 = nn.BatchNorm2d(1712, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer17.conv1 = nn.Conv2d(1712, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer17.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer18.norm1 = nn.BatchNorm2d(1753, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer18.conv1 = nn.Conv2d(1753, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer18.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer19.norm1 = nn.BatchNorm2d(1794, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer19.conv1 = nn.Conv2d(1794, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer19.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer20.norm1 = nn.BatchNorm2d(1835, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer20.conv1 = nn.Conv2d(1835, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer20.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer21.norm1 = nn.BatchNorm2d(1876, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer21.conv1 = nn.Conv2d(1876, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer21.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer22.norm1 = nn.BatchNorm2d(1917, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer22.conv1 = nn.Conv2d(1917, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer22.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer23.norm1 = nn.BatchNorm2d(1958, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer23.conv1 = nn.Conv2d(1958, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer23.conv2 = nn.Conv2d(192, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer24.norm1 = nn.BatchNorm2d(1999, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.features.denseblock4.denselayer24.conv1 = nn.Conv2d(1999, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.features.denseblock4.denselayer24.conv2 = nn.Conv2d(192, 49, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
 
        self.backbone.features._modules['norm5'] = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.classifier = nn.Conv2d(2048, hidden // 4, 1)
 
        self.pos_encoder = PositionalEncoding(hidden, dropout)
        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=hidden*4, dropout=dropout, activation='relu')
 
        self.fc_out = nn.Linear(hidden, outtoken)
        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None
 
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
 
    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)
 
    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)
 
        x = self.backbone.features(src)
        x = F.relu(x, inplace=True)
        x = self.backbone.classifier(x)
 
        x = x.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)
        src_pad_mask = self.make_len_mask(x[:,:,0])
        src = self.pos_encoder(x)
 
        trg_pad_mask = self.make_len_mask(trg)
        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)
 
        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)
 
        return output

# Предсказания
def prediction():
    os.makedirs('/output', exist_ok=True)
    with open('dict.json') as json_file:
        words = json.load(json_file)
        
    model = TripleEnsemble(model1, model2, model3, letters, 1, device, words)
    
    with torch.no_grad():
        for filename in os.listdir(hp.test_dir):
            img = cv2.imread(hp.test_dir + filename,cv2.IMREAD_GRAYSCALE)#
            img = process_image(img)
            h, w, _ = img.shape

            if not ((w / h) == 8):
                if (w / h) < 8:
                    white = np.zeros([h, 8*h - w ,3], dtype=np.uint8)
                    white.fill(255)     
                    img = cv2.hconcat([img, white])
                elif (w / h) > 8:
                    white = np.zeros([(w - 8*h) // 16,w,3], dtype=np.uint8)
                    white.fill(255) 
                    img = cv2.vconcat([white, img, white])
            img = cv2.resize(img, (hp.width, hp.height))

            img = img.astype('uint8')
            img = img/img.max() 
            img = np.transpose(img, (2, 0, 1))

            src = torch.FloatTensor(img).unsqueeze(0).cuda()

            pred = model(src)
            pred = postprocess(pred, words)

            # print('pred:',p_values,pred)

            with open(os.path.join('/output', filename.replace('.jpg', '.txt').replace('.png', '.txt')), 'w', encoding="utf-8") as file:
                file.write(pred.strip())

# Загружаем гиперпараметры
hp = Hparams()

if __name__ == '__main__':
    # Фиксируем сиды
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ['PYTHONHASHSEED'] = "42"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создать папку с логами
    os.makedirs("log/img/", exist_ok=True)
    
    # Обучение или предсказание
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", default='generate', help=\
        "Enter the function you want to run | Введите функцию, которую надо запустить (train, generate)")
    parser.add_argument("-c", "--checkpoint", default='', help="Чекпоинт")
    parser.add_argument("-d", "--test_dir", default='', help="Чекпоинт")
    
    args = parser.parse_args()
    if args.run == 'train' or args.run == 't':
        it_train = True
    else:
        it_train = False
    if args.checkpoint:
        hp.chk = args.checkpoint
    if args.test_dir:
        hp.test_dir = args.test_dir

    # Загрузить частоту слов
    if it_train:
        if hp.chk:
            hp.chk = './log/' + hp.chk
        
        # Загружаем название файлов, список строк для обучения и алфавит
        names, lines, cnt, all_word = process_texts(hp.image_dir, hp.trans_dir)
        letters = set(cnt.keys())
        letters = sorted(list(letters))
        letters = ['PAD', 'SOS'] + letters + ['EOS']
    else:
        hp.chk = './' + hp.chk
        
        # Алфавит
        letters = ['PAD', 'SOS', ' ', '(', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','[', ']', 
                'i', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л','м', 'н', 'о', 'п', 'р', 
                'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ѣ', 'EOS']
   
    print('Символов:',len(letters),':', ' '.join(letters))

    # Перевод символов в индексы и наоборот
    p2idx = {p: idx for idx, p in enumerate(letters)}
    idx2p = {idx: p for idx, p in enumerate(letters)} 
    
    # Создадим обучающую и валидационную выборки.
    if it_train:
        
        lines_train = []
        names_train = []

        lines_val = []
        names_val = []

        for num,(line, name) in enumerate(zip(lines,names)):
            # файлы оканчивающиеся на 9 в валидацию
            if name[-5] == '9':
                lines_val.append(line)
                names_val.append(name)
            else:
                lines_train.append(line)
                names_train.append(name)

        image_train = generate_data(names_train,hp.image_dir) 
        image_val = generate_data(names_val,hp.image_dir) 
        
        # Датасеты
        train_dataset = TextLoader(image_train,lines_train,hp.image_dir, eval=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                  batch_size=hp.batch_size, pin_memory=True,
                                  drop_last=True, collate_fn=TextCollate())

        val_dataset = TextLoader(image_val,lines_val,hp.image_dir, eval=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
                                                 batch_size=1, pin_memory=False,
                                                 drop_last=False, collate_fn=TextCollate())

    valid_loss_all, train_loss_all,eval_accuracy_all,eval_loss_cer_all = [],[],[],[]
    epochs, best_eval_loss_cer = 0, float('inf')
    
    # Создаём модель
    model1 = TransformerModelDnet('densenet161', len(letters), hidden=hp.hidden, enc_layers=hp.enc_layers, dec_layers=hp.dec_layers, nhead = hp.nhead, dropout=hp.dropout, pretrained=it_train).to(device)
    model2 = TransformerModel('resnext101_32x8d', len(letters), hidden=hp.hidden, enc_layers=hp.enc_layers, dec_layers=hp.dec_layers, nhead = hp.nhead, dropout=hp.dropout, pretrained=it_train).to(device)
    model3 = TransformerModel('resnext101_32x8d', len(letters) - 1, hidden=hp.hidden, enc_layers=hp.enc_layers, dec_layers=hp.dec_layers, nhead = hp.nhead, dropout=hp.dropout, pretrained=it_train).to(device)
    
    # Загружаем веса
    ckpt = torch.load(hp.weights3)
    if 'model' in ckpt:
        model1.load_state_dict(ckpt['model'])
    else:
        model1.load_state_dict(ckpt)
    if 'epochs' in ckpt:
        epochs = int(ckpt['epoch'])
    if 'valid_loss_all' in ckpt:
        valid_loss_all   = ckpt['valid_loss_all']
    if 'best_eval_loss_cer' in ckpt:
        best_eval_loss_cer   = ckpt['best_eval_loss_cer']        
    if 'train_loss_all' in ckpt:
        train_loss_all   = ckpt['train_loss_all']
    if 'eval_accuracy_all' in ckpt:
        eval_accuracy_all   = ckpt['eval_accuracy_all']
    if 'eval_loss_cer_all' in ckpt:
        eval_loss_cer_all   = ckpt['eval_loss_cer_all']

    ckpt = torch.load(hp.weights2)
    if 'model' in ckpt:
        model2.load_state_dict(ckpt['model'])
    else:
        model2.load_state_dict(ckpt)
    if 'epochs' in ckpt:
        epochs = int(ckpt['epoch'])
    if 'valid_loss_all' in ckpt:
        valid_loss_all   = ckpt['valid_loss_all']
    if 'best_eval_loss_cer' in ckpt:
        best_eval_loss_cer   = ckpt['best_eval_loss_cer']        
    if 'train_loss_all' in ckpt:
        train_loss_all   = ckpt['train_loss_all']
    if 'eval_accuracy_all' in ckpt:
        eval_accuracy_all   = ckpt['eval_accuracy_all']
    if 'eval_loss_cer_all' in ckpt:
        eval_loss_cer_all   = ckpt['eval_loss_cer_all']

    ckpt = torch.load(hp.weights3)
    if 'model' in ckpt:
        model3.load_state_dict(ckpt['model'])
    else:
        model3.load_state_dict(ckpt)
    if 'epochs' in ckpt:
        epochs = int(ckpt['epoch'])
    if 'valid_loss_all' in ckpt:
        valid_loss_all   = ckpt['valid_loss_all']
    if 'best_eval_loss_cer' in ckpt:
        best_eval_loss_cer   = ckpt['best_eval_loss_cer']        
    if 'train_loss_all' in ckpt:
        train_loss_all   = ckpt['train_loss_all']
    if 'eval_accuracy_all' in ckpt:
        eval_accuracy_all   = ckpt['eval_accuracy_all']
    if 'eval_loss_cer_all' in ckpt:
        eval_loss_cer_all   = ckpt['eval_loss_cer_all']
        
    prediction()
