import json
import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from tqdm import tqdm
import torch.nn.functional as F

from hparams import Hparams, process_texts, text_to_labels, labels_to_text, \
    phoneme_error_rate, generate_data, count_parameters
from custom_functions import ExtraLinesAugmentation, postprocess, SmartResize


class TextCollate:
    """Makes the text in a batch of the same length."""

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


class TextLoader(torch.utils.data.Dataset):
    """Dataset for uploading images and texts."""

    def __init__(self, name_image, label, image_dir, eval=False):
        self.name_image = name_image
        self.label = label
        self.image_dir = image_dir
        self.eval = eval
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            ExtraLinesAugmentation(number_of_lines=hp.number_of_lines,
                                   width_of_lines=hp.width_of_lines),
            SmartResize(int(hp.width * 1.05), int(hp.height * 1.05)),
            transforms.RandomAffine(degrees=0, scale=(0.935, 0.935),
                                    fillcolor=255),
            transforms.RandomCrop((hp.height, hp.width)),
            transforms.RandomRotation(degrees=(-2, 2), fill=255),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = self.name_image[index]
        if not self.eval:
            img = self.transform(img)
            img = img / img.max()
            img = img ** (random.random() * 0.7 + 0.6)
        else:
            img = np.transpose(img, (2, 0, 1))
            img = img / img.max()

        label = text_to_labels(self.label[index], p2idx)
        return torch.FloatTensor(img), torch.LongTensor(label)

    def __len__(self):
        return len(self.label)


class PositionalEncoding(nn.Module):
    """Character position encoding."""

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


class TransformerModel(nn.Module):
    def __init__(self, name, outtoken, hidden=128, enc_layers=1, dec_layers=1, nhead= 1, dropout=0.1, pretrained=False):
        super(TransformerModel, self).__init__()
        self.backbone = models.__getattribute__(name)(pretrained=pretrained)
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

    def is_not_used(self):
        pass

    def generate_square_subsequent_mask(self, sz):
        self.is_not_used()
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        self.is_not_used()
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


def train(model, optimizer, criterion, iterator):
    """Training."""
    model.train()
    epoch_loss = 0

    for (src, trg) in tqdm(iterator):
        src, trg = src.cuda(), trg.cuda()

        optimizer.zero_grad()
        output = model(src, trg[:-1, :])

        loss = criterion(output.view(-1, output.shape[-1]),
                         torch.reshape(trg[1:, :], (-1,)))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Saving the images
        if random.random() < 0.01:
            img = np.moveaxis(src[0].cpu().numpy(), 0, 2)
            img = np.array(img)
            img = cv2.convertScaleAbs(img, alpha=255.0)
            cv2.imwrite(os.path.join(os.path.join(hp.dir, "img/"),
                                     labels_to_text(trg[1:, 0].cpu().numpy(),
                                                    idx2p) + ".jpg"), img)

    return epoch_loss / len(iterator)


def evaluate(model, criterion, iterator):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for (src, trg) in tqdm(iterator):
            src, trg = src.cuda(), trg.cuda()
            output = model(src, trg[:-1, :])
            loss = criterion(output.view(-1, output.shape[-1]),
                             torch.reshape(trg[1:, :], (-1,)))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def validate(model, dataloader, show=50):
    model.eval()
    show_count = 0
    error_w = 0
    error_p = 0
    with torch.no_grad():
        for (src, trg) in tqdm(dataloader):
            src = src.cuda()
            x = model.backbone.features(src)
            x = F.relu(x, inplace=True)
            x = model.backbone.classifier(x)

            x = x.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)

            memory = model.transformer.encoder(model.pos_encoder(x))

            out_indexes = [p2idx['SOS'], ]

            for i in range(100):
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(
                    device)

                output = model.fc_out(model.transformer.decoder(
                    model.pos_decoder(model.decoder(trg_tensor)), memory))
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == p2idx['EOS']:
                    break

            out_p = labels_to_text(out_indexes[1:], idx2p)
            out_p = postprocess(out_p, words)
            real_p = labels_to_text(trg[1:, 0].numpy(), idx2p)
            error_w += int(real_p != out_p)
            if out_p:
                cer = phoneme_error_rate(real_p, out_p)
            else:
                cer = 1

            error_p += cer
            if show > show_count:
                show_count += 1
                print('Real:', real_p)
                print('Pred:', out_p)
                print(cer)

    return error_p / len(dataloader) * 100, error_w / len(dataloader) * 100


def train_all(best_evalloss_cer):
    """Main train loop including validation."""
    count_bad = 0
    for epoch in range(epochs, 1000):
        print(f'Epoch: {epoch + 1:02}')
        start_time = time.time()
        print("-----------train------------")
        train_loss = train(model, optimizer, criterion, train_loader)
        print("-----------valid------------")
        valid_loss = evaluate(model, criterion, val_loader)
        print("-----------eval------------")
        eval_loss_cer, eval_accuracy = validate(model, val_loader, show=10)
        scheduler.step(eval_loss_cer)
        valid_loss_all.append(valid_loss)
        train_loss_all.append(train_loss)
        eval_loss_cer_all.append(eval_loss_cer)
        eval_accuracy_all.append(eval_accuracy)
        if eval_loss_cer < best_evalloss_cer:
            count_bad = 0
            best_evalloss_cer = eval_loss_cer
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'best_eval_loss_cer': best_evalloss_cer,
                'valid_loss_all': valid_loss_all,
                'train_loss_all': train_loss_all,
                'eval_loss_cer_all': eval_loss_cer_all,
                'eval_accuracy_all': eval_accuracy_all,
            },  os.path.join(hp.dir, 'resnet50_trans_%.3f.pt' % (
                best_evalloss_cer)))
            print('Save best model')
        else:
            count_bad += 1
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'best_eval_loss_cer': best_evalloss_cer,
                'valid_loss_all': valid_loss_all,
                'train_loss_all': train_loss_all,
                'eval_loss_cer_all': eval_loss_cer_all,
                'eval_accuracy_all': eval_accuracy_all,
            }, os.path.join(hp.dir, 'resnet50_trans_last.pt'))
            print('Save model')
        print(f'Time: {time.time() - start_time}s')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val   Loss: {valid_loss:.4f}')
        print(f'Eval  CER: {eval_loss_cer:.4f}')
        print(f'Eval accuracy: {eval_accuracy:.4f}')
        plt.clf()
        plt.plot(valid_loss_all[-20:])
        plt.plot(train_loss_all[-20:])
        plt.savefig(os.path.join(hp.dir, 'all_loss.png'))
        plt.clf()
        plt.plot(eval_loss_cer_all[-20:])
        plt.savefig(os.path.join(hp.dir,  'loss_cer.png'))
        plt.clf()
        plt.plot(eval_accuracy_all[-20:])
        plt.savefig(os.path.join(hp.dir, 'eval_accuracy.png'))
        with open(os.path.join(hp.dir, "textlog.txt"), "a") as file:
            file.write("epoch: " + str(epoch) + " count_bad: " + str(
                count_bad) + " lr: " +
                       str(optimizer.param_groups[0]['lr']) + "\n")
        if count_bad > 25:
            break


# Getting hyperparameters.
hp = Hparams()

if __name__ == '__main__':
    # Setting seeds
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ['PYTHONHASHSEED'] = "42"
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('dict.json') as json_file:
        words = json.load(json_file)

    if not os.path.exists(hp.dir):
        os.makedirs(hp.dir)

    # Creating a folder with logs
    os.makedirs(os.path.join(hp.dir, "img/"), exist_ok=True)

    # Training or prediction
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", default='generate',
                        help="Enter the function you want to run (train, generate)")
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

    names, lines, cnt, all_word = process_texts(hp.image_dir, hp.trans_dir)
    letters = set(cnt.keys())
    letters = sorted(list(letters))
    letters = ['PAD', 'SOS'] + letters + ['EOS']

    print('Символов:', len(letters), ':', ' '.join(letters))

    # Converting characters to indexes and Vice versa.
    p2idx = {p: idx for idx, p in enumerate(letters)}
    idx2p = {idx: p for idx, p in enumerate(letters)}

    # Creating training and validation samples.

    lines_train = []
    names_train = []

    lines_val = []
    names_val = []

    for num, (line, name) in enumerate(zip(lines, names)):
        # Files ending in 9 will be used for validation
        if name[-5] == '9':
            lines_val.append(line)
            names_val.append(name)
        else:
            lines_train.append(line)
            names_train.append(name)

    image_train = generate_data(names_train, hp.image_dir)
    image_val = generate_data(names_val, hp.image_dir)

    # Initialing  dataloaders
    train_dataset = TextLoader(image_train, lines_train, hp.image_dir,
                                eval=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                                batch_size=hp.batch_size,
                                                pin_memory=True,
                                                drop_last=True,
                                                collate_fn=TextCollate())

    val_dataset = TextLoader(image_val, lines_val, hp.image_dir, eval=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
                                                batch_size=1, pin_memory=False,
                                                drop_last=False,
                                                collate_fn=TextCollate())

    valid_loss_all, train_loss_all, eval_accuracy_all, eval_loss_cer_all = [], [], [], []
    epochs, best_eval_loss_cer = 0, float('inf')

    # Creating model
    model = TransformerModel('densenet161', len(letters), hidden=hp.hidden,
                             enc_layers=hp.enc_layers, dec_layers=hp.dec_layers,
                             nhead=hp.nhead, dropout=hp.densenetdropout,
                             pretrained=True).to(device)

    # Uploading weights
    if hp.chk:
        ckpt = torch.load(hp.chk)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        if 'epochs' in ckpt:
            epochs = int(ckpt['epoch'])
        if 'valid_loss_all' in ckpt:
            valid_loss_all = ckpt['valid_loss_all']
        if 'best_eval_loss_cer' in ckpt:
            best_eval_loss_cer = ckpt['best_eval_loss_cer']
        if 'train_loss_all' in ckpt:
            train_loss_all = ckpt['train_loss_all']
        if 'eval_accuracy_all' in ckpt:
            eval_accuracy_all = ckpt['eval_accuracy_all']
        if 'eval_loss_cer_all' in ckpt:
            eval_loss_cer_all = ckpt['eval_loss_cer_all']

    optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=p2idx['PAD'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           patience=8,
                                                           factor=0.1,
                                                           verbose=True)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    # print(model)

    train_all(best_eval_loss_cer)
