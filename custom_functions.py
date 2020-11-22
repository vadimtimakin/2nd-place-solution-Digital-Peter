import numpy as np
from PIL import Image, ImageDraw
import cv2
import PIL
import random
import torch
import torch.nn.functional as F


class ExtraLinesAugmentation:
    '''
    Add random black lines to an image
    Args:
        number_of_lines (int): number of black lines to add
        width_of_lines (int): width of lines
    '''

    def __init__(self, number_of_lines: int = 1, width_of_lines: int = 10):
        self.number_of_lines = number_of_lines
        self.width_of_lines = width_of_lines
      
    def __call__(self, img):
        '''
        Args:
          img (PIL Image): image to draw lines on
        Returns:
          PIL Image: image with drawn lines
        '''
        draw = ImageDraw.Draw(img)
        for _ in range(self.number_of_lines):
            x1 = random.randint(0, np.array(img).shape[1]); y1 = random.randint(0, np.array(img).shape[0])
            x2 = random.randint(0, np.array(img).shape[1]); y2 = random.randint(0, np.array(img).shape[0])
            draw.line((x1, y1, x2 + 100, y2), fill=0, width=self.width_of_lines)

        return img


class SmartResize:
    """
    Resizes image avoiding changing its aspect ratio.
    Includes stretching and squeezing augmentations.
    """

    def __init__(self, width, height, stretch=(1, 1), fillcolor=255):
        """
        Args:
        
            width (int): target width of the image.
            
            height (int): target height of the image.
            
            stretch (tuple): defaults to (1, 1) - turned off.
            Parameter for squeezing / stretching augmentation,
            tuple containining to values which represents
            the range of queezing / stretching.
            Values less than 1 compress the image from the sides (squeezing),
            values greater than 1 stretch the image to the sides (stretching).
            Use range (1, 1) to avoid squeezing / stretching augmentation.
            
            fillcolor (int): defults to 255 - white. Number in range [0, 255]
            representing fillcolor.
        """

        assert len(
            stretch) == 2, "stretch has to contain only two values " \
                           "representing range. "
        assert stretch[0] >= 0 and stretch[
            1] >= 0, "stretch has to contain only positive values."
        assert 0 <= fillcolor <= 255, "fillcolor has to contain values in " \
                                      "range [0, 255]. "

        self.width = int(width)
        self.height = int(height)
        self.stretch = stretch
        self.ratio = int(width / height)
        self.color = fillcolor

    def __call__(self, img) -> PIL.Image.Image:
        """
        Transformation.
        Args:
            img (PIL.Image.Image): RGB PIL image which has to be transformed.
        """

        stretch = random.uniform(self.stretch[0], self.stretch[1])

        img = np.array(img)
        h, w, _ = img.shape
        img = cv2.resize(img, (w, int(w / (w * stretch / h))))
        h, w, _ = img.shape

        if not (w / h) == self.ratio:
            if (w / h) < self.ratio:
                white = np.zeros([h, self.ratio * h - w, 3], dtype=np.uint8)
                white.fill(self.color)
                img = cv2.hconcat([img, white])
            elif (w / h) > self.ratio:
                white = np.zeros(
                    [(w - self.ratio * h) // (self.ratio * 2), w, 3],
                    dtype=np.uint8)
                white.fill(self.color)
                img = cv2.vconcat([white, img, white])
        img = cv2.resize(img, (self.width, self.height))

        img = Image.fromarray(img.astype(np.uint8))
        return img

    def __repr__(self) -> str:
        """Representation."""
        return f'{self.__class__.__name__}(width={self.width}, height="{self.height}", stretch={self.stretch}, ratio={self.ratio}) '


def postprocess(pred, wdict):
    """
    Getting rid of extra spaces in post processing by trying to
    merge neighboring words and checking their presence in the dictionary.
    """
    strspl = pred.split()

    for x in range(len(strspl) - 1):
        if (strspl[x] in wdict) and (strspl[x + 1] in wdict):
            continue
        if (strspl[x] + strspl[x + 1]) in wdict:
            strspl[x] = strspl[x] + strspl[x + 1]
            strspl[x + 1] = ''

    for x in range(len(strspl) - 2):
        if ((strspl[x] in wdict) and (strspl[x + 1] in wdict) and (
                strspl[x + 2] in wdict)):
            continue
        if (strspl[x] + strspl[x + 1] + strspl[x + 2]) in wdict:
            strspl[x] = strspl[x] + strspl[x + 1] + strspl[x + 2]
            strspl[x + 1] = ''
            strspl[x + 2] = ''

    pred = ' '.join([s for s in strspl])
    pred = pred.replace("   ", " ")
    pred = pred.replace("  ", " ")

    return pred


class TripleEnsemble:
    """
    Character-by-character ensemble of three models.

    Iterates over all symbols predicted by each model and
    selects the most frequent one. Takes into account different
    sample lengths and different placement of spaces.

    Available models: ResNeXt101, DenseNet161.
    """

    def __init__(self, model1, model2, model3, characters, priority=1,
                 device=torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu'), words=None):
        """
        Args:

            model1: the first model - ResNeXt101 or DenseNet161.

            model2: the second model - ResNeXt101 or DenseNet161.

            model2: the third model - ResNeXt101 or DenseNet161.

            characters (list): the list of all the unique characters which model might predict.

            priority (int): Optional, defaults to 1. Determines which model to prefer in the case of prediction
            of all models of different characters. Available values: 1, 2, 3.

            device (torch.device): device used for prediction,
            torch.device("cuda") or torch.device("cpu").

            words (list): defaults to None. Dictionary for post processing.
        """

        assert model1.backbone.__class__.__name__ in ("DenseNet",
                                                      "ResNet"), "Backbone of model1 has to be ResNeXt of DenseNet."
        assert model2.backbone.__class__.__name__ in ("DenseNet",
                                                      "ResNet"), "Backbone of model2 has to be ResNeXt of DenseNet."
        assert model3.backbone.__class__.__name__ in ("DenseNet",
                                                      "ResNet"), "Backbone of model3 has to be ResNeXt of DenseNet."
        assert priority in (
        1, 2, 3), "priority value has to be equal 1, 2 or 3."
        assert type(
            characters) == list, "characters should be passed as a list."
        assert type(
            device) == torch.device, "device should be passed as torch.device object."

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.priority = priority
        self.p2idx = {p: idx for idx, p in enumerate(characters)}
        self.idx2p = {idx: p for idx, p in enumerate(characters)}
        self.device = device
        self.words = words

    def labels_to_text(self, s, idx2p) -> str:
        """Labels to text."""

        S = "".join([idx2p[i] for i in s])
        if S.find('EOS') == -1:
            return S
        else:
            return S[:S.find('EOS')]

    def resnextforward(self, model, src) -> str:
        """Forward for prediction of ResNeXt."""

        model.eval()

        x = model.backbone.conv1(src)
        x = model.backbone.bn1(x)
        x = model.backbone.relu(x)
        x = model.backbone.maxpool(x)

        x = model.backbone.layer1(x)
        x = model.backbone.layer2(x)
        x = model.backbone.layer3(x)
        x = model.backbone.layer4(x)

        x = model.backbone.fc(x)
        x = x.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)
        memory = model.transformer.encoder(model.pos_encoder(x))

        p_values = 1
        out_indexes = [self.p2idx['SOS'], ]
        for i in range(100):
            trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(
                self.device)
            output = model.fc_out(model.transformer.decoder(
                model.pos_decoder(model.decoder(trg_tensor)), memory))

            out_token = output.argmax(2)[-1].item()
            p_values = p_values * torch.sigmoid(output[-1, 0, out_token]).item()
            out_indexes.append(out_token)
            if out_token == self.p2idx['EOS']:
                break

        pred = self.labels_to_text(out_indexes[1:], self.idx2p)

        return pred

    def densenetforward(self, model, src) -> str:
        """Forward for prediction of DenseNet."""

        model.eval()

        x = model.backbone.features(src)
        x = F.relu(x, inplace=True)
        x = model.backbone.classifier(x)
        x = x.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)
        memory = model.transformer.encoder(model.pos_encoder(x))

        p_values = 1
        out_indexes = [self.p2idx['SOS'], ]
        for i in range(100):
            trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(
                self.device)
            output = model.fc_out(model.transformer.decoder(
                model.pos_decoder(model.decoder(trg_tensor)), memory))

            out_token = output.argmax(2)[-1].item()
            p_values = p_values * torch.sigmoid(output[-1, 0, out_token]).item()
            out_indexes.append(out_token)
            if out_token == self.p2idx['EOS']:
                break

        pred = self.labels_to_text(out_indexes[1:], self.idx2p)

        return pred

    def ensemble(self, pred1, pred2, pred3) -> str:
        """Ensemble metod implementation."""

        preds = [pred1, pred2, pred3]
        if len(set(preds)) == 3:
            return preds[self.priority - 1]
        else:
            return max(set(preds), key=preds.count)

    def __call__(self, src) -> str:
        """
        Prediction.

        Args:

            src (torch.FloatTensor): source image.
        """

        if self.model1.backbone.__class__.__name__ == "DenseNet":
            pred1 = self.densenetforward(self.model1, src)
        else:
            pred1 = self.resnextforward(self.model1, src)

        if self.model2.backbone.__class__.__name__ == "DenseNet":
            pred2 = self.densenetforward(self.model2, src)
        else:
            pred2 = self.resnextforward(self.model2, src)

        if pred1 == pred2:
            return postprocess(pred1, self.words)

        if self.model3.backbone.__class__.__name__ == "DenseNet":
            pred3 = self.densenetforward(self.model3, src)
        else:
            pred3 = self.resnextforward(self.model3, src)

        pred = self.ensemble(pred1, pred2, pred3)
        return postprocess(pred, self.words)

    def __repr__(self) -> dict:
        """Representation."""
        return {"model1": self.model1.backbone.__class__.__name__,
                "model2": self.model2.backbone.__class__.__name__,
                "model3": self.model3.backbone.__class__.__name__,
                "priority": self.priority}
