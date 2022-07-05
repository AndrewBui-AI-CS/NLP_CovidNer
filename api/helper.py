import argparse
import sys

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# import transformersm
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE
# from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm.notebook import trange
from transformers import *
from transformers import (AdamW, RobertaConfig, RobertaForTokenClassification,
                          get_linear_schedule_with_warmup)


class Ner(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            config.hidden_size*4, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        output = torch.cat(
            (outputs[2][-1], outputs[2][-2], outputs[2][-3], outputs[2][-4]), dim=-1)
        sequence_output = self.dropout(output)
        logits = self.classifier(sequence_output)
        outputs = logits

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(
                    loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss, logits)
        return outputs


model2 = torch.load("model/best_ner_syllable.pt", map_location='cpu')

df_test = []
i = 1
for line in open("data/test_word.conll", "r").readlines():
    if len(line.split()) < 2:
        i += 1
    else:
        tmp = line.split()
        tmp.append('s'+str(i))
        df_test.append(tmp)

data_test = pd.DataFrame(df_test, columns=['Word', 'Tag', 'Sentence#'])


def concatWord(data):
    def tuple_func(f): return [(w, t)
                               for w, t in zip(f['Word'].values, f['Tag'].values)]
    sentences_with_tag = data.groupby('Sentence#').apply(tuple_func)
    # print(sentences_with_tag)
    sentences_with_tag = [sent for sent in sentences_with_tag]
    return sentences_with_tag


sentences_with_tag_test = concatWord(data_test)

sentences_test = [' '.join([word[0] for word in sent])
                  for sent in sentences_with_tag_test]
labels_test = [[word[1] for word in sent] for sent in sentences_with_tag_test]


df_train = []
i = 1
for line in open("data/train_syllable.conll", "r").readlines():
    if len(line.split()) < 2:
        i += 1
    else:
        tmp = line.split()
        tmp.append('s'+str(i))
        df_train.append(tmp)
data_train = pd.DataFrame(df_train, columns=['Word', 'Tag', 'Sentence#'])
label2idx = {k: v for v, k in enumerate(data_train.Tag.unique())}
ids_to_labels = {v: k for v, k in enumerate(data_train.Tag.unique())}
label2idx['PAD'] = 20
label2idx['[CLS]'] = 21
label2idx['[SEP]'] = 22
label2idx['X'] = 23
ids_to_labels[20] = 'PAD'
ids_to_labels[21] = '[CLS]'
ids_to_labels[22] = '[SEP]'
ids_to_labels[23] = 'X'

labels_value = ['PAD', '[CLS]', '[SEP]', 'X']+data_train.Tag.unique().tolist()
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes',
                    default="phobert-base/bpe.codes",
                    required=False,
                    type=str,
                    help='path to fastBPE BPE'
                    )
args, unknown = parser.parse_known_args()

bpe = fastBPE(args)
vocab = Dictionary()
vocab.add_from_file("phobert-base/vocab.txt")


def text2output(text, vocab):
    sentences_test
    subwords_test = ['<s> '+bpe.encode(text)+' </s>']
    input_ids_test = pad_sequences([vocab.encode_line(sent, append_eos=False, add_if_not_exist=False).long().tolist() for sent in subwords_test],
                                   truncating='post', padding='post', maxlen=90, value=1.0, dtype='long')
    attenion_mask_test = [[float(val != 1) for val in sent]
                          for sent in input_ids_test]

    X_test = input_ids_test
    test_mask = attenion_mask_test

    X_test = torch.tensor(X_test)
    test_mask = torch.tensor(test_mask)
    pred_labels_ids = []
    with torch.no_grad():
        output_test = model2(X_test[0:1], test_mask[0:1])

    logit = output_test.detach().cpu().numpy()
    pred_labels_ids.extend([list(pred_label)
                           for pred_label in np.argmax(logit, axis=2)])
    pred = [ids_to_labels[pred_labels_ids[j][i]] for j in range(
        len(pred_labels_ids)) for i in range(len(pred_labels_ids[j]))]
    return list(zip(subwords_test[0].split(), pred))
    # return subwords_test[0].split(), pred

bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)


def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            # for index2 in nguyen_am_index:
            #     if index2 != index:
            #         x, y = nguyen_am_to_ids[chars[index]]
            #         chars[index2] = bang_nguyen_am[x][0]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
        # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
        # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
    return ''.join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def chuan_hoa_dau_cau_tieng_viet(sentence):
    """
        Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
        :param sentence:
        :return:
        """
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)',
                    r'\1/\2/\3', word).split('/')
        # print(cw)
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)


def chuan_hoa_chuam_cau(text):
    tmp = []
    for t in text.split():
        if t[-1] in ',.:!?;*#$%^*()!~':
            t = t[:-1]+' '+t[-1]
        tmp.append(t)
    return ' '.join(tmp)

import unicodedata
text = chuan_hoa_dau_tu_tieng_viet(sys.argv[1])
text = chuan_hoa_chuam_cau(text)
text = unicodedata.normalize('NFC', text)
final_result = text2output(text, vocab)
print(sys.argv[1])
print(text)
print(final_result)
# f = open("data/first_text.txt", "w")
# f.write(sys.argv[1])
# f = open("data/final_result.txt", "w")
# f.write(' '.join([str(item) for item in final_result]))
# f.close()

