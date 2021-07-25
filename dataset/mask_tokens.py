import json

from torch.utils.data import Dataset
import tqdm
import random
import torch
from vocab import WordVocab

class maksed_data(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and on_memory:
                self.corpus_lines=json.load(f)['tokens']



    def __len__(self):
        return len(self.corpus_lines)

    def __getitem__(self):
        t1_random, t1_label = self.random_word(self.corpus_lines)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]

        bert_input = (t1)[:self.seq_len]
        bert_label = (t1_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label}
        print("the output is ", output)

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        sentence=" ".join(sentence)
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

if __name__=="__main__":
    vocab_path = "/mnt/data/competition/pythonProject/pythonProject/layoutlmv2/funds_data/pretrained_data/vocab.pkl"
    corpus_path="/mnt/data/competition/pythonProject/pythonProject/layoutlmv2/funds_data/extracted_data/00040534.json"
    vocab = WordVocab.load_vocab(vocab_path)
    data = maksed_data(corpus_path, vocab, seq_len=100, encoding="utf-8", corpus_lines=None, on_memory=True)
    print(data.__getitem__())