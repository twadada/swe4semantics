import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer
import scipy.stats
import copy
from tqdm import tqdm
import argparse
import pickle
import string
from datasets import load_dataset
from scipy import stats

# Punctuation list for filtering
PUNCTLIST = set(list(string.punctuation) + ["¡"] + ["○"] + ["¦"] + ["–"] + ["—"] + ["”"] + ["…"] + ['’'] + ['“']+["。"]+["、"]+["？"]+["！"]+["「"]+["」"]+["（"]+["）"]+["："]+["・"]+["，"])

def load_w2v(file):
    word2vec = {}
    with open(file, 'r', errors='ignore') as f:
        dim = None
        for line in f:
            line = line.rstrip('\n').rstrip(' ').split(' ')
            if len(line) == 2:
                print("skip the first line")
                continue
            w = line[0]
            vec = line[1:]
            if dim is None:
                dim = len(vec)
            else:
                if len(vec) == 0:
                    print("skip the zero vector")
                    continue
                assert len(vec) == dim
            word2vec[w] = np.array([float(x) for x in vec])
    return word2vec, dim

class Net(nn.Module):
    def __init__(self, word2vec, model_path):
        super().__init__()
        self.embd = len(next(iter(word2vec.values())))

        self.vocab2id = {}
        for w in word2vec:
            self.vocab2id[w] = len(self.vocab2id) + 1

        self.emb = nn.Embedding(len(word2vec) + 1, self.embd, padding_idx=0)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_path)

        with torch.no_grad():
            print("initialise embeddings")
            for w in word2vec:
                wid = self.vocab2id[w]
                assert wid != 0
                self.emb.weight.data[wid] = torch.FloatTensor(word2vec[w])

    def forward(self, sents, detector, langcode = None):
        embeddings = []
        for i in range(len(sents)):
            if langcode is None:
                detected_lang = detector.detect_language_of(sents[i])
                if detected_lang is None:
                    langcode = "en"
                else:
                    langcode = detected_lang.iso_code_639_1.name.lower()

            if langcode in ["ja", "zh"]: # NOTE: Hard coded for Japanese/Chinese as specified in the paper
                # subword segmentation using mGTE tokeniser
                words = [x.replace("▁", "") for x in self.model_tokenizer.tokenize(sents[i]) if x != "▁"]
            else:
                # word segmentation using BERT basic tokeniser
                words = [x[0] for x in self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sents[i])]

            n_words = len(words)
            idx = 0
            sent_emb = []
            while idx < n_words:
                w = words[idx]
                # Skip punctuation
                if w in PUNCTLIST:
                    idx += 1
                    continue

                if w in self.vocab2id or w.lower() in self.vocab2id:
                    if w not in self.vocab2id:
                        w = w.lower()
                    emb = self.emb.weight[self.vocab2id[w]]
                    sent_emb.append(emb)
                    idx += 1
                    continue
                else:
                    subwords = self.model_tokenizer.tokenize(w)
                    if len(subwords):
                        found_subword = None
                        while len(subwords) > 1:
                            subwords = subwords[:-1]  # Remove last subword
                            # Reconstruct the string and clean tokenizer artifacts
                            subwords_str = "".join(subwords).replace("##", "").replace("▁", "")

                            # Check if this prefix exists in vocab (case-insensitive)
                            if subwords_str in self.vocab2id:
                                found_subword = subwords_str
                                break
                            elif subwords_str.lower() in self.vocab2id:
                                found_subword = subwords_str.lower()
                                break

                        # If we found a matching subword prefix, add its embedding
                        if found_subword is not None:
                            emb = self.emb.weight[self.vocab2id[found_subword]]
                            sent_emb.append(emb)
                idx += 1

            if len(sent_emb) > 0:
                sent_emb = torch.stack(sent_emb)
                sent_emb = sent_emb.sum(dim=0)
            else:
                sent_emb = torch.FloatTensor(np.zeros(self.embd)).to("cuda")
            embeddings.append(sent_emb)

        embeddings = torch.stack(embeddings)
        return embeddings

def contrastive_loss(sim_s, sim_t):
    p = F.log_softmax(sim_s, dim=-1)
    q = F.softmax(sim_t, dim=-1)
    loss = (-(q * p).nansum() / q.nansum()).mean()
    return loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', default=128, type=int)
    parser.add_argument('-model', required=True, type=str)
    parser.add_argument('-vec_path', required=True, type=str)
    parser.add_argument('-output_folder', required=True, type=str)
    parser.add_argument('-epoch', default=1, type=int)
    parser.add_argument('-temp', default=0.05, type=float)
    parser.add_argument('-parallel_sents', default="", nargs='+')
    parser.add_argument('-lang', choices=['ende', 'enja', 'enzh'])
    return parser.parse_args()

def main():
    args = parse_args()

    # Load word vectors
    word2vec, edim = load_w2v(args.vec_path)
    print(f"Loaded {len(word2vec)} word vectors with dimension {edim}")

    # Setup language detector
    from lingua import Language, LanguageDetectorBuilder
    if args.lang == "enja":
        languages = [Language.ENGLISH, Language.JAPANESE]
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
    elif args.lang == "enzh":
        languages = [Language.ENGLISH, Language.CHINESE]
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
    elif args.lang == "ende":
        languages = [Language.ENGLISH, Language.GERMAN]
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
    else:
        raise ValueError("Accept enja/enzh/ende only")

    print(f"Language detector: {detector}")

    # Initialize model
    model = Net(word2vec, args.model)
    print(f"Embedding dimension: {model.embd}")
    model.to("cuda")

    # Setup Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load MT data
    assert len(args.parallel_sents) == 2

    l1_lines = []
    for line in open(args.parallel_sents[0]):
        line = line.strip('\n')
        l1_lines.append(line)

    l2_lines = []
    for line in open(args.parallel_sents[1]):
        line = line.strip('\n')
        l2_lines.append(line)

    l1_lines_train = l1_lines[:-10000]
    l1_lines_dev = l1_lines[-10000:]
    l2_lines_train = l2_lines[:-10000]
    l2_lines_dev = l2_lines[-10000:]

    assert len(l1_lines_train) == len(l2_lines_train)
    assert len(l1_lines_dev) == len(l2_lines_dev)

    print(f"Train: {len(l1_lines_train)} parallel sentences")
    print(f"Dev: {len(l1_lines_dev)} parallel sentences")


    best_dev_loss = float('inf')
    early_stop_count = 5
    no_improvement = 0
    best_epoch = 0
    early_stop_flag = False

    # Training loop
    total_n = 128 * 2000
    step_size = 128 * 40
    bs = args.bs

    for epoch_word in range(args.epoch):
        if early_stop_flag:
            break
        print(f"\nEpoch {epoch_word + 1}/{args.epoch}")
        new_idx = np.random.permutation(len(l1_lines_train))
        l1_lines_train = [l1_lines_train[i] for i in new_idx]
        l2_lines_train = [l2_lines_train[i] for i in new_idx]
        for batch_idx in range(0, total_n, step_size):
            print(f"  Step {batch_idx}/{total_n}")

            # Sample training data
            l1_lines_tmp = l1_lines_train[batch_idx:batch_idx+step_size]
            l2_lines_tmp = l2_lines_train[batch_idx:batch_idx+step_size]

            model.train()
            running_loss = 0.0

            # Training on MT loss
            for j in tqdm(range(0, step_size, bs), desc="Training"):
                l1_sents = l1_lines_tmp[j:j + bs]
                l2_sents = l2_lines_tmp[j:j + bs]

                if len(l1_sents) == 0:
                    continue

                optimizer.zero_grad()

                l1_static_embeddings = model(l1_sents, detector)
                l2_static_embeddings = model(l2_sents, detector)
                l1_static_embeddings = F.normalize(l1_static_embeddings, dim=-1)
                l2_static_embeddings = F.normalize(l2_static_embeddings, dim=-1)

                # Create gold labels (diagonal matrix)
                goldlabel = np.full((len(l1_static_embeddings), len(l1_static_embeddings)), float("-inf"))
                for i in range(len(goldlabel)):
                    goldlabel[i][i] = 0
                goldlabel = torch.FloatTensor(goldlabel).to("cuda")

                cossim_static = torch.matmul(l1_static_embeddings, l2_static_embeddings.T)

                loss = contrastive_loss(cossim_static / args.temp, goldlabel)
                loss += contrastive_loss(cossim_static.T / args.temp, goldlabel.T)

                loss.backward()
                running_loss += loss.item()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            print(f"  Training loss: {running_loss:.4f}")

            # Validation
            model.eval()
            dev_loss_total = 0

            with torch.no_grad():
                for j in tqdm(range(0, len(l1_lines_dev), bs), desc="Validation"):
                    l1_sents = l1_lines_dev[j:j + bs]
                    l2_sents = l2_lines_dev[j:j + bs]

                    if len(l1_sents) == 0:
                        continue

                    l1_static_embeddings = model(l1_sents, detector)
                    l2_static_embeddings = model(l2_sents, detector)
                    l1_static_embeddings = F.normalize(l1_static_embeddings, dim=-1)
                    l2_static_embeddings = F.normalize(l2_static_embeddings, dim=-1)

                    goldlabel = np.full((len(l1_static_embeddings), len(l1_static_embeddings)), float("-inf"))
                    for i in range(len(goldlabel)):
                        goldlabel[i][i] = 0
                    goldlabel = torch.FloatTensor(goldlabel).to("cuda")

                    cossim_static = torch.matmul(l1_static_embeddings, l2_static_embeddings.T)
                    loss = contrastive_loss(cossim_static / args.temp, goldlabel)
                    loss += contrastive_loss(cossim_static.T / args.temp, goldlabel.T)
                    dev_loss_total += loss

            print(f"  Dev loss: {dev_loss_total:.4f}")

            if best_dev_loss > dev_loss_total:
                best_dev_loss = dev_loss_total
                best_epoch = batch_idx + total_n * epoch_word
                best_model = copy.deepcopy(model.to("cpu"))
                model.to("cuda")
                print(f"  *** New best model at epoch {best_epoch}")
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement == early_stop_count:
                    print(f"  Early stopping after {early_stop_count} steps without improvement")
                    early_stop_flag = True
                    break

    print(f"\nTraining complete. Best epoch: {best_epoch}")

    # Save embeddings
    output_file = f"{args.output_folder}_{args.epoch}epoch_best_epoch_{best_epoch}_bs{args.bs}.txt"
    with open(output_file, "w") as f:
        for w in word2vec:
            vec = best_model.emb.weight[best_model.vocab2id[w]].data.numpy()
            f.write(w + " " + " ".join([str(x) for x in vec]))
            f.write("\n")

    print(f"Saved embeddings to: {output_file}")

if __name__ == "__main__":
    main()