import numpy as np
import string

def load_w2v(file):
    word2vec = {}
    with open(file, 'r', errors='ignore') as f:
        dim = None
        for line in f:
            line = line.rstrip('\n')
            line = line.rstrip(' ').split(' ')
            if len(line) == 2:
                print("skip the first line")
                continue
            w = line[0]
            vec = line[1:]
            if dim is None:
                dim = len(vec)
            else:
                if len(vec) == 0:
                    print("skip the empty vector")
                    continue
                assert len(vec) == dim, str(len(vec)) + "::" + str(dim)
            word2vec[w] = np.array([float(x) for x in vec])
    return word2vec, dim


def encode_text(text, word_tokenizer, model_tokenizer, punct_list, word2vec, edim, normalise = True):
    eps =  0.00000001
    tokens = [x[0] for x in word_tokenizer(text)]
    sent_emb = []
    for w in tokens:
        if w not in punct_list:
            word_vec = word2vec.get(w)
            if word_vec is None:
                word_vec = word2vec.get(w.lower())
            if word_vec is not None:
                sent_emb.append(word_vec)
            elif model_tokenizer is not None:
                subwords = model_tokenizer.tokenize(w)
                if len(subwords):
                    while len(subwords) > 1:
                        subwords = subwords[:-1]
                        subwords_str = "".join(subwords).replace("##", "").replace("▁", "") #remove word boudnary indicators
                        word_vec = word2vec.get(subwords_str)
                        if word_vec is None:
                            word_vec = word2vec.get(subwords_str.lower())
                        if word_vec is not None:
                            sent_emb.append(word_vec)
                            break
    if len(sent_emb) > 0:
        sent_emb = np.sum(sent_emb, axis=0)  # n, edim
        if normalise:
            sent_emb = sent_emb / (np.linalg.norm(sent_emb) + eps)
    else:
        sent_emb = np.zeros(edim)
    return sent_emb



def encode_text_xling(lang, text, word_tokenizer, model_tokenizer, punct_list, word2vec, edim, normalise = True):
    assert lang in set(["en","de","ja","zh"])
    eps =  0.00000001
    if lang =='ja' or lang == "zh": #We do not use word_tokenizer for ja/zh because these languages do not have explicit word boundary, unlike en/de
        tokens = [x.replace("▁", "")  for x in model_tokenizer(text) if x != "▁"]
    else: #word tokenisation
        tokens = [x[0] for x in word_tokenizer(text)]
    sent_emb = []
    for w in tokens:
        if w not in punct_list:
            word_vec = word2vec.get(w)
            if word_vec is None:
                word_vec = word2vec.get(w.lower())
            if word_vec is not None:
                sent_emb.append(word_vec)
            elif model_tokenizer is not None:
                subwords = model_tokenizer.tokenize(w)
                if len(subwords):
                    while len(subwords) > 1:
                        subwords = subwords[:-1]
                        subwords_str = "".join(subwords).replace("##", "").replace("▁", "") #remove word boudnary indicators
                        word_vec = word2vec.get(subwords_str)
                        if word_vec is None:
                            word_vec = word2vec.get(subwords_str.lower())
                        if word_vec is not None:
                            sent_emb.append(word_vec)
                            break
    if len(sent_emb) > 0:
        sent_emb = np.sum(sent_emb, axis=0)  # n, edim
        if normalise:
            sent_emb = sent_emb / (np.linalg.norm(sent_emb) + eps)
    else:
        sent_emb = np.zeros(edim)
    return sent_emb
