import argparse
import os
import pickle
import string

import numpy as np
from lingua import Language, LanguageDetectorBuilder
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoTokenizer
from util import  load_w2v

def encode_bilingual(tokenizer, punct_list, word2vec, sents, model_tokenizer):
    """Encode sentences using bilingual word embeddings."""
    embeddings = []
    
    for sent in tqdm(sents):
        lang = detector.detect_language_of(sent)
        langcode = "en" if lang is None else lang.iso_code_639_1.name.lower()
        
        # Tokenize based on language
        if langcode in ['ja', 'zh']:
            words = [x.replace("▁", "") for x in model_tokenizer.tokenize(sent) if x != "▁"]
        else:
            words = [x[0] for x in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)]
        
        sent_emb = []
        idx = 0
        n_words = len(words)
        
        while idx < n_words:
            w = words[idx]
            
            # Skip punctuation
            if w in punct_list:
                idx += 1
                continue
            
            # Try exact match or lowercase
            if w in word2vec:
                sent_emb.append(word2vec[w])
                idx += 1
                continue
            elif w.lower() in word2vec:
                sent_emb.append(word2vec[w.lower()])
                idx += 1
                continue
            
            # Try subword tokenization
            if model_tokenizer is not None:
                subwords = model_tokenizer.tokenize(w)
                while len(subwords) > 1:
                    subwords = subwords[:-1]
                    subwords_str = "".join(subwords).replace("##", "").replace("▁", "")
                    
                    if subwords_str in word2vec:
                        sent_emb.append(word2vec[subwords_str])
                        break
                    elif subwords_str.lower() in word2vec:
                        sent_emb.append(word2vec[subwords_str.lower()])
                        break
            
            idx += 1
        
        if sent_emb:
            embeddings.append(np.vstack(sent_emb).mean(axis=0))
        else:
            print(f"No embeddings found for: {words}")
    
    return np.vstack(embeddings)


def collect_sentences(word2sent_path, target_lang, detector, max_sents=100000):
    """Collect sentences in target language from word2sent dictionary."""
    with open(word2sent_path, 'rb') as f:
        word2sent = pickle.load(f)
    
    lines = []
    total = 0
    
    for word, sents in word2sent.items():
        if len(sents) < 10: #ignore rare words with less than 10 example sentences
            continue
        
        sents = np.random.permutation(sents)
        found = 0
        
        for sent in sents:
            total += 1
            lang = detector.detect_language_of(sent)

            # Filter out sentences that are not in target_lang
            if lang is not None and lang.iso_code_639_1.name.lower() == target_lang:
                found += 1
                if len(lines) < max_sents:
                    lines.append(sent)
                if found == 5: #5 sents for each word
                    break
        
        if len(lines) == max_sents:
            break
    
    print(f"{target_lang} sentences: {100*max_sents/total:.3f}% " if total > 0 else f"{target_lang}: No sentences found")
    return lines


def merge_word_vectors(word2vec1, word2vec2):
    """Merge two word vector dictionaries by averaging shared words."""
    vocab = set(word2vec1.keys())
    vocab.update(word2vec2.keys())
    vocab = list(vocab)
    print(f"Total vocab: {len(vocab)}")
    
    word2vec = {}
    for w in vocab:
        if w in word2vec1 and w in word2vec2: #Exist in both langs
            word2vec[w] = (word2vec1[w] + word2vec2[w]) / 2
        elif w in word2vec1: #Exist only in lang1
            word2vec[w] = word2vec1[w]
        else: #Exist only in lang2
            word2vec[w] = word2vec2[w]
    
    return word2vec


def save_word_vectors(word2vec, pca, output_path, d_remove, embd):
    """Apply PCA transformation and save word vectors."""
    vocab = list(word2vec.keys())
    emb_list = np.array([word2vec[w] for w in vocab])
    
    # Apply PCA transformation (skipping first d_remove components, keeping embd components)
    new_emb = (emb_list - pca.mean_).dot(pca.components_[d_remove:d_remove+embd].T)
    
    with open(output_path, "w") as f:
        for i, w in enumerate(vocab):
            vec = new_emb[i]
            f.write(w + " " + " ".join([str(x) for x in vec]) + "\n")
    return new_emb

def main():
    parser = argparse.ArgumentParser(description="Create bilingual word embeddings with PCA")
    parser.add_argument('-model', required=True, help='Tokenizer model name')
    parser.add_argument('-vec_path1', required=True, help='Path to first word2vec file')
    parser.add_argument('-vec_path2', required=True, help='Path to second word2vec file')
    parser.add_argument('-lang1', required=True, help='First language code (should be "en")')
    parser.add_argument('-lang2', required=True, help='Second language code')
    parser.add_argument('-word2sent1', required=True, help='Path to word2sent pickle for lang1')
    parser.add_argument('-word2sent2', required=True, help='Path to word2sent pickle for lang2')
    parser.add_argument('-output_folder', required=True, help='Output folder for PCA files')
    parser.add_argument('-embd', required=True, type=int,help='the number of components to remove (ABTT)')
    parser.add_argument('-d_remove', required=True, type=int,help='Embedding dimension')

    args = parser.parse_args()
    
    # Validate inputs
    assert args.lang1 == "en", "lang1 must be 'en'"
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load word vectors
    print("Loading word vectors...")
    word2vec1, edim = load_w2v(args.vec_path1)
    word2vec2, _ = load_w2v(args.vec_path2)
    print(f"word2vec1: {len(word2vec1)} words")
    print(f"word2vec2: {len(word2vec2)} words")
    
    # Setup language detector
    global detector
    language_map = {
        'de': Language.GERMAN,
        'zh': Language.CHINESE,
        'ja': Language.JAPANESE
    }
    assert args.lang1 == "en"
    languages = [Language.ENGLISH, language_map[args.lang2]]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    
    # Merge or keep separate vocabularies
    word2vec = merge_word_vectors(word2vec1, word2vec2)
    # Load tokenizers
    print("Loading tokenizers...")
    model_tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # Define punctuation list
    punct_list = list(string.punctuation) + ["¡"] + ["○"] + ["¦"] + ["–"] + ["—"] + ["”"] + ["…"] + ['’'] + ['“'] + [
        "。"] + ["、"] + ["？"] + ["！"] + ["「"] + ["」"] + ["（"] + ["）"] + ["："] + ["・"] + ["，"]
    punct_list = set(punct_list)

    # Collect sentences
    print("Collecting sentences...")
    lines = collect_sentences(args.word2sent1, args.lang1, detector, max_sents=100000)
    lines += collect_sentences(args.word2sent2, args.lang2, detector, max_sents=100000)
    print(f"Total sentences collected: {len(lines)}")
    
    # Encode sentences
    print("Encoding sentences...")
    sent_embs = encode_bilingual(tokenizer, punct_list, word2vec, lines, model_tokenizer)
    print(f"Sentence embeddings shape: {sent_embs.shape}")
    
    # Fit PCA
    print("Fitting PCA...")
    edim = sent_embs.shape[1]
    pca = PCA(n_components=edim)
    pca.fit(sent_embs)
    
    # Save PCA parameters
    np.save(os.path.join(args.output_folder, "all_pca_mean.npy"), pca.mean_)
    np.save(os.path.join(args.output_folder, "all_pca_components.npy"), pca.components_)
    
    # Print variance statistics
    print(f"\nPCA variance explained:")
    print(f"First 1 component: {pca.explained_variance_ratio_[:1].sum():.4f}")
    print(f"First 3 components: {pca.explained_variance_ratio_[:3].sum():.4f}")
    print(f"First 300 components: {pca.explained_variance_ratio_[:300].sum():.4f}")
    print(f"First 512 components: {pca.explained_variance_ratio_[:512].sum():.4f}")
    print(f"All components: {pca.explained_variance_ratio_.sum():.4f}")
    
    for i in range(len(pca.explained_variance_ratio_)):
        if pca.explained_variance_ratio_[:i].sum() > 0.99:
            print(f"99% variance at component: {i}")
            break
    
    # Save transformed word vectors
    print("\nSaving transformed word vectors...")
    output_vec_path = os.path.join(args.output_folder, "vec.txt")
    new_emb = save_word_vectors(word2vec, pca, output_vec_path, args.d_remove, args.embd)
    print(f"Word vectors saved to: {output_vec_path}")
    embd = new_emb.shape[1]
    print(f"Embedding dim: {embd}")


if __name__ == "__main__":
    main()
