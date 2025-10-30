
import numpy as np
import pickle
import os
import argparse
import string
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from util import load_w2v

# Punctuation marks to filter out during tokenization
PUNCTLIST = list(string.punctuation) + ["¡"] + ["○"] + ["¦"] + ["–"] + ["—"] + ["”"] + ["…"] + ['’'] + ['“']

def encode_sentences(tokenizer, word2vec, sents, model_tokenizer):
    """
    Encode sentences to embeddings by averaging word vectors.
    
    For each sentence:
    1. Tokenize into words
    2. Look up each word in vocabulary (with subword fallback)
    3. Average all found word embeddings
    4. Return stacked array (num_sentences, embedding_dim)
    
    Args:
        tokenizer: Hugging Face tokenizer for word segmentation
        word2vec: Dictionary mapping words to embeddings
        sents: List of sentences to encode
        model_tokenizer: Tokenizer for subword fallback
    
    Returns:
        embeddings: numpy array (num_sentences, embedding_dim)
    """
    embeddings = []
    
    for i in tqdm(range(len(sents)), desc="Encoding sentences"):
        # Tokenize sentence using BERT Basic tokenizer
        words = [x[0] for x in 
                tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sents[i])]
        
        n_words = len(words)
        sent_emb = []
        idx = 0
        
        # Process each word
        while idx < n_words:
            w = words[idx]
            
            # Skip punctuation
            if w not in PUNCTLIST:
                # Try exact match or lowercase match
                if w in word2vec or w.lower() in word2vec:
                    w = w if w in word2vec else w.lower()
                    sent_emb.append(word2vec[w])
                    idx += 1
                    continue
                
                # Try subword matching if word not found
                subwords = model_tokenizer.tokenize(w)
                
                # Try progressively shorter prefixes (longest first)
                # Prefer longer matches to preserve semantic/grammatical info
                while len(subwords) > 1:
                    subwords = subwords[:-1]  # Remove last subword
                    subwords_str = "".join(subwords).replace("##", "")
                    # Check if this prefix exists in vocabulary
                    if subwords_str in word2vec or subwords_str.lower() in word2vec:
                        subwords_str = subwords_str if subwords_str in word2vec else subwords_str.lower()
                        sent_emb.append(word2vec[subwords_str])
                        break
            
            idx += 1
        
        # Average word embeddings for this sentence
        if len(sent_emb):
            sent_emb = np.vstack(sent_emb)
            embeddings.append(sent_emb.mean(axis=0))
        else:
            # If no words found, skip this sentence
            print(f"Warning: No words found in sentence {i}")
    
    if len(embeddings) == 0:
        raise ValueError("No valid sentences encoded. Check your input data.")
    
    embeddings = np.vstack(embeddings)
    return embeddings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute sentence-level PCA components from word embeddings'
    )
    parser.add_argument('-model', required=True,
                       help='Hugging Face model for subword tokenization')
    parser.add_argument('-vec_path', required=True,
                       help='Path to input word vectors')
    parser.add_argument('-output_folder', required=True,
                       help='Output folder for PCA components')
    parser.add_argument('-word2sent', required=True,
                       help='Pickle file mapping words to example sentences')
    parser.add_argument('-embd', required=True, type=int, help='the number of components to remove (ABTT)')
    parser.add_argument('-d_remove', required=True, type=int, help='Embedding dimension')

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # ============================================================================
    # 1. Load Word Embeddings
    # ============================================================================
    
    print(f"Loading word vectors from {args.vec_path}")
    word2vec, edim = load_w2v(args.vec_path)
    print(f"Vocabulary size: {len(word2vec)}")
    print(f"Embedding dimension: {edim}")
    
    # ============================================================================
    # 2. Load Sentences and Compute Sentence Embeddings
    # ============================================================================
    
    print("Computing sentence embeddings for PCA")
    
    # Initialize tokenizers
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased") #Used for word tokenisation
    model_tokenizer = AutoTokenizer.from_pretrained(args.model) #Used for subword tokenisation
    
    # Load sentences for PCA computation
    print(f"Loading sentences from {args.word2sent}")
    with open(args.word2sent, 'rb') as f:
        word2sent_dict = pickle.load(f)
    
    # Sample one sentence per word (up to 100k sentences)
    # This provides diverse coverage for computing PCA
    lines = []
    for w in word2sent_dict:
        # Only use words with at least 3 example sentences (quality filter)
        if len(word2sent_dict[w]) >= 3:
            sents = word2sent_dict[w]
            sents = np.random.permutation(sents)
            lines.append(sents[0])  # Take first random sentence
            if len(lines) == 100000:
                break
    print(f"Collected {len(lines)} sentences for PCA")
    
    # Encode all sentences
    sent_embs = encode_sentences(tokenizer, word2vec, lines, model_tokenizer)
    print(f"Encoded sentence embeddings shape: {sent_embs.shape}")
    
    # ============================================================================
    # 3. Compute PCA
    # ============================================================================
    
    edim = len(sent_embs[0])
    print(f"\nFitting PCA with {edim} components...")
    
    pca = PCA(n_components=edim, whiten=False)
    pca.fit(sent_embs)
    
    print(f"PCA components shape: {pca.components_.shape}")
    
    # ============================================================================
    # 4. Save PCA Components
    # ============================================================================
    
    print(f"\nSaving PCA components to {args.output_folder}")
    
    # Save mean vector for centering
    np.save(f"{args.output_folder}/all_pca_mean.npy", pca.mean_)
    
    # Save principal components
    np.save(f"{args.output_folder}/all_pca_components.npy", pca.components_)
    
    # ============================================================================
    # 5. Report Variance Explained
    # ============================================================================
    
    print("\n" + "="*80)
    print("Variance Explained by Principal Components:")
    print("="*80)
    print(f"Top 1 component:     {pca.explained_variance_ratio_[:1].sum():.4f}")
    print(f"Top 3 components:    {pca.explained_variance_ratio_[:3].sum():.4f}")
    print(f"Top 300 components:  {pca.explained_variance_ratio_[:300].sum():.4f}")
    
    if edim >= 512:
        print(f"Top 512 components:  {pca.explained_variance_ratio_[:512].sum():.4f}")
    
    print(f"All {edim} components: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Find number of components needed for 99% variance
    for i in range(len(pca.explained_variance_ratio_)):
        if pca.explained_variance_ratio_[:i].sum() > 0.99:
            print(f"\n✓ 99% variance explained by {i} components")
            break
    
    print("="*80 + "\n")
    
    # ============================================================================
    # 6. Save PCA-Transformed Word Vectors
    # ============================================================================
    
    print("Transforming and saving word vectors...")

    # Collect all word embeddings
    vocab = list(word2vec.keys())
    emb_list = np.array([word2vec[w] for w in vocab])

    # Center and project onto principal components
    new_emb = (emb_list - pca.mean_).dot(pca.components_[args.d_remove:args.d_remove+args.embd].T)
    print(f"Transformed embeddings shape: {new_emb.shape}")

    # Save in word2vec text format
    output_file = f"{args.output_folder}/vec.txt"
    print(f"Saving transformed vectors to {output_file}")

    with open(output_file, "w") as f:
        for i, w in enumerate(vocab):
            vec = new_emb[i]
            f.write(w + " " + " ".join([str(x) for x in vec]) + "\n")

    print(f"✓ Saved {len(vocab)} transformed word vectors")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
