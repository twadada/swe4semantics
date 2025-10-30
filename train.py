import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer
import copy
from tqdm import tqdm
import string
import pickle
import argparse
from scipy import stats
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from util import  load_w2v
# Constants
PUNCTLIST = set(list(string.punctuation) + ["。", "、", "？", "！", "「", "」", "（", "）", "：", "・", "，"])

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train sentence embeddings')
    parser.add_argument('-bs', default=128, type=int, help='Batch size')
    parser.add_argument('-edim', type=int, required=True, help='Embedding dimension')
    parser.add_argument('-prompt', default="", type=str, help='Prompt for sentence transformer')
    parser.add_argument('-vec_path', required=True, type=str, help='Path to word vectors')
    parser.add_argument('-model', required=True, type=str, help='Sentence transformer model')
    parser.add_argument('-output_folder', required=True, type=str, help='Output folder')
    parser.add_argument('-epoch', default=1, type=int, help='Number of epochs')
    parser.add_argument('-temp', default=0.05, type=float, help='Temperature for distillation')
    parser.add_argument('-word2sent', required=True, type=str, help='Phrase to sentence mapping file')
    parser.add_argument('-sts_eval', action='store_true',
                        help='Evaluate on STS-B')
    return parser.parse_args()

class Net(nn.Module):
    """Neural network for sentence embedding."""
    
    def __init__(self, word2vec, model_path):
        super().__init__()
        
        # Get embedding dimension
        self.embd = len(next(iter(word2vec.values())))
        
        # Build vocabulary
        self.vocab2id = {w: idx + 1 for idx, w in enumerate(word2vec.keys())}
        
        # Embedding layer
        self.emb = nn.Embedding(len(word2vec) + 1, self.embd, padding_idx=0)
        
        # Initialize embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.punct_list = PUNCTLIST
        
        with torch.no_grad():
            print("Initializing embeddings...")
            for w, vec in word2vec.items():
                wid = self.vocab2id[w]
                self.emb.weight.data[wid] = torch.FloatTensor(vec)
    
    def encode(self, sents):
        """Encode sentences to embeddings."""
        embeddings = []
        
        for sent in sents:
            words = [x[0] for x in 
                    self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)]
            
            sent_emb = []
            idx = 0
            n_words = len(words)
            
            while idx < n_words:
                w = words[idx]
                
                if w not in self.punct_list:
                    # Try exact match or lowercase
                    if w in self.vocab2id or w.lower() in self.vocab2id:
                        w = w if w in self.vocab2id else w.lower()
                        emb = self.emb.weight[self.vocab2id[w]]
                        sent_emb.append(emb)
                        idx += 1
                        continue
                    else:
                        # Try subword matching
                        subwords = self.model_tokenizer.tokenize(w)
                        if len(subwords):
                            subword_flag = False
                            while len(subwords) > 1:
                                subwords = subwords[:-1]
                                subwords_str = "".join(subwords).replace("##", "").replace("▁", "")
                                
                                if subwords_str in self.vocab2id or subwords_str.lower() in self.vocab2id:
                                    subwords_str = subwords_str if subwords_str in self.vocab2id else subwords_str.lower()
                                    emb = self.emb.weight[self.vocab2id[subwords_str]]
                                    sent_emb.append(emb)
                                    subword_flag = True
                                    break
                            
                            if subword_flag:
                                idx += 1
                                continue
                
                idx += 1
            
            if len(sent_emb) > 0:
                sent_emb = torch.stack(sent_emb).sum(dim=0)
            else:
                sent_emb = torch.zeros(self.embd).to("cuda")
            
            embeddings.append(sent_emb)
        
        return torch.stack(embeddings)
    
    def forward(self, sents):
        return self.encode(sents)


def distill_loss(sim_s, sim_t):
    """Compute distillation loss."""
    p = F.log_softmax(sim_s, dim=-1)
    q = F.softmax(sim_t, dim=-1)
    loss = (-(q * p).nansum() / q.nansum()).mean()
    return loss


def compute_batch_loss(batch, st_model, model, temp):
    """Compute loss for a batch."""
    # Get teacher embeddings
    with torch.no_grad():
        if st_model.prompt:
            batch_st = [st_model.prompt + s for s in batch]
        else:
            batch_st = batch
        
        embeddings = st_model.encode(batch_st, convert_to_numpy=False)
        if isinstance(embeddings, list):
            embeddings = torch.stack(embeddings)
        embeddings = embeddings.to("cuda")
    
    embeddings = F.normalize(embeddings, dim=-1)
    
    # Get student embeddings
    static_embeddings = model(batch)
    static_embeddings = F.normalize(static_embeddings, dim=-1)
    
    # Compute similarity matrices
    cossim = torch.matmul(embeddings, embeddings.T)
    cossim_static = torch.matmul(static_embeddings, static_embeddings.T)
    
    # Mask diagonal
    for k in range(len(cossim_static)):
        cossim[k][k] = float("-inf")
        cossim_static[k][k] = float("-inf")
    
    # Compute distillation loss
    loss = distill_loss(cossim_static / temp, cossim / temp)
    return loss


def evaluate_sts(model, sentence1, sentence2, gold_score):
    """Evaluate on STS-B dataset."""
    cossim_list = []
    
    for j in tqdm(range(0, len(sentence1), 256), desc="Evaluating"):
        sent1_tmp = sentence1[j:j + 256]
        sent2_tmp = sentence2[j:j + 256]
        
        static_embeddings1 = model(sent1_tmp)
        static_embeddings2 = model(sent2_tmp)
        
        static_embeddings1 = F.normalize(static_embeddings1, dim=-1)
        static_embeddings2 = F.normalize(static_embeddings2, dim=-1)
        
        cossim = torch.sum(static_embeddings1 * static_embeddings2, dim=-1).cpu().tolist()
        cossim_list.extend(cossim)
    
    pearson_score = stats.pearsonr(cossim_list, gold_score)[0]
    return pearson_score


def main():
    args = parse_args()
    
    # Load word vectors
    print(f"Loading word vectors from {args.vec_path}")
    word2vec, edim = load_w2v(args.vec_path)

    # Load training data
    print(f"Loading training data from {args.word2sent}")
    with open(args.word2sent, 'rb') as f:
        word2sent_dict = pickle.load(f)
    
    sents_train = []
    sents_dev = []
    n_sent_per_word = 3 #Adjust this number according to the vocab size
    
    for w in word2sent_dict.keys(): # Sample sentences from each word and use them for knowledge distillation
        if len(word2sent_dict[w]) >= 5:
            sents = np.random.permutation(word2sent_dict[w])
            for k in range(n_sent_per_word):  
                if k < 2:
                    sents_train.append(sents[k])
                else:
                    sents_dev.append(sents[k])
    
    sents_train = list(set(sents_train))
    sents_dev = set(sents_dev) - set(sents_train)
    sents_dev = np.random.permutation(list(sents_dev))[:10000].tolist() # Use 10k sents for validation
    
    print(f"Training samples: {len(sents_train)}")
    print(f"Dev samples: {len(sents_dev)}")
    
    # Initialize model
    print(f"Initializing model with embedding dim: {args.edim}")
    model = Net(word2vec, args.model)
    model.to("cuda")
    
    # Load sentence transformer
    print(f"Loading sentence transformer: {args.model}")
    st_model = SentenceTransformer(args.model, trust_remote_code=True)
    st_model.prompt = args.prompt
    st_model.to("cuda")
    st_model.eval()
    
    for param in st_model.parameters():
        param.requires_grad = False
    
    # Setup Adam optimizer
    optimizer = optim.Adam(model.parameters())

    # Load STS-B for monitoring scores
    # You may choose to use this score for model selection instead of the validation loss
    print("Loading STS-B dataset")
    stsb_train = load_dataset("sentence-transformers/stsb", split="train")
    sentence1 = stsb_train['sentence1']
    sentence2 = stsb_train['sentence2']
    gold_score = stsb_train['score']

    if args.sts_eval:
        # Initial evaluation
        model.eval()
        with torch.no_grad():
            pearson_score = evaluate_sts(model, sentence1, sentence2, gold_score)

        print(f"Initial Pearson score: {pearson_score:.4f}")
    
    # Training loop
    best_dev_loss = float('inf')
    pearson_score_best = pearson_score
    best_step = 0
    best_model = None
    early_stop_count = 5
    no_improvement_count = 0

    total_n = 128 * 2000
    step_size = 128 * 40

    for epoch in range(args.epoch):
        print(f"\n=== Epoch {epoch + 1}/{args.epoch} ===")

        for train_step in range(0, total_n, step_size):
            print(f"Step: {train_step}/{total_n}")

            # Sample training data
            sents_train_tmp = np.random.permutation(sents_train)[:step_size].tolist()

            # Training
            model.train()
            running_loss = 0.0

            for j in tqdm(range(0, step_size, args.bs), desc="Training"):
                batch = sents_train_tmp[j:j + args.bs]

                optimizer.zero_grad()
                loss = compute_batch_loss(batch, st_model, model, args.temp)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                running_loss += loss.item()

            print(f"Training loss: {running_loss:.4f}")

            # Validation
            model.eval()
            dev_loss_total = 0.0

            with torch.no_grad():
                for j in tqdm(range(0, len(sents_dev), 256), desc="Validation"):
                    batch = sents_dev[j:j + 256]
                    dev_loss = compute_batch_loss(batch, st_model, model, args.temp)
                    dev_loss_total += dev_loss.item()

                # Evaluate on STS-B
                if args.sts_eval:
                    pearson_score = evaluate_sts(model, sentence1, sentence2, gold_score)
                    print(f"Pearson score: {pearson_score:.4f} (best: {pearson_score_best:.4f})")

            print(f"Dev loss: {dev_loss_total:.4f}")


            # Update best score just for the purpose of monitoring model performance
            if pearson_score > pearson_score_best:
                pearson_score_best = pearson_score

            # Save best model based on validation loss
            if dev_loss_total < best_dev_loss:
                best_dev_loss = dev_loss_total
                best_step = train_step + total_n * epoch
                best_model = copy.deepcopy(model.to("cpu"))
                model.to("cuda")
                print(f"New best model at step {best_step}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_count:
                    print("Early stopping triggered")
                    break

        if no_improvement_count >= early_stop_count:
            break

    # Save final embeddings
    print(f"\nSaving embeddings to {args.output_folder}")
    output_file = (f"{args.output_folder}_{args.epoch}epoch_"
                  f"{''.join(args.prompt.split()).strip().strip(':')}_"
                  f"best_step_{best_step}_bs{args.bs}.txt")

    with open(output_file, "w") as f:
        for w in word2vec:
            vec = best_model.emb.weight[best_model.vocab2id[w]].data.cpu().numpy()
            f.write(w + " " + " ".join([str(x) for x in vec]) + "\n")

    print(f"Best epoch: {best_step}")
    print("Done!")


if __name__ == "__main__":
    main()
