import torch
import os
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def read_existing_vocabulary(count_file):
    """
    Read already processed vocabulary from output folder to avoid reprocessing.
    """
    vocab_list = []

    if os.path.isfile(count_file):
        with open(count_file) as f:
            for line in f:
                parts = line.strip('\n').split()
                assert len(parts) == 2, "Expected format: word count"
                vocab_list.append(parts[0])

        # Exclude last entry (might be incomplete)
        vocab_list = set(vocab_list[:-1])

    return vocab_list


def identify_word_indices(batch_ids, token, token_ids, tokenizer, init_idx=0, subword=False):
    """
    Find the token positions where a word/token appears in each sentence.

    For each sentence, searches for the token and returns the indices.
    """
    token_len = len(token_ids)
    assert token_len != 0, "token must have at least one token"

    token_col_idx = []
    valid_sent_id = []

    for sid, sent in enumerate(batch_ids):
        flag = False

        # Skip very long sentences (for computation efficiency)
        if len(sent) >= 500:
            continue

        sent_tmp = sent.copy()

        # Search for token starting from init_idx
        for i in range(init_idx, len(sent_tmp)):
            if i + len(token_ids) <= len(sent_tmp):
                if subword:
                    # Subword matching mode: Check if phrase is contained in a single token
                    # Used when the phrase is a subword that appears within a single token
                    token_in_sent = [sent_tmp[i]]  # Look at one token at a time
                    decoded = "".join(tokenizer.convert_ids_to_tokens(token_in_sent)).replace("▁", "").strip()
                    if decoded == token:
                        flag = True
                else:
                    # Standard mode: Match exact token ID sequence
                    if all([sent_tmp[i + j] == token_ids[j] for j in range(len(token_ids))]):
                        flag = True

                if flag:
                    # Record the token positions
                    # In subword mode, this will be a single index
                    # In standard mode, this will be the full sequence
                    token_idx = [i + j for j in range(len(token_ids))]
                    token_col_idx.append(token_idx)
                    valid_sent_id.append(sid)
                    break

    token_col_idx = np.array(token_col_idx)  # shape: (num_valid, token_length)
    return token_col_idx, valid_sent_id


def encode_batch_with_padding(tokenizer, model, sent_list, sent_len_list, col_idx,
                              max_tokens=8192):
    """
    Encode sentences in batches with dynamic padding to maximize GPU utilization.

    Groups sentences by length to minimize padding waste, then extracts
    contextualized representations at specific token positions (col_idx).

    Args:
        tokenizer: Hugging Face tokenizer
        model: Sentence Transformer model
        sent_list: List of sentences (sorted by length, descending)
        sent_len_list: List of sentence lengths
        col_idx: Token indices to extract (shape: num_sents, token_length)
        max_tokens: Maximum total tokens per batch (controls memory usage)

    Returns:
        Array of contextualized embeddings (shape: num_sents, token_length, hidden_dim)
    """
    all_token_states = []
    max_len = None
    idx_list_batch = []

    for i in range(len(sent_list)):
        sent = sent_list[i]
        assert isinstance(sent, str), "Each sentence must be a string"

        idx_list_batch.append(i)

        # Track maximum length in current batch
        if max_len is None:
            max_len = sent_len_list[i]
        else:
            # Since sorted descending, all subsequent sentences should be shorter
            assert sent_len_list[i] <= max_len

        # Flush batch when we exceed max_tokens or reach the end
        if len(idx_list_batch) * max_len >= max_tokens or i == len(sent_list) - 1:
            # Extract batch
            sent_batch = [sent_list[k] for k in idx_list_batch]
            col_idx_batch = col_idx[idx_list_batch]
            row_idx_batch = np.arange(len(col_idx_batch))[:, None]  # (batch_size, 1)

            # Tokenize batch
            batch = tokenizer(sent_batch, padding=True, return_tensors='pt')
            batch = batch.to("cuda")

            # Forward pass
            outputs = model(**batch)
            hidden_states = outputs["last_hidden_state"]  # (batch_size, seq_len, hidden_dim)

            # Extract token positions: (batch_size, token_length, hidden_dim)
            token_states = hidden_states[row_idx_batch, col_idx_batch].data.cpu().numpy()

            all_token_states.append(token_states)

            # Reset for next batch
            max_len = None
            idx_list_batch = []

    # Concatenate all batches
    all_token_states = np.concatenate(all_token_states, axis=0)  # (num_sents, token_length, hidden_dim)
    return all_token_states


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate static word embeddings from contextualized models'
    )
    parser.add_argument('-model', required=True,
                       help='Hugging Face model name or path')
    parser.add_argument('-output_folder', required=True,
                       help='Output folder for embeddings')
    parser.add_argument('-word2sent', required=True,
                       help='Pickle file mapping words/tokens to example sentences')
    parser.add_argument('-nsent', default=100, type=int,
                       help='Maximum number of sentences to use per word')
    parser.add_argument('-prompt', default="", type=str,
                       help='Prompt to prepend to each sentence (usually empty)')
    parser.add_argument('-subword', action='store_true',
                       help='Enable subword matching mode')

    # Unused but kept for backwards compatibility
    parser.add_argument('-mono_file', default="")

    return parser.parse_args()


def main():
    args = parse_args()

    # ============================================================================
    # 1. Setup Output Folder
    # ============================================================================

    folder = args.output_folder
    os.makedirs(folder, exist_ok=True)
    print(f"Output folder: {folder}")

    # ============================================================================
    # 2. Load Model and Tokenizer
    # ============================================================================

    model_path = args.model
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to("cuda")
    model.eval()

    # ============================================================================
    # 3. Calculate Prompt Offset
    # ============================================================================

    # If using a prompt, we need to skip those tokens when finding the token
    # In your case, prompt is empty (""), so init_idx = 0
    init_idx = 0

    if args.prompt != "":
        init_idx = len(tokenizer.tokenize(args.prompt))
        print(f"Prompt token offset: {init_idx}")

    # ============================================================================
    # 4. Load token-to-Sentence Mapping
    # ============================================================================

    print(f"Loading token-to-sentence mapping from {args.word2sent}")
    with open(args.word2sent, 'rb') as f:
        word2sent = pickle.load(f)

    vocab = list(word2sent.keys())
    print(f"Vocabulary size: {len(vocab)}")

    # Check if vocabulary is lowercase
    lowercase = all(w == w.lower() for w in vocab)
    print(f"Vocabulary is lowercase: {lowercase}")
    print(f"Subword matching mode: {args.subword}")

    # ============================================================================
    # 5. Read Already Processed Vocabulary
    # ============================================================================
    count_file = f"{folder}/count_old.txt"
    vocab_processed = read_existing_vocabulary(count_file)
    print(f"Already processed: {len(vocab_processed)} words")

    # ============================================================================
    # 6. Process Each Word/token
    # ============================================================================

    with open(f"{folder}/count.txt", "w") as f_count:
        with open(f"{folder}/vec.txt", "w") as f_vec:
            for token in tqdm(vocab, desc="Processing vocabulary"):
                # Create key for output (join multi-word tokens with ▁)
                veckey = "▁".join(token.lstrip(" ").split(" "))

                # Skip if already processed
                if veckey in vocab_processed:
                    continue

                assert token[0] != " ", "token should not start with space"

                # Get example sentences for this token
                unlabelled_sents = word2sent[token]
                # Skip if no sentences available
                if len(unlabelled_sents) < 1:
                    continue

                # Prepare sentences (lowercase if needed, add prompt if specified)
                sentences_prepared = []
                for s in unlabelled_sents:
                    if lowercase:
                        s = s.lower()

                    if args.prompt != "":
                        s = args.prompt + " " + s

                    sentences_prepared.append(s)

                # ----------------------------------------------------------------
                # Tokenize token
                # ----------------------------------------------------------------

                # Subword mode: don't add leading space
                # Standard mode: add space before token for proper tokenization
                if args.subword:
                    token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
                else:
                    token_ids = tokenizer(" " + token, add_special_tokens=False)["input_ids"]

                # Skip if tokenization produces no tokens
                if len(tokenizer.convert_ids_to_tokens(token_ids)) == 0:
                    continue

                # Remove leading ▁ token if present
                if tokenizer.convert_ids_to_tokens(token_ids)[0] == '▁':
                    token_ids = token_ids[1:]

                # ----------------------------------------------------------------
                # Tokenize Sentences and Find token Positions
                # ----------------------------------------------------------------

                sentences_tokens = tokenizer(sentences_prepared, padding=True, return_tensors='pt')
                sentences_tokens = sentences_tokens["input_ids"].tolist()
                assert len(sentences_tokens) == len(sentences_prepared)

                # Find where the token appears in each sentence
                col_idx, valid_sent_id = identify_word_indices(
                    sentences_tokens, token, token_ids, tokenizer,
                    init_idx=init_idx, subword=args.subword
                )

                assert len(col_idx) == len(valid_sent_id)

                # Limit to args.nsent sentences
                valid_sent_id = valid_sent_id[:args.nsent]
                col_idx = col_idx[:args.nsent]

                # Skip if token not found in any sentence
                if len(valid_sent_id) == 0:
                    print(f"\nWarning: token not found in any sentence: {token}")
                    print(f"Tokens: {tokenizer.convert_ids_to_tokens(token_ids)}")
                    f_count.write(veckey + " NOT_FOUND\n")
                    continue

                # Filter to valid sentences only
                sentences_prepared = [sentences_prepared[k] for k in valid_sent_id]
                sentences_tokens = [sentences_tokens[k] for k in valid_sent_id]

                # ----------------------------------------------------------------
                # Sort by Length and Extract Embeddings
                # ----------------------------------------------------------------

                # Sort sentences by length (descending) for efficient batching
                sent_len_list = [len(tokens) for tokens in sentences_tokens]
                sorted_idx = np.argsort(sent_len_list)[::-1]

                sentences_sorted = [sentences_prepared[k] for k in sorted_idx]
                sent_len_list = [sent_len_list[k] for k in sorted_idx]
                col_idx = col_idx[sorted_idx]

                # Extract contextualised representations
                with torch.no_grad():
                    token_states = encode_batch_with_padding(
                        tokenizer, model, sentences_sorted, sent_len_list, col_idx,
                        max_tokens=1024 * 50
                    )

                # token_states shape: (num_sents, token_length, hidden_dim)
                assert len(token_states) == len(sentences_sorted)

                # ----------------------------------------------------------------
                # Average Across Tokens and Sentences
                # ----------------------------------------------------------------

                # Average across token tokens (for multi-token tokens)
                token_states = token_states.mean(axis=1)  # (num_sents, hidden_dim)

                # Average across all sentences to get static embedding
                static_embedding = token_states.mean(axis=0)  # (hidden_dim,)

                # ----------------------------------------------------------------
                # Write to Output Files
                # ----------------------------------------------------------------

                # Write to count file: word num_sentences
                f_count.write(f"{veckey} {len(token_states)}\n")

                # Write to vector file: word vec[0] vec[1] ... vec[n]
                f_vec.write(veckey + " ")
                f_vec.write(" ".join([str(x) for x in static_embedding]))
                f_vec.write("\n")

    print(f"\n✓ Done! Embeddings saved to {folder}/vec.txt")
    print(f"✓ Sentence counts saved to {folder}/count.txt")


if __name__ == "__main__":
    main()
