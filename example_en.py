import string
from transformers import AutoTokenizer,AutoModel
from util import load_w2v, encode_text

st_model_path = "Alibaba-NLP/gte-base-en-v1.5"
vec_path = "embeddings/swe_gte256_en.txt" # Make sure you unzip the embedding file in advance
punct_list = set(list(string.punctuation) + ["¡"] + ["○"] + ["¦"] + ["–"] + ["—"] + ["”"] + ["…"] + ['’'] + ['“']+["。"]+["、"]+["？"]+["！"]+["「"]+["」"]+["（"]+["）"]+["："]+["・"]+["，"])
bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
word_tokenizer = bert_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
model_tokenizer = AutoTokenizer.from_pretrained(st_model_path)
word2vec, edim = load_w2v(vec_path)

text1 = "We proposed word embeddings for sentence semantic representation."
text2 = "Our word embeddings represent the meaning of sentences effectively"

text1_emb = encode_text(text1, word_tokenizer, model_tokenizer, punct_list, word2vec, edim)
text2_emb = encode_text(text2, word_tokenizer, model_tokenizer, punct_list, word2vec, edim)

similarity = text1_emb.dot(text2_emb.T)
print(similarity) #0.8049839511057257