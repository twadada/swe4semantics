import string
from transformers import AutoTokenizer,AutoModel
from util import load_w2v, encode_text_xling

st_model_path = "Alibaba-NLP/gte-multilingual-base"
punct_list = set(list(string.punctuation) + ["¡"] + ["○"] + ["¦"] + ["–"] + ["—"] + ["”"] + ["…"] + ['’'] + ['“']+["。"]+["、"]+["？"]+["！"]+["「"]+["」"]+["（"]+["）"]+["："]+["・"]+["，"])
bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
word_tokenizer = bert_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
model_tokenizer = AutoTokenizer.from_pretrained(st_model_path)

#English-German embeddings
vec_path = "embeddings/swe_mgte256_ende.txt" #Make sure you unzip the embedding file in advance
word2vec_ende, edim = load_w2v(vec_path)
# A pair of parallel sentences sampled from BUCC
en_text = "Let us try to apply these principles to international conflict."
de_text = "Versuchen wir, diese Grundsätze auf internationale Konflikte zu übertragen."
lang1 = 'en'
lang2 = 'de'
en_emb = encode_text_xling(lang1, en_text, word_tokenizer, model_tokenizer, punct_list, word2vec_ende, edim)
de_emb = encode_text_xling(lang2, de_text, word_tokenizer, model_tokenizer, punct_list, word2vec_ende, edim)
similarity = en_emb.dot(de_emb.T)
print(similarity) #0.7179491409990943


#English-Japanese embeddings
vec_path = "embeddings/swe_mgte256_enja.txt" #Make sure you unzip the embedding file in advance
word2vec_enja, edim = load_w2v(vec_path)
# A pair of parallel  sentences sampled from Tatoeba
en_text = "Tomorrow, I'm going to study at the library."
ja_text = "明日図書館で勉強するつもりです。"
lang1 = 'en'
lang2 = 'ja'
en_emb = encode_text_xling(lang1, en_text, word_tokenizer, model_tokenizer, punct_list, word2vec_enja, edim)
ja_emb = encode_text_xling(lang2, ja_text, word_tokenizer, model_tokenizer, punct_list, word2vec_enja, edim)
similarity = en_emb.dot(ja_emb.T)
print(similarity) #0.7353363024886208
