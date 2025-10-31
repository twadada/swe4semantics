# swe4semantics
This repositiory provides code and models proposed in our paper "[Static Word Embeddings for Sentence Semantic Representation](https://arxiv.org/abs/2506.04624)" (EMNLP 25 Main).


# How to use SWEs for encoding sentences
English and cross-lingual (English-{German/Japanese/Chinese}) SWEs (static word embeddings) are both stored in the "embeddings" folder.  **All SWEs, except for the English-Japanese one ("swe_mgte256_enja.txt"), are released under the Apache license 2.0. The English-Japanese one follows [the license of JParaCrawl](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/).**

Refer to "example.py" for how to use English SWEs, and "example_xling.py" for cross-lignual ones.

# Train English SWEs
First, prepare the **"word2sent.pkl"** file that pickles the python dictionary where keys are a list of words in (pre-defined) vocabulary and values are a list of N unlabelled sentences (**not passages or documents**) that contain the key word. In our paper, we employ [CC-100](https://data.statmt.org/cc-100/) and split text in each line into sentences using [BlingFire](https://github.com/microsoft/BlingFire), and then sample N=100 sentences for each word in the 150k vocab.

**Some pieces of code are hard-coded for the BERT-style tokenisation that specifies the subword boundary with "##". Modify relevant parts if necessary.**

## 1. Extract English SWEs from GTE-base
```
model=Alibaba-NLP/gte-base-en-v1.5
word2sent=path_to_word2sent.pkl
output_folder=output_folder_path
nsent=100
CUDA_VISIBLE_DEVICES=0 python extract_embs.py  -prompt "" -output_folder ${output_folder} -model ${model} -word2sent ${word2sent}  -nsent ${nsent}
```

## 2. Apply Sentence-level PCA
```
model="Alibaba-NLP/gte-base-en-v1.5"
word2sent=path_to_word2sent.pkl
vec_path=output_folder_path/vec.txt
output_folder=output_pca_folder_path
python apply_pca.py -d_remove 7 -embd 256 -word2sent ${word2sent} -vec_path ${vec_path} -model ${model} -output_folder ${output_folder} 
```

## 3. Fine-tune SWEs using knowledge distillation
```
model="Alibaba-NLP/gte-base-en-v1.5"
word2sent=path_to_word2sent.pkl
vec_path=output_pca_folder_path/vec.txt
output_folder=final_output_folder_path
CUDA_VISIBLE_DEVICES=0 python train.py -prompt "" -word2sent ${word2sent} -epoch 15 -bs 128  -model ${model} -vec_path ${vec_path} -output_folder ${output_folder}
```

# Train Cross-lingual SWEs
As in monolingual SWEs, prepare the "word2sent.pkl" file that pickles the python dictionary where keys are a list of words in a pre-defined vocabulary and values are a list of N unlabelled sentences (**not passages or documents**) that contain the key word. In our paper, we use  [CCMatrix](https://opus.nlpl.eu/CCMatrix/corpus/version/CCMatrix) and sample N=100 sentences for each word.

**Some pieces of code are hard-coded for language pairs used in our paper (en-de, en-zh, en-ja); modify relevant parts if necessary.**

## 1. Extract English-German SWEs from mGTE-base
```
model=Alibaba-NLP/gte-multilingual-base
word2sent_en=path_to_english_word2sent
folder=output_english_folder_path
CUDA_VISIBLE_DEVICES=0 python extract_embs.py -prompt "" -folder ${folder} -model ${model} -word2sent ${word2sent_en}  -nsent 100 

word2sent_de=path_to_german_word2sent
folder=output_german_folder_path
CUDA_VISIBLE_DEVICES=0 python extract_embs.py  -prompt "" -folder ${folder} -model ${model} -word2sent ${word2sent_de} -nsent 100 
```

(If the input language is Japanese/Chinese, use enable the "-subword" option)

## 2. Merge embeddings and apply Sentence-level PCA

For bilingual SWEs
```
langs="en de"
vec_path="output_english_folder_path/vec.txt output_german_folder_path/vec.txt"
model="Alibaba-NLP/gte-multilingual-base"
word2sent="${word2sent_en} ${word2sent_de}"
output_folder=output_pca_folder_path
python apply_pca_xling.py -d_remove 7 -embd 256 -langs ${langs} -word2sent ${word2sent}  -vec_path ${vec_path} -model ${model}  -output_folder ${output_folder}
```

For multilingual SWEs (e.g. aligned across English, German, Chinese, Japanese)
```
langs="en de zh ja"
vec_path="output_english_folder_path/vec.txt output_german_folder_path/vec.txt output_chinese_folder_path/vec.txt output_japanese_folder_path/vec.txt"
model="Alibaba-NLP/gte-multilingual-base"
word2sent="${word2sent_en} ${word2sent_de} ${word2sent_zh} ${word2sent_ja}"
output_folder=output_pca_folder_path
python apply_pca_xling.py -d_remove 7 -embd 256 -langs ${langs} -word2sent ${word2sent}  -vec_path ${vec_path} -model ${model}  -output_folder ${output_folder}
```

## 3. Fine-tune SWEs with contrastive learning
Prepare the **"en.txt" and "de.txt"**, where each line is a sentence that is parallel (translation) to each language (hence, both files must have the same numbner of lines). These files are used for contrastive learning. In our paper, we use [CCMatrix](https://opus.nlpl.eu/CCMatrix/corpus/version/CCMatrix) as well.

```
vec_path=output_pca_folder_path/vec.txt
lang=ende
output_folder=final_output_folder_path
model="Alibaba-NLP/gte-multilingual-base"
parallel_sents="en.txt de.txt"
CUDA_VISIBLE_DEVICES=0 python train_xling.py -parallel_sents ${parallel_sents} -lang ${lang} -epoch 15 -bs 128 -model ${model} -vec_path ${vec_path} -output_folder ${output_folder}
```

**Note: The code used in step 2 and 3 are designed for training bilingual SWEs (as done in our paper), but can be easily extended to mulitlingual training. For step 2, merge word embeddings of multiple languages and apply PCA, which we did to produce the results shown in Table 10 and 11. For step 3, feed paralell sentences of multiple language pairs and jointly minimise the contrastive learning loss.**
