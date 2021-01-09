# nlp-deeplearning-ai-examples

---
This repository started as a place to store Colab notebooks for a Coursera NLP Specialization series that I was auditing. Because I am auditing, I don't have access to the exact assignments or datasets, so I improvised a bit, using the opportunity to try out appropriate illustrative toy examples. I also wanted to teach myself [Pytorch](https://pytorch.org/), so all examples use Pytorch.

* [Natural Language Processing with Classification and Vector Spaces](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/home/welcome).
  * [Logistic Regression](01_01_logistic_regression.ipynb).
  * [Sentiment Analysis](01_01b_lstm_sentiment_analysis.ipynb).
  * [Word Translation with Embeddings](01_04_word_translation.ipynb).
* [Natural Language Processing with Probabilistic Models](https://www.coursera.org/learn/probabilistic-models-in-nlp/home/welcome).
* [Natural Language Processing with Sequence Models](https://www.coursera.org/learn/sequence-models-in-nlp/home/welcome).
  * [GPT2 based autocomplete and fine tuning email autocomplete](03_02_lm_autocomplete.ipynb).
  * [BioNLP Named Entity Recognizer](03_03_tx_bionlp_ner.ipynb).
  * [Siamese Network to detect Duplicate Questions](03_04_siamese_network.ipynb).
* [Natural Language Processing with Attention Models](https://www.coursera.org/learn/attention-models-in-nlp/home/welcome).
  * [Neural Machine Translation using seq2seq + attention](04_01_nmt_seq2seq_attn).
  * [Transformer seq2seq for summarization](04_02_tx_seq2seq_summarizer.ipynb).
  * [Summarization with Pegasus (Inference only)](04_02a_pegasus_summarizer.ipynb).
  * [Question Answering with T5 (Inference and fine-tuning)](04_03_t5_qanda.ipynb).
  * [Multilabel Document classification with Longformer (fine-tuning)](04_04_longformer_multilabel.ipynb).

---
Before I did that, though, I realized that I needed to learn Pytorch basics first, so I picked up an elementary Pytorch book (don't remember which unfortunately, and tried out some of the exercises with toy datasets I found here and there.

* [Introduction to Pytorch notebook](x1_intro.ipynb).
* [Pytorch Building Blocks](x2_building_blocks.ipynb).
* [Credit Card Repayment Classifier - FCN](x3_classification.ipynb).
* [CIFAR-10 Image Classifier - CNN](x4_cnn.ipynb).
* [Time Series Prediction - RNN](x6_rnn.ipynb).

---
I also wanted to get familiar with [Pytorch-Lightning](https://www.pytorchlightning.ai/), so I rebuilt the last 3 from the list above to use Pytorch Lightning.

* [Credit Card Repayment Classifier - FCN + Lightning](xl3_classification.ipynb).
* [CIFAR-10 Image Classifier - CNN + Lightning](xl4_cnn.ipynb).
* [Time Series Prediction - RNN + Lightning](xl6_rnn.ipynb).

---
Here are some experiments with Longformer as part of research for my talk on [Transformer Mods for Document Length Inputs](https://www2.slideshare.net/sujitpal/transformer-mods-for-document-length-inputs) for the #nlp-embeddings group on the [TWIML Slack Channel](twimlai.slack.com).

* [Pretrained Longformer and Document Similarity](lf1_longformer_pretrained.ipynb).
* [Fine-tuning Longformer for Sentiment Analysis](lf2_longformer_sentiment_training.ipynb).

---
Random notebooks, usually to back up blog posts or satisfy my own curiosity.

* [Language Model or Knowledge Base?](https://arxiv.org/pdf/1909.01066.pdf) (Petroni, 2019).
  * [Generating masked words with pre-trained BERT Language Model](arxiv_1909_01066_lm_as_kb.ipynb).
* [Word Sense Disambiguation using BERT as a Language Model](https://sujitpal.blogspot.com/2020/11/word-sense-disambiguation-using-bert-as.html) - blog post (Salmon Run, 2020).
  * [WSD Using Raw BERT Embeddings](blog_tds_fd905cb22df7_bert_embeddings_wsd.ipynb).
  * [WSD Using BERT Masked Language Model](blog_tds_fd905cb22df7_bert_mlm_wsd.ipynb).
