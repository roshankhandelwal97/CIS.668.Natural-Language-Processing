**Description**

This repository contains the assignments, labs, and dataset used in the Natural Language Processing course (CIS.668) at Syracuse University. The course offers an in-depth exploration of linguistic and computational aspects of NLP technologies. Key topics covered include text processing at various levels, Python programming in the Jupyter notebook environment, and real-world applications of NLP. The curriculum is designed to develop proficiency in NLP techniques for practical and research applications, including deep learning approaches like CNN, RNN, LSTM, and attention-based methods. The repository showcases 2 comprehensive assignments and 12 labs, each reflecting a unique aspect of NLP.

**Dataset**

This repository includes two key datasets:

1. **True News Dataset** : This dataset comprises authentic news articles, covering various topics and dates. Each record contains details like the title, text, subject, and date of publication. It serves as a benchmark for analyzing and understanding the structure and content of genuine news articles.

1. **Fake News Dataset** : In contrast, this dataset contains fabricated news articles. Similar to the True News Dataset, it includes information such as the title, text, subject, and date, but these articles are examples of misinformation. This dataset is crucial for studying the characteristics of fake news and for developing algorithms to distinguish between true and false information.

Both datasets provide a comprehensive platform for exploring various NLP techniques and models, particularly in the context of news authenticity classification.

**Assignments**

This repository includes two Assignments:

1. **Assignments 1**

**Summary:** Analyze real and fake news articles from the ISOT dataset, focusing on pre-processing (like tokenization and lemmatization) and conducting frequency analysis of various textual features. The goal is to distinguish patterns between true and fake news, using metrics such as word, content word, and punctuation counts.

**Key NLP Concepts**

- Tokenization
- Lemmatization
- Bigrams
- Frequency Analysis

**Python Libraries**

- NLTK
- Pandas
- Spacy
- Google Colab

1. **Assignments 2**

**Summary:** Involves sentiment analysis on the "text" in Fake.csv and True.csv files. The main task is to build a classifier to classify the sentiment polarity of sentences as positive or negative, using "bag-of-words" features. The assignment requires the use of the NaiveBayes classifier and multi-fold cross-validation, with a focus on obtaining precision, recall, and F-measure scores. Students are encouraged to experiment with different sets of features, like unigram word features, and to compare the results. The assignment also includes an analysis of sentiment polarity in the first 50 fake and real news articles in the datasets.

**Key NLP Concepts**

- Sentiment Analysis
- Feature Extraction (e.g., Bag-of-Words)
- Naive Bayes Classification

**Python Libraries**

- NLTK
- Scikit-learn
- Pandas
- NumPy

**Labs**

This repository includes 10 Labs:

Here are the summaries and key features for each lab:

**Lab 1**

- **Summary** : Introduction to Python and NLTK for basic syntactic analysis of text.
- **Key Concepts** : NLTK, Python

**Lab 2**

- **Summary** : Focus on corpus-linguistic analyses, starting with text pre-processing and tokenizing.
- **Key Concepts** : Corpus analysis, Tokenization

**Lab 3**

- **Summary** : Exploration of morphology, stemmers, part of speech tagging, and word representation as high dimensional vectors.
- **Key Concepts** : Morphology, Stemmers, POS Tagging, Word Vectors

**Lab 4**

- **Summary** : Understanding of context-free and dependency grammar, and parsing using NLTK.
- **Key Concepts** : Context-free Grammar, Dependency Grammar, Parsing

**Lab 5**

- **Summary** : Sentiment analysis using traditional machine learning approaches.
- **Key Concepts** : Sentiment Analysis, Machine Learning

**Lab 6**

- **Summary** : Focus on spaCy's architecture and capabilities, exploring its pipeline metaphor, including tokenization, part of speech tagging, named entity recognition, and word vectors.
- **Key**** Concepts**: spaCy, Pipeline, Tokenization, POS Tagging, NER, Word Vectors, Dependency Parsing, Sentence Similarity

**Lab 8**

- **Summary** : Focuses on various word embedding techniques, such as GloVe, Word2Vec, ELMo, BERT, spaCy, and others
- **Key Concepts** : GloVe, Word2Vec, ELMo, BERT, XLNet

**Lab 9**

- **Summary** : Use of convolutional neural network (CNN) for recognizing emotions in tweets, combining deep learning and language processing.
- **Key Concepts** : CNN, Emotion Recognition, Deep Learning

**Lab 10**

- **Summary** : Text generation using a character-based RNN, learning sequences of characters from a corpus of text.
- **Key Concepts** : RNN, Text Generation

**Lab 12**

- **Summary** : Application of embedding techniques to build BiLSTM classifiers for sentiment analysis, with an added Attention layer.
- **Key Concepts** : BiLSTM, Sentiment Analysis, Attention Mechanism

**Contributors**

- Roshan Khandelwal
