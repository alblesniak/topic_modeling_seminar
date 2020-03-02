from gensim.models import LdaModel, LdaMulticore
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser
from functions import preprocess_files, create_current_path
import csv
import os
import re

if __name__ == '__main__':
    # # POLITYKA CORPUS
    # data = preprocess_files(corpus_path='corpora/1000_articles_Polityka/',
    #                              stopwords_path='stopwords.txt',
    #                              tags_to_exclude=['interp'])

    # POETRY CORPUS
    with open('stopwords_ru.txt', 'r') as f:
        stopwords_ru = f.read().split('\n')
    data = []
    for p in os.listdir('corpora/ru19_plain_lemm/'):
        with open(os.path.join('corpora/ru19_plain_lemm/', p), 'r') as p_file:
            poetry = p_file.read()
        lemmas = [lemma for lemma in re.findall(r'\{(.*?)\}', poetry) if lemma not in stopwords_ru]
        data.append(lemmas)
    
    current_path = create_current_path()
    model_path = os.path.join(current_path, 'models')
    if not os.path.exists(model_path):
            os.mkdir(model_path)

    # Polityka
    print('Generating set of bigrams for corpus..')
    bigram = Phrases(data, min_count=5, threshold=100)
    bigram_mod = Phraser(bigram)
    text_data = [bigram_mod[doc] for doc in data]
    print('Creating dictionary..')
    dictionary = Dictionary(text_data)
    print('Creating corpus...')
    corpus = [dictionary.doc2bow(text) for text in text_data]
    topics_range=range(5, 101, 5)
    alpha_range=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    beta_range=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    with open(os.path.join(current_path, 'distributions_ru.tsv'), 'a') as dist_file:
        writer = csv.writer(dist_file, delimiter='\t')
        writer.writerow(['topics_n', 'alpha', 'beta', 'doc', 'dist'])
    for num_topics in topics_range:
            print('Topics:', num_topics)
            # iterate through alpha values
            for a in alpha_range:
                print('Alpha:', a)
                # iterare through beta values
                for b in beta_range:
                    print('Beta', b)
                    model = LdaMulticore(corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    passes=5,
                    alpha=a,
                    eta=b,
                    minimum_probability=0)
                    for doc in range(len(corpus)):
                        distributions = model[corpus[doc]]
                        with open(os.path.join(current_path, 'distributions_ru.tsv'), 'a') as dist_file:
                            writer = csv.writer(dist_file, delimiter='\t')
                            writer.writerow([num_topics, a, b, doc, distributions])
                    