from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import date
from shutil import copyfile
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import pickle
from collections import Counter

def create_current_path():
    data_path = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    data_content = os.listdir(data_path)
    today = date.today().strftime('%Y-%m-%d')
    if True in [today in str(working_dir) for working_dir in data_content]:
        i = max([int(num) for num in [working_dir.split('_ver')[1] for working_dir in data_content if working_dir.split('_ver')[0] == today and '_ver' in working_dir]])
        current_path = os.path.join(data_path, today + '_ver' + str(i+1))
        os.mkdir(current_path)
    else:
        current_path = os.path.join(data_path, today + '_ver1')
        os.mkdir(current_path)
    return current_path

def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

def process_ccl(ccl_files, corpus_path, stopwords, tags_to_exclude, n):
    print('Processing ccl...')
    data = []
    for ccl_file in ccl_files:
        ccl_words = []
        with open(os.path.join(corpus_path, ccl_file), 'r') as f:
            soup = BeautifulSoup(f.read(), 'lxml')
        lex = [tok.find('lex', {'disamb' : 1}) for tok in soup.find_all('tok')]
        for element in lex:
            ctag = element.find('ctag')
            if ctag.get_text() not in tags_to_exclude:
                ctag.decompose()
                lemma = element.get_text()
                if lemma not in tags_to_exclude:
                    ccl_words.append(lemma)
                else:
                    continue
            else:
                continue
        if n != None:
            [data.append(chunk) for chunk in chunks(l=ccl_words, n=n)]
        else:
            data.append(ccl_words)
    print('Done.')
    return data

def process_vrt(vrt_files, corpus_path, stopwords, tags_to_exclude, n, delimiter='\t'):
    print('Processing vrt...')
    data = []
    for vrt_file in vrt_files:
        with open(os.path.join(corpus_path, vrt_file), "r", encoding='utf-8') as vrt_file:
            file_content = vrt_file.read()
        soup = BeautifulSoup(file_content, 'lxml')
        for doc_soup in soup.find_all('doc'):
            content = []
            for sentence in doc_soup.find_all('s'):
                for line in sentence.get_text(strip=True).split('\n'):
                    elem = line.split(delimiter)
                    if len(elem) == 3:
                        if elem[1] not in stopwords and elem[2] not in tags_to_exclude:
                            content.append(elem[1].lower())
                    else:
                        continue
            if n != None:
                [data.append(chunk) for chunk in chunks(l=content, n=n)]
            else:
                data.append(content)
    print('Done.')
    return data

def process_txt(txt_files, corpus_path, stopwords, n):
    print('Processing txt...')
    data = []
    for txt_file in txt_files:
        with open(os.path.join(corpus_path, txt_file), "r", encoding='utf-8') as txt_file:
            file_content = txt_file.read()
        content = file_content.split()
        if n != None:
            [data.append(chunk) for chunk in chunks(l=content, n=n)]
        else:
            data.append(content)
    print('Done.')
    return data
        
def preprocess_files(corpus_path, stopwords_path, tags_to_exclude, n=None):
    if stopwords_path:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = f.read().split('\n')
    corpus_files = os.listdir(corpus_path)
    if  corpus_files[0].endswith('.ccl'):
        documents = [doc for doc in corpus_files if doc.endswith('.ccl')]
        return process_ccl(ccl_files=documents, corpus_path=corpus_path, stopwords=stopwords, tags_to_exclude=tags_to_exclude, n=n)
    elif corpus_files[0].endswith('.vrt'):
        documents = [doc for doc in corpus_files if doc.endswith('.vrt')]
        return process_vrt(vrt_files=documents, corpus_path=corpus_path, stopwords=stopwords, tags_to_exclude=tags_to_exclude, n=n)
    elif corpus_files[0].endswith('.txt'):
        documents = [doc for doc in corpus_files if doc.endswith('.txt')]
        return process_txt(txt_files=documents, corpus_path=corpus_path, stopwords=stopwords, n=n)
    else:
        print('Don\'t know the format.')


def plot_topics(text_data, topics_data, path_to_save):
    data_flat = [w for w_list in text_data for w in w_list]
    counter = Counter(data_flat)
    out = []
    for i, topic in topics_data:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])
    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])
    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(len(topics_data), 1, figsize=(16, len(topics_data)*5), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    cols_i = 0
    for i, ax in enumerate(axes.flatten()):
        if cols_i == len(cols):
            cols_i = 0
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[cols_i], width=0.5, alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[cols_i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[cols_i])
        ax.set_title('Topic: ' + str(i), color=cols[cols_i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper center'); ax_twin.legend(loc='upper right')
        cols_i += 1
    fig.tight_layout()    
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22)    
    plt.savefig(path_to_save)
    return plt.show()

def topics_to_excel(topics_obj, path_to_save):
    writer = pd.ExcelWriter(path_to_save)
    for topic in topics_obj:
        df = pd.DataFrame(topic[1], columns=['keyword', 'weight'])
        df.to_excel(writer, str(topic[0]), index=0)
    writer.save()
    print('Excel saved.')

def dominant_topics(ldamodel, corpus, texts, path_to_save, n=None):
    sent_topics_df = pd.DataFrame()
    for i, row_list in enumerate(tqdm(ldamodel[corpus][:n])):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df.to_excel(path_to_save, index=0)
    print('Excel saved.')

def compute_coherence_values(corpus, texts, dictionary, num_topics, alpha, beta, coherence):
    lda_model = LdaMulticore(corpus=corpus,
                           id2word=dictionary,
                           num_topics=num_topics,
                           passes=10,
                           alpha=alpha,
                           eta=beta,
                           workers=4)
    topics = lda_model.show_topics(num_topics=num_topics, num_words=20, log=False, formatted=False)
    coherence_path = os.path.join(current_path, 'coherence_models')
    if not os.path.exists(coherence_path):
        os.mkdir(coherence_path)
    coherence_models_path = os.path.join(coherence_path, 'a{}_b{}_k{}_{}.txt'.format(alpha, beta, num_topics, coherence))
    with open(coeherence_models_path, 'w') as topics_file:
        topics_file.write(str(topics))
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence=coherence)   
    return coherence_model_lda.get_coherence(), coherence_models_path


def coherence_heatmap(topics_range, alpha_range, beta_range, corpus, texts, dictionary, coherence, current_path):
    model_results = {
        'Topics': [],
        'Alpha': [],
        'Beta': [],
        'Coherence': []
    }
    # iterate through number of topics
    for num_topics in topics_range:
        print('Topics:', num_topics)
        # iterate through alpha values
        for a in alpha_range:
            print('Alpha:', a)
            # iterare through beta values
            for b in beta_range:
                print('Beta', b)
                # get the coherence score for the given parameters
                print('Computing coherence value...')
                cv, topics_path = compute_coherence_values(corpus=corpus, texts=texts, dictionary=dictionary, alpha=a, beta=b, num_topics=num_topics, coherence=coherence)
                print('Coherence value:', cv)
                # Save the model results
                model_results['Topics'].append(num_topics)
                model_results['Alpha'].append(a)
                model_results['Beta'].append(b)
                model_results['Coherence'].append(cv)
    data = pd.DataFrame(model_results)
    data.to_pickle(os.path.join(current_path, 'coherence_c_v_data.pickle'))
    for topic_file in os.listdir(topics_path):
        with open(os.path.join(topics_path, topic_file), 'r') as f:
            file_content = f.read()
        topics = ast.literal_eval(file_content)
        pattern = 'a([0-9]\.?[0-9]*)_b([0-9]\.?[0-9]*)_k([0-9]+)_(.*)\.txt'
        matches = re.match(pattern, topic_file)
        alpha = float(matches.group(1))
        beta = float(matches.group(2))
        topics_number = int(matches.group(3))
        coherence_method = matches.group(4)
        topics_df = topics_df.append({
            'Alpha' : alpha,
            'Beta' : beta,
            'Topics_number' : topics_number,
            'Coherence_method' : coherence_method,
            'Topics' : topics
        }, ignore_index=True)
    df = topics_df.merge(coherence_df, on=['Topics_number', 'Alpha', 'Beta'])
    df.set_index(['Topics_number', 'Alpha', 'Beta'])
    topics = df['Topics_number'].unique()
    fig = go.Figure()
    for topic in topics:
        df_topic = df[df['Topics_number'] == topic]
        df_pivoted = df_topic.pivot_table(columns='Alpha', index='Beta', values='Coherence_value')
        tooltips = []
        for alpha in df_pivoted.columns:
            for beta in df_pivoted.index:
                tooltips_beta = []
                topics_string = ''
                topics_series = df[(df['Alpha'] == alpha) & (df['Beta'] == beta) & (df['Topics_number'] == topic)]['Topics']
                topics_list = list(topics_series)[0]
                for topic_name, topic_values in topics_list:
                    topics_string += str(topic_name) + ': ' + ", ".join([str(value[0]) for value in topic_values][:16]) + '<br>'
                tooltips_beta.append(topics_string)
            tooltips.append(tooltips_beta)
    
    fig.add_trace(go.Heatmap(
        z=df_pivoted.transpose().values.tolist(),
        x=df_pivoted.columns,
        y=df_pivoted.index,
        colorscale='Viridis',
        name=topic,
        hovertemplate='Alpha: %{x}<br>Beta: %{y}<br>Coherence: %{z}<br><br><extra></extra>'
    ))
    # Make 10th trace visible
    fig.data[10].visible = True
    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
        )
        step['label'] = i + 30
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Topics: "},
        pad={"t": 20},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )
    plotly.offline.plot(fig, filename = os.path.join(current_path, 'heatmap.html'), auto_open=True)

