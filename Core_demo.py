import streamlit as st
import re
import nltk
import heapq
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from re import sub
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
from gensim.utils import simple_preprocess
from scipy.special import rel_entr
import pandas as pd
import base64



def get_base64_of_bin_file2(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file2(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: contain;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('nn-core3.gif')






#st.write(st.__version__)
st.title('How do People Cite?')
st.sidebar.title('Similarity Check Methods')

co_lst=[]
for i in range(50):
    co_lst.append(i+1)


#drop_down = st.sidebar.selectbox("Select References: ",co_lst)

level = st.sidebar.slider("Select the Threshold:", 0.0, 1.0)
#st.write(type(level))
check_box1=st.sidebar.checkbox(label='Cosine Similarity')
check_box2=st.sidebar.checkbox(label='Kl-Divergence')
check_box3=st.sidebar.checkbox(label='Word Embeddedings Similarity')

#main_name=st.text_input("Enter the Main Abstract:")
#ref_name=st.text_input("Enter the Reference Abstracts Seperated by @ :")

col1,col2=st.columns(2)

#col1.success('1st Column')

ref_name_lst=[]
with col1:
    #nm='Enter Main Abstract:'
    #st.text("Hello GeeksForGeeks!!!")
    #st.subheader("This is a subheader")
    st.markdown("##### Enter Main Abstract:")
    main_name=st.text_area(label='',height=200, max_chars=None, key=None)

with col2:
    st.markdown("##### Enter Reference Abstract:")
    ref_name=st.text_area(label=' ',height=200, max_chars=None, key=None)


ref_name_lst=str(ref_name).split('@')


#ref_name_lst.append(ref_name)

#ref_name=st.text_area(label='Enter Reference Abstract {}:'.format(i+1), height=275, max_chars=None,key=random.choice(string.ascii_uppercase)+str(random.randint(0,999999)))
#ch=random.choice(string.ascii_uppercase)+str(random.randint(0,999999))

#st.write(main_name)
#st.write(ref_name_lst)



#for i in range(drop_down):
    #ref_name_lst.append(st.text_area(label='Enter Reference Abstract {}:'.format(i+1), height=275, max_chars=None,key=1))


#st.write(ref_name_lst)












def get_summary(reqs):
    # base1=re.sub(r'\([^()]*\)',' ',reqs)
    base1 = re.sub(r'[\(\[].*?[\)\]]', ' ', reqs)
    # print(base1)
    # base1=re.sub(r'-',' ',b1)
    base2 = re.sub(r'\s+', ' ', base1)
    reqs1 = re.sub(r'[^0-9A-Za-z]', ' ', base2)
    # print(reqs1)
    stopwords = nltk.corpus.stopwords.words('english')
    sentence_list = nltk.sent_tokenize(base2)
    # print(sentence_list)
    q = nltk.word_tokenize(reqs1)
    removed_stop = []
    for i in q:
        if (i not in stopwords):
            removed_stop.append(i)

    word_freq = {}
    for ele in removed_stop:
        i = ele.lower()
        if (i in list(word_freq.keys())):
            word_freq[i] = word_freq[i] + 1

        else:
            word_freq[i] = 1

    max_value = max(list(word_freq.values()))

    for ele in list(word_freq.keys()):
        word_freq[ele] = word_freq[ele] / max_value
    # print(word_freq)

    sentence_freq = {}
    for sen in sentence_list:
        # print(sen)
        # print(len(sen.split(' ')))
        # print(sen.split(' '))
        if ((len(sen.split(' ')) < 70) and (len(sen.split(' ')) > 3)):
            lst1 = nltk.word_tokenize(sen.lower())
            # print(lst1)

            for i in lst1:
                if (i in list(word_freq.keys())):
                    if (sen in list(sentence_freq.keys())):
                        sentence_freq[sen] = sentence_freq[sen] + word_freq[i]

                    else:
                        sentence_freq[sen] = word_freq[i]

            # print(sen)
            sentence_freq[sen] = sentence_freq[sen] / len(sen.split(' '))
    # print(sentence_freq)

    # avg=np.mean(list(sentence_freq.values()))

    # keys=list(sentence_freq.keys())
    # hw=[]
    # for i in keys:
    #   if(sentence_freq[i]>avg):
    #      hw.append(i)
    # else:
    #    continue

    summary_sen = heapq.nlargest(3, sentence_freq, key=sentence_freq.get)
    # print(summary_sen)

    final_summary = ''
    for i in sentence_list:
        for j in summary_sen:
            if (i == j):
                final_summary = final_summary + j
                break
            else:
                continue

                # final_summary=''
    # for i in hw:
    #   final_summary=final_summary+i

    return final_summary


if(check_box1):
    input1=str(main_name)
    input2=ref_name_lst

    # main_abs
    #main_summary = []

    main_su = get_summary(input1)
    #main_summary.append(su)


    ref_sum_lst=[]
    for i in input2:
        ref_sum_lst.append(get_summary(i))


    #st.write(main_su)
    #st.write(ref_sum_lst)



    # Cosine Similarity


    cos_X=main_su
    cos_ref=ref_sum_lst

    cos_score=[]

    final=[cos_X]+cos_ref

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(final).todense()

    for i in range(1, len(features), 1):
        sc = cosine_similarity(X=features[0], Y=features[i], dense_output=True)

        cos_score.append(round(sc[0][0], 5))

    if(st.button('Check')):





        ind_lst=[]

        for i in range(len(cos_score)):
            ind_lst.append('Ref '+str(i+1))
        out_df= pd.DataFrame({'Similarity Scores':cos_score},index=ind_lst)

        #with st.container():
        st.write(out_df[out_df['Similarity Scores'] >= level])

        #for i in range(len(cos_score)):
         #   st.write('Ref'+' '+str(i)+' : '+str(cos_score[i]))



# Glove_model

if(check_box3):
    input1=str(main_name)
    input2=ref_name_lst

    # main_abs
    #main_summary = []

    main_su = get_summary(input1)
    #main_summary.append(su)


    ref_sum_lst=[]
    for i in input2:
        ref_sum_lst.append(get_summary(i))

    stopwords = ['the', 'and', 'are', 'a']


    # Glove word-embedding similarity

    def preprocess(doc):
        # Tokenize, clean up input document string
        doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
        # print(doc)
        doc = sub(r'<[^<>]+(>|$)', " ", doc)
        # print(doc)
        doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
        # print(doc)
        doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
        # print(doc)
        return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]


    def get_simscore(query, document):

        glove = api.load("glove-wiki-gigaword-50")
        similarity_index = WordEmbeddingSimilarityIndex(glove)

        query_string = query
        documents = document
        # print(type(documents))

        corpus = [preprocess(document) for document in documents]
        # print(len(corpus))
        query = preprocess(query_string)

        dictionary = Dictionary(corpus + [query])
        # print(dictionary)
        tfidf = TfidfModel(dictionary=dictionary)
        # print(tfidf)
        # Create the term similarity matrix.
        similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

        query_tf = tfidf[dictionary.doc2bow(query)]

        index = SoftCosineSimilarity(
            tfidf[[dictionary.doc2bow(document) for document in corpus]],
            similarity_matrix)

        doc_similarity_scores = index[query_tf]
        # print(doc_similarity_scores)

        # Output the sorted similarity scores and documents
        # sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
        # print(sorted_indexes)
        # for idx in sorted_indexes:
        # print(f'{doc_similarity_scores[idx]:0.3f} \t {documents[idx]}')

        return doc_similarity_scores

    query=main_su

    document=ref_sum_lst

    glove_score=get_simscore(query,document)


    #glove_final=glove_score.reshape((1,glove_score.shape[0]))
    if (st.button('Check')):

        embed_lst=[]
        for i in glove_score:
            embed_lst.append(i)

        ind_lst = []

        for i in range(len(embed_lst)):
            ind_lst.append('Ref ' + str(i + 1))
        out_df = pd.DataFrame({'Similarity Scores': embed_lst}, index=ind_lst)

        st.write(out_df[out_df['Similarity Scores'] >= level])



        #st.write(glove_score)

        #st.write(glove_score[0])




# KL_Divergence

if(check_box2):
    input1 = str(main_name)
    input2 = ref_name_lst

    # main_abs
    # main_summary = []

    main_su = get_summary(input1)
    # main_summary.append(su)

    ref_sum_lst = []
    for i in input2:
        ref_sum_lst.append(get_summary(i))


    stopwords = ['the', 'and', 'are', 'a']


    def preprocess(doc):
        # Tokenize, clean up input document string
        doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
        # print(doc)
        doc = sub(r'<[^<>]+(>|$)', " ", doc)
        # print(doc)
        doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
        # print(doc)
        doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
        # print(doc)
        return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]


    def get_wordvectors(query, document):

        glove = api.load("glove-wiki-gigaword-50")
        similarity_index = WordEmbeddingSimilarityIndex(glove)

        query_string = query
        documents = document
        # print(type(documents))

        corpus = [preprocess(document) for document in documents]
        # print(len(corpus))
        query = preprocess(query_string)

        dictionary = Dictionary(corpus + [query])
        # print(dictionary)
        tfidf = TfidfModel(dictionary=dictionary)
        # print(tfidf)
        # Create the term similarity matrix.
        similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

        query_tf = tfidf[dictionary.doc2bow(query)]

        index = SoftCosineSimilarity(
            tfidf[[dictionary.doc2bow(document) for document in corpus]],
            similarity_matrix)

        return list(query_tf), list(tfidf[[dictionary.doc2bow(document) for document in corpus]])


    query=main_su
    document=ref_sum_lst

    v1,v2=get_wordvectors(query,document)

    v1_lst=[]
    v2_lst=[]
    KL_div_lst=[]

    for i in v1:
        v1_lst.append(i[1])

    for j in v2:
        temp_lst = []
        for k in j:
            temp_lst.append(k[1])

        v2_lst.append(temp_lst)

    v1_shape = len(v1_lst)
    v1_lst = np.array(v1_lst)
    v1_lst = v1_lst.reshape((1, v1_shape))

    for i in range(0, len(v2_lst), 1):
        v2_shape = len(v2_lst[i])
        v2_temp = np.array(v2_lst[i])
        v2_temp = v2_temp.reshape((v2_shape, 1))
        kl_pq = rel_entr(v1_lst, v2_temp)
        n = np.array(kl_pq)
        arr = list(n[0])

        print('KL(P || Q): %.3f nats' % sum(arr))
        KL_div_lst.append(sum(arr))

    if (st.button('Check')):

        ind_lst = []

        for i in range(len(KL_div_lst)):
            ind_lst.append('Ref ' + str(i + 1))
        out_df = pd.DataFrame({'Similarity Scores': KL_div_lst}, index=ind_lst)

        st.write(out_df[out_df['Similarity Scores'] >= level])

        #st.write(KL_div_lst)



















