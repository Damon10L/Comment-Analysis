from gensim.models.word2vec import Word2Vec
import numpy as np
import jieba
from sklearn.externals import joblib
import pandas as pd

# read Word2Vec and check out the new data
def average_vec(words):
    w2v = Word2Vec.load('data/w2v_model.pkl')
    vec = np.zeros(300).reshape(1,300)
    for word in words:
        try:
            vec += w2v[word].reshape((1,300))
        except KeyError:
            continue
    return vec

# categorize the emotions that the comments indicate
def svm_predict():
# read comments
    df = pd.read_csv('comments.csv',header = 0)
    comment_sentiment = []
    for string in df['评论内容']:
        # split the comments
        words = jieba.lcut(str(string))
        words_vec = average_vec(words)
        # read svm model
        model = joblib.load('data/svm_model.pkl')
        result = model.predict(words_vec)
        comment_sentiment.append(result[0])
# return the results
        if int(result[0])==1:
            print(string,'[积极]')
        else:
            print(string,'[消极]')
# merge the result with the original document
    merged = pd.concat([df, pd.Series(comment_sentiment,name = '用户情绪')], axis=1)
# store the document
    pd.DataFrame.to_csv(merged,'comment_sentiment.csv')
    print('done')

#execute
svm_predict()
