# 自作のデータ読み込み&前処理用ライブラリ
from lib import scraping
from lib.tfidf import TfidfModel
from lib.doc2vec import Doc2Vec
from lib.utils import stems
from lib.text_tiling import TextTiling
from lib import utils
import datetime
import sys


if __name__ == '__main__':

    # ハイパーパラメータ
    train = False
    no_below = 10
    no_above=0.1
    keep_n=100000

    tfidf = TfidfModel(no_below=no_below, no_above=no_above, keep_n=keep_n, train=train)

    # docs: インタビュー全体
    print('Load data')
    path = './data/test.txt'
    # path = './data/interview-text_01-26_all.txt'
    data = utils.load_data(path)
    data = utils.to_sentence(data)
    docs = [row[1] for row in data]
    print('Done')

    # モデルを訓練する場合
    if train:
        print('===TFIDFモデル生成===')
        print('Train')
        # docs
        # for train
        print(docs[:1])
        tfidf.train([stems(doc) for doc in docs])
        print('Done')

    # 要約する単位 文 or 発言
    print(docs[:1])

    # GensimのTFIDFモデルを用いた文のベクトル化
    sent_vecs = tfidf.to_vector([stems(doc) for doc in docs])

    # print('===セグメンテーション===')
    # TODO
    text_tiling = TextTiling(sent_vecs)


    # with open('./result/segmentation/tfidf/' + str(datetime.date.today()) + '.txt', 'w') as f:
    #     print("no_below: " + str(no_below) + ", no_above: " + str(no_above) + ", keep_n: " + str(keep_n) + ", threshold: " + str(threshold), file=f)
    #     for i, docs in enumerate(docs_summary):
    #         print(str(i) + ': ' + docs.strip(), file=f)
