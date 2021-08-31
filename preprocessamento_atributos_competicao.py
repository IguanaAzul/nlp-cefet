import pandas as pd
from bags import BagOfWords, BagOfItems
import unidecode
from nltk.stem.snowball import SnowballStemmer
import nltk
from string import punctuation
from nltk import tokenize

nltk.download('stopwords')


def gerar_atributos_ator(df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame):
    obj_bag_of_actors = BagOfItems(min_occur=3)
    df_treino_boa = obj_bag_of_actors.cria_bag_of_items(df_treino, ["ator_1", "ator_2", "ator_3", "ator_4", "ator_5"])
    df_data_to_predict_boa = obj_bag_of_actors.aplica_bag_of_items(df_data_to_predict,
                                                                   ["ator_1", "ator_2", "ator_3", "ator_4", "ator_5"])
    return df_treino_boa, df_data_to_predict_boa


def gerar_atributos_diretor(df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame):
    obj_bag_of_actors = BagOfItems(min_occur=3)
    df_treino_boa = obj_bag_of_actors.cria_bag_of_items(df_treino, ["dirigido_por"])
    df_data_to_predict_boa = obj_bag_of_actors.aplica_bag_of_items(df_data_to_predict,
                                                                   ["dirigido_por"])
    return df_treino_boa, df_data_to_predict_boa


def gerar_atributos_escritores(df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame):
    obj_bag_of_actors = BagOfItems(min_occur=3)
    df_treino_boa = obj_bag_of_actors.cria_bag_of_items(df_treino, ["escrito_por_1", "escrito_por_2"])
    df_data_to_predict_boa = obj_bag_of_actors.aplica_bag_of_items(df_data_to_predict,
                                                                   ["escrito_por_1", "escrito_por_2"])
    return df_treino_boa, df_data_to_predict_boa


def gerar_atributos_geral(df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame):
    obj_bag_of_actors = BagOfItems(min_occur=3)
    df_treino_boa = obj_bag_of_actors.cria_bag_of_items(df_treino, ["ator_1", "ator_2", "ator_3", "ator_4", "ator_5",
                                                                    "dirigido_por", "escrito_por_1", "escrito_por_2"])
    df_data_to_predict_boa = obj_bag_of_actors.aplica_bag_of_items(df_data_to_predict,
                                                                   ["ator_1", "ator_2", "ator_3", "ator_4", "ator_5",
                                                                    "dirigido_por", "escrito_por_1", "escrito_por_2"])
    return df_treino_boa, df_data_to_predict_boa


def tratar_resumo(dataframe):
    dataframe["resumo"] = dataframe["resumo"].astype("str")
    pontuacao = [i for i in punctuation]
    tokenizer_pontuacao = tokenize.WordPunctTokenizer()
    pontuacao_stopwords = pontuacao + nltk.corpus.stopwords.words("english")
    stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]
    stemmer = SnowballStemmer("english")

    dataframe["resumo_tratado"] = [unidecode.unidecode(texto) for texto in dataframe["resumo"]]
    frase_processada = list()
    for resumo in dataframe["resumo_tratado"]:
        nova_frase = list()
        resumo = resumo.lower()
        palavras_texto = tokenizer_pontuacao.tokenize(resumo)
        for palavra in palavras_texto:
            if palavra not in stopwords_sem_acento:
                nova_frase.append(stemmer.stem(palavra))
        frase_processada.append(' '.join(nova_frase))
    dataframe["resumo_tratado"] = frase_processada
    return dataframe


def gerar_atributos_resumo(df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame):
    bow_amostra = BagOfWords()

    df_treino = tratar_resumo(df_treino)
    df_data_to_predict = tratar_resumo(df_data_to_predict)

    df_bow_treino = bow_amostra.cria_bow(df_treino, "resumo_tratado")
    df_bow_data_to_predict = bow_amostra.aplica_bow(df_data_to_predict, "resumo_tratado")

    return df_bow_treino, df_bow_data_to_predict
