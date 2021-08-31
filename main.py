import numpy as np
from mlmethod import MLMethod
import pandas as pd
from sklearn.neural_network import MLPClassifier
from metrics import Metrics
from sklearn.model_selection import train_test_split


def gerar_saida_teste(df_treino, df_teste, col_classe):
    ml_method = MLMethod(MLPClassifier(hidden_layer_sizes=[100], solver="adam"))
    y_to_predict, arr_predictions = ml_method.eval(df_treino, df_teste, col_classe)
    for movie, gender, prediction in zip(df_teste["titulo"].tolist(), df_teste["genero"].tolist(), arr_predictions):
        print(f"Filme: {movie}, Genero: {gender}, Predito: {ml_method.dic_int_to_nom_classe[prediction]}")
    resultados = Metrics(list(np.array(arr_predictions).astype(int)), y_to_predict)
    print("acc: ", resultados.acuracia)
    print("precis: ", resultados.precisao)
    print("revoc: ", resultados.revocacao)
    print("conf_matrix: ", resultados.mat_confusao)


df_movies = pd.read_csv("./datasets/movies_amostra.csv")
treino, teste = train_test_split(df_movies)
gerar_saida_teste(treino, teste, "genero")
