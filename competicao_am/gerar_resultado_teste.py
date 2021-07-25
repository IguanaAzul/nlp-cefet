import numpy as np
from competicao_am.metodo_competicao import MetodoCompeticao
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from base_am.resultado import Resultado


def gerar_saida_teste(df_data_to_predict:pd.DataFrame, col_classe:str, num_grupo):
    """
    Assim como os demais códigos da pasta "competicao_am", esta função 
    só poderá ser modificada na fase de geração da solução. 
    """
    #o treino será sempre o dataset completo - sem nenhum dado a mais e sem nenhum preprocessamento
    #esta função que deve encarregar de fazer o preprocessamento
    dataset = pd.read_csv("datasets/movies_amostra.csv")

    # ml_method_novo = MetodoCompeticao((MLPClassifier(hidden_layer_sizes=100, solver="adam", random_state=2),
    #                                   RandomForestClassifier(n_estimators=50, max_depth=100,min_samples_split=10,
    #                                                          min_samples_leaf=1, random_state=2)))

    ml_method = MetodoCompeticao(MLPClassifier(hidden_layer_sizes=100, solver="adam", random_state=2))
    #gera as representações e seu resultado
    # y_to_predict_novo, arr_predictions_novo = ml_method_novo.eval_novo(dataset, df_data_to_predict, col_classe)
    y_to_predict, arr_predictions = ml_method.eval(dataset, df_data_to_predict, col_classe)

    #grava o resultado obtido
    # with open(f"predict_grupo_{num_grupo}.txt","w") as file_predict:
    #     for predict in arr_predictions_novo:
    #         file_predict.write(ml_method_novo.dic_int_to_nom_classe[predict]+"\n")

    with open(f"predict_grupo_{num_grupo}.txt","w") as file_predict:
        for predict in arr_predictions:
            file_predict.write(ml_method.dic_int_to_nom_classe[predict]+"\n")

    if y_to_predict is not None:
        resultados = Resultado(list(np.array(arr_predictions).astype(int)), y_to_predict)
        print("acc: ", resultados.acuracia)
        print("precis: ", resultados.precisao)
        print("revoc: ", resultados.revocacao)
        print("conf_matrix: ", resultados.mat_confusao)

