import pandas as pd
from preprocessing import gerar_atributos_ator, gerar_atributos_resumo, \
    gerar_atributos_diretor, gerar_atributos_escritores, gerar_atributos_geral
from typing import List
import numpy as np


class MLMethod:
    def __init__(self, ml_method):
        self.ml_method = ml_method
        self.dic_int_to_nom_classe = {}
        self.dic_nom_classe_to_int = {}

    def class_to_number(self, y):
        arr_int_y = []

        for rotulo_classe in y:
            if rotulo_classe not in self.dic_nom_classe_to_int:
                int_new_val_classe = len(self.dic_nom_classe_to_int.keys())
                self.dic_nom_classe_to_int[rotulo_classe] = int_new_val_classe
                self.dic_int_to_nom_classe[int_new_val_classe] = rotulo_classe

            arr_int_y.append(self.dic_nom_classe_to_int[rotulo_classe])

        return arr_int_y

    def obtem_y(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        y_treino = self.class_to_number(df_treino[col_classe])
        y_to_predict = None
        if col_classe in df_data_to_predict.columns:
            y_to_predict = self.class_to_number(df_data_to_predict[col_classe])
        return y_treino, y_to_predict

    def obtem_x(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        x_treino = df_treino.drop(col_classe, axis=1)
        x_to_predict = df_data_to_predict
        if col_classe in df_data_to_predict.columns:
            x_to_predict = df_data_to_predict.drop(col_classe, axis=1)
        return x_treino, x_to_predict

    def eval_actors(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        x_treino, x_to_predict = self.obtem_x(df_treino, df_data_to_predict, col_classe)
        y_treino, y_to_predict = self.obtem_y(df_treino, df_data_to_predict, col_classe)

        df_treino_ator, df_to_predict_ator = gerar_atributos_ator(x_treino, x_to_predict)

        arr_df_to_remove_id = [df_treino_ator, df_to_predict_ator]
        for df_data in arr_df_to_remove_id:
            df_data.drop("id", axis=1)

        self.ml_method.fit(df_treino_ator, y_treino)
        arr_predict = self.ml_method.predict(df_to_predict_ator)
        return y_to_predict, arr_predict

    def eval_director(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        x_treino, x_to_predict = self.obtem_x(df_treino, df_data_to_predict, col_classe)
        y_treino, y_to_predict = self.obtem_y(df_treino, df_data_to_predict, col_classe)
        df_treino_diretor, df_to_predict_diretor = gerar_atributos_diretor(x_treino, x_to_predict)
        arr_df_to_remove_id = [df_treino_diretor, df_to_predict_diretor]
        for df_data in arr_df_to_remove_id:
            df_data.drop("id", axis=1)
        self.ml_method.fit(df_treino_diretor, y_treino)
        arr_predict = self.ml_method.predict(df_to_predict_diretor)
        return y_to_predict, arr_predict

    def eval_writers(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        x_treino, x_to_predict = self.obtem_x(df_treino, df_data_to_predict, col_classe)
        y_treino, y_to_predict = self.obtem_y(df_treino, df_data_to_predict, col_classe)
        df_treino_escritores, df_to_predict_escritores = gerar_atributos_escritores(x_treino, x_to_predict)
        arr_df_to_remove_id = [df_treino_escritores, df_to_predict_escritores]
        for df_data in arr_df_to_remove_id:
            df_data.drop("id", axis=1)
        self.ml_method.fit(df_treino_escritores, y_treino)
        arr_predict = self.ml_method.predict(df_to_predict_escritores)
        return y_to_predict, arr_predict

    def eval_bow(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        x_treino, x_to_predict = self.obtem_x(df_treino, df_data_to_predict, col_classe)
        y_treino, y_to_predict = self.obtem_y(df_treino, df_data_to_predict, col_classe)
        df_treino_bow, df_to_predict_bow = gerar_atributos_resumo(x_treino, x_to_predict)
        self.ml_method.fit(df_treino_bow, y_treino)
        arr_predict = self.ml_method.predict(df_to_predict_bow)
        return y_to_predict, arr_predict

    def eval(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        y_to_predict, y_actor = self.eval_actors(df_treino, df_data_to_predict, col_classe)
        y_to_predict, y_director = self.eval_director(df_treino, df_data_to_predict, col_classe)
        y_to_predict, y_writer = self.eval_writers(df_treino, df_data_to_predict, col_classe)
        y_to_predict, y_bow = self.eval_bow(df_treino, df_data_to_predict, col_classe)
        arr_predictions = list(np.array(self.combine_predictions(y_actor, y_director, y_writer, y_bow)).astype(int))
        return y_to_predict, arr_predictions

    def combine_predictions(self, arr_predictions_1: List[int], arr_predictions_2: List[int],
                            arr_predictions_3: List[int], arr_predictions_4: List[int]) -> List[int]:
        y_final_predictions = []
        for i1, i2, i3, i4 in zip(arr_predictions_1, arr_predictions_2, arr_predictions_3, arr_predictions_4):
            pred = round((i1 + i2 + i3 + i4 * 4) / 7)
            y_final_predictions.append(pred)
        return y_final_predictions

    def eval_novo(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        x_treino, x_to_predict = self.obtem_x(df_treino, df_data_to_predict, col_classe)
        y_treino, y_to_predict = self.obtem_y(df_treino, df_data_to_predict, col_classe)
        df_treino_geral, df_to_predict_geral = gerar_atributos_geral(x_treino, x_to_predict)
        df_treino_bow, df_to_predict_bow = gerar_atributos_resumo(x_treino, x_to_predict)
        self.ml_method[1].fit(df_treino_geral, y_treino)
        self.ml_method[0].fit(df_treino_bow, y_treino)
        arr_predict_rf = self.ml_method[1].predict_proba(df_to_predict_geral)
        arr_predict_nn = self.ml_method[0].predict_proba(df_to_predict_bow)
        arr_predictions = list(np.array(self.combine_predictions_novo(arr_predict_rf, arr_predict_nn)).astype(int))
        return y_to_predict, arr_predictions

    def combine_predictions_novo(self, arr_predictions_1: List[int], arr_predictions_2: List[int]) -> List[int]:
        y_final_predictions = []
        for i1, i2 in zip(arr_predictions_1, arr_predictions_2):
            pred = round((i1[1] * 0.1 + i2[1] * 0.9))
            y_final_predictions.append(pred)
        return y_final_predictions
