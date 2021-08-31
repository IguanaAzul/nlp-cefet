from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import pandas as pd
import warnings
from typing import List


class Metricas:
    def __init__(self, y: List[float], predict_y: List[float]):
        self.y = y
        self.predict_y = predict_y
        self._mat_confusao = None
        self._precisao = None
        self._revocacao = None

    @property
    def mat_confusao(self):
        if self._mat_confusao is not None:
            return self._mat_confusao
        max_class_val = max([max(self.y), max(self.predict_y)])
        self._mat_confusao = np.zeros((max_class_val + 1, max_class_val + 1))
        for i, classe_real in enumerate(self.y):
            self._mat_confusao[classe_real][self.predict_y[i]] += 1
        return self._mat_confusao

    @property
    def precisao(self):
        if self._precisao is not None:
            return self._precisao
        self._precisao = np.zeros(len(self.mat_confusao))
        for classe in range(len(self.mat_confusao)):
            num_previstos_classe = 0
            for classe_real in range(len(self.mat_confusao)):
                num_previstos_classe += self.mat_confusao[classe_real][classe]
            if num_previstos_classe != 0:
                self._precisao[classe] = self.mat_confusao[classe][classe] / num_previstos_classe
            else:
                self._precisao[classe] = 0
                warnings.warn(
                    "Não há elementos previstos para a classe " + str(classe) + " precisão foi definida como zero.",
                    UndefinedMetricWarning)
        return self._precisao

    @property
    def revocacao(self):
        if self._revocacao is not None:
            return self._revocacao

        self._revocacao = np.zeros(len(self.mat_confusao))
        for classe in range(len(self.mat_confusao)):
            num_elementos_classe = 0
            for classe_prevista in range(len(self.mat_confusao)):
                num_elementos_classe += self.mat_confusao[classe][classe_prevista]
            if num_elementos_classe != 0:
                self._revocacao[classe] = self.mat_confusao[classe][classe] / num_elementos_classe
            else:
                self._revocacao[classe] = 0
                warnings.warn("Não há elementos da classe " + str(classe) + " revocação foi definida como zero.",
                              UndefinedMetricWarning)
        return self._revocacao

    @property
    def f1_por_classe(self):
        f1 = np.zeros(len(self.mat_confusao))
        for classe in range(len(self.mat_confusao)):
            if self.precisao[classe] + self.revocacao[classe] == 0:
                f1[classe] = 0
            else:
                f1[classe] = 2 * (self.precisao[classe] * self.revocacao[classe]) / (
                            self.precisao[classe] + self.revocacao[classe])
        return f1

    @property
    def macro_f1(self):
        return np.average(self.f1_por_classe)

    @property
    def acuracia(self):
        num_previstos_corretamente = 0
        for classe in range(len(self.mat_confusao)):
            num_previstos_corretamente += self.mat_confusao[classe][classe]

        return num_previstos_corretamente / len(self.y)


class Fold:
    def __init__(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame,
                 col_classe: str, num_folds_validacao: int = 0, num_repeticoes_validacao: int = 0):
        self.df_treino = df_treino
        self.df_data_to_predict = df_data_to_predict
        self.col_classe = col_classe
        if num_folds_validacao > 0:
            self.arr_folds_validacao = self.gerar_k_folds(df_treino, num_folds_validacao, col_classe,
                                                          num_repeticoes_validacao)
        else:
            self.arr_folds_validacao = []

    @staticmethod
    def gerar_k_folds(df_dados, val_k: int, col_classe: str, num_repeticoes: int = 1, seed: int = 1,
                      num_folds_validacao: int = 0, num_repeticoes_validacao: int = 1):
        num_instances_per_partition = len(df_dados) // val_k
        arr_folds = []
        for num_repeticao in range(num_repeticoes):
            df_dados_rand = df_dados.sample(frac=1, random_state=seed + num_repeticao)
            for num_fold in range(val_k):
                ini_fold_to_predict = num_instances_per_partition * num_fold
                if num_fold < val_k - 1:
                    fim_fold_to_predict = num_instances_per_partition * (num_fold + 1)
                else:
                    fim_fold_to_predict = len(df_dados_rand)
                df_to_predict = df_dados_rand[ini_fold_to_predict:fim_fold_to_predict]
                df_treino = df_dados_rand.drop(df_to_predict.index)
                fold = Fold(df_treino, df_to_predict, col_classe, num_folds_validacao, num_repeticoes_validacao)
                arr_folds.append(fold)

        return arr_folds

    def __str__(self):
        return f"Treino: \n{self.df_treino}\n Dados a serem avaliados (teste ou validação): {self.df_data_to_predict}"

    def __repr__(self):
        return str(self)
