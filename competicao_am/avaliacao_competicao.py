from base_am.avaliacao import OtimizacaoObjetivo, Experimento
from base_am.metodo import MetodoAprendizadoDeMaquina
from base_am.resultado import Fold, Resultado
from competicao_am.metodo_competicao import MetodoCompeticao
import optuna
from sklearn.svm import LinearSVC
import numpy as np
from typing import List,Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class OtimizacaoObjetivoSVMCompeticao(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, num_arvores_max:int=5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        #Um custo adequado para custo pode variar muito, por ex, para uma tarefa 
        #o valor de custo pode ser 10, para outra, 32000. 
        #Assim, normalmente, para conseguir valores mais distintos,
        #usamos c=2^exp_cost
        exp_cost = trial.suggest_uniform('min_samples_split', 0, 7) 

        scikit_method = LinearSVC(C=2**exp_cost, random_state=2)

        return MetodoCompeticao(scikit_method)

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1
