import logging

import numpy as np

from AlgoritmoGenetico.BaseDatos import BD
from AlgoritmoGenetico.Operadores.evaluador import Evaluador
from AlgoritmoGenetico.Poblacion.individuo import Individuo
from AlgoritmoGenetico.Poblacion.poblacion import Poblacion

logger = logging.getLogger(__name__)
# handdle to write in file
formatter = logging.Formatter('%(asctime)s | %(message)s')
file_handler = logging.FileHandler('spam.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Iterador:
    def __init__(self, p_elitismo: float, prob_mutacion: float, p_crossover: float):
        """
        :param p_elitismo: Porcentaje de la poblacion con mejor valor pasa directamente a la sgte generacion.
        :param prob_mutacion: Probabilidad de que un gen mute durante el crossover.
        :param p_crossover: Porcentaje de la población con mejor valor que es cantidata para crossover.
        """
        self.p_elitismo = p_elitismo
        self.p_mutacion = prob_mutacion
        self.p_crossover = p_crossover

    def avanzar(self, poblacion: Poblacion, bd_size: int, OD_matrix, evaluador: Evaluador, bd: BD) -> Poblacion:
        """
        Avanza el algoritmo a la siguiente generación aplicando elitismo, crossover y mutación.
        :param OD_matrix: Matriz de demanda.
        :param poblacion: Poblacion.
        :param bd_size: Int. Tamaño de la base de datos de líneas.
        :return:
        """
        # Logging
        logger.info('Creando siguiente generación')

        # Ordenar la poblacion segun MVRC
        pobl_sorted = sorted(poblacion.get_population(), key=lambda x: x.MVRC)
        poblacion.set_population(pobl_sorted)

        # Elitismo
        nueva_poblacion_lista = []
        s = int((self.p_elitismo * poblacion.size))
        nueva_poblacion_lista.extend(poblacion.get_population()[:s])
        for ind in poblacion.get_population()[:s]:
            edl = ind.get_id_lineas()
            logger.info(f'EDL Agregada por elitismo {edl}, MVRC: {ind.get_MVRC()}')

        # Crossover y mutación. From 50% of fittest population, Individuals will mate to produce offspring
        s = poblacion.size - s
        z = int(self.p_crossover * poblacion.size)
        for _ in range(s):
            child, parent1, parent2 = None, None, None
            status = False
            while not status: # Iterar hasta encontrar un resultado factible
                parent1 = np.random.choice(poblacion.get_population()[:z])
                while not parent1.optimizado:
                    parent1 = np.random.choice(poblacion.get_population()[:z])
                parent2 = np.random.choice(poblacion.get_population()[:z])
                while not parent2.optimizado:
                    parent2 = np.random.choice(poblacion.get_population()[:z])
                child = self.crossover(parent1, parent2, bd_size)
                evaluador.construir_un_individuo(child, bd)
                status = child.validate(OD_matrix)
            nueva_poblacion_lista.append(child)
            # Logging
            logger.info(f'Crossover, padres: {parent1.get_id_lineas()} y {parent2.get_id_lineas()}, hijo'
                        f' {child.get_id_lineas()}')

        nueva_poblacion = Poblacion(size=poblacion.size, max_densitiy=poblacion.max_density)
        nueva_poblacion.set_population(nueva_poblacion_lista)
        nueva_poblacion.set_gen(poblacion.get_gen() + 1)
        return nueva_poblacion

    def crossover(self, madre: Individuo, padre: Individuo, bd_size: int) -> Individuo:
        """
        Ejecuta el crossover. Para cada gen, elige con probabilidades dadas entre los padres o mutación.
        :param madre: Individuo. Individuo precursor del crossover.
        :param padre: Individuo. Individuo precursor del crossover.
        :param bd_size: Int. Tamaño de la base de datos de líneas.
        :return: Individuo.
        """
        child_lines = []
        child_freq = {}
        s1 = (1-self.p_mutacion)/2
        s2 = (1-self.p_mutacion)
        len_m = len(madre.get_id_lineas()) # Largo madre
        len_p = len(padre.get_id_lineas())  # Largo padre

        for i in range(max(len_m, len_p)):
            # random probability
            prob = np.random.random()

            # if prob is less than s1, insert gene from mother
            if prob < s1:
                if i < len_m:
                    id = madre.get_id_lineas()[i]
                    child_lines.append(id)
                    child_freq[id] = madre.get_freq()[id]

            # if prob is between s1 and s2, insert gene from father
            elif prob < s2:
                if i < len_p:
                    id = padre.get_id_lineas()[i]
                    child_lines.append(id)
                    child_freq[id] = padre.get_freq()[id]

            # otherwise insert random gene(mutate)
            else:
                id = np.random.randint(bd_size)
                child_lines.append(id)
                child_freq[id] = 12

        child = Individuo(id_lineas=child_lines)
        child.set_freq(child_freq)
        return child
