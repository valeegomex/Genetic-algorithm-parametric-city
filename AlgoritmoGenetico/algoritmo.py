import logging
import time

from AlgoritmoGenetico.BaseDatos.BD import BD
from AlgoritmoGenetico.Operadores.divisor import Divisor
from AlgoritmoGenetico.Operadores.evaluador import Evaluador
from AlgoritmoGenetico.Operadores.iterador import Iterador
from AlgoritmoGenetico.Poblacion.poblacion import Poblacion

# Desactivar mensajes de sidermit
logging.getLogger("sidermit").setLevel(logging.WARNING)
# Crear mensajes propios
logger = logging.getLogger(__name__)
logger_r = logging.getLogger('resultados')
# handdle to write in file
formatter = logging.Formatter('%(asctime)s | %(message)s')
file_handler = logging.FileHandler('spam.log')
file_handler_r = logging.FileHandler('Resultados.log')
file_handler.setLevel(logging.DEBUG)
file_handler_r.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
file_handler_r.setFormatter(formatter)

logger.addHandler(file_handler)
logger_r.addHandler(file_handler_r)

class Algoritmo_genetico:
    def __init__(self, n_zonas: int,  poblacion: Poblacion, gen_max: int, evaluador: Evaluador, iterador: Iterador,
                 divisor: Divisor, id:str, n_procesos:int = 1, name:str=None):
        """
        Inicializa y ejecuta automaticamente el algoritmo genetico. Retorna los resultados en un archivo de texto y
        graficos de las lineas de la EDL optima.
        :param n_zonas: Cantidad de zonas de la ciudad parametrica.
        :param poblacion: Poblacion sin construir.
        :param gen_max: Cantidad maxima de generaciones, es decir, iteraciones.
        :param evaluador: Evaluador.
        :param iterador: Iterador.
        :param divisor: Divisor.
        """

        self.n = n_zonas
        self.size_poblacion = poblacion.size
        self.densidad_max = poblacion.max_density
        self.gen_max = gen_max
        self.p_elitismo = iterador.p_elitismo
        self.prob_mutacion = iterador.p_mutacion
        self.p_crossover = iterador.p_crossover

        t0 = time.time()

        # Crear bd y poblacion
        bd = BD(n_zonas)
        self.bd = bd
        if name is None:
            poblacion.build_random(bd)
            name = 'Na'
        else:
            poblacion.build_from_file(name)
        # Reemplazamos los infactibles por factibles
        evaluador.construir_individuos(poblacion, bd)
        OD_matrix = evaluador.matriz_demanda(n_zonas)
        # evaluador.quitar_infactibles(bd, poblacion)

        resultados = []
        logger_r.info(f'Ciudad {n_zonas} zonas, Y={evaluador.Y}, a={evaluador.a}, alpha={evaluador.alpha}, beta={evaluador.beta} \n'
                          f'Archivo poblacion inicial: {name}, usando {n_procesos} procesadores \n'
                          f'Estrategia: {divisor.get_name()}, d1={divisor.d1}, d2={divisor.d2}, adicionales: {divisor.get_info_adicional()} \n'
                          f'Tamaño población {self.size_poblacion},elitismo {self.p_elitismo}, densidad máxima EDL {self.densidad_max}, \n'
                          f'crossover para los {self.p_crossover} mejores y probabilidad mutación {self.prob_mutacion} \n')
        logger.info(f'Ciudad {n_zonas} zonas, Y={evaluador.Y}, a={evaluador.a}, alpha={evaluador.alpha}, beta={evaluador.beta} \n'
                    f'Archivo poblacion inicial: {name} \n'
                    f'Estrategia: {divisor.get_name()}, d1={divisor.d1}, d2={divisor.d2}, adicionales: {divisor.get_info_adicional()} \n'
                    f'Tamaño población {self.size_poblacion},elitismo {self.p_elitismo}, densidad máxima EDL {self.densidad_max}, \n'
                    f'crossover para los {self.p_crossover} mejores y probabilidad mutación {self.prob_mutacion} \n')

        # Iterar para avanzar en las generaciones
        for i in range(gen_max):
            logger.info(f'GENERACION {i} \n')
            evaluador.evaluar_poblacion_multiprocess(poblacion, bd, n_procesos=n_procesos)
            divisor.dividir_multiprocess(poblacion, evaluador, bd, n_procesos=n_procesos)
            poblacion = iterador.avanzar(poblacion, bd.get_size(), OD_matrix, evaluador, bd)
            txt = f'Generacion: {poblacion.get_gen()}, mejor MVRC {evaluador.get_mvrc_min()}, MVRC promedio {evaluador.get_mvrc_mean()}, ' \
                  f'linea ganadora: {evaluador.get_edl_minimal().get_id_lineas()}'
            logger.info(txt)
            logger_r.info(txt)
            # Checkpoint
            poblacion.save_edl_population('checkpoint')

        # Calcular las estadísticas de la última generación
        evaluador.evaluar_poblacion_multiprocess(poblacion, bd, n_procesos=n_procesos)
        txt = f'Generacion: {poblacion.get_gen()}, mejor MVRC {evaluador.get_mvrc_min()}, MVRC promedio {evaluador.get_mvrc_mean()}, ' \
              f'linea ganadora: {evaluador.get_edl_minimal().get_id_lineas()}'
        logger.info(txt)
        logger_r.info(txt)

        tf = time.time()

        logger.info(f'Tiempo total ejecución: {tf-t0}')
        logger_r.info(f'Tiempo total ejecución: {tf - t0}')

        logger_r.info(f'Mejor EDL {evaluador.get_edl_minimal().get_id_lineas()}')
        logger_r.info(f'Tiempo total ejecución: {tf-t0}')

        # Graficar las lineas del mejor
        # edl = evaluador.get_edl_minimal()
        # for route in edl.network_sidermit.get_routes():
        #     edl.network_sidermit.plot(f'sidermit{n_zonas}zonasLinea{route.id}id{id}.png', list_routes=[route.id])
