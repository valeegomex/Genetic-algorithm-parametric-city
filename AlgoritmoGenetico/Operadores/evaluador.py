import logging
from logging.handlers import QueueHandler

import numpy as np
import multiprocessing
from sidermit.city import Demand, Graph
from sidermit.optimization import Optimizer
from sidermit.publictransportsystem import Passenger, TransportMode

from AlgoritmoGenetico.BaseDatos.BD import BD
from AlgoritmoGenetico.Poblacion.individuo import Individuo
from AlgoritmoGenetico.Poblacion.poblacion import Poblacion

# Desactivar logging de sidermit
logging.getLogger("sidermit").setLevel(logging.WARNING)

class Evaluador:
    def __init__(self, passenger_obj: Passenger, custom_tmode: TransportMode, L: float, g: float, P:float, Y: float,
                 a: float, alpha: float, beta: float, n_zonas: int):
        """
        :param passenger_obj: Pasajero.
        :param custom_tmode: Modo de transporte.
        """
        self.pasajero = passenger_obj
        self.tmode = custom_tmode
        self.L = L
        self.g = g
        self.P = P
        self.Y = Y
        self.a = a
        self.alpha = alpha
        self.beta = beta
        self.mvrc_min = 0
        self.edl_minimal = None
        self.mvrc_mean = 0
        self.n_zonas = n_zonas

    def get_mvrc_min(self):
        return self.mvrc_min

    def set_mvrc_min(self, new_mvrc_min):
        self.mvrc_min = new_mvrc_min
        pass

    def get_mvrc_mean(self):
        return self.mvrc_mean

    def set_mvrc_mean(self, mean: float):
        self.mvrc_mean = mean
        pass

    def get_edl_minimal(self) -> Individuo:
        return self.edl_minimal

    def set_lineas_minimal(self, new_lineas_minimal):
        self.edl_minimal = new_lineas_minimal
        pass

    def actualizar_reporte(self, poblacion: Poblacion):
        mvrc_min = np.inf
        lineas_minimal = None
        mvrc_mean = []

        for ind in poblacion.get_population():
            # Buscar MVRC minimal
            ind_mvrc = ind.get_MVRC()
            if ind_mvrc < np.inf:
                mvrc_mean.append(ind_mvrc)
            if ind_mvrc < mvrc_min:
                mvrc_min = ind_mvrc
                lineas_minimal = ind

        # Guardar minimal y promedio
        mean = sum(mvrc_mean)/max(len(mvrc_mean),1)
        self.set_mvrc_mean(mean)
        self.set_mvrc_min(mvrc_min)
        self.set_lineas_minimal(lineas_minimal)
        pass

    def matriz_demanda(self, n: int):
        """
        Retorna la matriz de demanda.
        :param n: Cantidad de zonas de la ciudad.
        :return:
        """
        graph_obj = Graph.build_from_parameters(n, self.L, self.g, self.P)
        demand_obj = Demand.build_from_parameters(graph_obj, self.Y, self.a, self.alpha, self.beta)

        return demand_obj.get_matrix()

    def construir_individuos(self, poblacion: Poblacion, bd: BD):
        """
        Construye todos los individuos.
        :param poblacion: Conjunto de individuos.
        :param bd: Base de datos de línea.
        :return:
        """
        for ind in poblacion.get_population():
            ind.build_network(n=bd.get_n(), L=self.L, g=self.g, P=self.P, custom_tmode=self.tmode, bd=bd)
        pass

    def quitar_infactibles(self, bd: BD, poblacion: Poblacion):
        """
        Reemplaza los individuos infactibles de la población por otros factibles aleatoriamente.
        :param poblacion: Conjunto de individuos.
        :param bd: Base de datos de línea.
        :return:
        """
        OD_matrix = self.matriz_demanda(self.n_zonas)
        new_individuos = []
        for ind in poblacion.get_population():
            validator = ind.validate(OD_matrix)
            if validator: # Si es válido pasa directo
                new_individuos.append(ind)
            else: # Iterar hasta encontrar uno válido.
                ind_new = None
                while not validator:
                    ind_new = poblacion.ind_nuevo_azar(bd)
                    self.construir_un_individuo(ind_new, bd)
                    validator = ind_new.validate(OD_matrix)
                # Reemplazar con el nuevo
                new_individuos.append(ind_new)

        poblacion.set_population(new_individuos)
        pass


    def construir_un_individuo(self, ind: Individuo, bd: BD):
        """
        Construye el individuo entregado.
        :param poblacion: Conjunto de individuos.
        :param bd: Base de datos de línea.
        :return:
        """
        ind.build_network(n=bd.get_n(), L=self.L, g=self.g, P=self.P, custom_tmode=self.tmode, bd=bd)
        pass

    def evaluar_poblacion(self, poblacion: Poblacion, bd: BD):
        """
        Calcula el MVRC de cada individuo de la población y guarda el menor valor. Si hay individuos ya
        optimizados, simplemente consulta su MVRC.
        :param poblacion: Conjunto de individuos.
        :param bd: Base de datos de línea.
        :return:
        """
        mvrc_min = np.inf
        lineas_minimal = None
        mvrc_mean = []

        # create a logger
        logger = logging.getLogger(__name__)
        # log all messages, debug and up
        logger.setLevel(logging.INFO)

        # Iterar sobre los individuos
        for ind in poblacion.get_population():
            if not ind.optimizado:
                ind.build_network(n=bd.get_n(), L=self.L, g=self.g, P=self.P, custom_tmode=self.tmode, bd=bd)
                demand_obj = Demand.build_from_parameters(ind.graph_sidermit, self.Y, self.a, self.alpha, self.beta)
                # Optimizar
                ind.optimize(demand_obj, self.pasajero, bd)
                logger.info(f'Optimizando: {ind.get_id_lineas()} ')
            # Buscar MVRC minimal
            ind_mvrc = ind.get_MVRC()
            if ind_mvrc < np.inf:
                mvrc_mean.append(ind_mvrc)
            if ind_mvrc < mvrc_min:
                mvrc_min = ind_mvrc
                lineas_minimal = ind

        # Guardar minimal y promedio
        mean = sum(mvrc_mean)/max(len(mvrc_mean),1)
        self.set_mvrc_mean(mean)
        self.set_mvrc_min(mvrc_min)
        self.set_lineas_minimal(lineas_minimal)
        pass

    def evaluar_poblacion_multiprocess_interno(self, individuos: list[Individuo], queue: multiprocessing.Queue,
                                               logger_queue: multiprocessing.Queue):
        """
        Calcula el MVRC de cada individuo de la población y guarda el menor valor. Si hay individuos ya
        optimizados, simplemente consulta su MVRC. Función adaptada para multiplocesos. Los individuos nuevos
        no están construidos.
        :param queue:
        :param individuos: Lista de individuos a optimizar, ya deben estar construidos.
        :param bd: Base de datos de línea.
        :return: individuos_new, mvrc_mean, mvrc_min, edl_minimal
        """
        # create a logger
        logger = logging.getLogger(__name__)
        # add a handler that uses the shared queue
        logger.addHandler(QueueHandler(logger_queue))
        # log all messages, debug and up
        logger.setLevel(logging.INFO)

        # Iterar sobre los individuos
        for ind in individuos:
            if not ind.optimizado:
                demand_obj = Demand.build_from_parameters(ind.graph_sidermit, self.Y, self.a, self.alpha, self.beta)
                msge = ind.optimize_multiprocess(demand_obj, self.pasajero)
                logger.info(f'Optimizando: {ind.get_id_lineas()} ' + msge)
            # Guardar en el objeto compartido
            queue.put(ind)
        pass

    def evaluar_poblacion_multiprocess(self, poblacion: Poblacion, bd: BD, n_procesos: int):
        """
        Evalúa la población usando la cantidad de procesos solicitada. Si hay individuos ya optimizados, simplemente
        consulta su MVRC.
        :param poblacion: Población a evaluar.
        :param bd: Base de datos de líneas.
        :param n_procesos: Cantidad de procesos simultáneos a usar.
        :return:
        """

        # poblacion_no_optimizada = []
        new_population = []
        # Construir individuos y quitarles las variables que no se pueden migrar a la memoria compartida
        for ind in poblacion.get_population():
            ind.reset()
            ind.build_network(n=bd.get_n(), L=self.L, g=self.g, P=self.P, custom_tmode=self.tmode, bd=bd)

        # Crear el conjunto de individuos que irá a cada proceso
        step = len(poblacion.get_population())//n_procesos
        actual = 0
        sets_ind = []  # Conjuntos individuos
        sets_sizes = []   # Tamaño de los conjuntos
        for _ in range(n_procesos - 1):
            s = poblacion.get_population()[actual:actual+step]
            sets_ind.append(s)
            sets_sizes.append(len(s))
            actual += step
        # Ahora el último, que podría ser de largo variable
        s = poblacion.get_population()[actual:]
        sets_ind.append(s)
        sets_sizes.append(len(s))

        # Creamos el logger compartido de los procesos
        logger_queue = multiprocessing.Queue()
        # start the logger process
        logger_p = multiprocessing.Process(target=self.logger_process, args=(logger_queue,))
        logger_p.start()

        # Creamos los procesos y las colas de memoria compartida para rescatar los resultados de los procesos.
        queques = []
        procesos = []
        for i in range(n_procesos):
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=self.evaluar_poblacion_multiprocess_interno,
                                         args=(sets_ind[i], q,logger_queue))
            queques.append(q)
            procesos.append(p)

        # Iniciamos los procesos
        for p in procesos:
            p.start()

        # Recibimos los resultados
        for i in range(n_procesos):
            for _ in range(sets_sizes[i]):
                new_population.append(queques[i].get())

        # Finalizamos los procesos
        for p in procesos:
            p.join()

        # shutdown the queue correctly
        logger_queue.put(None)
        logger_p.join()

        # Actualizar la población nuevos y actualizar sus atributos, simultáneamente buscamos VRC minimal.
        poblacion.set_population(new_population)
        mvrc_mean = []
        mvrc_min = np.inf
        edl_minimal = None
        for ind in poblacion.get_population():
            if ind.optimizado:
                # Buscar MVRC minimal
                ind_mvrc = ind.get_MVRC()
                mvrc_mean.append(ind_mvrc)
                if ind_mvrc < mvrc_min:
                    mvrc_min = ind_mvrc
                    edl_minimal = ind

        # Guardar minimal y promedio
        mean = sum(mvrc_mean) / max(len(mvrc_mean), 1)
        self.set_mvrc_mean(round(mean,3))
        self.set_mvrc_min(round(mvrc_min,3))
        self.set_lineas_minimal(edl_minimal)
        pass

    def evaluar_individuo(self, ind: Individuo, bd: BD):
        """
        Construye la red del individuo, la demanda y lo optimiza.
        :param ind: Individuo a optimizar.
        :param bd: Base de datos de línea.
        :return:
        """
        if not ind.optimizado:
            ind.build_network(n=bd.get_n(), L=self.L, g=self.g, P=self.P, custom_tmode=self.tmode, bd=bd)
            demand_obj = Demand.build_from_parameters(ind.graph_sidermit, self.Y, self.a, self.alpha, self.beta)
            # Optimizar
            ind.optimize(demand_obj, self.pasajero, bd)
        pass

    @staticmethod
    # executed in a process that performs logging
    def logger_process(queue):
        # create a logger
        logger = logging.getLogger(__name__)
        # handdle to write in file
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        file_handler = logging.FileHandler('spam.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # configure a stream handler
        logger.addHandler(logging.StreamHandler())
        # log all messages, debug and up
        logger.setLevel(logging.DEBUG)
        # run forever
        while True:
            # consume a log message, block until one arrives
            message = queue.get()
            # check for shutdown
            if message is None:
                break
            # log the message
            logger.handle(message)
