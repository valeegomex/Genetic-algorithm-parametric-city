import logging
import multiprocessing
import multiprocessing.process
from logging.handlers import QueueHandler

import numpy as np
from collections import defaultdict

from sidermit.city import Demand
from sidermit.publictransportsystem import TransportNetwork, TransportMode, passenger

from AlgoritmoGenetico.BaseDatos.BD import BD
from AlgoritmoGenetico.Operadores.evaluador import Evaluador
from AlgoritmoGenetico.Poblacion.individuo import Individuo
from AlgoritmoGenetico.Poblacion.poblacion import Poblacion

defaultdict3_float = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

class Divisor:
    def __init__(self, d1: float, d2:float, L: float):
        """
        :param d1: Constante relacionada a los transbordos.
        :param d2: Constante relacionada a el largo de la línea.
        :param umbral: Umbral para decidir si dividir o no.
        :param L: Parametro para construir la CP. Average distance in km of subcenters to (0,0) of the Cartesian plane.
        """
        self.d1 = d1
        self.d2 = d2
        self.L = L

    def dividir(self, population: Poblacion, evaluador: Evaluador, bd: BD):
        pass

    def dividir_interno(self, individuo_obj: Individuo, bd: BD) -> (bool,Individuo):
        pass

    @staticmethod
    def sequences_to_list(sequence: str) -> list[int]:
        """
        convert a string of node id sequence to a list
        :param sequence: String
        :return: List[node id]
        """
        if sequence == "" or sequence is None:
            return []

        nodes_split = sequence.split(",")
        nodes = []

        for node in nodes_split:
            nodes.append(int(node.rstrip("\n")))

        return nodes

    def actualizar_indice_divisibilidad(self, ind: Individuo):
        """
        Actualiza el valor del índice de divisibilidad de cada individuo en la poblacion.
        :param ind: Individuo. Objeto a modificar.
        :return:
        """
        indice_div = defaultdict(list)

        # Obtener subidas (z), bajadas (v) y carga por tramos (loaded_section_route).
        z, v, loaded_section_route = ind.obtain_z_v_loaded()
        nodes_stops = Divisor.itinerary(ind.get_network())
        nodes_sequence = Divisor.itinerary_sequences(ind.get_network())
        # Ordenamos las subidas, bajadas y carga según orden en el itinerario de la ruta y dirección.
        z_new, v_new, loaded_section_route_new = Divisor.sort_z_v_loaded_section_route(z, v, loaded_section_route, nodes_stops)
        # Revisar simetría
        simetria_dict = {}
        for route_id in z:
            simetria = [abs(i-r) for i,r in zip(loaded_section_route_new[route_id]['I'], loaded_section_route_new[route_id]['R'])]
            simetria = all([x<10 for x in simetria])
            simetria_dict[route_id] = simetria

        # Obtener términos
        diferencia_factor = Divisor.direncia_factor_carga(loaded_section_route_new, simetria_dict)
        transbordos_obj = Divisor.transbordos(v_new, loaded_section_route_new, simetria_dict)
        largos_linea_obj = Divisor.largos_linea(ind.graph_sidermit, loaded_section_route_new, nodes_stops,
                                                nodes_sequence, self.L, simetria_dict)

        # Obtener indice de divisibilidad
        for route_id in z_new:
            indice_div[route_id] = [dfc*(1-tau*self.d1)*(1+ll*self.d2) for dfc, tau, ll in
                                    zip(diferencia_factor[route_id], transbordos_obj[route_id], largos_linea_obj[route_id])]

        ind.set_indice_divisibilidad(indice_div)

        pass

    @staticmethod
    def largos_linea(graph_obj, loaded_section_route_new, itinerario, nodes_sequence, L, simetria):
        """
        Calcula (|l_i| + |l_i|/|l|)/L para cada nodo de cada ruta.
        :param nodes_sequence: Itinerary with or without stops. dic[route_id][direction] = list(nodes ID).
        :param loaded_section_route_new:  Itinerary. dic[route_id][direction] = list(nodes ID).
        :param graph_obj: Graph.
        :param itinerario: Lista de nodos ordenados. dic[route_id][direction] = list(nodes ID).
        :param L: Atributo del L grafo. Float.
        :param simetria: dict[route_id] = bool. True if the route is simetric, False otherwise.
        :return: dict[route_id] = list(float)
        """
        largo_linea_dif = defaultdict(list)

        len_edges = graph_obj.get_edges_distance()

        for route_id in loaded_section_route_new:
            largos_linea_dif_lista = [0] * (len(loaded_section_route_new[route_id]['I']) + 1)

            # Calculamos el largo total y acumulado de la línea
            largo_acumulado = []
            sgte_parada_index = 1
            sgte_parada = itinerario[route_id]['I'][sgte_parada_index]
            l_aux = 0
            ultima_parada = itinerario[route_id]['I'][-1]
            for i in range(len(nodes_sequence[route_id]['I']) - 1):
                nodei = nodes_sequence[route_id]['I'][i]
                nodej = nodes_sequence[route_id]['I'][i + 1]
                l_aux += len_edges[nodei][nodej]
                if nodej == sgte_parada: # Cuando alcanza la siguiente parada
                    if nodej == ultima_parada:
                        largo_acumulado.append(l_aux)
                        continue
                    else:
                        largo_acumulado.append(l_aux)
                        sgte_parada_index += 1
                        sgte_parada = itinerario[route_id]['I'][sgte_parada_index]
                        l_aux = 0
            largo_total = sum(largo_acumulado)
            largo_acumulado = np.cumsum(largo_acumulado)

            # Buscamos el tramo de carga máxima
            if simetria[route_id]:
                ida = loaded_section_route_new[route_id]['I']
                index_max = ida.index(max(ida))
            else:
                ida = loaded_section_route_new[route_id]['I']
                regreso = loaded_section_route_new[route_id]['R'].copy()
                regreso.reverse()
                if max(ida) > max(regreso):
                    index_max = ida.index(max(ida))
                else:
                    index_max = regreso.index(max(regreso))

            # Para cada nodo de la línea
            for i in range(1, len(loaded_section_route_new[route_id]['I'])):  # Ignorar el primer y último nodo
                if i < index_max: # maximo a la izquierda
                    # calular largo de l_i
                    l_i = largo_acumulado[i - 1]
                    largos_linea_dif_lista[i] = (l_i + l_i / largo_total) / L
                else:
                    l_i = largo_acumulado[-1] - largo_acumulado[i - 1]
                    largos_linea_dif_lista[i] = (l_i + l_i / largo_total) / L

            largo_linea_dif[route_id] = largos_linea_dif_lista

        return largo_linea_dif

    @staticmethod
    def transbordos(v_new, loaded_section_route_new, simetria):
        """
        Calcula la cantidad de transbordos en el nodo y ruta.
        :param v_new: Alight. dic[route_id][direction] = list(pax [pax/veh]).
        :param loaded_section_route_new: Itinerary. dic[route_id][direction] = list(nodes_id)
        :param simetria: dict[route_id] = bool. True if the route is simetric, False otherwise.
        :return: dict[route_id] = list(float)
        """
        transbord = defaultdict(list)

        for route_id in v_new:
            transbord_list = [0] * len(v_new[route_id]['I'])
            largo = len(v_new[route_id]['I']) - 1
            for i in range(1, largo+1):
                # Transbordos en la ida
                trans_ida = loaded_section_route_new[route_id]['I'][i - 1] - v_new[route_id]['I'][i]
                # Transbordos en la vuelta
                if simetria[route_id]:
                    trans_vue = 0
                else:
                    trans_vue = loaded_section_route_new[route_id]['R'][(largo - i) - 1] - v_new[route_id]['R'][largo - i]
                # Se suman
                transbord_list[i] = trans_vue + trans_ida
            transbord[route_id] = transbord_list

        return transbord

    @staticmethod
    def direncia_factor_carga(loaded_section_route_new, simetria):
        """
        For every route and stop node, calculate the load factor's differences beetwen the max load factor right and
        left the stop node in the route. If the load is simetric, then only one direction is considered.
        :param simetria: dict[route_id] = bool. True if the route is simetric, False otherwise.
        :param loaded_section_route_new: loaded of every edge in the route and direction.
        dic[route_id][direction] = list(pax [pax/veh])
        :return: dict[route_id] = list(load factor difference)
        """
        fc_diference = defaultdict(list)

        for route_id in loaded_section_route_new:
            if simetria[route_id]:
                # Capacidad
                k = max(max(loaded_section_route_new[route_id]['I']), max(loaded_section_route_new[route_id]['R']))
                ida = loaded_section_route_new[route_id]['I']
                fc = [0] * (len(ida) + 1)
                for i in range(1, len(ida)):
                    max_izq = max(ida[:i])
                    max_der = max(ida[i:])
                    fc[i] = abs(max_izq - max_der) / k
                fc_diference[route_id] = fc

            else:
                # Capacidad
                k = max(max(loaded_section_route_new[route_id]['I']), max(loaded_section_route_new[route_id]['R']))
                ida = loaded_section_route_new[route_id]['I']
                regreso = loaded_section_route_new[route_id]['R'].copy()
                regreso.reverse()
                fc = [0] * (len(ida) + 1)
                for i in range(1, len(ida)):
                    max_izq = max([max(regreso[:i]), max(ida[:i])])
                    max_der = max([max(regreso[i:]), max(ida[i:])])
                    fc[i] = abs(max_izq - max_der) / k
                fc_diference[route_id] = fc

        return fc_diference


    @staticmethod
    def sort_z_v_loaded_section_route(z: defaultdict3_float, v:  defaultdict3_float,
                                      loaded_section_route: defaultdict3_float, nodes_secuence: dict[dict[list]]):
        """
        Reduce una dimensión de los diccionarios z, v y loaded_section_route, ya que el último nivel lo transforma en
        una lista ordenada por el itinerario de la ruta y dirección respectiva.
        :param z: subidas z = dic[route_id][direction][stop: StopNode] = pax [pax/veh].
        :param v: bajadas v = dic[route_id][direction][stop: StopNode] = pax [pax/veh].
        :param loaded_section_route: carga por arista dic[route_id][direction][stop: StopNode] = pax [pax/veh].
        :param nodes_secuence: dic[route_id][direction] = list(nodes ID).
        :return: z_new = dic[route_id][direction] = list[pax [pax/veh]],
                 v_new = dic[route_id][direction] = list[pax [pax/veh]],
                 loaded_section_route_new =  dic[route_id][direction] = list[pax [pax/veh]].
        """
        z_new = defaultdict(lambda: defaultdict(list))
        v_new = defaultdict(lambda: defaultdict(list))
        loaded_section_route_new = defaultdict(lambda: defaultdict(list))

        for route_id in nodes_secuence:
            for direction in nodes_secuence[route_id]:
                nodes_id_seq = nodes_secuence[route_id][direction]
                # Not circular lines
                if len(nodes_id_seq) == 0:
                    continue
                if nodes_id_seq[0] == nodes_id_seq[-1]:
                    continue
                z_new[route_id][direction] = [0] * len(nodes_id_seq)
                v_new[route_id][direction] = [0] * len(nodes_id_seq)
                loaded_section_route_new[route_id][direction]  = [0] * (len(nodes_id_seq) - 1)

        for route_id in z_new:
            for direction in z_new[route_id]:
                #
                # # Secuencia id_nodos ordenado por la direccion
                nodes_id_seq = nodes_secuence[route_id][direction]
                # # Not circular lines
                # if nodes_id_seq[0] == nodes_id_seq[-1]:
                #     continue
                # # List for save the sort loaded secion route
                # z_list = [0] * len(nodes_id_seq)
                # v_list = [0] * len(nodes_id_seq)
                # loaded_section_route_list = [0] * (len(nodes_id_seq) - 1)

                for stop_node in z[route_id][direction]:
                    # Search the node position in the itinerary
                    index = nodes_id_seq.index(stop_node.city_node.id)
                    # Save loaded value in the right position
                    z_new[route_id][direction][index] = z[route_id][direction][stop_node]

                for stop_node in v[route_id][direction]:
                    # Search the node position in the itinerary
                    index = nodes_id_seq.index(stop_node.city_node.id)
                    # Save loaded value in the right position
                    v_new[route_id][direction][index] = v[route_id][direction][stop_node]

                for stop_node in loaded_section_route[route_id][direction]:
                    # Search the node position in the itinerary
                    index = nodes_id_seq.index(stop_node.city_node.id)
                    # Save loaded value in the right position
                    loaded_section_route_new[route_id][direction][index] = loaded_section_route[route_id][direction][stop_node]

                # # Save list
                # z_new[route_id][direction] = z_list
                # v_new[route_id][direction] = v_list
                # loaded_section_route_new[route_id][direction] = loaded_section_route_list

        return z_new, v_new, loaded_section_route_new

    @staticmethod
    def itinerary(network_obj: TransportNetwork) -> dict[dict[list]]:
        """
        Para cada ruta y dirección entrega una lista del ID de las paradas ordenadas.
        :param network_obj: Network
        :return: dic[route_id][direction] = list(nodes ID)
        """
        routes_nodes_index = defaultdict(lambda: defaultdict(list))

        # get all routes in network:obj
        routes = network_obj.get_routes()

        for route in routes:
            routes_nodes_index[route.id]['I'] = route.stops_sequence_i
            routes_nodes_index[route.id]['R'] = route.stops_sequence_r

        return routes_nodes_index

    @staticmethod
    def itinerary_sequences(network_obj: TransportNetwork) -> dict[dict[list]]:
        """
        Para cada ruta y dirección entrega una lista del ID de la secuencia ordenada.
        :param network_obj: Network
        :return: dic[route_id][direction] = list(nodes ID)
        """
        routes_nodes_index = defaultdict(lambda: defaultdict(list))

        # get all routes in network:obj
        routes = network_obj.get_routes()

        for route in routes:
            routes_nodes_index[route.id]['I'] = route.nodes_sequence_i
            routes_nodes_index[route.id]['R'] = route.nodes_sequence_r

        return routes_nodes_index

    @staticmethod
    # executed in a process that performs logging
    def logger_process(queue):
        # create a logger
        logger = logging.getLogger(__name__)
        # if (logger.hasHandlers()):
        #     logger.handlers.clear()
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
            logger.info(message)

    def dividir_multiprocess(self, poblacion: Poblacion, evaluador: Evaluador, bd: BD, n_procesos: int):
        pass


class Divisor_umbral(Divisor):

    def __init__(self, d1: float, d2:float, L: float, umbral:float):
        self.umbral = umbral
        Divisor.__init__(self, d1, d2, L)

    def dividir(self, population: Poblacion, evaluador: Evaluador, bd: BD):
        logger = logging.getLogger(__name__)
        # handdle to write in file
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        file_handler = logging.FileHandler('spam.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info('Inicia division')
        for ind in population.get_population():
            logger.info(f'Dividiendo: {ind.get_id_lineas()}, MVRC: {ind.get_MVRC()}')
            status_opt = ind.get_optimizado()
            if status_opt:
                # calcular índice de divisibilidad
                self.actualizar_indice_divisibilidad(ind)
                # repetir hasta que no se deba seguir cambiando
                status = self.dividir_interno(ind, bd)
                while status:
                    evaluador.evaluar_individuo(ind, bd)
                    logger.info(f'dividido a {ind.get_id_lineas()}, MVRC: {ind.get_MVRC()}')
                    if not ind.get_optimizado(): # Si la optimizacion falla
                        break
                    self.actualizar_indice_divisibilidad(ind)
                    status = self.dividir_interno(ind, bd)
            logger.info(f'{ind.get_id_lineas()} no se divide')

        # Actualizamos los datos de la poblacion para el reporte
        evaluador.evaluar_poblacion(population, bd)
        pass

    def dividir_multiprocess_interno(self, individuos_lista: list[Individuo], evaluador: Evaluador, bd: BD,
                                     queue: multiprocessing.Queue, logger_queue: multiprocessing.Queue):
        """
        Solamente debería recibir individuos ya optimizados y con el índice de divisibilidad calculado.
        :param individuos_lista:
        :param evaluador:
        :param bd:
        :return:
        """
        # # create a logger
        # logger = logging.getLogger(__name__)
        # # add a handler that uses the shared queue
        # logger.addHandler(QueueHandler(logger_queue))
        # # log all messages, debug and up
        # logger.setLevel(logging.INFO)

        process = multiprocessing.current_process()

        # Los mensajes se almacenarán y enviarán a la cola al final para que aparezcan en la consola de forma consecutiva.
        for ind in individuos_lista:
            mensajes = []
            mensajes.append(f'{process.name} Dividiendo: {ind.get_id_lineas()}, MVRC: {ind.get_MVRC()}')
            # Hacer la primera división y repetir hasta que el índice lo indique o la optimización falle.
            status = self.dividir_interno(ind, bd)
            while status:
                evaluador.evaluar_individuo(ind, bd)
                mensajes.append(f'{process.name} dividido a {ind.get_id_lineas()}, MVRC: {ind.get_MVRC()}')
                if not ind.get_optimizado(): # Si la optimizacion falla
                    break
                self.actualizar_indice_divisibilidad(ind)
                status = self.dividir_interno(ind, bd)
            mensajes.append(f'{process.name} {ind.get_id_lineas()} no se divide')
            # Enviar mensajes a logger
            for msje in mensajes:
                logger_queue.put(msje)
                # logger.info(msje)
            # Guardar en el objeto compartido
            ind.reset_multiplocess()
            queue.put(ind)
        pass

    def dividir_multiprocess(self, poblacion: Poblacion, evaluador: Evaluador, bd: BD, n_procesos: int):

        if n_procesos > 4:
            print('Número de procesos demasiado alto, intente menor que 5')
            pass

        # Actualizar indice de divisibilidad y resetear para poder migrar los datos a la memoria compartida
        divisibles = []
        new_population = []

        for _ in range(poblacion.size):
            ind = poblacion.get_population().pop()
            if ind.get_optimizado():
                self.actualizar_indice_divisibilidad(ind)
                ind.reset_multiplocess()
                divisibles.append(ind)
            else:
                ind.reset()
                new_population.append(ind)

        # Crear el conjunto de individuos que irá a cada proceso
        step = len(divisibles) // n_procesos
        actual = 0
        sets_ind = []  # Conjuntos individuos
        sets_sizes = []  # Tamaño de los conjuntos
        for _ in range(n_procesos - 1):
            s = divisibles[actual:actual + step]
            sets_ind.append(s)
            sets_sizes.append(len(s))
            actual += step
        # Ahora el último, que podría ser de largo variable
        s = divisibles[actual:]
        sets_ind.append(s)
        sets_sizes.append(len(s))

        # Creamos el logger compartido de los procesos
        logger_queue = multiprocessing.Queue()
        # create a logger
        # logger = logging.getLogger(__name__)
        # # add a handler that uses the shared queue
        # logger.addHandler(QueueHandler(logger_queue))
        # # log all messages, debug and up
        # logger.setLevel(logging.INFO)

        logger_p = multiprocessing.Process(target=self.logger_process, args=(logger_queue,))
        logger_p.start()

        # Creamos los procesos y las colas de memoria compartida para rescatar los resultados de los procesos.
        queques = []
        procesos = []
        for i in range(n_procesos):
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=self.dividir_multiprocess_interno,
                                        args=(sets_ind[i], evaluador, bd, q, logger_queue))
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

        # shutdown the queue correctly and the logger process
        logger_queue.put(None)
        logger_p.join()

        # Actualizamos la población
        poblacion.set_population(new_population)
        # Actualizamos los datos de la poblacion para el reporte
        evaluador.actualizar_reporte(poblacion)
        pass

    def dividir_interno(self, individuo_obj: Individuo, bd: BD) -> bool:
        """
        Realiza una división sobre el individuo; lo reinicia si es necesario.
        :param individuo_obj: Individuo a dividir.
        :return: True si se realizó la división, False en caso contrario.
        """
        idx_div = individuo_obj.get_indice_divisibilidad()
        # Debemos encontrar la línea y nodo de mayor índice de divisibilidad en el individuo
        linea_id = max(idx_div, key= lambda x: max(idx_div[x])) # Linea
        idx = max(idx_div[linea_id]) # valor del indice max
        i = idx_div[linea_id].index(idx) # posicion en la linea del nodo

        # Hay que dividir en i-ésimo nodo de la línea_id
        if idx > self.umbral:
            # Preparar paradas nuevas
            paradas_old = bd.getByIndex(linea_id).get_paradas_i()
            paradas_old = self.sequences_to_list(paradas_old)
            parada_nueva1 = paradas_old[:i+1]
            parada_nueva2 = paradas_old[i:]

            # Buscar el índice de las nuevas líneas
            id_nueva1 = bd.buscar_linea_lista(parada_nueva1)
            id_nueva2 = bd.buscar_linea_lista(parada_nueva2)

            # Modificar la lista de lineas
            id_lineas_nueva = individuo_obj.get_id_lineas().copy()
            id_lineas_nueva.remove(linea_id)
            id_lineas_nueva.append(id_nueva1)
            id_lineas_nueva.append(id_nueva2)

            # Modificar frecuencias opt
            # (ya no serán opt pero ayudan al optimizador a converger más rapido)
            freq_nueva = individuo_obj.get_freq().copy()
            freq_linea_old = freq_nueva.pop(linea_id)
            freq_nueva[id_nueva1] = freq_linea_old
            freq_nueva[id_nueva2] = freq_linea_old

            ## creae un metodo individuo.actualizar()
            individuo_obj.set_freq(freq_nueva)
            individuo_obj.id_lineas = id_lineas_nueva
            individuo_obj.reset()

            # Retorna True
            return True

        # Nunca entró en el if
        return False


class Divisor_formula(Divisor):

    def dividir_multiprocess_interno(self, individuos_lista: list[Individuo], evaluador: Evaluador, bd: BD,
                                     queue: multiprocessing.Queue, logger_queue: multiprocessing.Queue):
        """
        Solamente debería recibir individuos ya optimizados y con el índice de divisibilidad calculado.
        :param individuos_lista:
        :param evaluador:
        :param bd:
        :return:
        """

        process = multiprocessing.current_process()

        # Los mensajes se almacenarán y enviarán a la cola al final para que aparezcan en la consola de forma consecutiva.
        for ind in individuos_lista:
            mensajes = []
            mensajes.append(f'{process.name} Dividiendo: {ind.get_id_lineas()}, MVRC: {ind.get_MVRC()}')
            # Hacer la primera división y repetir hasta que el MVRC lo indique o la optimización falle.
            ind_new = self.dividir_interno(ind, bd)
            while True:
                evaluador.evaluar_individuo(ind_new, bd)
                if not ind_new.get_optimizado():  # Si la optimizacion falla
                    break
                if ind_new.get_MVRC() < ind.get_MVRC(): # La division tiene costo menor
                    ind = ind_new
                    mensajes.append(f'{process.name} dividido a {ind.get_id_lineas()}, MVRC: {ind.get_MVRC()}')
                    self.actualizar_indice_divisibilidad(ind)
                    ind_new = self.dividir_interno(ind, bd)
                else:
                    break
            mensajes.append(f'{process.name} {ind.get_id_lineas()} no se divide')
            # Enviar mensajes a logger
            for msje in mensajes:
                logger_queue.put(msje)
            # Guardar en el objeto compartido
            ind.reset_multiplocess()
            queue.put(ind)
        pass

    def dividir_multiprocess(self, poblacion: Poblacion, evaluador: Evaluador, bd: BD, n_procesos: int):

        if n_procesos > 4:
            print('Número de procesos demasiado alto, intente menor que 5')
            pass

        # Actualizar indice de divisibilidad y resetear para poder migrar los datos a la memoria compartida
        divisibles = []
        new_population = []

        for _ in range(poblacion.size):
            ind = poblacion.get_population().pop()
            if ind.get_optimizado():
                self.actualizar_indice_divisibilidad(ind)
                ind.reset_multiplocess()
                divisibles.append(ind)
            else:
                ind.reset()
                new_population.append(ind)

        # Crear el conjunto de individuos que irá a cada proceso
        step = len(divisibles) // n_procesos
        actual = 0
        sets_ind = []  # Conjuntos individuos
        sets_sizes = []  # Tamaño de los conjuntos
        for _ in range(n_procesos - 1):
            s = divisibles[actual:actual + step]
            sets_ind.append(s)
            sets_sizes.append(len(s))
            actual += step
        # Ahora el último, que podría ser de largo variable
        s = divisibles[actual:]
        sets_ind.append(s)
        sets_sizes.append(len(s))

        # Creamos el logger compartido de los procesos
        logger_queue = multiprocessing.Queue()
        logger_p = multiprocessing.Process(target=self.logger_process, args=(logger_queue,))
        logger_p.start()

        # Creamos los procesos y las colas de memoria compartida para rescatar los resultados de los procesos.
        queques = []
        procesos = []
        for i in range(n_procesos):
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=self.dividir_multiprocess_interno,
                                        args=(sets_ind[i], evaluador, bd, q, logger_queue))
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

        # shutdown the queue correctly and the logger process
        logger_queue.put(None)
        logger_p.join()

        # Actualizamos la población
        poblacion.set_population(new_population)
        # Actualizamos los datos de la poblacion para el reporte
        evaluador.actualizar_reporte(poblacion)
        pass

    def dividir_interno(self, individuo_obj: Individuo, bd: BD) -> Individuo:
        """
        Realiza una división sobre una copia del individuo y la retorna.
        :param individuo_obj: Individuo a dividir.
        :return: Individuo.
        """
        idx_div = individuo_obj.get_indice_divisibilidad()
        # Debemos encontrar la línea y nodo de mayor índice de divisibilidad en el individuo
        linea_id = max(idx_div, key= lambda x: max(idx_div[x])) # Linea
        idx = max(idx_div[linea_id]) # valor del indice max
        i = idx_div[linea_id].index(idx) # posicion en la linea del nodo

        # Hay que dividir en i-ésimo nodo de la línea_id

        # Preparar paradas nuevas
        paradas_old = bd.getByIndex(linea_id).get_paradas_i()
        paradas_old = self.sequences_to_list(paradas_old)
        parada_nueva1 = paradas_old[:i+1]
        parada_nueva2 = paradas_old[i:]

        # Buscar el índice de las nuevas líneas
        id_nueva1 = bd.buscar_linea_lista(parada_nueva1)
        id_nueva2 = bd.buscar_linea_lista(parada_nueva2)

        # Modificar la lista de lineas
        id_lineas_nueva = individuo_obj.get_id_lineas().copy()
        id_lineas_nueva.remove(linea_id)
        id_lineas_nueva.append(id_nueva1)
        id_lineas_nueva.append(id_nueva2)

        # Modificar frecuencias opt
        # (ya no serán opt pero ayudan al optimizador a converger más rapido)
        freq_nueva = individuo_obj.get_freq().copy()
        freq_linea_old = freq_nueva.pop(linea_id)
        freq_nueva[id_nueva1] = freq_linea_old
        freq_nueva[id_nueva2] = freq_linea_old

        # Crear individuo nuevo
        individuo_obj_new = Individuo(id_lineas_nueva)
        individuo_obj_new.set_freq(freq_nueva)

        # Retorna
        return individuo_obj_new


class Divisor_intervalo(Divisor):

    def __init__(self,  d1: float, d2:float, L: float, umbral:float, tolerancia: float):
        """
        En el intervalo [umbral*(1-tolerancia), umbral*(1+tolerancia)] usa fórmula. En caso contrario umbral.
        :param d1:
        :param d2:
        :param umbral:
        :param L:
        :param tolerancia:
        """
        self.tolerancia = tolerancia
        self.umbral = umbral
        Divisor.__init__(self, d1, d2, L)

    def paso_umbral(self, individuo_obj: Individuo, bd: BD):
        """
        Realiza sólo un paso de división con umbral.  Si la división se ejecuta retorna True, individuo_obj_dividido,
        y si no, entonces retorna False, individuo_obj.
        :return:
        """
        idx_div = individuo_obj.get_indice_divisibilidad()
        # Debemos encontrar la línea y nodo de mayor índice de divisibilidad en el individuo
        linea_id = max(idx_div, key=lambda x: max(idx_div[x]))  # Linea
        idx = max(idx_div[linea_id])  # valor del indice max
        i = idx_div[linea_id].index(idx)  # posicion en la linea del nodo

        # Hay que dividir en i-ésimo nodo de la línea_id
        if idx > self.umbral:
            # Preparar paradas nuevas
            paradas_old = bd.getByIndex(linea_id).get_paradas_i()
            paradas_old = self.sequences_to_list(paradas_old)
            parada_nueva1 = paradas_old[:i + 1]
            parada_nueva2 = paradas_old[i:]

            # Buscar el índice de las nuevas líneas
            id_nueva1 = bd.buscar_linea_lista(parada_nueva1)
            id_nueva2 = bd.buscar_linea_lista(parada_nueva2)

            # Modificar la lista de lineas
            id_lineas_nueva = individuo_obj.get_id_lineas().copy()
            id_lineas_nueva.remove(linea_id)
            id_lineas_nueva.append(id_nueva1)
            id_lineas_nueva.append(id_nueva2)

            # Modificar frecuencias opt
            # (ya no serán opt pero ayudan al optimizador a converger más rapido)
            freq_nueva = individuo_obj.get_freq().copy()
            freq_linea_old = freq_nueva.pop(linea_id)
            freq_nueva[id_nueva1] = freq_linea_old
            freq_nueva[id_nueva2] = freq_linea_old

            ## creae un metodo individuo.actualizar()
            individuo_obj_new = Individuo(id_lineas_nueva)
            individuo_obj_new.set_freq(freq_nueva)

            # Retorna True
            return True, individuo_obj_new

        # Nunca entró en el if
        return False, individuo_obj


    def paso_formula(self,  individuo_obj: Individuo, evaluador: Evaluador, bd: BD):
        """
        Realiza sólo un paso de división con fórmula. Si la división se ejecuta retorna True, individuo_obj_dividido,
        y si no, entonces retorna False, individuo_obj.
        :param individuo_obj:
        :param evaluador:
        :param bd:
        :return:
        """

        ind_new = self.dividir_interno(individuo_obj, bd)
        evaluador.evaluar_individuo(ind_new, bd)
        if not ind_new.get_optimizado():  # Si la optimizacion falla
            return False, individuo_obj
        if ind_new.get_MVRC() < individuo_obj.get_MVRC():  # La division tiene costo menor
            return True, ind_new
        else: # La división tiene costo mayor
            return False, individuo_obj



    def dividir_multiprocess_interno(self, individuos_lista: list[Individuo], evaluador: Evaluador, bd: BD,
                                     queue: multiprocessing.Queue, logger_queue: multiprocessing.Queue):
        """
        Solamente debería recibir individuos ya optimizados y con el índice de divisibilidad calculado.
        :param individuos_lista:
        :param evaluador:
        :param bd:
        :return:
        """

        process = multiprocessing.current_process()

        # Limites intervalo
        min_ = self.umbral*(1-self.tolerancia)
        max_ = self.umbral*(1+self.tolerancia)

        # Los mensajes se almacenarán y enviarán a la cola al final para que aparezcan en la consola de forma consecutiva.
        for ind in individuos_lista:

            mensajes = []
            mensajes.append(f'{process.name} Dividiendo: {ind.get_id_lineas()}, MVRC: {ind.get_MVRC()}')

            status = True
            while status:
                # Revisar su índice de divisibilidad máximo
                idx_div = ind.get_indice_divisibilidad()
                # Debemos encontrar la línea y nodo de mayor índice de divisibilidad en el individuo
                linea_id = max(idx_div, key=lambda x: max(idx_div[x]))  # Linea
                idx = max(idx_div[linea_id])  # valor del indice max

                # Decidir si usar fórmula o umbral
                if (min_ < idx) and (idx < max_):
                    status, ind = self.paso_formula(ind, evaluador, bd)
                    if status:
                        self.actualizar_indice_divisibilidad(ind)
                        mensajes.append(f'{process.name} dividido a {ind.get_id_lineas()}, con fórmula, MVRC: {ind.get_MVRC()}')
                else:
                    status, ind = self.paso_umbral(ind, bd)
                    evaluador.evaluar_individuo(ind, bd)
                    if status:
                        self.actualizar_indice_divisibilidad(ind)
                        mensajes.append(f'{process.name} dividido a {ind.get_id_lineas()}, con umbral, MVRC: {ind.get_MVRC()}')

            mensajes.append(f'{process.name} {ind.get_id_lineas()} no se divide')
            # Enviar mensajes a logger
            for msje in mensajes:
                logger_queue.put(msje)
            # Guardar en el objeto compartido
            ind.reset_multiplocess()
            queue.put(ind)
        pass

    def dividir_multiprocess(self, poblacion: Poblacion, evaluador: Evaluador, bd: BD, n_procesos: int):

        if n_procesos > 4:
            print('Número de procesos demasiado alto, intente menor que 5')
            pass

        # Actualizar indice de divisibilidad y resetear para poder migrar los datos a la memoria compartida
        divisibles = []
        new_population = []

        for _ in range(poblacion.size):
            ind = poblacion.get_population().pop()
            if ind.get_optimizado():
                self.actualizar_indice_divisibilidad(ind)
                ind.reset_multiplocess()
                divisibles.append(ind)
            else:
                ind.reset()
                new_population.append(ind)

        # Crear el conjunto de individuos que irá a cada proceso
        step = len(divisibles) // n_procesos
        actual = 0
        sets_ind = []  # Conjuntos individuos
        sets_sizes = []  # Tamaño de los conjuntos
        for _ in range(n_procesos - 1):
            s = divisibles[actual:actual + step]
            sets_ind.append(s)
            sets_sizes.append(len(s))
            actual += step
        # Ahora el último, que podría ser de largo variable
        s = divisibles[actual:]
        sets_ind.append(s)
        sets_sizes.append(len(s))

        # Creamos el logger compartido de los procesos
        logger_queue = multiprocessing.Queue()
        logger_p = multiprocessing.Process(target=self.logger_process, args=(logger_queue,))
        logger_p.start()

        # Creamos los procesos y las colas de memoria compartida para rescatar los resultados de los procesos.
        queques = []
        procesos = []
        for i in range(n_procesos):
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=self.dividir_multiprocess_interno,
                                        args=(sets_ind[i], evaluador, bd, q, logger_queue))
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

        # shutdown the queue correctly and the logger process
        logger_queue.put(None)
        logger_p.join()

        # Actualizamos la población
        poblacion.set_population(new_population)
        # Actualizamos los datos de la poblacion para el reporte
        evaluador.actualizar_reporte(poblacion)
        pass

    def dividir_interno(self, individuo_obj: Individuo, bd: BD) -> Individuo:
        """
        Realiza una división sobre una copia del individuo y la retorna.
        :param individuo_obj: Individuo a dividir.
        :return: Individuo.
        """
        idx_div = individuo_obj.get_indice_divisibilidad()
        # Debemos encontrar la línea y nodo de mayor índice de divisibilidad en el individuo
        linea_id = max(idx_div, key= lambda x: max(idx_div[x])) # Linea
        idx = max(idx_div[linea_id]) # valor del indice max
        i = idx_div[linea_id].index(idx) # posicion en la linea del nodo

        # Hay que dividir en i-ésimo nodo de la línea_id

        # Preparar paradas nuevas
        paradas_old = bd.getByIndex(linea_id).get_paradas_i()
        paradas_old = self.sequences_to_list(paradas_old)
        parada_nueva1 = paradas_old[:i+1]
        parada_nueva2 = paradas_old[i:]

        # Buscar el índice de las nuevas líneas
        id_nueva1 = bd.buscar_linea_lista(parada_nueva1)
        id_nueva2 = bd.buscar_linea_lista(parada_nueva2)

        # Modificar la lista de lineas
        id_lineas_nueva = individuo_obj.get_id_lineas().copy()
        id_lineas_nueva.remove(linea_id)
        id_lineas_nueva.append(id_nueva1)
        id_lineas_nueva.append(id_nueva2)

        # Modificar frecuencias opt
        # (ya no serán opt pero ayudan al optimizador a converger más rapido)
        freq_nueva = individuo_obj.get_freq().copy()
        freq_linea_old = freq_nueva.pop(linea_id)
        freq_nueva[id_nueva1] = freq_linea_old
        freq_nueva[id_nueva2] = freq_linea_old

        # Crear individuo nuevo
        individuo_obj_new = Individuo(id_lineas_nueva)
        individuo_obj_new.set_freq(freq_nueva)

        # Retorna
        return individuo_obj_new


class Divisor_sin_division(Divisor):
    """
    No hace nada.
    """
    pass