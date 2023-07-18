from collections import defaultdict

import numpy as np
from sidermit.exceptions import TransportNetworkException, SIDERMITException
from sidermit.optimization.preoptimization import ExtendedGraph, Hyperpath, Assignment
from sidermit.publictransportsystem import TransportNetwork, TransportMode, Route, Passenger, passenger
from sidermit.city import Graph, Demand
from AlgoritmoGenetico.BaseDatos.BD import BD
from sidermit.optimization import Optimizer



class Individuo:
    def __init__(self, id_lineas: list):
        """
        :param id_lineas: list(int). Lista del índice de las lineas de la BD.
        """
        self.id_lineas = id_lineas
        self.freq = None
        self.MVRC = np.inf
        self.network_sidermit = None
        self.graph_sidermit = None
        self.optimizado = False
        self.Vij = None
        self.hyperpaths = None
        self.successors = None
        self.assignment = None
        self.indice_divisibilidad = None

    def set_Vij(self, new):
        self.Vij = new

    def set_hyperpaths(self, new):
        self.hyperpaths = new

    def set_successors(self, new):
        self.successors = new

    def set_assignment(self, new):
        self.assignment = new

    def get_optimizado(self):
        return self.optimizado

    def set_optimizado(self, new: bool):
        self.optimizado = new
        pass

    def reset_multiplocess(self):
        """
        Reset all the values except id_linea, freq, optimizado, indice_divisibilidad and MVRC..
        :return:
        """
        # self.network_sidermit = None
        # self.graph_sidermit = None
        self.Vij = None
        self.hyperpaths = None
        self.successors = None
        self.assignment = None

    def reset(self):
        """
        Reset all the values except id_linea and freq.
        :return:
        """
        self.MVRC = np.inf
        self.network_sidermit = None
        self.graph_sidermit = None
        self.optimizado = False
        self.Vij = None
        self.hyperpaths = None
        self.successors = None
        self.assignment = None
        self.indice_divisibilidad = None
        pass

    def get_id_lineas(self):
        return self.id_lineas

    def set_indice_divisibilidad(self, new_indice):
        self.indice_divisibilidad = new_indice
        pass

    def get_indice_divisibilidad(self) -> dict[list]:
        """
        :return: dict[linea_id] = list[float]
        """
        return self.indice_divisibilidad

    def build_network(self, n: int, L: float, g: float, P: float, custom_tmode: TransportMode, bd: BD):
        """
        Construye la red en sidermit y guarda los atributos "network_sidermit" y "graph_sidermir".
        :param n: int. Cantidad de zonas de la ciudad.
        :param L: float. Para construir el grafo, average distance in km of subcenters to (0,0) of the Cartesian plane.
        :param g: float. Para construir el grafo, (>=0) to represent average distance of peripheries (= l + g * l).
        :param P: float. Para construir el grafo, node width in km, takes part in the demand
        assignment considering lateral access time.
        :param custom_tmode: TransportMode. Objeto del transporte bus usado.
        :param bd: BD. Base de datos líneas.
        :return:
        """
        # build city graph
        graph_obj = Graph.build_from_parameters(n, L, g, P)
        # build a network without routes
        network_obj = TransportNetwork(graph_obj)

        # Eliminar id lineas duplicados
        id_lineas_new = list(set(self.id_lineas))
        self.id_lineas = id_lineas_new

        # Agregar las rutas
        for id in self.get_id_lineas():
            # string with node_id sequences
            nodes_sequence_i = bd.getByIndex(id).get_secuencia_i()
            nodes_sequence_r = bd.getByIndex(id).get_secuencia_r()
            # string with node_id sequences (STOPS)
            stops_sequence_i = bd.getByIndex(id).get_paradas_i()
            stops_sequence_r = bd.getByIndex(id).get_paradas_r()
            # custom route
            custom_route = Route(id, custom_tmode, nodes_sequence_i, nodes_sequence_r, stops_sequence_i,
                                 stops_sequence_r)

            # add route and its respective mode of transport to the network
            network_obj.add_route(custom_route)

        self.network_sidermit = network_obj
        self.graph_sidermit = graph_obj
        pass

    def update_network(self, bd: BD):
        """
        Update the network with new lines_id and frequency in self.
        :return:
        """

        custom_tmode = self.network_sidermit.get_modes()[0]
        # build a network without routes
        network_obj = TransportNetwork(self.graph_sidermit)

        # Eliminar id lineas duplicados
        id_lineas_new = list(set(self.id_lineas))
        self.id_lineas = id_lineas_new

        # Agregar las rutas
        for id in self.get_id_lineas():
            # string with node_id sequences
            nodes_sequence_i = bd.getByIndex(id).get_secuencia_i()
            nodes_sequence_r = bd.getByIndex(id).get_secuencia_r()
            # string with node_id sequences (STOPS)
            stops_sequence_i = bd.getByIndex(id).get_paradas_i()
            stops_sequence_r = bd.getByIndex(id).get_paradas_r()
            # custom route
            custom_route = Route(id, custom_tmode, nodes_sequence_i, nodes_sequence_r, stops_sequence_i,
                                 stops_sequence_r)

            # add route and its respective mode of transport to the network
            network_obj.add_route(custom_route)

        self.network_sidermit = network_obj
        pass

    def get_MVRC(self):
        return self.MVRC

    def get_network(self):
        return self.network_sidermit

    def set_MVRC(self, mvrc: float):
        self.MVRC = mvrc
        pass

    def set_freq(self, freq_new: dict):
        self.freq = freq_new
        pass

    def get_freq(self):
        return self.freq

    def set_id_lineas(self, lineas_new: list[int]):
        self.id_lineas = lineas_new
        pass

    def validate(self, OD_matrix: defaultdict[lambda: defaultdict[float]]) -> bool:

        """
        Revisa si la EDL conecta a todos los nodos son demanda en el sentido correcto. El individuo debe estar contruido.
        :return: True si la EDL es convexa, False de otra forma.
        """
        passenger_obj = Passenger.get_default_passenger()  # Para lo que necesitamos, el pasajero no es relevante.
        extended_graph = ExtendedGraph(self.graph_sidermit, self.network_sidermit.get_routes(), passenger_obj.pt , self.freq)
        hyperpath = Hyperpath(extended_graph, passenger_obj)
        try:
            status = hyperpath.network_validator(OD_matrix)
        except TransportNetworkException:
            status = False

        return status


    def optimize(self, demand_obj: Demand, passenger_obj: Passenger, bd:BD):
        """
        Optimiza el individuo y actualiza la frecuencia y VMRC
        :param demand_obj: Demand.
        :param passenger_obj: Passenger.
        :return:
        """
        if not self.optimizado:
            try:
                opt_obj = Optimizer.network_optimization(self.graph_sidermit, demand_obj, passenger_obj, self.network_sidermit,
                                                         max_number_of_iteration=5, f=self.freq, tolerance=0.1)
                # Guardar los resultados
                res = opt_obj.better_res
                fopt, success, status, message, constr_violation, vrc = res
                dict_f = opt_obj.fopt_to_f(fopt)
                self.set_MVRC(vrc)
                self.optimizado = True

                # Quitar las lineas que son cero
                id_lineas_nuevo = []
                freq_nuevo = {}
                for id in self.id_lineas:
                    if dict_f[id] > 0.01:
                        id_lineas_nuevo.append(id)
                        freq_nuevo[id] = dict_f[id]
                self.set_freq(freq_nuevo)
                self.set_id_lineas(id_lineas_nuevo)

                # Con la frecuencia optima, obtenemos los nuevos objetos
                self.update_network(bd)
                opt_obj = Optimizer(self.graph_sidermit, demand_obj, passenger_obj, self.network_sidermit, self.freq)
                self.hyperpaths = opt_obj.hyperpaths
                self.successors = opt_obj.successors
                self.Vij = opt_obj.Vij
                self.assignment = opt_obj.assignment
            except SIDERMITException:
                self.optimizado = False
        pass

    def optimize_multiprocess(self, demand_obj: Demand, passenger_obj: Passenger):
        """
        Optimiza el individuo y actualiza la frecuencia y VMRC. No actualiza network, hyperpaths, successors, Vij
        ni assignment.
        :param demand_obj: Demand.
        :param passenger_obj: Passenger.
        :return:
        """
        if not self.optimizado:
            try:
                opt_obj = Optimizer.network_optimization(self.graph_sidermit, demand_obj, passenger_obj, self.network_sidermit,
                                                         max_number_of_iteration=5, f=self.freq, tolerance=0.1)
                # Guardar los resultados
                res = opt_obj.better_res
                fopt, success, status, message, constr_violation, vrc = res
                dict_f = opt_obj.fopt_to_f(fopt)
                self.set_MVRC(vrc)
                self.optimizado = True

                # Quitar las lineas que son cero
                id_lineas_nuevo = []
                freq_nuevo = {}
                for id in self.id_lineas:
                    if dict_f[id] > 0.01:
                        id_lineas_nuevo.append(id)
                        freq_nuevo[id] = dict_f[id]
                self.set_freq(freq_nuevo)
                self.set_id_lineas(id_lineas_nuevo)
                return (f'Optimización exitosa, MVRC: {round(vrc,3)}')
            except SIDERMITException:
                self.optimizado = False
                return (f'Optimización fallida')

        pass

    def delete_bad_lines(self):
        """
        Remueve del individuo las líneas con frecuencia cero.
        :return:
        """
        for id in self.id_lineas:
            if self.freq[id] < 0.01:
                freq_new = self.get_freq().copy()
                freq_new.pop(id)
                self.set_freq(freq_new)
                self.id_lineas.remove(id)
        pass

    def obtain_z_v_loaded(self):
        """
        Obtiene las matrices de subidas, bajadas y carga por arista.
        :return: subidas z = dic[route_id][direction][stop: StopNode] = pax [pax/veh],
         bajadas v = dic[route_id][direction][stop: StopNode] = pax [pax/veh],
         carga por arista dic[route_id][direction][stop: StopNode] = pax [pax/veh].
        """
        # if (self.hyperpaths is None):
        #     g = self.optimizado
        #     h = 1
        z, v, loaded_section_route = Assignment.get_alighting_and_boarding(Vij=self.Vij, hyperpaths=self.hyperpaths,
                                                                           successors=self.successors,
                                                                           assignment=self.assignment,
                                                                           f=self.freq)
        return z, v, loaded_section_route

