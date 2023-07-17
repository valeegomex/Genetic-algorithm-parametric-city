import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sidermit.city import Graph, Demand
from sidermit.optimization import Optimizer
from sidermit.optimization.preoptimization import Assignment
from sidermit.publictransportsystem import TransportNetwork
from sidermit.publictransportsystem import Passenger, TransportMode

from AlgoritmoGenetico.Operadores.divisor import Divisor


class Calibrador():
    def __init__(self, n: int, graph_obj: Graph, tmode: TransportMode, passenger_obj: Passenger, Y: int, L: float, build=True):
        self.n = n
        self.graph = graph_obj
        self.tmode = tmode
        self.Y = Y
        self.L = L
        # Hacer las simulaciones
        if build:
            edl_completa = self.simular_edl_completa(passenger_obj)
            edl_dividida = self.simular_edl_dividida(passenger_obj)
            self.edl_completa = edl_completa
            self.edl_dividida = edl_dividida

    def set_edl_compelta(self, new: pd.DataFrame):
        self.edl_completa = new

    def set_edl_dividida(self, new: pd.DataFrame):
        self.edl_dividida = new

    def obtener_frontera(self) -> pd.DataFrame:
        """
        Calcula el dataframe frontera, a partir de los datos simulados.
        :return:
        """

        edl_completa = self.edl_completa
        edl_dividida = self.edl_dividida

        # Agregamos una columna para identificarlas después de la unión
        edl_completa['EDL_id'] = 0
        edl_dividida['EDL_id'] = 1
        # Unir ambas tablas
        cols = ['a', 'alpha', 'beta', 'VRC', 'EDL_id']
        data = pd.concat([edl_completa[cols], edl_dividida[cols]], axis=0)
        # Eliminar los puntos infactibles
        data = data[~data['VRC'].isna()]
        data = data.reset_index(drop=True)
        # Eliminamos los puntos donde alguna de las opt no terminó
        aux = data.groupby(['a', 'alpha', 'beta']).VRC.transform('count') > 1
        data = data[aux]
        # Encontrar mínimo para cada a, alpha y beta.
        data_min_index = data.groupby(['a', 'alpha', 'beta'])['VRC'].idxmin()
        data_min = data.loc[data_min_index]

        # Nuevas columnas para detectar cambios de la EDL optima arriba/abajo/derecha/izquierda
        data_min['shift_alpha_foward'] = data_min.groupby(['a', 'alpha']).EDL_id.shift(1)
        data_min['shift_alpha_backward'] = data_min.groupby(['a', 'alpha']).EDL_id.shift(-1)
        data_min['shift_beta_foward'] = data_min.groupby(['a', 'beta']).EDL_id.shift(1)
        data_min['shift_beta_backward'] = data_min.groupby(['a', 'beta']).EDL_id.shift(-1)

        # Guardar los casos en que la EDL optima no coincide con alguno de sus vecinos
        frontera = data_min.copy()
        for col in ['shift_alpha_foward', 'shift_alpha_backward', 'shift_beta_foward', 'shift_beta_backward']:
            frontera[col] = frontera[col].fillna(frontera['EDL_id'])
        frontera = frontera[(frontera['EDL_id'] != frontera['shift_alpha_foward']) | (
                    frontera['EDL_id'] != frontera['shift_alpha_backward'])
                            | (frontera['EDL_id'] != frontera['shift_beta_foward']) | (
                                        frontera['EDL_id'] != frontera['shift_beta_backward'])]
        frontera = frontera[['a', 'alpha', 'beta', 'EDL_id']]

        return frontera

    def simular_edl_completa(self, passenger_obj: Passenger):
        """
        Obtiene las frecuencias óptimas y MVRC en la EDL completa para a= [0.3, 0.5, 0.7] con una malla de paso 0.1
        en el espacio (alpha, beta). Retorna un DataFrame con solumnas ['a', 'alpha', 'beta', 'D3_bus_1', 'D3_bus_2',
        'D3_bus_3', 'CIR_I_bus', 'CIR_R_bus', 'VRC'].
        :param passenger_obj: Passenger.
        :return: DataFrame.
        """
        # Build network
        network_obj = TransportNetwork(self.graph)
        circular_route = network_obj.get_circular_routes(mode_obj=self.tmode)
        diametral_route = network_obj.get_diametral_routes(mode_obj=self.tmode, jump=int(self.n / 2), short=False,
                                                           express=False)

        # to add diametral and circular routes to transport network
        for route in diametral_route:
            network_obj.add_route(route)
        for route in circular_route:
            network_obj.add_route(route)

        # demand parameters
        Y = self.Y  # [trips/hr]
        a_list = [0.3, 0.5, 0.7]
        alpha_list = np.linspace(0, 1, 10, endpoint=False).round(3)
        beta_list = np.linspace(0, 1, 10, endpoint=False).round(3)

        # To save the results
        routes = network_obj.get_routes()
        routes_list = [route.id for route in routes]
        routes_list.extend(['a', 'alpha', 'beta', 'VRC', 'factor_carga', 'transbordos', 'largo_lineas'])
        data = pd.DataFrame(columns=routes_list)
        flast = None

        # Iterate
        for a in a_list:
            for alpha in alpha_list:
                for beta in beta_list:
                    if alpha + beta < 0.99:  # Factibility
                        demand_obj = Demand.build_from_parameters(self.graph, Y, a, alpha, beta)  # Define demand
                        # try:
                        opt_obj = Optimizer.network_optimization(self.graph, demand_obj, passenger_obj, network_obj,
                                                                 max_number_of_iteration=5, f=flast, tolerance=0.1)
                        # Save general information
                        res = opt_obj.better_res
                        fopt, success, status, message, constr_violation, vrc = res
                        flast = opt_obj.fopt_to_f(fopt)
                        f_last = flast.copy()
                        # Calcular los terminos del indice de divisibilidad
                        df, tb, ll = self.terminos_divisibilidad(opt_obj)
                        dict_aux = {'alpha': alpha, 'beta': beta, 'a': a, 'VRC': vrc, 'factor_carga': df,
                                    'transbordos': tb, 'largo_lineas':ll}
                        f_last |= dict_aux
                        # except:
                        #     f_last = {'alpha': alpha, 'beta': beta, 'a': a}
                        data.loc[len(data)] = f_last
                        print(f'a={a}, alpha={alpha}, beta={beta}', end='\r')
                    else:  # Infactiability
                        f = {'alpha': alpha, 'beta': beta, 'a': a}
                        data.loc[len(data)] = f


        return data

    def terminos_divisibilidad(self, opt_obj: Optimizer):
        """
        Calculamos los términos de divisibilidad en los subcentros según la línea diametral.
        :param opt_obj: Optimizer.
        :return: float, float, float. Factor_carga_diferecnia, ransbordos_diferencia y largo_linea_termino.
        """

        z, v, loaded_section_route = self.obtain_z_v_loaded(opt_obj)
        nodes_stops = Divisor.itinerary(opt_obj.network_obj)
        nodes_sequence = Divisor.itinerary_sequences(opt_obj.network_obj)
        # Ordenamos las subidas, bajadas y carga según orden en el itinerario de la ruta y dirección.
        z_new, v_new, loaded_section_route_new = Divisor.sort_z_v_loaded_section_route(z, v, loaded_section_route,
                                                                                       nodes_stops)
        # Revisar simetría
        simetria_dict = {}
        for route_id in z:
            # simetria = [abs(i - r) for i, r in
            #             zip(loaded_section_route_new[route_id]['I'], loaded_section_route_new[route_id]['R'])]
            # simetria = all([x < 10 for x in simetria])
            simetria_dict[route_id] = True

        # Obtener términos
        diferencia_factor = Divisor.direncia_factor_carga(loaded_section_route_new, simetria_dict)
        transbordos_obj = Divisor.transbordos(v_new, loaded_section_route_new, simetria_dict)
        largos_linea_obj = Divisor.largos_linea(self.graph, loaded_section_route_new, nodes_stops,
                                                nodes_sequence, self.L, simetria_dict)

        # En la linea diametral (no importa la dirección) los SC son los nodos 1 y 3, y como se considera simétrica,
        # nos quedamos sólo con 1.
        df = []
        tb = []
        ll = []
        for route_id in diferencia_factor:
            for dic, lis in zip([diferencia_factor, transbordos_obj, largos_linea_obj],[df, tb, ll]):
                lis.append(dic[route_id][1])
                # lis.append(dic[route_id][3])

        df = sum(df)/len(df)
        tb = sum(tb)/len(tb)
        ll = sum(ll)/len(ll)

        return df, tb, ll


    def obtain_z_v_loaded(self, opt_obj: Optimizer):
        """
        Obtiene las matrices de subidas, bajadas y carga por arista.
        :return: subidas z = dic[route_id][direction][stop: StopNode] = pax [pax/veh],
         bajadas v = dic[route_id][direction][stop: StopNode] = pax [pax/veh],
         carga por arista dic[route_id][direction][stop: StopNode] = pax [pax/veh].
        """
        res = opt_obj.better_res
        fopt, success, status, message, constr_violation, vrc = res
        dict_f = opt_obj.fopt_to_f(fopt)

        z, v, loaded_section_route = Assignment.get_alighting_and_boarding(Vij=opt_obj.Vij, hyperpaths=opt_obj.hyperpaths,
                                                                           successors=opt_obj.successors,
                                                                           assignment=opt_obj.assignment,
                                                                           f= dict_f)
        return z, v, loaded_section_route

    def simular_edl_dividida(self, passenger_obj: Passenger):
        """
        Obtiene las frecuencias óptimas y MVRC en la EDL dividida para a= [0.3, 0.5, 0.7] con una malla de paso 0.1
        en el espacio (alpha, beta). Retorna un DataFrame con solumnas ['a', 'alpha', 'beta', 'D3_bus_1', 'D3_bus_2',
        'D3_bus_3', 'CIR_I_bus', 'CIR_R_bus', 'VRC'].
        :param passenger_obj: Passenger.
        :return: DataFrame.
        """
        # build network
        network_obj = TransportNetwork(self.graph)

        feeder_route = network_obj.get_feeder_routes(mode_obj=self.tmode)
        circular_route = network_obj.get_circular_routes(mode_obj=self.tmode)
        diametral_route = network_obj.get_diametral_routes(mode_obj=self.tmode, jump=int(self.n/2), short=True, express=False)

        for route in circular_route:
            network_obj.add_route(route)
        for route in feeder_route:
            network_obj.add_route(route)
        for route in diametral_route:
            network_obj.add_route(route)

        # demand parameters
        Y = self.Y  # [trips/hr]
        a_list = [0.3, 0.5, 0.7]
        alpha_list = np.linspace(0, 1, 10, endpoint=False).round(3)
        beta_list = np.linspace(0, 1, 10, endpoint=False).round(3)

        # To save the results
        routes = network_obj.get_routes()
        routes_list = [route.id for route in routes]
        routes_list.extend(['a', 'alpha', 'beta', 'VRC'])
        data = pd.DataFrame(columns=routes_list)
        flast = None

        # Iterate
        for a in a_list:
            for alpha in alpha_list:
                for beta in beta_list:
                    if alpha + beta < 0.99:  # Factibility
                        demand_obj = Demand.build_from_parameters(self.graph, Y, a, alpha, beta)  # Define demand
                        try:
                            opt_obj = Optimizer.network_optimization(self.graph, demand_obj, passenger_obj, network_obj,
                                                                     max_number_of_iteration=5, f=flast, tolerance=0.1)
                            # Save frecuency
                            res = opt_obj.better_res
                            fopt, success, status, message, constr_violation, vrc = res
                            flast = opt_obj.fopt_to_f(fopt)
                            f_last = flast.copy()
                            f_last['alpha'] = alpha
                            f_last['beta'] = beta
                            f_last['a'] = a
                            f_last['VRC'] = vrc
                        except:
                            f_last = {'alpha': alpha, 'beta': beta, 'a': a}
                        data.loc[len(data)] = f_last
                        print(f'a={a}, alpha={alpha}, beta={beta}', end='\r')
                    else:  # Infactiability
                        f = {'alpha': alpha, 'beta': beta, 'a': a}
                        data.loc[len(data)] = f

        return data

    def error_prediccion(self, d1: float, d2: float, umbral: float, frontera: pd.DataFrame):
        """
        Entrega la predicción (divide o no) de la línea completa en la frontera.
        :return: DataFrame.
        """
        completa = self.edl_completa.drop('EDL_id', axis=1)
        pred = completa.merge(frontera[['a', 'alpha', 'beta', 'EDL_id']], on=['a', 'alpha', 'beta'], how='right')
        pred['indice'] = pred['factor_carga']*(1-pred['transbordos']*d1)*(1+pred['largo_lineas']*d2)
        pred['error'] = (pred['indice'] - umbral)*(1-2*pred['EDL_id'])
        pred.loc[pred['error'] > 0, 'error'] = np.cbrt(pred[pred['error'] > 0]['error'])

        # cuando error es positivo, la predicción es equivocada
        error = pred[pred['error'] > 0]['error'].sum()

        return error

    def calibrar_frontera(self, frontera: pd.DataFrame):
        """
        Funcion para encontrar d1, d2 y umbral óptimos.
        :param frontera:
        :return:
        """

        def minimizar_error_interno(x):
            d1, d2, umbral = x
            err = self.error_prediccion(d1, d2, umbral, frontera)
            return err

        # Minimizar el error
        bnds = ((0, 10), (0, 10), (0,10))
        res = minimize(minimizar_error_interno, (0.8, 0.2, 0.56), bounds=bnds, method='Nelder-Mead')

        if res.success:
            return res.x

        else:
            print('Minimización no terminó con éxito')
            return [0, 0, 0]