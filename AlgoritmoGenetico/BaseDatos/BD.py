import math
from collections import defaultdict

from AlgoritmoGenetico.BaseDatos.linea import Linea
import numpy as np
import pandas as pd
import itertools as it



class BD:
    def __init__(self, n: int, build=True):
        """
        Base de líneas.
        :param n: Cantidad de zonas de la ciudad.
        :param build: Si es verdadero, se construye la bd, si es faso, se crea un objeto con lista_lineas y
        largos_index vacío.
        """
        self.n = n
        self.lista_lineas = []
        self.size = 0
        self.largos_index = None

        if build:
            # Creamos la BD referencial como tabla pandas
            raw = self.crearBDraw(n)
            # Para cada fila, crear la linea correspondiente.
            for index, row in raw.iterrows():
                # Stop sequence
                nodes_i_stop = row['stops']
                nodes_r_stop = nodes_i_stop[::-1]
                # Route sequence
                nodes_i_seq = row['sequence']
                nodes_r_seq = nodes_i_seq[::-1]
                # To string
                nodes_i_stop = self.lineaString(nodes_i_stop)
                nodes_r_stop = self.lineaString(nodes_r_stop)
                nodes_i_seq = self.lineaString(nodes_i_seq)
                nodes_r_seq = self.lineaString(nodes_r_seq)
                # Crear linea
                lin = Linea(index, nodes_i_stop, nodes_r_stop, nodes_i_seq, nodes_r_seq)
                self.lista_lineas.append(lin)

            self.size = len(self.lista_lineas)

        else:
            pass

    def get_lineas(self):
        return self.lista_lineas

    def get_size(self):
        return self.size

    def getByIndex(self, i) -> Linea:
        return self.get_lineas()[i]

    def get_n(self):
        return self.n

    def build_from_csv(self, path_df: str, path_np: str):
        """
        Construye la base desde los archivos dados.
        :param path_df: Directorio del csv de las lineas raw.
        :param path_np: directorio de la lista largo index.
        :return:
        """
        # Traemos la BD referencial como tabla pandas
        raw = pd.read_csv(path_df, index_col=0)
        # Para cada fila, crear la linea correspondiente.
        for index, row in raw.iterrows():
            # Stop sequence
            nodes_i_stop = row['stops']
            nodes_r_stop = nodes_i_stop[::-1]
            # Route sequence
            nodes_i_seq = row['sequence']
            nodes_r_seq = nodes_i_seq[::-1]
            # To string
            nodes_i_stop = self.lineaString(nodes_i_stop)
            nodes_r_stop = self.lineaString(nodes_r_stop)
            nodes_i_seq = self.lineaString(nodes_i_seq)
            nodes_r_seq = self.lineaString(nodes_r_seq)
            # Crear linea
            lin = Linea(index, nodes_i_stop, nodes_r_stop, nodes_i_seq, nodes_r_seq)
            self.lista_lineas.append(lin)

        self.size = len(self.lista_lineas)

        # Traer lineas_largo
        largos_index = []
        with open(path_np) as f:
            largos_index = f.readlines()
        largos_index = [int(l) for l in largos_index]
        self.largos_index = largos_index


        pass


    def get_largos_index(self) -> list[int]:
        """
        Retorna una lista que contiene los indices de posición en la BD en que hay líneas de determinado largo.
        Por ej. las lineas de largo 4 están entre self.get_largos_index(4) y self.get_largos_index(5).
        Para el largo de la línea, solo se consideran los nodos con detención.
        :return: list[int].
        """
        return self.largos_index

    def generarLineasAmbosImpar(self, n, l):
        """
        Genera las líneas de largo l y n zonas, sin paradas intermedias en la periferia ni líneas espejo, que empiezan y terminan en impar.
        :param n: Int. Cantidad de zonas de la ciudad.
        :param l: Int. Largo de las líneas.
        :return: DataFrame.
        """
        # Lista de paradas separadas por paridad
        pares = np.arange(0, 2 * n + 1, 2)
        impares = np.arange(1, 2 * n + 1, 2).tolist()

        # Combinaciones de largo l-2 de paradas pares
        pares_lm2 = pd.DataFrame(list(it.permutations(pares, l - 2)))
        pares_lm2['key'] = 0

        # Iteratemos sobre los impares
        nuevo_lista = []
        while len(impares) > 1:
            i = impares.pop()
            # Copia del dataframe para trabajar
            nuevo = pares_lm2.copy()
            # Fijar la última parada
            nuevo[l - 1] = i
            # Iterar sobre la primera parada
            frame_impares = pd.DataFrame(impares)
            frame_impares['key'] = 0
            nuevo = frame_impares.merge(nuevo, on='key', how='outer')
            # Guardar
            nuevo_lista.append(nuevo)

        # Ajustamos las columnas
        resultado = []
        for df in nuevo_lista:
            df = df.drop('key', axis=1)
            nombres = df.columns
            dict = {nombres[i]: i for i in range(len(nombres))}
            df = df.rename(dict, axis=1)
            resultado.append(df)

        # Unir todos
        result = pd.concat(resultado)
        result = result.reset_index(drop=True)

        result = result[~result.isna().any(axis=1)]
        result = result.reset_index(drop=True)

        return result

    def generarLineasPares(self, n, l):
        """
        Genera las líneas de largo l y n zonas, sin paradas intermedias en la periferia ni líneas espejo, sólo con paradas pares.
        :param n: Int. Cantidad de zonas de la ciudad.
        :param l: Int. Largo de las líneas.
        :return: DataFrame.
        """

        # Lista de los pares
        pares = np.arange(0, 2 * n + 1, 2)

        # Empezamos con las líneas que no tienen impares
        result = list(it.permutations(pares, l))
        result = pd.DataFrame(result)
        if result.shape[0] > 0:
            result = result[result[0] < result[l - 1]]

        return result

    def generarLineasUnImpar(self, n, l):
        """
        Genera las líneas de largo l y n zonas, sin paradas intermedias en la periferia ni líneas espejo, que empiezan impar y terminan par.
        :param n: Int. Cantidad de zonas de la ciudad.
        :param l: Int. Largo de las líneas.
        :return: DataFrame.
        """
        # Cramos las que empiezan impar y terminan par
        pares = np.arange(0, 2 * n + 1, 2)
        impares = np.arange(1, 2 * n + 1, 2)

        # Hay que hacer un merge con los pares totales de largo l-1
        frame_impares = pd.DataFrame(impares)
        frame_impares['key'] = 0
        # Merge
        soloParesAnterior = list(it.permutations(pares, l - 1))
        soloParesAnterior = pd.DataFrame(soloParesAnterior)
        soloParesAnterior['key'] = 0
        if soloParesAnterior.shape[0] == 0:
            return pd.DataFrame()
        result = frame_impares.merge(soloParesAnterior, on='key', how='outer')

        # Ajustamos las columnas
        result = result.drop('key', axis=1)
        nombres = result.columns
        dict = {nombres[i]: i for i in range(len(nombres))}
        result = result.rename(dict, axis=1)

        return result

    def generarLineas(self, n, l):
        """
        Genera las líneas de largo l y n zonas, sin paradas intermedias en la periferia ni líneas espejo.
        :param n: Int. Cantidad de zonas de la ciudad.
        :param l: Int. Largo de las líneas.
        :return: DataFrame.
        """
        ambosImpar = self.generarLineasAmbosImpar(n, l)
        ambosPar = self.generarLineasPares(n, l)
        empiezaImpar = self.generarLineasUnImpar(n, l)

        # Unir todos
        result = pd.concat([ambosImpar, ambosPar, empiezaImpar])
        result = result.reset_index(drop=True)

        return result

    def secuenciaSubcentros(self, p, a, b, n):
        """
        Calcula la secuencia de nodos para unir dos subcentros no adjacentes a y b.
        :param p: Int. Cantidad máxima de zonas tal que ir por el anillo circular es más eficiente que pasar por el CBD.
        :param a: Int. Subcentro origen.
        :param b: Int. Subcentro destino.
        :param n: Int. Cantidad de zonas de la ciudad.
        :return: array(Int).
        """
        # Calcular zonas de distancia, paso y sentido de distancia mínima
        if abs(b - a) <= n:
            z = abs(b - a) / 2
            step = 2 * (b - a) / abs(b - a)
            s = 1
        else:
            z = n - abs(b - a) / 2
            step = (-2) * (b - a) / abs(b - a)
            s = 0
        # Completar secuencia
        if z > p:  # Es mas conveniente pasar por el CBD
            return np.array([a, 0, b])
        else:  # Es mas conveniente rodear el CBD
            if s == 1:  # Seguir sentido natural
                return np.arange(a, b + step, step)
            else:  # Pasar por 2n
                if step == -2:  # Dependiendo de la direccion, hay que considerar cero o 2n
                    aux1 = np.arange(a, 0 + step, step)
                    aux1[-1] = 2 * n
                    aux2 = np.arange(2 * n, b + step, step)
                else:
                    aux1 = np.arange(a, 2 * n + step, step)
                    aux2 = np.arange(0, b + step, step)
                return np.concatenate([aux1, aux2[1:]])

    def secuenciaConsecutivos(self, li, li1, n, p=None):
        """
        Retorna el camino más corto entre li y li1 en una ciudad de n zonas. El camino no incluye a li.
        :param li: Int. Id nodo origen.
        :param li1: Int. Id nodo destino.
        :param n: Int. Cantidad de zonas de la ciudad.
        :param p: Int. Max zonas de distancia entre SC para no pasar por el CBD
        :return: List(int).
        """
        nueva = []
        if p is None:
            p = math.floor(1 / math.sin(math.pi / n))  # Max zonas de distancia entre SC para no pasar por el CBD

        # Nodo actual periferia
        if li % 2 == 1:
            if li1 == 0:  # sgte parada es el CBD
                nueva.append(li + 1)
                nueva.append(li1)
            elif li + 1 == li1:  # sgte parada es el SC propio
                nueva.append(li1)
            elif li1 % 2 == 0:  # sgte parada es un SC externo
                aux = self.secuenciaSubcentros(p, li + 1, li1, n).tolist()
                nueva = nueva + aux
            else:  # sgte parada es una periferia
                aux = self.secuenciaSubcentros(p, li + 1, li1 + 1, n).tolist()
                nueva = nueva + aux
                nueva.append(li1)

        # Nodo actual SC
        elif li != 0:
            if li1 == 0:  # sgte parada es el CBD
                nueva.append(li1)
            elif li1 % 2 == 0:  # sgte parada es otro SC
                aux = self.secuenciaSubcentros(p, li, li1, n).tolist()
                nueva = nueva + aux[1:]
            elif li1 + 1 == li:  # sgte parada es la periferia propia
                nueva.append(li1)
            else:  # sgte parada es una periferia externa
                aux = self.secuenciaSubcentros(p, li, li1 + 1, n).tolist()
                nueva = nueva + aux[1:]
                nueva.append(li1)

        # Nodo actual CBD
        else:
            if li1 % 2 == 0:  # sgte parada es un SC
                nueva.append(li1)
            else:  # es una periferia
                nueva.append(li1 + 1)
                nueva.append(li1)

        nueva = [int(i) for i in nueva]
        return nueva

    def crearDiccionario(self, n):
        """
        Crea un diccionario que devuelve la secuencia para unir dos nodos cualquiera, sin retornar el nodo origen.
        :param n: Int. Cantidad de zonas de la ciudad.
        :return: dict[id_origen][id_destino] = list(int)
        """
        p = math.floor(1 / math.sin(math.pi / n))  # Max zonas de distancia entre SC para no pasar por el CBD
        seq = defaultdict(lambda: defaultdict(list))
        for i in range(2 * n + 1):
            for j in range(2 * n + 1):
                if i != j:
                    s = self.secuenciaConsecutivos(i, j, n, p)
                    seq[i][j] = s
        return seq

    def secuenciaLineaDic(self, seq_dict, linea, n):
        """
        Completa la secuencia de una línea usando los valores del diccionario.
        :param seq_dict: Dict[id_origen][id_destino] = list(int). Secuencia nodos entre origen y destino.
        :param linea: List. Lista de los nodos de parada.
        :param n: Int. Cantiad de zonas de la ciudad.
        :return:
        """
        nueva = [linea[0]]
        for i in range(len(linea) - 1):
            li = linea[i]  # nodo actual (ya en la lista nueva)
            li1 = linea[i + 1]  # nodo siguiente
            seq = seq_dict[li][li1]
            nueva = nueva + seq

        return nueva

    def lineaString(self, linea):
        """
        Convierte una línea tipo list, en string.
        :param linea: List.
        :return: String.
        """
        nodes = ' '.join([str(elem) + ',' for j, elem in enumerate(linea)])
        nodes = nodes[:-1]

        return nodes

    def crearParadasSecuencias(self, lineas, n):
        """
        Crea las listas de paradas y secuencia, para cada fila en lineas.
        :param lineas: DataFrame. Cada fila es una lista de paradas.
        :param n: Int. Cantidad de zonas de la ciudad.
        :return: DataFrame.
        """

        def contains_duplicates(X):
            return len(np.unique(X)) != len(X)

        lineasNew = pd.DataFrame()

        lista_paradas = []
        lista_secuencia = []
        lista_duplicates = []

        seq_dict = self.crearDiccionario(n)

        for i in lineas.index:
            paradas = lineas.loc[i].values
            paradas = [int(j) for j in paradas]
            secuencia = self.secuenciaLineaDic(seq_dict, paradas, n)
            duplicates = contains_duplicates(secuencia)

            lista_paradas.append(paradas)
            lista_secuencia.append(secuencia)
            lista_duplicates.append(duplicates)

        lineasNew['stops'] = lista_paradas
        lineasNew['sequence'] = lista_secuencia
        lineasNew['duplicates'] = lista_duplicates

        lineasNew = lineasNew[~lineasNew['duplicates']].reset_index(drop=True)
        lineasNew = lineasNew.drop('duplicates', axis=1)

        return lineasNew

    def crearBDraw(self, n) -> pd.DataFrame:
        """
        Crea la base de datos de líneas, esto es, para cada línea sus paradas y secuencia.
        :param n: Int. Cantidad de zonas de la ciudad.
        :return: DataFrame.
        """
        bd = []
        # Lista para guardar las posiciones según largo de línea
        largos_index = [0]*(n+5)
        # Generar iterativamente las lineas de largo l
        for l in range(2, n + 4):
            lineas = self.generarLineas(n, l)
            lineasNew = self.crearParadasSecuencias(lineas, n)
            largos_index[l+1] = lineasNew.shape[0]
            bd.append(lineasNew)

        result = pd.concat(bd)
        result = result.reset_index(drop=True)
        largos_index= np.cumsum(largos_index)
        self.largos_index = largos_index
        return result

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

    def buscar_linea(self, paradas: str) -> int:
        """
        Retorna el índice en la BD de la línea con las paradas ida o regreso entregadas.
        :param paradas: str con las paradas ida.
        :return: int.
        """
        # Reverso
        paradas_reverso = paradas[::-1].replace(',','.').replace(' ', ',').replace('.', ' ')
        # Calcular la cantidad de nodos
        l = len(self.sequences_to_list(paradas))
        # Buscar los indices en la bd para lineas de ese largo
        idx_min = self.get_largos_index()[l]
        idx_max = self.get_largos_index()[l+1]
        # Recorrer la bd comparando
        for idx in range(idx_min, idx_max):
            actual = self.getByIndex(idx)
            if actual.paradas_iguales_str(paradas):
                # indice encontrado
                return idx
            if actual.paradas_iguales_str(paradas_reverso):
                # indice encontrado
                return idx
        # Elemento no se encontró en la bd
        return -1

    def buscar_linea_lista(self, paradas: list[int]) -> int:
        """
        Retorna el índice en la BD de la línea con las paradas ida o regreso entregadas.
        :param paradas: Lista con las paradas ida.
        :return: int.
        """
        # Reverso
        paradas_reverso = paradas.copy()
        paradas_reverso.reverse()
        # Calcular la cantidad de nodos
        l = len(paradas)
        # Buscar los indices en la bd para lineas de ese largo
        idx_min = self.get_largos_index()[l]
        idx_max = self.get_largos_index()[l+1]
        # Recorrer la bd comparando
        for idx in range(idx_min, idx_max):
            actual = self.getByIndex(idx)
            if actual.paradas_iguales_lista(paradas):
                # indice encontrado
                return idx
            if actual.paradas_iguales_lista(paradas_reverso):
                # indice encontrado
                return idx
        # Elemento no se encontró en la bd
        return -1

    @staticmethod
    def guardar_bd_raw(n:int, path_df:str, path_np:str):
        """
        Guarda el dataframe de las edl en raw y la lista de largos_index.
        :param n: Cantidad de zonas ciudad.
        :param path_df: Directorio del dataframe.
        :param path_np: Directorio de largos_index.
        :return:
        """
        bd = BD(n=n)
        # Guardar pandas
        df = bd.crearBDraw(n)
        df.to_csv(path_df)
        # Guardar largos_index
        save = bd.get_largos_index()
        items = [str(s) for s in save]
        file = open(path_np, 'w')
        for item in items:
            file.write(item + "\n")
        file.close()



