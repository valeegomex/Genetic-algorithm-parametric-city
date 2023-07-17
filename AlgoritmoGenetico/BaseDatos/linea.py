class Linea:

    def __init__(self, id: int, paradas_i: str, paradas_r: str, secuencia_i: str, secuencia_r: str):
        """
        Linea.
        :param id: Identificador en la BD.
        :param paradas_i: Nodos con parada en la dirección ida.
        :param paradas_r: Nodos con parada en la dirección regreso.
        :param secuencia_i: Secuencia nodos en la dirección ida.
        :param secuencia_r: Secuencia nodos en la dirección regreso.
        """
        self.id = id
        self.paradas_i = paradas_i
        self.paradas_r = paradas_r
        self.secuencia_i = secuencia_i
        self.secuencia_r = secuencia_r
        self.paradas_i_lista = self.sequences_to_list(paradas_i)

    def get_paradas_i(self):
        return self.paradas_i

    def get_paradas_r(self):
        return self.paradas_r

    def get_secuencia_i(self):
        return self.secuencia_i

    def get_secuencia_r(self):
        return self.secuencia_r

    def get_id(self):
        return self.id

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

    def paradas_iguales_lista(self, paradas: list[int]) -> bool:
        """
        Compueba si las paradas ida del objeto son iguales a las entregadas.
        :param paradas: Lista de paradas a comparar.
        :return: bool.
        """
        return paradas == self.paradas_i_lista

    def paradas_iguales_str(self, paradas: str) -> bool:
            """
            Compueba si las paradas ida del objeto son iguales a las entregadas.
            :param paradas: str de paradas a comparar.
            :return: bool.
            """
            return paradas == self.get_paradas_i()
