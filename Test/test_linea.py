from unittest import TestCase

from AlgoritmoGenetico.BaseDatos.linea import Linea


class TestLinea(TestCase):
    def runTest(self):
        paradas_i = '1, 2, 4'
        paradas_r = '4, 2, 1'
        secuencia_i = '1, 2, 3, 4'
        secuencia_r = '4, 3, 2, 1'
        linea = Linea(0, paradas_i, paradas_r, secuencia_i, secuencia_r)
        self.assertEqual(linea.get_paradas_i(), paradas_i, 'Paradas incorrectas')


