from unittest import TestCase
from AlgoritmoGenetico.BaseDatos.BD import BD
from AlgoritmoGenetico.BaseDatos.linea import Linea

class TestBD(TestCase):
    def runtest(self):
        n = 3
        bd = BD(n)
        self.assertIsInstance(BD.get_lineas()[0], Linea)

