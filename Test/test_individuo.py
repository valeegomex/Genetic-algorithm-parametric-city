from unittest import TestCase

from AlgoritmoGenetico.BaseDatos.BD import BD
from AlgoritmoGenetico.Poblacion.individuo import Individuo
import numpy as np


class TestIndividuo(TestCase):
    def test_delete_bad_lines(self):
        n = 3
        bd = BD(n)

        # Create individual
        id_lineas = np.random.choice(bd.get_size(), size=5, replace=False).tolist()
        individuo1 = Individuo(id_lineas)

        # Set frecuences
        f = {}
        for id in individuo1.get_id_lineas():
            f[id] = 10
        f[individuo1.get_id_lineas()[0]] = 0
        individuo1.set_freq(f)

        # Delete bad lines
        individuo1.delete_bad_lines()

        self.assertEqual(len(individuo1.get_id_lineas()), 4)
