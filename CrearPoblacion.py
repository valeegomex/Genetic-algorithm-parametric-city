
from sidermit.publictransportsystem import TransportMode, passenger

from AlgoritmoGenetico.BaseDatos.BD import BD
from AlgoritmoGenetico.Operadores.divisor import Divisor_umbral, Divisor_formula, Divisor_intervalo
from AlgoritmoGenetico.Operadores.evaluador import Evaluador
from AlgoritmoGenetico.Operadores.iterador import Iterador
from AlgoritmoGenetico.Poblacion.poblacion import Poblacion
from AlgoritmoGenetico.algoritmo import Algoritmo_genetico

# Crear bd y poblacion aleatoria

# n_zonas = 6
# bd = BD(n_zonas)
#
# size_poblacion = 200
# densidad_max_edl = 12
# poblacion = Poblacion(size=size_poblacion, max_densitiy=densidad_max_edl)
# poblacion.build_random(bd)
#
# # Guardar en un archivo
# poblacion.save_edl_population('zonas6size200')

n_zonas = 4
bd = BD(n_zonas)

size_poblacion = 20
densidad_max_edl = 10
poblacion = Poblacion(size=size_poblacion, max_densitiy=densidad_max_edl)
# poblacion.build_random(bd)

# Guardar en un archivo
poblacion.build_from_file('ejemplo')
# poblacion.save_edl_population('ejemplo')
for ind in poblacion.get_population():
    print(ind.get_id_lineas())


