import logging

from sidermit.publictransportsystem import TransportMode, passenger

from AlgoritmoGenetico.BaseDatos.BD import BD
from AlgoritmoGenetico.Operadores.divisor import Divisor_umbral, Divisor_formula, Divisor_intervalo
from AlgoritmoGenetico.Operadores.evaluador import Evaluador
from AlgoritmoGenetico.Operadores.iterador import Iterador
from AlgoritmoGenetico.Poblacion.poblacion import Poblacion
from AlgoritmoGenetico.algoritmo import Algoritmo_genetico

# Crear bd y poblacion aleatoria

n_zonas = 7
bd = BD(n_zonas)

size_poblacion = 200
densidad_max_edl = 40
poblacion = Poblacion(size=size_poblacion, max_densitiy=densidad_max_edl)
poblacion.build_random(bd)

# Pasajeros y modo de transporte
pasajero=  passenger.Passenger(va=4, pv=2.74, pw=5.48, pa=0, pt=16, spv=2.74, spw=5.48, spa=0, spt=16)
tmode=  TransportMode(name='bus', bya=0, co=8.61 , c1=0.15, c2=0, v=20, t=2.5, fmax=150, kmax=160, theta=0.5, tat=0, d=1, fini=15)

# Parámetros del evaluador
L, g, P = 11.65, 0.79, 1
G_inner = 1
G_outer = 5.35
Gi = [G_inner, G_outer, G_inner, G_outer, G_inner, G_inner, G_outer]
Hi = Gi
Y, a, alpha, beta = 4500000, 0.91, 0.0033, 0.287
theta_inner = 0.249*n_zonas
theta_outer = 0.0004*n_zonas
theta = [theta_inner, theta_outer, theta_inner, theta_outer, theta_inner, theta_inner, theta_outer]
evaluador = Evaluador(passenger_obj=pasajero, custom_tmode=tmode, L=L, g=g, P=P, Y=Y, a=a, alpha=alpha, beta=beta,
                      n_zonas=n_zonas, theta=theta, Gi=Gi, Hi=Hi)

# Desactivar logging de sidermit
# logging.getLogger("sidermit").setLevel(logging.WARNING)
logging.getLogger("sidermit").setLevel(logging.INFO)

# Crear mensajes propios
logger = logging.getLogger(__name__)
# handdle to write in file
formatter = logging.Formatter('%(asctime)s | %(message)s')
file_handler = logging.FileHandler('InfoCrearPoblacion.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# Reemplazar infactibles
evaluador.construir_individuos(poblacion, bd)
evaluador.quitar_infactibles(bd, poblacion, logger)

# Guardar en un archivo
poblacion.save_edl_population('LosAngelesProblacionInicial')

# n_zonas = 4
# bd = BD(n_zonas)
#
# size_poblacion = 20
# densidad_max_edl = 10
# poblacion = Poblacion(size=size_poblacion, max_densitiy=densidad_max_edl)
# # poblacion.build_random(bd)
#
# # Guardar en un archivo
# poblacion.build_from_file('checkpoint')
# # poblacion.save_edl_population('ejemplo')
# for ind in poblacion.get_population():
#     print(ind.get_id_lineas())


