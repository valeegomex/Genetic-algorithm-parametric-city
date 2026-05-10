import logging

from sidermit.publictransportsystem import TransportMode, passenger

from AlgoritmoGenetico.BaseDatos.BD import BD
from AlgoritmoGenetico.Operadores.evaluador import Evaluador
from AlgoritmoGenetico.Poblacion.poblacion import Poblacion

# Crear bd y poblacion aleatoria

n_zonas = 7
bd = BD(n_zonas)

size_poblacion = 200
densidad_max_edl = 20
poblacion = Poblacion(size=size_poblacion, max_densitiy=densidad_max_edl)
poblacion.build_random(bd)

# Pasajeros y modo de transporte
pasajero=  passenger.Passenger(va=4, pv=2.74, pw=5.48, pa=0, pt=16, spv=2.74, spw=5.48, spa=0, spt=16)
tmode=  TransportMode(name='bus', bya=0, co=8.61 , c1=0.15, c2=0, v=20, t=2.5, fmax=150, kmax=160, theta=0.5, tat=0, d=1, fini=15)

# Parámetros del evaluador
L, g, P = 10, 0.85, 1
Y, a, alpha, beta = int(2565622/100), 0.78, 0.25, 0.22
evaluador = Evaluador(passenger_obj=pasajero, custom_tmode=tmode, L=L, g=g, P=P, Y=Y, a=a, alpha=alpha, beta=beta,
                      n_zonas=n_zonas)

# Desactivar logging de sidermit
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
poblacion.save_edl_population('SantiagoPoblacionInicial3')


