# Create data base
from sidermit.publictransportsystem import TransportMode, passenger
import time

from AlgoritmoGenetico.BaseDatos.BD import BD
from AlgoritmoGenetico.Operadores.evaluador import Evaluador
from AlgoritmoGenetico.Poblacion.poblacion import Poblacion
import logging

if __name__ ==  '__main__':
    # Desactivar logging de sidermit
    logging.getLogger("sidermit").setLevel(logging.WARNING)

    t0 = time.process_time()

    n = 4
    bd = BD(n)

    # Build network
    tmode = TransportMode(name='bus', bya=0, co=8.61 , c1=0.15, c2=0, v=20, t=2.5, fmax=150, kmax=160, theta=0.5, tat=0, d=1, fini=12)
    passenger_obj = passenger.Passenger(va=4, pv=2.74, pw=5.48, pa=0, pt=16, spv=2.74, spw=5.48, spa=0, spt=16)

    # Create population
    poblacion = Poblacion(size=10, max_densitiy=8)
    poblacion.quitar_infactibles(bd)

    # Evaluate
    evaluador = Evaluador(passenger_obj=passenger_obj, custom_tmode=tmode, L=10, g=1.8, P=1, Y=3000, a=0.4, alpha=0.3, beta=0.4)
    evaluador.evaluar_poblacion_multiprocess(poblacion, bd, 2)
    # evaluador.evaluar_poblacion(poblacion, bd)

    tf = time.process_time()
    print(tf-t0)
    print(evaluador.get_mvrc_min())
    print(evaluador.get_mvrc_mean())