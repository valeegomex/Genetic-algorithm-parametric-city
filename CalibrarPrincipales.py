if __name__ ==  '__main__':
    import logging
    from sidermit.city import Graph
    from sidermit.publictransportsystem import TransportMode, passenger
    from AlgoritmoGenetico.Operadores.calibrador import Calibrador

    logger = logging.getLogger(__name__)
    # handdle to write in file
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    file_handler = logging.FileHandler('spamCalibrador.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    tmode = TransportMode(name='bus', bya=0, co=8.61 , c1=0.15, c2=0, v=20, t=2.5, fmax=150, kmax=160, theta=0.5, tat=0, d=1, fini=12)
    passenger_obj = passenger.Passenger(va=4, pv=2.74, pw=5.48, pa=0, pt=16, spv=2.74, spw=5.48, spa=0, spt=16)

    n, L, g,  P = 6, 10, 1.8, 1
    graph_obj = Graph.build_from_parameters(n=n, l=L, g=g, p=P)
    n_pross = 6

    Y = 3750
    calibrador = Calibrador(n=n, graph_obj=graph_obj, tmode=tmode, passenger_obj=passenger_obj, Y=Y, L=L,
                            build=True, n_procesos=n_pross)
    frontera = calibrador.obtener_frontera()
    terminos_optimos = calibrador.calibrar_frontera(frontera)
    logger.info(f'Los terminos optimos para {n} zonas e Y={Y} [pax/hora], son d1={terminos_optimos[0]},'
                f' d2={terminos_optimos[1]} y umbral={terminos_optimos[2]}')

   
