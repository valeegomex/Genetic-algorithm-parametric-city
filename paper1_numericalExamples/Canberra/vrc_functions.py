
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def transfers_calculate(demand: pd.DataFrame) -> list[int]:
    """
    For each stop, calculate the potential transfers adding the two directions.
    """
    transfers_list = []
    for i in demand.index:
        # The transfers are the passengers traveling from before i to after i.
        trans = demand.loc[:i-1, i+1:].values.sum()
        # Also, the transfers are the passengers traveling from after i to before i.
        trans += demand.loc[i+1:, :i-1].values.sum()
        transfers_list.append(trans)

    return transfers_list

def max_flow_calculate(demand: pd.DataFrame) -> list[tuple[int, int]]:
    """
    For each stop, calculate the maximum flow before and after, including both directions.
    """
    # Calculate flow both directions in each segment
    flow_left = []
    flow_right = []
    for i in range(len(demand.index)-1):
        # Passengers traveling from before i to after i.
        flow_l = demand.loc[:i, i+1:].values.sum()
        flow_left.append(flow_l)
        # Passengers traveling from after i to before i.
        flow_r = demand.loc[i+1:, :i].values.sum()
        flow_right.append(flow_r)

    # Maximum flow by arc
    flow_max = [max(flow_left[i], flow_right[i]) for i in range(len(flow_left))]
    flow_max = np.array(flow_max)

    # Maximum flow both side division
    flow_max_division = [0]*len(demand.index)
    for i in range(1, len(demand.index)-1):
        flow_max_division[i] = (flow_max[:i].max(), flow_max[i:].max())

    return flow_max_division

def length_term_calculate(travel_time: pd.DataFrame, flow_max_division: list[tuple[int, int]]) -> list[int]:
    """
    For each stop, calculate |l_i| + |l_i|/|l|
    """
    length = [0]*len(travel_time.index)
    l = travel_time.loc[0,:].values.sum()
    for i in range(1, len(travel_time.index)-1):
        # The maximum flow segment is in the right
        if flow_max_division[i][0] < flow_max_division[i][1]:
            li = travel_time.loc[0,i]
        # The maximum flow segment is in the left
        else:
            li = travel_time.loc[i,travel_time.index[-1]]
        # Calculate and salve
        length[i] = li + li/l

    return length

def di_calculation(demand: pd.DataFrame, travel_time: pd.DataFrame, d1: float, d2: float) -> list[int]:
    """
    Divisibility index for each stop.
    """
    # Calculate the terms
    transfers = transfers_calculate(demand)
    flow_max_list = max_flow_calculate(demand)
    flow_max = max([item for f in flow_max_list[1:-2] for item in (f[0], f[1])])
    length = length_term_calculate(travel_time, flow_max_list)

    # Calculate the DI
    total_demand = demand.values.sum()
    divisibility = [0]*len(demand.index)
    for i in range(1, len(travel_time.index)-1):
        term1 = 1 - d1*transfers[i]/total_demand
        term2 = abs(flow_max_list[i][0] - flow_max_list[i][1]) / flow_max
        term3 = 1 + d2*length[i]
        divisibility[i] = term1 * term2 * term3

    return divisibility
def cost_users(demand: pd.DataFrame,
               travel_time: pd.DataFrame,
               frequency: list[float],
               board_alight_time: float,
               piv: float,
               pw: float,
               pr: float,
               divided_nodes: list[int]) -> float:
    """
    Total users cost, given the divisions of the corridor.
    """
    # Transfer cost
    transfers = np.array(transfers_calculate(demand))
    transfers = [transfers[i] if i in divided_nodes
                 else 0
                 for i in range(len(demand.index))]
    transfers_totales = sum(transfers)

    # Waiting time cost
    waiting_time = 0
    for i in range(len(demand.index)):
        line = find_line(divided_nodes=divided_nodes, stop=i)
        first_boarding_right = demand.loc[i, i+1:].sum()/(2*frequency[line])
        if i in divided_nodes:
            boarding_due_transfers = demand.loc[:i-1, i+1:].values.sum()/(2*frequency[line]) + demand.loc[i+1:, :i-1].values.sum()/(2*frequency[line-1])
            first_boarding_left = demand.loc[i, :i-1].sum()/(2*frequency[line-1])
        else:
            first_boarding_left = demand.loc[i, :i-1].sum()/(2*frequency[line])
            boarding_due_transfers = 0
        waiting_time += boarding_due_transfers + first_boarding_right + first_boarding_left

    # Travel time cost
    stop_state_right, stop_state_left = board_alight_calculate(demand)
    total_travel_time = 0
    # In-vehicle moving time
    for i in demand.index:
        for j in demand.columns:
            total_travel_time += demand.loc[i,j]*travel_time.loc[i,j]
    # In-vehicle stopped time
    for i in demand.index:
        line = find_line(divided_nodes=divided_nodes, stop=i)
        # Right
        # boarding and alighting passengers
        boarding = stop_state_right.loc[i, 'subidas']*board_alight_time/frequency[line]
        alight = stop_state_right.loc[i, 'bajadas']*board_alight_time/frequency[line]
        # Passengers on board waiting
        stopped_passengers = demand.loc[:i-1, i+1:].values.sum()
        total_travel_time += stopped_passengers*(boarding+alight)
        # Passengers alight
        avr_alight_time = stop_state_right.loc[i, 'bajadas']*board_alight_time/(2*frequency[line])
        total_travel_time += stop_state_right.loc[i, 'bajadas']*avr_alight_time

        if i in divided_nodes:
            line = line-1
        # Left
        # boarding and alighting passengers
        boarding = stop_state_left.loc[i, 'subidas']*board_alight_time/frequency[line]
        alight = stop_state_left.loc[i, 'bajadas']*board_alight_time/frequency[line]
        # Passengers on board waiting
        stopped_passengers = demand.loc[i+1:, :i-1].values.sum()
        total_travel_time += stopped_passengers*(boarding+alight)
        # Passengers alight
        avr_alight_time = stop_state_left.loc[i, 'bajadas']*board_alight_time/(2*frequency[line])
        total_travel_time += stop_state_left.loc[i, 'bajadas']*avr_alight_time

    return total_travel_time*piv + waiting_time*pw + transfers_totales*pr

def board_alight_calculate(demand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    It calculares the board and alight for each stop, for both directions.
    """

    # flow through the right
    subidas = []
    for i in demand.index:
        subidas.append(demand.loc[i, i:].values.sum())

    bajadas = []
    for i in demand.index:
        bajadas.append(demand.loc[:i, i].values.sum())

    flujos_explicito_right = pd.DataFrame({'subidas': subidas, 'bajadas':bajadas})

    # flow through the left
    subidas = []
    for i in demand.index:
        subidas.append(demand.loc[i, :i].values.sum())

    bajadas = []
    for i in demand.index:
        bajadas.append(demand.loc[i:, i].values.sum())

    flujos_explicito_left = pd.DataFrame({'subidas': subidas, 'bajadas':bajadas})

    return flujos_explicito_right, flujos_explicito_left

def find_line(divided_nodes: list[int], stop: int) -> int:
    """
    Given the nodes where the corridor is divided, it gives you the line where a given stop belongs. The lines are numbered from 0 and growing to the right.
    """
    if len(divided_nodes) == 0:
        return 0
    for i in range(len(divided_nodes)):
        if stop < divided_nodes[i]:
            return i
    return len(divided_nodes)

def cost_operators(demand: pd.DataFrame,
                   travel_time: pd.DataFrame,
                   frequency: list[float],
                   board_alight_time: float,
                   c0: float,
                   c1: float,
                   divided_nodes: list[int]) -> float:
    """
    It calculates the operator cost.
    """
    # Calculate flow both directions in each segment
    flow_left = []
    flow_right = []
    for i in range(len(demand.index)-1):
        # Passengers traveling from before i to after i.
        flow_l = demand.loc[:i, i+1:].values.sum()
        flow_left.append(flow_l)
        # Passengers traveling from after i to before i.
        flow_r = demand.loc[i+1:, :i].values.sum()
        flow_right.append(flow_r)
    # Maximum flow by arc
    flow_max = [max(flow_left[i], flow_right[i]) for i in range(len(flow_left))]
    flow_max = np.array(flow_max)

    # Calculate transfers
    transfers = np.array(transfers_calculate(demand))

    total_cost = 0
    # Calculate the cost for each line
    for i in range(len(frequency)):

        # Line terminal stops
        # Only one line
        if len(frequency) == 1:
            first_node = 0
            last_node = len(demand.index)-1
        # First (not only) line
        elif i == 0:
            first_node = 0
            last_node = divided_nodes[i]
        # Other than first
        else:
            first_node = divided_nodes[i-1]
            if i == len(frequency) - 1:
                last_node = demand.index[-1]
            else:
                last_node = divided_nodes[i]

        # Maximum flow in the line
        flow_max_line = flow_max[first_node:last_node].max()
        # Capacity
        capacity = flow_max_line/frequency[i]

        # Cycle time
        total_travel_time = travel_time.loc[first_node, last_node]*2
        stop_state_right, stop_state_left = board_alight_calculate(demand)
        boarding_time = (stop_state_left.loc[first_node+1: last_node, 'subidas'].values.sum() +
                         stop_state_right.loc[first_node: last_node-1, 'subidas'].values.sum())
        alight_time = (stop_state_left.loc[first_node: last_node-1, 'bajadas'].values.sum() +
                         stop_state_right.loc[first_node+1: last_node, 'bajadas'].values.sum())
        extra_boarding_time_transfers = transfers[first_node]
        cycle_time = total_travel_time + (boarding_time + alight_time + extra_boarding_time_transfers)*board_alight_time/frequency[i]

        # Fleet size
        fleet = cycle_time*frequency[i]

        # Add cost
        total_cost += fleet*(c0 + c1*capacity)

    return total_cost

def vrc(demand: pd.DataFrame,
        travel_time: pd.DataFrame,
        frequency: list[float],
        divided_nodes: list[int],
        piv: float,
        pw: float,
        pr: float,
        board_alight_time: float,
        c0: float,
        c1: float) -> float:
    """
    It calculates the VRC.
    """

    users = cost_users(demand=demand,
                       travel_time=travel_time,
                       frequency=frequency,
                       board_alight_time=board_alight_time,
                       piv=piv,
                       pw=pw,
                       pr=pr,
                       divided_nodes=divided_nodes)

    operators = cost_operators(demand=demand,
                               travel_time=travel_time,
                               frequency=frequency,
                               board_alight_time=board_alight_time,
                               c0 = c0,
                               c1 = c1,
                               divided_nodes=divided_nodes)

    return users + operators

def vrc_fix(demand: pd.DataFrame,
            travel_time: pd.DataFrame,
            piv: float,
            board_alight_time: float,
            c0: float,
            c1: float) -> float:
    """
    Calculates the VRC components that doesn't vary with the frequency or the line divisions.
    """

    # Users travel time in motion
    travel_time_motion = 0
    # In-vehicle moving time
    for i in demand.index:
        for j in demand.columns:
            travel_time_motion += demand.loc[i,j]*travel_time.loc[i,j]
    users_vrc_fix = travel_time_motion*piv

    # Operators
    cicle_time_motion = travel_time.loc[0, travel_time.index[-1]]
    total_demand = demand.values.sum()
    # Calculate flow both directions in each segment
    flow_left = []
    flow_right = []
    for i in range(len(demand.index)-1):
        # Passengers traveling from before i to after i.
        flow_l = demand.loc[:i, i+1:].values.sum()
        flow_left.append(flow_l)
        # Passengers traveling from after i to before i.
        flow_r = demand.loc[i+1:, :i].values.sum()
        flow_right.append(flow_r)
    # Minimum flow by arc
    flow_min = min(flow_left + flow_right)
    operators_vrc_fix = 2*board_alight_time*c0*total_demand + cicle_time_motion*flow_min*c1

    return users_vrc_fix + operators_vrc_fix

# def vrc_optimization(demand: pd.DataFrame,
#                      travel_time: pd.DataFrame,
#                      divided_nodes: list[int],
#                      piv: float,
#                      pw: float,
#                      pr: float,
#                      board_alight_time: float,
#                      c0: float,
#                      c1: float) -> tuple[tuple[float], float]:
#     """
#     It calculates the optimal frequency to minimize the VRC.
#     """
#     def vrc_auxiliar(freq):
#         vrc_value = vrc(demand=demand,
#                         travel_time=travel_time,
#                         frequency=freq,
#                         divided_nodes=divided_nodes,
#                         piv=piv,
#                         pw=pw,
#                         pr=pr,
#                         board_alight_time=board_alight_time,
#                         c0=c0,
#                         c1=c1)
#
#         return vrc_value
#
#     # Minimization
#     initial_freq = [30]*(len(divided_nodes)+1)
#     bounds = [(1, 300)]*len(initial_freq)
#     result = minimize(vrc_auxiliar, initial_freq, bounds=bounds)
#
#     optimal_freq = result.x
#     optimal_vrc = result.fun
#
#     return optimal_freq, optimal_vrc

def shift_dataframe(df: pd.DataFrame):
    df.index = range(len(df))
    df.columns = range(df.shape[1])

    return df

def vrc_optimization(demand: pd.DataFrame,
                     travel_time: pd.DataFrame,
                     divided_nodes: list[int],
                     piv: float,
                     pw: float,
                     pr: float,
                     board_alight_time: float,
                     c0: float,
                     c1: float) -> tuple[tuple[float], float]:
    """
    It calculates the optimal frequency to minimize the VRC.
    """
    if len(divided_nodes) == 0:
        def vrc_auxiliar(freq):
            vrc_value = vrc(demand=demand,
                            travel_time=travel_time,
                            frequency=freq,
                            divided_nodes=[],
                            piv=piv,
                            pw=pw,
                            pr=pr,
                            board_alight_time=board_alight_time,
                            c0=c0,
                            c1=c1)

            return vrc_value

        initial_freq = [30]
        bounds = [(1, 300)]
        result = minimize(vrc_auxiliar, initial_freq, bounds=bounds)

        optimal_freq = result.x
        optimal_vrc = result.fun

    else:
        div = divided_nodes[0]
        demand_modified = demand.copy()
        for i in range(0, div):
            for j in range(div + 1, len(demand.index)):
                demand_modified.loc[div, j] = demand_modified.loc[div, j] + demand.loc[i, j]
                demand_modified.loc[i, div] = demand_modified.loc[i, div] + demand.loc[i, j]
        for i in range(div + 1, len(demand.index)):
            for j in range(0, div):
                demand_modified.loc[div, j] = demand_modified.loc[div, j] + demand.loc[i, j]
                demand_modified.loc[i, div] = demand_modified.loc[i, div] + demand.loc[i, j]

        demand_modified_left = shift_dataframe(demand_modified.loc[:div, :div])
        demand_modified_right = shift_dataframe(demand_modified.loc[div:, div:])
        travel_time_left = shift_dataframe(travel_time.loc[:div, :div])
        travel_time_right = shift_dataframe(travel_time.loc[div:, div:])

        # Left freq
        def vrc_auxiliar(freq):
            vrc_value = vrc(demand=demand_modified_left,
                            travel_time=travel_time_left,
                            frequency=freq,
                            divided_nodes=[],
                            piv=piv,
                            pw=pw,
                            pr=pr,
                            board_alight_time=board_alight_time,
                            c0=c0,
                            c1=c1)

            return vrc_value

        initial_freq = [30]
        bounds = [(1, 300)]
        result_left = minimize(vrc_auxiliar, initial_freq, bounds=bounds)

        optimal_freq_left = result_left.x
        optimal_vrc_left = result_left.fun

        # Right freq
        def vrc_auxiliar(freq):
            vrc_value = vrc(demand=demand_modified_right,
                            travel_time=travel_time_right,
                            frequency=freq,
                            divided_nodes=[],
                            piv=piv,
                            pw=pw,
                            pr=pr,
                            board_alight_time=board_alight_time,
                            c0=c0,
                            c1=c1)

            return vrc_value

        initial_freq = [30]
        bounds = [(1, 300)]
        result = minimize(vrc_auxiliar, initial_freq, bounds=bounds)

        optimal_freq_right = result.x
        optimal_vrc_right = result.fun

        # Calculate transfers
        trans = transfers_calculate(demand)[div]
        # Join results
        optimal_freq = [optimal_freq_left, optimal_freq_right]
        optimal_vrc = optimal_vrc_left + optimal_vrc_right + trans*pr

    return optimal_freq, optimal_vrc