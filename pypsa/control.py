"""importing important libraries."""
from .descriptors import get_switchable_as_dense
import collections
import re
import logging
import numpy as np
logger = logging.getLogger(__name__)


def find_allowable_q(p, power_factor, s_nom):
    """
    Some times the reactive power that controller want to compensate using
    (p*tan(arccos(power_factor))) can go higher than what inverter can provide
    based on inverter "s_nom" and the provided "power_factor", in this case:
        - calculate reactive power that the formula gives "q"
        - calcualte reactive power max available capacity that inverter can
          provide based on the power factor given "q_inv_cap".
        - check q if it is less than "q_inv_cap" ok, if not take the value from
          "q_allowable" instead.
        - Return all (q, q_inv_cap, q_allowable) to the controller for further
          calculations and considerations.

    This values are returned to controller in order to check and make sure that
    the inverter equation s_nom = np.sqrt((p**2 + q**2) is not violated.
    """
    # Calculate reactive power that controller want to compensate initially
    q = p.mul(np.tan((np.arccos(power_factor, dtype=np.float64)),
                     dtype=np.float64))
    # find inverter q capacity according to power factor provided
    q_inv_cap = s_nom*np.sin(np.arccos(power_factor, dtype=np.float64),
                             dtype=np.float64)
    # find max allowable q that is possible for controller to give as output
    q_allowable = np.where(q <= q_inv_cap, q, q_inv_cap)

    return q_inv_cap, q_allowable, q


def adjust_p_set(s_nom, q, p, c, control_strategy):
    """
    when the initial reactive power "q" calculated by controller together with
    the active power "p" violates inverter equation s_nom = np.sqrt((p**2 + q**2),
    in this case controller needs to reduce p in order to fulfil reactive power
    need. In this case p is reduced and calculated here "new_p_set" and return
    it to the controller to consider this as p_out and set it to the network.
    """
    adjusted_p_set = np.sqrt((s_nom**2 - q**2),  dtype=np.float64)
    new_p_set = np.where(abs(p) <= abs(adjusted_p_set), p, adjusted_p_set)
    # info for user that p_set has been changed
    log_info = np.where(
            control_strategy == 'fixed_cosphi', '"fixed_cosphi" control is adjusted',
            ' "q_v" control might be adjusted, if needed')

    logger.info(" Some p_set in '%s' component with %s due to reactive power "
                "compensation priority. ", c, log_info)

    return new_p_set


def apply_fixed_cosphi(n, snapshot, c, index):
    """
    fix power factor inverter controller.
    This controller provides a fixed amount of reactive power compensation to the
    grid as a function of the amount of injected power (p_set) and the chosen
    power factor value. 
    Controller will take care of inverter capacity and controlls that the sum
    of active and reactive power does not increase than the inverter capacity.
    When reactive power need is more than what controller calculate based on
    the provided power factor, controller decreases a portion of active power
    to meet reactive power need, in this case controller will have two outputs
    p_out and q_out where q_out is the reactive power output and p_out is the
    reduced p_set and will be updated in buses_t.p and components_t.p.
    
    Finally the controller outpus are passed to "_set_controller_outputs_to_n"
    to update the network.

    reference : https://ieeexplore.ieee.org/document/6096349
    DOI link  : 10.1109/JPHOTOV.2011.2174821

    Parameters
    ----------
    n : pypsa.components.Network
        Network
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    index : index of controlled elements
    c : string
        Component name, i.e. 'Load', 'StorageUnit'...

    Returns
    -------
    None
    """
    # needed parameters
    p_input = n.pnl(c).p.loc[snapshot, index]
    params = n.df(c).loc[index]
    power_factor = params['power_factor']
    s_nom = params['s_nom']

    p_out=None
    ctrl_p_out = False
    q_inv_cap, q_allowable, q = find_allowable_q(p_input, power_factor, s_nom)
    q_out = -q_allowable

    # check if the calculated q is not exceeding the inverter capacity if yes then
    # decrease p_input in order not to violate s_nom = np.sqrt((p**2 + q**2) .
    if (abs(q) > q_inv_cap).any().any():
        ctrl_p_out = True
        p_out = adjust_p_set(s_nom, q_out, p_input, c, 'fixed_cosphi')

    _set_controller_outputs_to_n(n, c, index, snapshot, ctrl_p_out=ctrl_p_out,
                                 ctrl_q_out=True, p_out=p_out, q_out=q_out)


def apply_cosphi_p(n, snapshot, c, index):
    """
    Power factor as a function of active power (cosphi_p) controller.
    This controller provides a variable power factor value based on the chosen
    parameters and the droop curve defined below. And then using the calculated
    power factor an amount of reactive power is calculated for reactive power
    compensation, controller works as follow:
        - calculate: p_set_per_p_ref = (p_set / p_ref)*100, where p_ref is a
          setpoint in MW.
        - Then controller compares "p_set_per_p_ref" with the "set_p1" and
          "set_p2" set points where set_p1 and set_p2 are percentage values.
        - Controller decides the power factor based on the defined droop below
          (power_factor = ...). i.e. if p_set_per_p_ref < set_p1 then power
          factor is 1, since p_set_per_p_ref < set_p1 shows low generation and
          controller think there might not be any need for reactive power
          with this amount of generation, thus power_factor=1 which means q = 0.
          For the other conditions power factor is calculated respectively.
    Finally the controller outpus are passed to "_set_controller_outputs_to_n"
    to update the network.

    reference : https://ieeexplore.ieee.org/document/6096349.
    DOI link  : 10.1109/JPHOTOV.2011.2174821

    Parameters
    ----------
    n : pypsa.components.Network
        Network
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    index : index of controlled elements
    c : string
        Component name, i.e. 'Load', 'StorageUnit'...

    Returns
    -------
    None
    """
    # parameters needed
    params = n.df(c).loc[index]
    p_input = n.pnl(c).p.loc[snapshot, index]
    # TODO: it would be better to change p_ref to s_nom which will lead to removing p_ref completely
    p_set_per_p_ref = (abs(p_input) / params['p_ref'])*100

    # choice of power_factor according to controller inputs and its droop curve
    power_factor = np.select([(p_set_per_p_ref < params['set_p1']), (
        p_set_per_p_ref >= params['set_p1']) & (p_set_per_p_ref <= params['set_p2']), (
            p_set_per_p_ref > params['set_p2'])], [1, (1 - ((1 - params['power_factor_min']) / (
             params['set_p2'] - params['set_p1']) * (p_set_per_p_ref - params['set_p1']))), params['power_factor_min']])

    # find q_set and avoid -0 apperance as the output when power_factor = 1
    q_out = np.where(power_factor == 1, 0, -p_input.mul(np.tan((np.arccos(
                          power_factor, dtype=np.float64)), dtype=np.float64)))

    S = np.sqrt((p_input)**2 + q_out**2)
    assert ((S < params['s_nom']).any().any()), (
        "The resulting reactive power (q)  while using 'cosphi'_p control  "
        "with the chosen attr 'power_factor_min' in '%s' component results a  "
        "complex power (S = sqrt(p**2 + q**2))) which is greater than 's_nom') "
        "of the inverter, please choose the right power_factor_min value" % (c))

    _set_controller_outputs_to_n(
        n, c, index, snapshot, ctrl_q_out=True, q_out=q_out)


def apply_q_v(n, snapshot, c, index, n_trials_max, n_trials):
    """
    Reactive power as a function of voltage Q(V).
    This contrller controller provide reactive power compensation based on the
    voltage information of the bus where inverter is connected, for this purpose
    the droop for reactive power calculation is divided in to 5 different reactive
    power calculation zones. Where v1, v2, v3, v4 attrs form the droop and the
    reactive power is calculated based on which zone the bus v_mag_pu is landing.
        - controller finds the zone where bus v_mag_pu lands on
        - Based on the zone and the droop provided it calcualtes "curve_q_set_in_percentage"
        - Using "curve_q_set_in_percentage" it calcualtes reactive power q_out.
    Controller will take care of inverter capacity and controlls that the sum
    of active and reactive power does not increase than the inverter capacity.
    When reactive power need is more than what controller calculate based on
    the provided power factor, controller decreases a portion of active power
    to meet reactive power need, in this case controller will have two outputs
    p_out and q_out.
    Finally the controller outpus are passed to "_set_controller_outputs_to_n"
    to update the network.

    reference : https://ieeexplore.ieee.org/document/6096349
    DOI link  : 10.1109/JPHOTOV.2011.2174821 

    Parameters:
    ----------
    n : pypsa.components.Network
        Network
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    index : index of controlled elements
    c : string
        Component name, i.e. 'Load', 'StorageUnit'...
    n_trials_max : integer
        It is the max number of outer loop (while loop in pf.py) trials until
        the controller converges.
    n_trials : integer
        It is the outer loop (while loop in pf.py) number of trials until
        the controller converges.

    Returns
    -------
    None
    """
    if n_trials == n_trials_max:
        logger.warning("The voltage difference at snapshot ' %s', in components"
                       " '%s', with 'q_v' controller exceeds x_tol_outer limit,"
                       " please apply (damper < 1) or expand controller"
                       " parameters range between v1 & v2 and or v3 & v4 to"
                       " avoid the problem." % (snapshot, index))
    #  curve parameters
    v_pu_bus = n.buses_t.v_mag_pu.loc[snapshot, n.df(c).loc[index, 'bus']].values
    params = n.df(c).loc[index]
    p_input = n.pnl(c).p.loc[snapshot, index]
    p_out = None
    ctrl_p_out = False
    q_inv_cap, q_allowable, q = find_allowable_q(p_input, params['power_factor'], params['s_nom'])
    dy_damper = np.select([n_trials>=45, n_trials>=40, n_trials>=30, n_trials>20, n_trials>15],
                          [0.1, 0.3, 0.5, 0.8, 0.9], default=params[
            'damper'])
    # calculation of maximum q compensation in % based on bus v_pu_bus
    curve_q_set_in_percentage = np.select([(v_pu_bus < params['v1']), (v_pu_bus >= params['v1']) & (
            v_pu_bus <= params['v2']), (v_pu_bus > params['v2']) & (v_pu_bus <= params['v3']), (v_pu_bus > params['v3'])
        & (v_pu_bus <= params['v4']), (v_pu_bus > params['v4'])], [100, 100 - 100 / (params['v2'] - params['v1']) * (
                v_pu_bus - params['v1']), 0, -100 * (v_pu_bus - params['v3']) / (params['v4'] - params['v3']), -100])
    # calculation of q
    q_out = (((curve_q_set_in_percentage * q_allowable) / 100) * params[
            'damper'] * params['sign'])*dy_damper
    # check if there is need to reduce p_set due to q need
    if (q > q_inv_cap).any().any():
        ctrl_p_out = True
        p_out = adjust_p_set(params['s_nom'], q, p_input, c, 'q_v')

    _set_controller_outputs_to_n(n, c, index, snapshot, ctrl_p_out=ctrl_p_out,
                                 ctrl_q_out=True, p_out=p_out, q_out=q_out)


def find_oltc_tap_side(n, row):
    """Find the tap_side of the controlled tranformer."""
    if row['type'] != '':
        tap_side = n.transformer_types.loc[row['type'], 'tap_side']
    else:
        tap_side = row['tap_side']

    c = np.where(tap_side == 0, 1, -1)

    return c


def apply_oltc(n, snapshot, index, calculate_Y, sub_network, skip_pre, n_iter_oltc):
    """
    On Load Tap Changer Transformer (OLTC). Supports three conditions. 1. if no
    bus is give as controled bus to "ctrl_buses" attribute of transformer, OLTC
    assumes that the bus at which secondary of transformer is connected is the
    controlled bus. OR if any other bus name is give as input in "ctrl_buses"
    attribute, controller will choose the tap position to bring the voltage withing
    the range which is determined by "deadband" and "v_set". 3 If multiple buses
    are given as controlled buses, controller finds the min and max voltages of
    the controlled buses and chooses an optimum tap to bring the measured v_min
    and v_max withing the deandband range determined by "v_min" and "v_max",
    attributes where "v_min" and "v_max" are the minimum and maximum allowed
    voltages of	 the network.

    ==> important for multiple bus control: controller takes the existing
    "tap position" as the initial input for transformer, which will be changed
    only if controller detects voltage violation based on the setted deadband
    (v_min and v_max). This behaviour might change some bus v_mag_pu values
    even if they are withing the deandband range, but this change stays within
    the deadband.

    Also sometimes there can be mismatch between what controller predicts grid
    voltage changes and what actually voltages change after pf. For instance:
    based on "tap_step" and available "tap_position" options, oltc predicts the
    voltage change in the grid and chooses an optimum tap position so that to
    avoid undervoltage problmes, but pf results a slight undervoltage violation
    due to oltc prediction mismatch with the actual changes after pf. oltc recognizes
    this as voltage violation and chooses another tap which cause overvoltage
    problem and again controller chooses the previous tap, this process keeps
    going on. To avoid it controller will go for the tap option where no under
    voltage happens.

    Parameters
    ----------
    n : pypsa.components.Network
        Network.
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    index : index of controlled elements
    calculate_Y : function
        it calculate admmittance matrix.
    sub_network : pypsa.components.Network.sub_network
        network.sub_networks.
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating dependent
        values and finding bus controls.
    n_iter_oltc : integer
        Counts the number of iteration for oltc

    Returns
    -------
    None.
    """
    n_iter_oltc += 1
    tap_changed = False
    for ind in index:

        row = n.transformers.loc[ind]
        # c is contant to consider tap side issue in the end
        c = find_oltc_tap_side(n, row)

        current_tap = row['tap_position']

        # extracting controlled nodes names
        buses = [x.strip() for x in n.transformers.loc[ind, 'ctrl_buses'].split(',')]
        # if no node is chosen take node of secondary of trafo as control node
        ctrl_buses = np.where(
                len(n.transformers.loc[ind, 'ctrl_buses']) == 0, [row['bus1']], buses)

        # find voltages of controlled nodes
        v_pu_ctrl_buses = n.buses_t.v_mag_pu.loc[snapshot, ctrl_buses]
        
        # Single node oltc control part:
        if len(ctrl_buses) == 1:
            opt_tap, tap_step = trafo_single_node_ctrl(
                                  n, snapshot, index, ind, row, v_pu_ctrl_buses)

        # Multiple node oltc control part:
        elif len(ctrl_buses) > 1:
            opt_tap, tap_step = trafo_multiple_bus_ctrl(
                                  n, snapshot, index, ind, row, v_pu_ctrl_buses)

        if n_iter_oltc > 4:
            logger.info("==> Due to unconsistancy of oltc parameters in %s "
                        "transformer with multiple bus control, oltc is jumping "
                        "around the two tap positions %s and %s which oscilates "
                        "around lower voltage limit '%s'. To avoid under-voltage "
                        "violation, oltc chooses %s as the optimum tap position."
                        " If this is undesirable please change the lower voltage"
                        " limit 'v_min' to avoid the problem.",
                        ind, current_tap, opt_tap, row['v_min'], current_tap)

            opt_tap = np.where(
                    v_pu_ctrl_buses.min() > row['v_min'], current_tap, opt_tap)

        n.transformers_t.opt_tap_position.loc[snapshot, ind] = opt_tap

        # Set the new tap position changed to tranformer
        if current_tap != opt_tap:
            tap_changed = True
            n.transformers.loc[ind, 'tap_position'] = opt_tap
            n.transformers_t.opt_tap_position.loc[snapshot, ind] = opt_tap
            ratio = (row['tap_ratio'] + (opt_tap-current_tap)*c*tap_step/100)
            n.transformers.loc[ind, 'tap_ratio'] = ratio
            # TODO I will dig into "calculate_Y" after my thesis to extract only the needed part.
            calculate_Y(sub_network, skip_pre=skip_pre)

    return  n_iter_oltc, tap_changed


def find_taps_and_tap_step(n, snapshot, index, row, ind):
    """
    Find the the available number of taps and tap_step based on if 'type' is
    empty or not and return them for further consideration.
    
    """
    # find taps and tap_steps from n.transformers or n.transformer_types
    if row['type'] != '':
        taps = (np.arange(n.transformer_types.loc[row['type'], 'tap_min'],
                        n.transformer_types.loc[row['type'], 'tap_max']+1))
        tap_step = n.transformer_types.loc[row['type'], 'tap_step']
    else:
        taps = np.arange(row['tap_min'], row['tap_max']+1)
        tap_step = row['tap_step']

    return taps, tap_step


def trafo_single_node_ctrl(n, snapshot, index, ind, row, v_pu_ctrl_buses):

    opt_tap= row['tap_position']
    taps, tap_step = find_taps_and_tap_step(n, snapshot, index, row, ind)

    deadband_range = row['v_set'] + np.array(
                                   [row['deadband']/100, -row['deadband']/100])

    value_in_range = (v_pu_ctrl_buses.values >= min(
      deadband_range) and v_pu_ctrl_buses.values <= max(deadband_range))

    if value_in_range:
        logger.info(" The voltage in node '%s' in snapshot '%s'  controlled by "
                    " oltc in %s is already within the deadband range. ",
                    v_pu_ctrl_buses.index, snapshot, ind)

    else:
        possible_tap_res = abs(row['v_set'] - v_pu_ctrl_buses.values -
                        (row['tap_position'] - taps)*tap_step/100*row['v_set'])

        opt_tap = taps[np.where(possible_tap_res == min(possible_tap_res))][0] 
        
        calc_v_pu = v_pu_ctrl_buses.values - opt_tap * tap_step*row['v_set']/100
        if (calc_v_pu >= min(
                deadband_range) and calc_v_pu <= max(deadband_range)):
            logger.info("The voltage in %s in snapshot %s controlled by oltc in"
                        "  %s,  using %s as the optimum tap position is now "
                        "withing the deadband range.",
                        v_pu_ctrl_buses.index.tolist(), snapshot, ind, opt_tap)
        else:
            logger.warning("Due to oltc tap position limits Voltage in "
                           "node %s in snapshot %s controlled by oltc in %s, "
                           " could not set within the deadband range, %s is "
                           "used as the optimum possible tap position,",
                           v_pu_ctrl_buses.index.tolist(), snapshot, ind, opt_tap)

    return opt_tap, tap_step


def trafo_multiple_bus_ctrl(n, snapshot, index, ind, row, v_pu_ctrl_buses):

    opt_tap = row['tap_position']
    taps, tap_step = find_taps_and_tap_step(n, snapshot, index, row, ind)
    # find the max and min voltage of the grid from the controlled buses
    meas_max = v_pu_ctrl_buses.values.max()
    meas_min = v_pu_ctrl_buses.values.min()

    # check if meas_max and meas_min are withing the range, if not follow 'else' condition
    if (meas_min > row['v_min'] and meas_max < row['v_max']):
        
        logger.info(" Voltage in nodes %s controlled by oltc in  %s are"
                    " already withing 'v_min' and 'v_max' ranges.",
                    v_pu_ctrl_buses.index.tolist(), ind)
        pass

    else:
        # calculate the possible voltage changes on meas_max and meas_min for each tap position
        max_voltage = meas_max + (row['tap_position'] - taps)*tap_step/100
        min_voltage = meas_min + (row['tap_position'] - taps)*tap_step/100

        # find all the optimum indexes of tap positions
        opt_ind = np.where(((min_voltage > row['v_min']) & (
                max_voltage < row['v_max'])))[0]

        if len(opt_ind) != 0:
            # 0 implies that if multiple index exist take the first from left (neg taps first)
            opt_tap = taps[opt_ind[0]]
        else:
            opt_ind = np.where(min_voltage > row['v_min'])[0]
            if len(opt_ind) != 0:
                # if multiple opt index exist take the last one (positive taps)
                opt_tap = taps[len(opt_ind)-1]

            else:
                # if the above is not met, then take the first tap position (neg taps are first)
                opt_tap = taps[0]

        logger.info("The voltage in %s controlled by oltc in %s, using "
                    " %s as the optimum tap position.",
                    v_pu_ctrl_buses.index.tolist(), ind, opt_tap)

    return opt_tap, tap_step


def apply_p_v(n, snapshot, c, index, n_trials_max, n_trials):
    """
    Reactive power as a function of voltage Q(V).
    reference : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6096349

    This controller basically limits power generation or power consumption for
    the controlled components based on the connected bus voltages to avoid grid
    voltage violation issues. The controllable components in general have two
    status: 1. "grid injection mode": when storage (Store, StorageUnit) are
     discharging, generators are injecting, and negative loads are connected to
    the grid. 2. "grid consumption mode": when storage (Store, StorageUnit) are,
    charging, generators are  with negative power sign and loads are connected.
    to the grid. Controller droop characteristic for these two cases are different,
    therefore, controller first determines the grid consumption and the grid
    injection modes and their indexes and then uses the respective droop to
    determine the amount of allowed power injection or power consumption to or
    from the grid using.

    Parameters:
    ----------
    n : pypsa.components.Network
        Network
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    c : string
        Component name, i.e. 'Load', 'StorageUnit'...
    index : index of controlled elements
    n_trials_max : integer
        It is the max number of outer loop (while loop in pf.py) trials until
        the controller converges.
    n_trials : integer
        It is the outer loop (while loop in pf.py) number of trials until
        the controller converges.

    Returns
    -------
    None

    """
    if n_trials == n_trials_max:
        logger.warning("The voltage difference at snapshot ' %s' ,in components"
                       " '%s', with 'p_v' control exceeds x_tol_outer limit, "
                       "please apply damper or change  controller parameters to"
                       " avoid the problem." % (snapshot, index))

    v_pu_bus = n.buses_t.v_mag_pu.loc[
                           snapshot, n.df(c).loc[index, 'bus']].values

    p_input = get_switchable_as_dense(n, c, 'p_set', inds=index).loc[snapshot]

    # Flag for the case where the component consumes active power from the grid
    grid_consumption = (((c == 'Load') and (np.sign(p_input) > 0).any()) or (
        (c == 'StorageUnit' or c == 'Generator' or c == 'Store') and (
            np.sign(p_input) < 0).any()))

    # Flag for the case where the component injects active power to the grid
    grid_injection = (((c == 'Load') and (np.sign(p_input) < 0).any()) or (
        (c == 'StorageUnit' or c == 'Generator' or c == 'Store') and (
            np.sign(p_input) > 0).any()))

    # when both consumption and injection exist, i.e. one storage_unit is
    # charging and other one is discharging ...
    if (grid_consumption & grid_injection):
        # consumption part
        if grid_consumption:
            # filter inputs and indexes of grid consumption indexes
            if c == 'Load':
                p_con = p_input[p_input > 0]
            else:
                p_con = p_input[p_input < 0]

            ind_con = n.df(c).loc[p_con.index]  # filtered indexes
            v_pu_bus_con = n.buses_t.v_mag_pu.loc[
                snapshot, n.df(c).loc[ind_con.index, 'bus']].values

            calculate_cosumption_p(v_pu_bus_con, p_con, c, n, snapshot)

        # injection part
        if grid_injection:
            # filter inputs and indexes of grid injection indexes
            if c == 'Load':
                p_inj = p_input[p_input < 0]
            else:
                p_inj = p_input[p_input > 0]

            ind_inj = n.df(c).loc[p_inj.index]  # filtered indexes
            v_pu_bus = n.buses_t.v_mag_pu.loc[
                snapshot, n.df(c).loc[ind_inj.index, 'bus']].values

            calculate_injection_p(v_pu_bus, p_inj, c, n, snapshot, n_trials)

    # if only grid consumption exis, i.e. battery charing or loads...
    elif grid_consumption:
        calculate_cosumption_p(v_pu_bus, p_input, c, n, snapshot)
    # if only grid injection exist, i.e. generators, battery discharging...
    elif grid_injection:
        calculate_injection_p(v_pu_bus, p_input, c, n, snapshot, n_trials)


def calculate_cosumption_p(v_pu_bus, p_con, c, n, snapshot):
    """
    This method is called from "p_v" controller when any of the controlled
    components is consuming active power from the grid. Given are bus voltages
    "v_pu_bus", all attributes and indexes, active power input "p_input"
    list name "c" of the such indexes. Using these data the amount of allowed
    power after it is curtailed is calculted in percentag using "pperpmax" which
    is the droop that contains multiple condition and choices. "pperpmax" in in
     %, it is multiplied to "p_input" to get the amount of allowed active power
    consumption in MW.
    """
    # required parameters
    v_pu_cr = n.df(c).loc[p_con.index]['v_pu_cr']
    damper = n.df(c).loc[p_con.index]['damper']
    v_max_curtail = n.df(c).loc[p_con.index]['v_max_curtail']
    # find the amount of allowed power consumption in % from the droop
    pperpmax = np.select([(v_pu_bus > v_pu_cr), (v_pu_bus < v_max_curtail), (
        (v_pu_bus >= v_max_curtail) & (v_pu_bus <= v_pu_cr))], [100, 0, (100/(
            v_pu_cr - v_max_curtail)*(v_pu_bus - v_max_curtail))])

    # find the amount of allowed power consumption in MW
    p_out = ((pperpmax*(p_con))/100)*damper
    # update the active power contribution of the controlled indexes in network
    _set_controller_outputs_to_n(
        n, c, p_con.index, snapshot, ctrl_p_out=True, p_out=p_out)

def calculate_injection_p(v_pu_bus, p_inj, c, n, snapshot, n_trials):
    """
    This method is called from "p_v" controller when any of the controlled
    components is injecting active power to the grid. Given are bus voltages
    "v_pu_bus", all attributes and indexes , active power input "p_input"
    list name "c" of the such indexes. Using these data the amount of allowed
    power after it is curtailed is calculted in percentag using "pperpmax" which
    is the droop that contains multiple condition and choices. "pperpmax" is in
     %, it is multiplied to "p_input" to get the amount of allowed active power
    injection in MW.
    """
    # required parameters
    damper = n.df(c).loc[p_inj.index, 'damper']
    damper = np.select([n_trials > 30, n_trials > 20, n_trials > 10 ],
                       [0.3, 0.5, 0.7], default=damper)
    v_pu_cr = n.df(c).loc[p_inj.index, 'v_pu_cr']
    v_max_curtail = n.df(c).loc[p_inj.index, 'v_max_curtail']
    # find the amount of allowed power consumption in % from the droop
    pperpmax = np.select([(v_pu_bus < v_pu_cr), (v_pu_bus > v_max_curtail), (
        (v_pu_bus >= v_pu_cr) & (v_pu_bus <= v_max_curtail))], [
            100, 0, (100-(100/(v_max_curtail-v_pu_cr))*(v_pu_bus-v_pu_cr))])

    # find the amount of allowed power consumption in MW
    p_out = ((pperpmax*(p_inj)) / 100)*damper
    # update the active power contribution of the controlled indexes in network
    _set_controller_outputs_to_n(
        n, c, p_inj.index, snapshot, ctrl_p_out=True, p_out=p_out)


def prioritization(dict_controlled_index, control, v_diff, x_tol_outer,
                   oltc_conv, ctrl_buses, tap_changed, n_trials, prev_priority):
    """
    Prioritization of controller are hard coded as follow:

    - if 'oltc' is combined with 'p_v' controller, the priority is given to,
        'oltc', such that 'oltc' applies and converges then 'p_v' takes action
        until it converges and finally a final load flow is run to apply both of
        them, if the controllers settings do not conflict each other, the final
        load flow should converge and the process jumps to the next snapshot.

    - if 'oltc' is combined with both 'p_v' and or 'q_v', in this case 'q_v'
        has the priority to converge and 'oltc' starts right after convergence
        of 'q_v'. As the last resort 'p_v' will take action after convergance
        of 'q_v' and 'oltc'. As 'p_v' also converged a last power flow is run
        to to apply all controller, if all controllers settings do not
        conflict each other, the final load flow should converge.
    """

    if "q_v" not in dict_controlled_index:  # "oltc" + "p_v"
        priority = np.select(
            [control == "oltc", control == "p_v"],
            # condition for oltc activation
            [tap_changed is None or tap_changed,
             # condition for p_v activation
             v_diff.max() > x_tol_outer and tap_changed is False])

    else:  # ("oltc" + "p_v" + "q_v") or ("oltc" + "q_v")
        priority = np.select(
            [control == "oltc", control == "p_v", control == "q_v"],
            # condition for oltc activation
            [v_diff.max() < x_tol_outer and tap_changed is None or tap_changed,
             # condition for p_v activation
             tap_changed is False and v_diff.max() < x_tol_outer or
             prev_priority == 'p_v' and v_diff.loc[ctrl_buses].max() > x_tol_outer,
             # condition for q_v activation
             v_diff.loc[ctrl_buses].max() > x_tol_outer and not tap_changed
             and (prev_priority in ['', 'q_v'])])

        # copy the priority as info for the next iteration
        prev_priority = np.where(
            control in ["p_v", "q_v"] and priority, control, prev_priority)

    # break the loop when all controllers are converged and give priority for all controller as the findal loop
    if v_diff.max() < x_tol_outer and tap_changed is False:
        oltc_conv = True
        priority = True
    else:
        oltc_conv = False

    return priority, oltc_conv, prev_priority


def apply_controller(n, now, n_trials, n_trials_max, dict_controlled_index,
                     v_diff, x_tol_outer, oltc_conv, calculate_Y,
                     sub_network, skip_pre, i, n_iter_oltc, tap_changed, prev_priority):
    """
    Iterate over chosen control strategies which exist as keys and the controlled
    indexes of each component as values in "dict_controlled_index" and call each
    controller to apply them for controlled components. And return the bus names
    that contain "q_v" controller attached for voltage difference comparison
    purpose in pf.py.

    Parameter:
    ----------
    n : pypsa.components.Network
        Network
    now : single snaphot
        Current  element of n.snapshots on which the power flow is run.
    n_trials : integer
        It is the outer loop (while loop in pf.py) number of trials until
        the controller converges.
    n_trials_max : integer
        It is the max number of outer loop (while loop in pf.py) trials until
        the controller converges.
    dict_controlled_index : dictionary
        Contains all the controlled indexes of controlled components as values
        inside "dict_controlled_index" where each controller is a key there.
    v_diff : pandas series
        Voltage difference between the two iterations of the bus voltages that
        are controlled with voltage dependent controllers such as "p_v" or "q_v".
    x_tol_outer : float
        Tolerance for outer loop voltage difference between the two successive
        power flow iterations as a result of applying voltage dependent controller
        such as reactive power as a function of voltage "q_v".
    oltc_conv : bool, default to None
        It shows the convergance loop of 'oltc' controller combined with one of
        the reactive power controller or active power curtailment control. If
        True, it reflects that oltc combined with any control is converged if
        not the loop will repeat.
    calculate_Y : function
        it calculate admmittance matrix.
    oltc_control : bool, default False
        If ``True``, activates outerloop which considers on load tap changer
        (oltc) transformer control on those transformers which their "oltc"
        attribute is activated (True).
    sub_network : pypsa.components.Network.sub_network
        network.sub_networks.
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating dependent
        values and finding bus controls.
    i : integer
        snapshot index which starts from zero.
    n_iter_oltc : integer
        Counts the number of needed iteration for oltc convegence
    tap_changed : bool, default None
        It shows the oltc tap changing status and convergence alone. If tap is
        changed is True, it implies that oltc controller did not converge yet and
        if tap_changed is False, this reflects that oltc controller converged.
                                                         
    Returns old_v_mag_pu, oltc_conv, n_iter_oltc, tap_changed
    -------
    old_v_mag_pu : pandas data frame
        Needed to compare v_mag_pu of the controlled buses with the voltage from
        previous iteration to decide for repeation of pf (in pf.py file).
    oltc_conv : bool
        It activates or deactivates the outer loop for 'oltc' and or oltc combined
        with other controller (explained above).
    n_iter_oltc : integer
        The counted number iterations for oltc, it is needed to keep track of
        oltc convergence (details is inside apply_oltc controller)
    tap_changed : bool
        It shows the convergance of oltc, it is need for the nex iteration
        prioritization.
    
    """
    v_dep_buses = np.array([])
    ctrl_buses = np.array([])
    prioritized = True
    # break oltc loop when oltc is not used, so that other controllers work fine alone
    oltc_conv = np.where('oltc' in dict_controlled_index, False, True)
    # oltc_conv = not oltc_control # keeps oltc loop running until oltc_conv is True or max trials is reached
    need_for_priority = (("q_v" in dict_controlled_index or "p_v" in dict_controlled_index)
                         and "oltc" in dict_controlled_index)

    for control in dict_controlled_index.keys():
        # parameter is the controlled indexes dataframe of a components
        for c, index in dict_controlled_index[control].items():

            if control in ["p_v", "q_v"]:
                ctrl_buses = np.unique(n.df(c).loc[index, 'bus'])
                v_dep_buses = np.unique(np.append(v_dep_buses, ctrl_buses))

            if need_for_priority:

                prioritized, oltc_conv, prev_priority = prioritization(
                    dict_controlled_index, control, v_diff, x_tol_outer,
                    oltc_conv, ctrl_buses, tap_changed, n_trials, prev_priority)

            # apply non voltage dependent controllers in the first trial
            if (control == "fixed_cosphi" and n_trials == 1):
                apply_fixed_cosphi(n, now, c, index)

            elif (control == "cosphi_p" and n_trials == 1):
                apply_cosphi_p(n, now, c, index)

            # apply voltage dependent controller at each iteration if prioritized
            elif (control == "q_v" and prioritized):
                apply_q_v(n, now, c, index, n_trials_max, n_trials)

            elif (control == "p_v" and prioritized):
                apply_p_v(n, now, c, index, n_trials_max, n_trials)

            # apply on load tap changer transforner if prioritized
            elif (control == "oltc" and prioritized and n_trials > 1):
                n_iter_oltc, tap_changed = apply_oltc(
                     n, now, index, calculate_Y, sub_network, skip_pre, n_iter_oltc)
                # break the outer loop (oltc_conv) based on transformer convergence
                # if need_for_priority is false. 
                oltc_conv = np.where(need_for_priority, oltc_conv, not tap_changed)

    # find the v_mag_pu of buses with v_dependent controller to return
    old_v_mag_pu = n.buses_t.v_mag_pu.loc[now, v_dep_buses]

    return old_v_mag_pu, oltc_conv, n_iter_oltc, tap_changed, prev_priority


def _set_controller_outputs_to_n(n, c, index, snapshot, ctrl_p_out=False,
                                 ctrl_q_out=False, p_out=None, q_out=None):
    """
    Set the controller outputs to the n (network). The controller outputs
    "p_out" and or "q_out" are set to buses_t.p or buses_t.q and component_t.p
    or component_t.q dataframes.

    Parameter:
    ----------
    n : pypsa.components.Network
        Network
    c : string
        Component name, i.e. 'Load', 'StorageUnit'...
    index : indexes of controlled elements
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    ctrl_p_out : bool default to False
        If ``True``, meaning that p_set is changed by controller due to reactive
        need and controller gives the effective p_out which needs to be set in
        power flow outputs.
    ctrl_q_out : bool default to False
        If ``True``, If controller has reactive power output then this flage
        activates in order to set the controller reactvie power output to the
        network.
    p_out : numpy array
        Active power output of the controller. note: "q_v" and "fixed_cosphi"
        have active power outputs only when a portion of active power is converted
        to reactive power due to reactive power need.
    q_out : numpy array defaut to None
        Reactive power output of the controller
        This behavior is in apply_cosphi and apply_q_v methods.

    Returns
    -------
    None
    """
    # input power before applying controller output to the network
    p_input = n.pnl(c).p.loc[snapshot, index]
    q_input = n.pnl(c).q.loc[snapshot, index]

    # empty dictrionary and adding attribute values to it in each snapshot
    p_q_dict = {}
    if ctrl_p_out:
        p_q_dict['p'] = p_out
    if ctrl_q_out:
        p_q_dict['q'] = q_out

    # setting p_out, q_out to component_t.(p or q) dataframes
    for attr in p_q_dict.keys():
        n.pnl(c)[attr].loc[snapshot, index] = p_q_dict[attr]

        # Finding the change in p and q for the connected buses
        if attr == 'q':
            power_change = -((q_input - n.pnl(c).q).loc[
                    snapshot, index] * n.df(c).loc[
                            index, 'sign']).groupby(n.df(c).loc[
                                    index, 'bus']).sum()

        if attr == 'p':
            power_change = -((p_input - n.pnl(c).p).loc[snapshot, index] *
                             n.df(c).loc[index, 'sign']).groupby(
                                 n.df(c).loc[index, 'bus']).sum()

        # adding the p and q change to the controlled buses
        n.buses_t[attr].loc[snapshot, power_change.index] += power_change


def prepare_controlled_index_dict(
        n, sub_network, inverter_control, snapshots, oltc_control):
    """
    For components of type "Transformer", "Generator", "Load", "Store" and
    "StorageUnit" collect the indices of controlled elements in the dictionary
    of dictionaries dict_controlled_index:
        - Any exisitng control strategy will be set as a key of dict_controlled_index
        - Each of these keys holds a dictionary as value, with:
            - the types of components it is enabled for as Keys
            - and the related indices of the components as values.
    If a "q_v" or 'p_v' controller is present, n_trial_max is set to 30
    which enables the outer loop of the power flow and sets the maximum allowed
    number of iterations.
    The returned dictionary is used in apply_controller().

    Parameter:
    ----------
    n : pypsa.components.Network
        Network
    sub_network : pypsa.components.Network.sub_network
        network.sub_networks.
    inverter_control : bool, default False
        If ``True``, activates outerloop which applies inverter control strategies
        (control strategy chosen in n.components.control_strategy) in the power flow.
    snapshots : list-like|single snapshot
        A subset or an elements of network.snapshots on which to run
        the power flow, defaults to network.snapshots
    oltc_control : bool, default False
        If ``True``, activates outerloop which considers on load tap changer
        (oltc) transformer control on those transformers which their "oltc"
        attribute is activated (True).

    Returns
    -------
    n_trials_max : int
        Shows the maximum allowed power flow iteration for convergance of voltage
        dependent controllers.
    dict_controlled_index : dictionary
        dictionary that contains each controller as key and controlled indexes
        as values.
    oltc_control : bool
        If True activates appliation apply_controller function in pf.py file
        which results to appliation of oltc controller, but in case of multiple
        subnetworks if any sub network does not contain controlled tranformer
        then oltc_control should be forced to False to avoid application of 
        unncessary oltc action.
    inverter_control : bool
        If True activates appliation apply_controller function in pf.py file
        which results to appliation of inverter controller, but in case of multiple
        subnetworks if any sub network does not contain controlled elements
        then inverter_control should be forced to False to avoid application of 
        unncessary inverter control action.
    """
    dict_controlled_index = {}
    ctrl_list = ['', 'q_v', 'p_v', 'cosphi_p', 'fixed_cosphi']
    if oltc_control:# res = [int(i) for i in test_string.split() if i.isdigit()]
        # extract sub_network integer from sub_network as 'sub_n'
        sub_n = int(re.search(r'\d+', str(sub_network)).group())
        
        if n.transformers.loc[n.transformers[
                n.transformers.sub_network == str(sub_n)].index, 'oltc'].any():

            # as controlled transformer take the one which is connected to sub_n
            ctr_index = n.transformers[n.transformers.sub_network == str(sub_n)].index

            dict_controlled_index['oltc'] = {}
            dict_controlled_index['oltc']['Transformer'] = ctr_index

            # TODO let opt tap position be done by pf.py file (line below)
            n.pnl('Transformer')['opt_tap_position'] = n.pnl('Transformer')[
                                 'opt_tap_position'].reindex(columns=ctr_index)

            # TODO: Hey @Jkaehler: one assertion both for oltc control and inverter control didnt work  so i made a function which is called in both spots, hope it is ok?
            # check if the entered control name spelling are correct
            check(dict_controlled_index, ctrl_list[1:5])
            # check if controlled buses are in the same subnetwork as the transformer
            check_ctrl_bus_existance_in_trafo_sub_n(n, ctr_index)
        else:
            # oltc loop should break if some sub_networks are not controlled
            oltc_control = False

    if inverter_control:
        # deactivate the outerloop in pf.py incase no controlled element exist, (it is updated below).
        inverter_control = False
        # loop through loads, generators, storage_units and stores if they exist
        for c in sub_network.iterate_components(n.controllable_one_port_components):
            if (c.df.loc[c.ind].control_strategy != '').any():

                inverter_control = True
                assert (c.df.loc[c.ind].control_strategy.isin(ctrl_list)).all(), (
                        "Not all given types of controllers are supported. "
                        "Elements with unknown controllers are:\n%s\nSupported "
                        "controllers are : %s." % (c.df.loc[c.ind].loc[
                            (~ c.df.loc[c.ind]['control_strategy'].isin(ctrl_list)),
                            'control_strategy'], ctrl_list[1:5]))

                # exclude slack generator to be controlled
                if c.list_name == 'generators':
                    c.df.loc[c.ind].loc[
                        c.df.loc[c.ind].control == 'Slack', 'control_strategy'] = ''
                # if voltage dep. controller exist,find the bus name

                for i in ctrl_list[1:5]:
                    # building a dictionary for each controller if they exist
                    if (c.df.loc[c.ind].control_strategy == i).any():
                        if i not in dict_controlled_index:
                            dict_controlled_index[i] = {}

                        dict_controlled_index[i][c.name] = c.df.loc[c.ind].loc[(
                                c.df.loc[c.ind].control_strategy == i)].index

                logger.info("We are in %s. These indexes are controlled:\n%s",
                            c.name, dict_controlled_index)
        check(dict_controlled_index, ctrl_list[1:5])

    n_trials_max = np.select([("q_v" in dict_controlled_index or "p_v"
                               in dict_controlled_index), 'oltc' in
                              dict_controlled_index], [50, 6], default = 1)

    dict_controlled_index = collections.OrderedDict(sorted(dict_controlled_index.items()))

    return n_trials_max, dict_controlled_index, oltc_control, inverter_control


def check(dict_controlled_index, ctrl_list):
    """
    If the user activates Inverter_control or oltc_control flags but forgets
    to activate the controller inside the component then the assertion below
    will break the power flow.
    """
    ctrl_list.append('oltc')
    assert (bool(dict_controlled_index)), (
        "Control loop is activated but no component is controlled,"
        " please choose the control_strategy in the desired "
        " component (Load, Generator, Store, StorageUnit, Transformer) indexes."
        " Supported type of controllers are:\n%s "
        "Where 'oltc' is specific for Transformer component."
        % (ctrl_list))


def check_ctrl_bus_existance_in_trafo_sub_n(n, ctr_index):
    """
    Check if the givien buses as controlled buses for oltc operation do exist
    in the same sub_network where transformer is located, if not stop the power
    flow.
    """
    for ind in ctr_index:
        row = n.transformers.loc[ind]

        current_tap = row['tap_position']
        # extracting controlled nodes names
        buses = [x.strip() for x in n.transformers.loc[ind, 'ctrl_buses'].split(',')]

        # if no node is chosen take node of secondary of trafo as control node
        ctrl_buses = np.where(
                len(n.transformers.loc[ind, 'ctrl_buses']) == 0, [row['bus1']], buses)
        if row['ctrl_buses'] != '' and row['ctrl_buses'] != 'All':
            assert (n.buses.loc[ctrl_buses, 'sub_network'] == n.transformers.loc[
                ind, 'sub_network']).all(), ("Not all the controlled buses %s, exist"
                " in the same sub_network of '%s' transformer, please choose the "
                "controlled buses withing the same sub network." % (ctrl_buses, ind))
