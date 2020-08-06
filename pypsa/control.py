"""importing important libraries."""
from .descriptors import get_switchable_as_dense
import logging
import numpy as np
logger = logging.getLogger(__name__)


def find_allowable_q(p, power_factor, s_nom):
    """
    Some times the reactive power that controller want to compensate using
    (p*tan(arccos(power_factor))) can go higher than what inverter can provide
    "q_inv_cap" based on the "power_factor" provided. for this purpose this
    fucntion calculates the reactive power "q" that controller want to compensate,
    and also the inverter reactive power capacity  "q_inv_cap". Then "q" is checked,
    if it is less than "q_inv_cap" take it, and for "q" values higher than "q_inv_cap"
    take "q_inv_cap" instead and name this new selection "q_allowable". finally
    return them all to the controller for further calculations(see controllers).
    This is done to make sure that the inverter equation s_nom = np.sqrt((p**2 + q**2)
    is not violated.
    """
    # Calculate reactive power that controller want ot compensate initially
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
    when compensated reactive power "q" by controller together with the generation
    "p" violates inverter equation s_nom = np.sqrt((p**2 + q**2), in this case
    controller needs to reduce p in order to fulfil reactive power need. In this
    case p is adjusted to "new_p_set" here and return it to the controller.
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


def apply_fixed_cosphi(n, snapshot, c, c_attrs):
    """
    fix power factor inverter controller.
    This controller provides a fixed amount of reactive power compensation to the
    grid as a function of the amount of injected power (p_set) and the chosen
    power factor value.

    reference : https://ieeexplore.ieee.org/document/6096349
    DOI link  : 10.1109/JPHOTOV.2011.2174821

    Parameters
    ----------
    n : pypsa.components.Network
        Network
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    c_attrs : pandas data frame
        Component attrs of controlled indexes, i.e. power_factor choice for
        generators, loads...
    c : string
        Component name, i.e. 'Load', 'StorageUnit'...

    Returns
    -------
    None
    """
    # needed parameters
    p_input = n.pnl(c).p.loc[snapshot, c_attrs.index]
    power_factor = c_attrs['power_factor']
    s_nom = c_attrs['s_nom']
    p_out=None
    ctrl_p_out = False
    q_inv_cap, q_allowable, q = find_allowable_q(p_input, power_factor, s_nom)
    q_out = -q_allowable

    # check if the calculated q is not exceeding the inverter capacity if yes then
    # decrease p_input in order not to violate s_nom = np.sqrt((p**2 + q**2) .
    if (abs(q) > q_inv_cap).any().any():
        ctrl_p_out = True
        p_out = adjust_p_set(s_nom, q_out, p_input, c, 'fixed_cosphi')

    _set_controller_outputs_to_n(n, c, c_attrs, snapshot, ctrl_p_out=ctrl_p_out,
                                 ctrl_q_out=True, p_out=p_out, q_out=q_out)


def apply_cosphi_p(n, snapshot, c, c_attrs):
    """
    Power factor as a function of active power (cosphi_p) controller.
    This controller provides reactive power compensation to the grid only when
    the amount of generated power (p_set) is more than a specific value (p_ref).
    controller chooses a variable power factor for reactive power calculation
    based on the amount of generation and the provided droop for power factor
    selection. Therefore for all generations less than p_ref no reactive power
    support is provided.

    reference : https://ieeexplore.ieee.org/document/6096349.
    DOI link  : 10.1109/JPHOTOV.2011.2174821

    Parameters
    ----------
    n : pypsa.components.Network
        Network
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    c_attrs : pandas data frame
        Component attrs of controlled indexes, i.e. power_factor choice for
        generators, loads...
    c : string
        Component name, i.e. 'Load', 'StorageUnit'...

    Returns
    -------
    None
    """
    # parameters needed
    set_p1 = c_attrs['set_p1']
    set_p2 = c_attrs['set_p2']
    s_nom = c_attrs['s_nom']
    p_input = n.pnl(c).p.loc[snapshot, c_attrs.index]
    power_factor_min = c_attrs['power_factor_min']
    p_set_per_p_ref = (abs(p_input) / c_attrs['p_ref'])*100

    # choice of power_factor according to controller inputs and its droop curve
    power_factor = np.select([(p_set_per_p_ref < set_p1), (
        p_set_per_p_ref >= set_p1) & (p_set_per_p_ref <= set_p2), (
            p_set_per_p_ref > set_p2)], [1, (1 - ((1 - power_factor_min) / (
             set_p2 - set_p1) * (p_set_per_p_ref - set_p1))), power_factor_min])

    # find q_set and avoid -0 apperance as the output when power_factor = 1
    q_out = np.where(power_factor == 1, 0, -p_input.mul(np.tan((np.arccos(
                          power_factor, dtype=np.float64)), dtype=np.float64)))

    S = np.sqrt((p_input)**2 + q_out**2)
    assert ((S < s_nom).any().any()), (
        "The resulting reactive power (q)  while using 'cosphi'_p control  "
        "with the chosen attr 'power_factor_min' in '%s' component results a  "
        "complex power (S = sqrt(p**2 + q**2))) which is greater than 's_nom') "
        "of the inverter, please choose the right power_factor_min value"
        % (c))
    n.pnl(c)['power_factor'].loc[snapshot, c_attrs.index] = power_factor

    _set_controller_outputs_to_n(
        n, c, c_attrs, snapshot, ctrl_q_out=True, q_out=q_out)


def apply_q_v(n, snapshot, c, c_attrs, n_trials_max, n_trials):
    """
    Reactive power as a function of voltage Q(V).
    This contrller controller provide reactive power compensation based on the
    voltage information of the bus where inverter is connected, for this purpose
    the droop for reactive power calculation is divided in to 5 different reactive
    power calculation. v1, v2, v3, v4 attrs form the droop and the reactive power
    is calculated based on where the the bus v_mag_pu is landing, as it is done
    here in "curve_q_set_in_percentage"

    reference : https://ieeexplore.ieee.org/document/6096349
    DOI link  : 10.1109/JPHOTOV.2011.2174821 

    Parameters:
    ----------
    n : pypsa.components.Network
        Network
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    c_attrs : pandas data frame
        Component attrs of controlled indexes, i.e. power_factor choice for
        generators, loads...
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
                       " avoid the problem." % (snapshot, c_attrs.index.values))
    #  curve parameters
    v_pu_bus = n.buses_t.v_mag_pu.loc[snapshot, c_attrs.loc[c_attrs.index, 'bus']].values
    v1 = c_attrs['v1']
    v2 = c_attrs['v2']
    v3 = c_attrs['v3']
    v4 = c_attrs['v4']
    s_nom = c_attrs['s_nom']
    power_factor = c_attrs['power_factor']
    p_input = n.pnl(c).p.loc[snapshot, c_attrs.index]
    p_out = None
    ctrl_p_out = False
    q_inv_cap, q_allowable, q = find_allowable_q(p_input, power_factor, s_nom)

    # calculation of maximum q compensation in % based on bus v_pu_bus
    curve_q_set_in_percentage = np.select([(v_pu_bus < v1), (v_pu_bus >= v1) & (
            v_pu_bus <= v2), (v_pu_bus > v2) & (v_pu_bus <= v3), (v_pu_bus > v3)
        & (v_pu_bus <= v4), (v_pu_bus > v4)], [100, 100 - 100 / (v2 - v1) * (
                v_pu_bus - v1), 0, -100 * (v_pu_bus - v3) / (v4 - v3), -100])
    # calculation of q
    q_out = (((curve_q_set_in_percentage * q_allowable) / 100) * c_attrs[
            'damper'] * c_attrs['sign'])

    # check if there is need to reduce p_set due to q need
    if (q > q_inv_cap).any().any():
        ctrl_p_out = True
        p_out = adjust_p_set(s_nom, q, p_input, c, 'q_v')

    _set_controller_outputs_to_n(n, c, c_attrs, snapshot, ctrl_p_out=ctrl_p_out,
                                 ctrl_q_out=True, p_out=p_out, q_out=q_out)


def apply_oltc(n, snapshot, c_attrs, calculate_Y, sub_network, skip_pre):
    """
    On Load Tap Changer Transformer.

    Parameters
    ----------
    n : pypsa.components.Network
        Network.
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    c_attrs : pandas data frame
        Component attrs of controlled transformers
    calculate_Y : function
        it calculate admmittance matrix.
    sub_network : pypsa.components.Network.sub_network
        network.sub_networks.
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating dependent
        values and finding bus controls.

    Returns
    -------
    None.

    """
    calc_admittance_matrix = True
    for ind, row in c_attrs.iterrows():
        opt_tap = row['tap_position']  # initial value for tap position
        # extracting controlled nodes names
        nodes = [x.strip() for x in n.transformers['ctrl_nodes'][0].split(',')]
        # if no node is chosen take node of secondary of trafo as control node
        ctrl_nodes = np.where(
                len(n.transformers['ctrl_nodes'][0]) == 0, [row['bus1']], nodes)
        # find voltages of controlled nodes
        v_pu_ctrl_nodes = n.buses_t.v_mag_pu.loc[snapshot, ctrl_nodes]
        # find taps and tap_steps
        if row['type'] != '':
            taps = (np.arange(n.transformer_types.loc[row['type'], 'tap_min'],
                             n.transformer_types.loc[row['type'], 'tap_max']+1))

            tap_side = n.transformer_types.loc[row['type'], 'tap_side']
            tap_step = n.transformer_types.loc[row['type'], 'tap_step']

        else:
            taps = np.arange(row['tap_min'], row['tap_max']+1)
            tap_side = row['tap_side']
            tap_step = row['tap_step']

        tap_side_cons = np.where(tap_side == 0, 1, -1)
        # Single node oltc control part:
        if len(ctrl_nodes) == 1:

            deadband_range = row['v_set'] + np.array([row['deadband']/100, -row[
                    'deadband']/100])

            value_in_range = (v_pu_ctrl_nodes.values >= min(
              deadband_range) and v_pu_ctrl_nodes.values <= max(deadband_range))

            if value_in_range:
                logger.info(" The voltage in node '%s' controlled by oltc in %s"
                            " is already within the deadband range. ",
                            v_pu_ctrl_nodes.index, ind)
                calc_admittance_matrix = False

            else:
                possible_tap_res = abs(row['v_set']-v_pu_ctrl_nodes.values +
                                       taps*tap_step/100*row['v_set'])
                opt_tap = taps[np.where(possible_tap_res == min(
                    possible_tap_res))][0]*tap_side_cons

                calc_v_pu = v_pu_ctrl_nodes.values - opt_tap * tap_side_cons * tap_step*row['v_set']/100
                if (calc_v_pu >= min(
                        deadband_range) and calc_v_pu <= max(deadband_range)):
                    logger.info("The voltage in %s controlled by oltc in %s, "
                                "using %s as the optimum tap position is now "
                                "withing the deadband range.",
                                v_pu_ctrl_nodes.index.tolist(), ind, opt_tap)
                else:
                    logger.warning("Due to oltc tap position limits Voltage in "
                                   "node %s controlled by oltc in %s, could not"
                                   " set within the deadband range, %s is "
                                   "used as the optimum possible tap position,",
                                   v_pu_ctrl_nodes.index.tolist(), ind, opt_tap)

        # Multiple node oltc control part:
        elif len(ctrl_nodes) > 1:
            # find the mean and max voltages of the controlled nodes
            meas_max = v_pu_ctrl_nodes.values.max()
            meas_min = v_pu_ctrl_nodes.values.min()
            # check if meas_max and meas_min are withing the range
            if (meas_min > row['v_min'] and meas_max < row['v_max']):
                logger.info(" Voltage in nodes %s controlled by oltc in  %s are"
                            " already withing 'v_min' and 'v_max' ranges.",
                            v_pu_ctrl_nodes.index.tolist(), ind)
                calc_admittance_matrix = False

            # if they are not withing the range then find optimum tap as follow:
            else:
                max_voltage = meas_max-taps*tap_step*row['v_set']/100
                min_voltage = meas_min-taps*tap_step*row['v_set']/100
                opt_ind = np.where(((min_voltage > row['v_min']) & (
                        max_voltage < row['v_max'])))[0]

                if len(opt_ind) != 0:
                    opt_tap = taps[opt_ind[0]]*tap_side_cons
                else:
                    opt_ind = np.where(min_voltage > row['v_min'])[0]
                    if len(opt_ind) != 0:
                        opt_tap = taps[len(opt_ind)-1]*tap_side_cons

                    else:
                        opt_tap = taps[0]*tap_side_cons

                logger.info("The voltage in %s controlled by oltc in %s, using "
                            " %s as the optimum tap position.",
                            v_pu_ctrl_nodes.index.tolist(), ind, opt_tap)

        # set the optimum tap position calculated either from single or multiple
        # node part, and also recalculte admittance matrix.
        if calc_admittance_matrix:
            n.transformers.loc[ind, 'tap_position'] = opt_tap
            n.transformers_t.opt_tap_position.loc[snapshot, ind] = opt_tap
            if row['type'] == '':
                n.transformers.loc[ind, 'tap_ratio'] = row[
                    'tap_ratio'] + (opt_tap * (tap_step / 100))

            calculate_Y(sub_network, skip_pre=False)


def apply_p_v(n, snapshot, c, c_attrs, n_trials_max, n_trials):
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
    c_attrs : pandas data frame
        Component attrs of controlled indexes, i.e. power_factor choice for
        generators, loads...
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
                       " avoid the problem." % (snapshot, c_attrs.index.values))

    v_pu_bus = n.buses_t.v_mag_pu.loc[
                           snapshot, c_attrs.loc[c_attrs.index, 'bus']].values

    p_input = get_switchable_as_dense(
                               n, c, 'p_set', inds=c_attrs.index).loc[snapshot]

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

            ind_con = c_attrs.loc[p_con.index]  # filtered indexes
            v_pu_bus_con = n.buses_t.v_mag_pu.loc[
                snapshot, ind_con.loc[ind_con.index, 'bus']].values

            calculate_cosumption_p(v_pu_bus_con, ind_con, p_con, c, n, snapshot)

        # injection part
        if grid_injection:
            # filter inputs and indexes of grid injection indexes
            if c == 'Load':
                p_inj = p_input[p_input < 0]
            else:
                p_inj = p_input[p_input > 0]

            ind_inj = c_attrs.loc[p_inj.index]  # filtered indexes
            v_pu_bus = n.buses_t.v_mag_pu.loc[
                snapshot, ind_inj.loc[ind_inj.index, 'bus']].values

            calculate_injection_p(v_pu_bus, ind_inj, p_inj, c, n, snapshot)

    # if only grid consumption exis, i.e. battery charing or loads...
    elif grid_consumption:
        calculate_cosumption_p(v_pu_bus, c_attrs, p_input, c, n, snapshot)
    # if only grid injection exist, i.e. generators, battery discharging...
    elif grid_injection:
        calculate_injection_p(v_pu_bus, c_attrs, p_input, c, n, snapshot)


def calculate_cosumption_p(v_pu_bus, c_attrs, p_input, c, n, snapshot):
    """
    This method is called from "p_v" controller when any of the controlled
    components is consuming active power from the grid. Given are bus voltages
    "v_pu_bus", all attributes and indexes "c_attrs", active power input "p_input"
    list name "c" of the such indexes. Using these data the amount of allowed
    power after it is curtailed is calculted in percentag using "pperpmax" which
    is the droop that contains multiple condition and choices. "pperpmax" in in
     %, it is multiplied to "p_input" to get the amount of allowed active power
    consumption in MW.
    """
    # required parameters
    v_pu_cr = c_attrs['v_pu_cr']
    v_max_curtail = c_attrs['v_max_curtail']
    # find the amount of allowed power consumption in % from the droop
    pperpmax = np.select([(v_pu_bus > v_pu_cr), (v_pu_bus < v_max_curtail), (
        (v_pu_bus >= v_max_curtail) & (v_pu_bus <= v_pu_cr))], [100, 0, (100/(
            v_pu_cr - v_max_curtail)*(v_pu_bus - v_max_curtail))])

    # find the amount of allowed power consumption in MW
    p_out = ((pperpmax*(p_input))/100)
    # update the active power contribution of the controlled indexes in network
    _set_controller_outputs_to_n(
        n, c, c_attrs, snapshot, ctrl_p_out=True, p_out=p_out)

def calculate_injection_p(v_pu_bus, c_attrs, p_input, c, n, snapshot):
    """
    This method is called from "p_v" controller when any of the controlled
    components is injecting active power to the grid. Given are bus voltages
    "v_pu_bus", all attributes and indexes "c_attrs", active power input "p_input"
    list name "c" of the such indexes. Using these data the amount of allowed
    power after it is curtailed is calculted in percentag using "pperpmax" which
    is the droop that contains multiple condition and choices. "pperpmax" is in
     %, it is multiplied to "p_input" to get the amount of allowed active power
    injection in MW.
    """
    # required parameters
    v_pu_cr = c_attrs['v_pu_cr']
    v_max_curtail = c_attrs['v_max_curtail']
    # find the amount of allowed power consumption in % from the droop
    pperpmax = np.select([(v_pu_bus < v_pu_cr), (v_pu_bus > v_max_curtail), (
        (v_pu_bus >= v_pu_cr) & (v_pu_bus <= v_max_curtail))], [
            100, 0, (100-(100/(v_max_curtail-v_pu_cr))*(v_pu_bus-v_pu_cr))])

    # find the amount of allowed power consumption in MW
    p_out = ((pperpmax*(p_input)) / 100)
    # update the active power contribution of the controlled indexes in network
    _set_controller_outputs_to_n(
        n, c, c_attrs, snapshot, ctrl_p_out=True, p_out=p_out)


def apply_controller(n, now, n_trials, n_trials_max, dict_controlled_index,
                     voltage_difference, x_tol_outer, i, oltc_control,
                     calculate_Y, sub_network, skip_pre):
    """
    Iterate over chosen control strategies which exist as keys in dict_controlled_index
    and call each, to be applied to the controlled indexes which are also inside
    each controller (key) as values that contain controlled indexes of controlled
    component with all their attributes. And return the bus names that contain
    "q_v" controller attached for voltage difference comparison purpose in pf.py.

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
        It is a dynamic dictionary that contains each controller as a key (if
        applied in any component) and each controlled component (filtered to
        only controlled indexes) as values.
    voltage_difference : pandas series
        Voltage difference between the two iterations of the bus voltages that
        are controlled with voltage dependent controllers such as "p_v" or "q_v".
    x_tol_outer : float
        Tolerance for outer loop voltage difference between the two successive
        power flow iterations as a result of applying voltage dependent controller
        such as reactive power as a function of voltage "q_v".
    i : integer
        snapshot index which starts from zero.
    oltc_control : bool, default False
        If ``True``, activates outerloop which considers on load tap changer
        (oltc) transformer control on those transformers which their "oltc"
        attribute is activated (True).
    calculate_Y : function
        it calculate admmittance matrix.
    sub_network : pypsa.components.Network.sub_network
        network.sub_networks.
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating dependent
        values and finding bus controls.
                                                         
    Returns
    -------
    v_mag_pu of voltage_dependent_controller : pandas data frame
        Needed to compare v_mag_pu of the controlled buses with the voltage from
        previous iteration to decide for repeation of pf (in pf.py file).
    oltc : bool
        Deactivate outerloop repeatation for the next iteration after oltc is applied
    """
    oltc = oltc_control
    v_dep_buses = np.array([])
    for controller in dict_controlled_index.keys():
        # parameter is the controlled indexes dataframe of a components
        for c, c_attrs in dict_controlled_index[controller].items():

            # call each controller
            if (controller == 'fixed_cosphi') and (n_trials == 1):
                apply_fixed_cosphi(n, now, c, c_attrs)

            elif (controller == 'cosphi_p') and (n_trials == 1):
                apply_cosphi_p(n, now, c, c_attrs)

            elif ((controller == 'q_v') and (voltage_difference > x_tol_outer)):
                v_dep_buses = np.append(v_dep_buses, np.unique(c_attrs.loc[(
                    c_attrs.control_strategy.isin(["q_v", "p_v"])), 'bus']))
                apply_q_v(n, now, c, c_attrs, n_trials_max, n_trials)

            elif ((controller == 'p_v') and (voltage_difference > x_tol_outer)):
                apply_p_v(n, now, c, c_attrs, n_trials_max, n_trials)
                v_dep_buses = np.append(v_dep_buses, np.unique(c_attrs.loc[(
                    c_attrs.control_strategy.isin(["q_v", "p_v"])), 'bus']))

            elif controller == 'oltc':
                # bool to check if non v dependen controllers are applied
                non_v_dep_ctr = any([ctr in dict_controlled_index for ctr in [
                        'cosphi_p', 'fixed_cosphi']])

                # switch gives the priority to inverter controller to be
                # converged (per load flow) and then oltc takes action.
                switch = np.select([n_trials_max > 0, (
                        non_v_dep_ctr or (not non_v_dep_ctr) & (i == 0)), (
                                (not non_v_dep_ctr) & (i > 0))], [
                 voltage_difference < x_tol_outer, n_trials > 1, n_trials > 0])

                if switch == 1:
                    apply_oltc(n, now, c_attrs, calculate_Y, sub_network, skip_pre)
                    oltc = 0
    # find the v_mag_pu of buses with v_dependent controller to return
    v_mag_pu_voltage_dependent_controller = n.buses_t.v_mag_pu.loc[
            now, v_dep_buses]
    
    return v_mag_pu_voltage_dependent_controller, oltc


def _set_controller_outputs_to_n(n, c, c_attrs, snapshot, ctrl_p_out=False,
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
    c_attrs : pandas data frame
        Component attrs of controlled indexes, i.e. power_factor choice for
        generators, loads...
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
    p_input = n.pnl(c).p.loc[snapshot, c_attrs.index]
    q_input = n.pnl(c).q.loc[snapshot, c_attrs.index]

    # empty dictrionary and adding attribute values to it in each snapshot
    p_q_dict = {}
    if ctrl_p_out:
        p_q_dict['p'] = p_out
    if ctrl_q_out:
        p_q_dict['q'] = q_out

    # setting p_out, q_out to component_t.(p or q) dataframes
    for attr in p_q_dict.keys():
        n.pnl(c)[attr].loc[snapshot, c_attrs.index] = p_q_dict[attr]

        # Finding the change in p and q for the connected buses
        if attr == 'q':
            power_change = -((q_input - n.pnl(c).q).loc[
                    snapshot, c_attrs.index] * c_attrs.loc[
                            c_attrs.index, 'sign']).groupby(c_attrs.loc[
                                    c_attrs.index, 'bus']).sum()

        if attr == 'p':
            power_change = -((p_input - n.pnl(c).p).loc[snapshot, c_attrs.index] *
                             c_attrs.loc[c_attrs.index, 'sign']).groupby(
                                 c_attrs.loc[c_attrs.index, 'bus']).sum()

        # adding the p and q change to the controlled buses
        n.buses_t[attr].loc[snapshot, power_change.index] += power_change


def prepare_controlled_index_dict(n, sub_network, inverter_control, snapshots, oltc_control):
    """
    Iterate over "Generator", "Load", "Store" and "StorageUnit" to check if they
    have inverter control strategy applied in any of their indexes and check
    if any oltc is activated in any transformer. If yes the name of control
    strategy will be set as key of the dictionary and the name of the controlled
    component will be as values which will contain the controlled indexes with
    their respective attributes. While preparing the dictionary if any "q_v"
    controller is used by any component n_trial_max is chosen 30 which is the
    maximum power flow trials for this controller to be converged. In the end it
    returns the dictionary which is then used in "apply_controller" fucntion
     to apply the chosen controllers to the chosen compoenents in the power flow.

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
    """
    n_trials_max = 0
    dict_controlled_index = {}
    ctrl_list = ['', 'q_v', 'p_v', 'cosphi_p', 'fixed_cosphi']
    if oltc_control:
        assert ((n.transformers.oltc).any()), (
             "On-load tap changer (oltc) power flow is activated but oltc inside"
             " the transformer is not activated, please activate oltc in the"
             " desired trasformer and then run the power flow.")

        if (n.transformers.oltc).any():
            dict_controlled_index['oltc'] = {}
            dict_controlled_index['oltc']['Transformer'] = n.transformers.loc[
                                                 (n.transformers.oltc == 1)]
            ctr_index = (n.transformers.oltc == 1).index
            n.pnl('Transformer')['opt_tap_position'] = n.pnl('Transformer')[
                                 'opt_tap_position'].reindex(columns=ctr_index)

    if inverter_control:
        # loop through loads, generators, storage_units and stores if they exist
        for c in sub_network.iterate_components(n.controllable_one_port_components):

            if (c.df.control_strategy != '').any():
                assert (c.df.control_strategy.isin(ctrl_list)).all(), (
                        "Not all given types of controllers are supported. "
                        "Elements with unknown controllers are:\n%s\nSupported "
                        "controllers are : %s." % (c.df.loc[(~ c.df[
                            'control_strategy'].isin(ctrl_list)),
                            'control_strategy'], ctrl_list[1:5]))

                if (c.df.control_strategy == 'cosphi_p').any():
                    # transfering power factors to n.component_t.power_factor
                    power_factor = get_switchable_as_dense(
                        n, c.name, 'power_factor', snapshots, c.ind)
                    c.pnl.power_factor = c.pnl.power_factor.reindex(columns=c.ind)
                    c.pnl['power_factor'].loc[snapshots, c.ind] = power_factor

                # exclude slack generator to be controlled
                if c.list_name == 'generators':
                    c.df.loc[c.df.control == 'Slack', 'control_strategy'] = ''
                # if voltage dep. controller exist,find the bus name
                n_trials_max = np.where(
                      c.df.control_strategy.isin(['q_v', 'p_v']).any(), 30, 0)

                for i in ctrl_list[1:5]:
                    # building a dictionary for each controller if they exist
                    if (c.df.control_strategy == i).any():
                        if i not in dict_controlled_index:
                            dict_controlled_index[i] = {}

                        dict_controlled_index[i][c.name] = c.df.loc[(
                                c.df.control_strategy == i)]

                logger.info("We are in %s. These indexes are controlled:\n%s",
                            c.name, dict_controlled_index)

        assert (bool(dict_controlled_index)), (
                "inverter_control is activated but no component is controlled,"
                " please choose the control_strategy in the desired "
                " component indexes. Supported type of controllers are:\n%s."
                % (ctrl_list[1:5]))

    return n_trials_max, dict_controlled_index
