"""importing important libraries."""
from .descriptors import get_switchable_as_dense
import logging
import numpy as np
logger = logging.getLogger(__name__)


def find_allowable_q(p, power_factor, s_nom):
    """
    Some times the reactive power that controller want to compensate using
    (p*tan(arccos(power_factor))) can go higher than what inverter can provide
    "q_inv_cap" based on the "power_factor" provided. for this purpose this fucntion
    calculates the reactive power "q" that controller want to compensate, and
    also the inverter reactive power capacity  "q_inv_cap". Then "q" is checked,
    if it is less than "q_inv_cap" take it, and for "q" values higher than "q_inv_cap"
    take "q_inv_cap" instead and name this new selection "q_allowable". finally
    return them all to the controller for further calculations(see controllers).
    This is done to make sure that the inverter equation s_nom = np.sqrt((p**2 + q**2)
    is not violated.
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
    reference : https://www.academia.edu/24772355/

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

    Example: if we have two controlled component c in  our subnetwork as below:
    --------
    >>> network.add(c, "PV1", bus="LV2 bus", p_set=1, power_factor=0.9, s_nom=1.5,
                   control_strategy="fixed_cosphi")

    >>> network.add(c, "PV2", bus="LV2 bus", p_set=0.7, power_factor=0.95, s_nom=1,
                   control_strategy="fixed_cosphi")

    >>>"c_attrs" is the filtered dataframe of c, where it contain "PV1" and PV2 as
        indexes and all attributes of c for "PV1" and PV2 as columns.
        controller uses the parameters and attributes in "c_attrs" as input to
        "find_allowable_q" function to calculate "q_inv_cap", "q_allowable" and
        "q". "q_allowable" is the controller output and is sent to
        "_set_controller_outputs_to_n" as "q_out". "q_allowable" and "q" are used
        to check if "q_allowable" does not contain any value that is exceeding
        the maximum reactive power capacity of the inverter "q_inv_cap", if yes,
        p_input will be reduced (in order not to exceed s_nom) using "adjust_p_set"
        function and the "new_p_set" will be also sent to "_set_controller_outputs_to_n".
        i.e. q for PV1 insdie "find_allowable_q": q=1*np.tan((np.arccos(0.9)))=0.4843
        q_inv_cap=1.5*np.tan((np.arccos(0.9)))=0.726. Now we see that 0.4843 is way
        less than q_inv_cap, therefore, q_out=0.4843 is the output and no need to
        adjust active power. Note: adjusting active power due to reactive power
        need is the worse case scenario.
    """
    # needed parameters
    p_input = n.pnl(c).p.loc[snapshot, c_attrs.index]
    power_factor = c_attrs['power_factor']
    s_nom = c_attrs['s_nom']
    p_changes = False
    new_p_set = 0
    q_inv_cap, q_allowable, q = find_allowable_q(p_input, power_factor, s_nom)
    q_out = -q_allowable
    # check if the calculated q is not exceeding the inverter capacity if yes then
    # decrease p_input in order not to violate s_nom = np.sqrt((p**2 + q**2) .
    if (abs(q) > q_inv_cap).any().any():
        p_changes = True
        new_p_set = adjust_p_set(s_nom, q_out, p_input, c, 'fixed_cosphi')

    _set_controller_outputs_to_n(n, c, c_attrs, snapshot, p_changes=p_changes,
                                 p_set=new_p_set, q_out=q_out, q_changes=True)


def apply_cosphi_p(n, snapshot, c, c_attrs):
    """
    Power factor as a function of active power (cosphi_p) controller.
    reference : https://ieeexplore.ieee.org/document/6096349.

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

        Examples: if we have two controlled component c in  our subnetwork as below:
    --------
    >>> network.add(c, "PV1", bus="LV2 bus", p_set=0.5, power_factor_min=0.9, set_p1=40,
                   set_p2=100, p_ref=1.3, s_nom=1.5, control_strategy="cosphi_p")

    >>> network.add(c, "PV1", bus="LV2 bus", p_set=1.5, power_factor_min=0.9, set_p1=40,
                   set_p2=100, p_ref=1.3, s_nom=1.5, control_strategy="cosphi_p")

    >>>"c_attrs" is the filtered dataframe of c, where it contain "PV1" and PV2 as
        indexes and all attributes of c for "PV1" and PV2 as columns.
        Based on the given parameters in the example which are stored in "c_attrs"
        controller forms a droop curve using "power_factor = np.select()" see below.
        Then controller finds the value of "(p_set/p_ref)*100" which is named as
        "p_set_per_p_ref" here and checks it on the droop characteristic to choose
        the right "power_factor" for q calculation using p_input.mul(np.tan((
        np.arccos(power_factor) equation. Finally controller has two outputs:
        first "power_factor" which is set to n inside the controller and "q_out"
        as reactive power output sent to "_set_controller_outputs_to_n".
        i.e. for PV1 power_factor=1, because p_set_per_p_ref=0.5/1.3*100 = 38.46%
        which is less than "set_p1=40", so according to the controller droop curve
        if "p_set_per_p_ref"< "set_p1" power_factor is 1 (check np.select())
        and 1 as the power_factor results to q_out=0.
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

    _set_controller_outputs_to_n(n, c, c_attrs, snapshot, q_out=q_out, q_changes=True)


def apply_q_v(n, snapshot, c, c_attrs, n_trials_max, n_trials):
    """
    Reactive power as a function of voltage Q(V).
    reference : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6096349

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

    Examples: if we have two controlled component c in  our subnetwork as below:
    --------
    >>> network.add(c, "PV1", bus="LV2 bus", p_set=0.5, v1=0.9, v2=0.95, v3=0.98,
                   v4=1.02, damper=1, power_factor=0.98, s_nom=1.5, control_strategy="q_v")

    >>> network.add(c, "PV1", bus="LV2 bus", p_set=1.5, power_factor_min=0.9, set_p1=40,
                   set_p2=100, p_ref=1.3, s_nom=1.5, control_strategy="cosphi_p")

    >>>"c_attrs" is the filtered dataframe of c, where it contains "PV1" and PV2 as
        indexes and all attributes of c for "PV1" and PV2 as columns.
        This controller gives reactive power q_out as output as a function voltage
        per unit "v_pu_bus" where this inverter is connected to. Controller has
         a droop curve with five different q choices with conditions according
         to the chosen v1, v2, v3, v4 values (see np.select(...) below). Controller
        checks the location of "v_pu_bus" on the droop curve and as output gives
        "curve_q_set_in_percentage" which shows the required percentage of
        "q_allowable". To find the actual q, they are multiplied togeather see
        "q_out" below. The initial calculated q (see "find_allowable_q" function)
        is checked if it is not exceeding "q_inv_cap" which is the inverter reactive
        power capacity, if yes, then adapt p_set accordingly (see "adjust_p_set" function)
        and finally send both "q_out" and "new_p_set" to _set_controller_outputs_to_n.
        i.e. if v_pu_bus is 1 p.u then q = 0 because according to the droop if
        "v_pu_bus" is between v2 and v3 then compensation percentage is zero, or
        it is considered as the normal range. Also There is a voltage difference
        limit condition in the loop for this controller, when the controller did
        not converge after n_trial_max (30) then the logger warning will pop up.
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
    new_p_set = None
    p_changes = False
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
        p_changes = True
        new_p_set = adjust_p_set(s_nom, q, p_input, c, 'q_v')

    _set_controller_outputs_to_n(n, c, c_attrs, snapshot, n_trials=n_trials,
                                 p_set=new_p_set, p_changes=p_changes,
                                 q_out=q_out, q_changes=True)


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
    _set_controller_outputs_to_n(n, c, c_attrs, snapshot,
                                 p_set=p_out, p_changes=True)


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
    p_out = ((pperpmax*(p_input))/100)
    # update the active power contribution of the controlled indexes in network
    _set_controller_outputs_to_n(n, c, c_attrs, snapshot,
                                 p_set=p_out, p_changes=True)


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


def _set_controller_outputs_to_n(n, c, c_attrs, snapshot, n_trials=0, p_set=None,
                                 p_changes=False, q_out=None, q_changes=False):
    """
    Set the controller outputs to the n (network).

    Parameter:
    ----------
    n : pypsa.components.Network
        Network
    c : string
        Component list_name, i.e. 'Load', 'StorageUnit'...
    c_attrs : pandas data frame
        Component attrs of controlled indexes, i.e. power_factor choice for
        generators, loads...
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    n_trials : integer
        It is the outer loop (while loop in pf.py) number of trials until the
        controller converges.
    p_set : numpy array defaut to None
        Active power output of controller, this will be used by "apply_p_v",
        "apply_cosphi" and "apply_q_v" controllers. for "apply_cosphi" and "apply_q_v"
        when reactive power need is more than the capacity of the inverter, in
        this case the inverter changes a bit of active power to reactive power
        to meet reactive power needs, therefore the reduced p_set needs to be
        updated in n.components_t.p and also in buses_t.p. For "apply_p_v" it
        will be used to updated the new allowed p_sets in the grid after curtailment.
    p_changes : bool default to False
        If ``True``, activates setting the new p_set or updated active power
        to the newtwork.
    q_out : numpy array
        Reactive power output of controller
    q_changes : bool default to False
        Activates setting reactive power output when any reactive power compensation
        controller is used.

    Returns
    -------
    None
    """
    # input power before applying controller output to the network
    p_input = n.pnl(c).p.loc[snapshot, c_attrs.index]
    q_input = n.pnl(c).q.loc[snapshot, c_attrs.index]
    p_q_dict = {}
    if p_changes:
        p_q_dict['p'] = p_set
    if q_changes:
        p_q_dict['q'] = q_out
    # setting p_set, q_out, p and q values to their respective dataframes
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

        # adding the change to the respective buses
        n.buses_t[attr].loc[snapshot, power_change.index] += power_change


def prepare_controlled_index_dict(n, sub_network, inverter_control, snapshots, oltc_control):
    """
    Iterate over "Generator", "Load", "Store" and "StorageUnit" to check if they
    have control strategy applied in any of their indexes. If yes the name
    of control strategy will be set as key of the dictionary and the name of the
    controlled component will as values which will contain the controlled indexes
    with their respective attributes. While preparing the dictionary if any "q_v"
    controller is used by any component n_trial_max is chosen 30 which is the
    maximum load flow trials for this controller to be converged. In the end it
    returns the dictionary which is then used in "apply_controller" to apply
    the chosen controllers to the chosen compoenents.

    Parameter:
    ----------
    Inverter_control : bool default to False
        Activate the outer loop for appliation controllers
    snapshots : list-like|single snapshot
        A subset or an elements of network.snapshots on which to run
        the power flow, defaults to network.snapshots

    Returns
    -------
    v_mag_pu of voltage_dependent_controller : pandas data frame
    Needed to compare v_mag_pu of the controlled buses with the voltage from
    previous iteration to decide for repeation of pf (in pf.py file).
    """
    # p_set = get_switchable_as_dense(n, c, 'p_set', snapshots, c_attrs.index)
    n_trials_max = 0
    parameter_dict = {}
    ctrl_list = ['', 'q_v', 'p_v', 'cosphi_p', 'fixed_cosphi']
    if oltc_control: #  TODO: if oltc control is chosen but oltc bool in n.transformer is not True then pf goes to infinity fix this
        if (n.transformers.oltc).any():
            parameter_dict['oltc'] = {}
            parameter_dict['oltc']['Transformer'] = n.transformers.loc[(n.transformers.oltc == True)]
            ctr_index = (n.transformers.oltc == 1).index
            n.pnl('Transformer')['opt_tap_position'] = n.pnl('Transformer')['opt_tap_position'].reindex(columns=ctr_index)
        # else:
        #     assert ((n.transformers.oltc).any()), (
        #     "inverter_control is activated but no component is controlled,"
        #     " please choose the control_strategy in the desired "
        #     " component indexes. Supported type of controllers are:\n%s."
        #     % (ctrl_list[1:5]))
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
                        if i not in parameter_dict:
                            parameter_dict[i] = {}

                        parameter_dict[i][c.name] = c.df.loc[(
                                c.df.control_strategy == i)]

#                logger.info("We are in %s. These indexes are controlled:\n%s",
#                            c.name, parameter_dict)

        assert (bool(parameter_dict)), (
                "inverter_control is activated but no component is controlled,"
                " please choose the control_strategy in the desired "
                " component indexes. Supported type of controllers are:\n%s."
                % (ctrl_list[1:5]))

    return n_trials_max, parameter_dict
