import os
import numpy as np
import pypsa
import sys

from numpy.testing import assert_array_almost_equal as arr_equal
from numpy.testing import assert_almost_equal as equal

solver_name = "glpk" if sys.platform == "win32" else "cbc"


def test_sclopf():
    csv_folder_name = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "scigrid-de",
        "scigrid-with-load-gen-trafos",
    )

    n = pypsa.Network(csv_folder_name)

    # test results were generated with GLPK and other solvers may differ

    # There are some infeasibilities without line extensions
    for line_name in ["316", "527", "602"]:
        n.lines.loc[line_name, "s_nom"] = 1200

    # choose the contingencies
    branch_outages = n.lines.index[:2]

    objectives = []
    for pyomo in [True, False]:

        n.sclopf(
            n.snapshots[0],
            branch_outages=branch_outages,
            pyomo=pyomo,
            solver_name=solver_name,
        )

        # For the PF, set the P to the optimised P
        n.generators_t.p_set = n.generators_t.p.copy()
        n.generators.loc[:, "p_set_t"] = True
        n.storage_units_t.p_set = n.storage_units_t.p.copy()
        n.storage_units.loc[:, "p_set_t"] = True

        # Check no lines are overloaded with the linear contingency analysis

        p0_test = n.lpf_contingency(
            n.snapshots[0], branch_outages=branch_outages
        )

        # check loading as per unit of s_nom in each contingency

        max_loading = (
            abs(p0_test.divide(n.passive_branches().s_nom, axis=0))
            .describe()
            .loc["max"]
        )

        arr_equal(max_loading, np.ones((len(max_loading))), decimal=4)

        objectives.append(n.objective)

    equal(*objectives, decimal=1)

if __name__ == "__main__":
    test_sclopf()
