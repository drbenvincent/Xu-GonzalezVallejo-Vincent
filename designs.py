import numpy as np
import pandas as pd


def generate_designs():
    """Generate designs (RA, DA, RB, DB). This should precisely match the
    set of questions we used in the actual experiment."""

    n = 50
    RA_vals = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 60])
    DB_vals = np.array([7, 15, 29, 56, 101])

    # define constant values
    DA = np.zeros(n)
    RB = np.full(n, 60)

    # shuffle index for DB
    DB_index = np.arange(len(DB_vals))
    np.random.shuffle(DB_index)

    # fill remaining design dimensions by iterating over DB (shuffled) and RA
    DB = []
    RA = []
    for db_index in DB_index:
        for ra in RA_vals:
            DB.append(DB_vals[db_index])
            RA.append(ra)

    DB = np.array(DB)
    RA = np.array(RA)

    designs = pd.DataFrame({"RA": RA, "DA": DA, "RB": RB, "DB": DB})
    return designs
