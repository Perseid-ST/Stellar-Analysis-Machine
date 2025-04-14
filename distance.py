import numpy as np

def calculate_dm(distance_kpc):
    """
    Calculate the distance modulus (dm) given the distance in parsecs.

    :param distance_pc: Distance in parsecs (float or array-like)
    :return: Distance modulus (dm)
    """
    return (5 * np.log10(distance_kpc * 1000)) - 5

def dm_to_kpc(dm):
    """Convert distance modulus (dm) to distance in kiloparsecs (kpc)."""
    return 10**((dm - 5) / 5)

# Example usage
#dm_values = [7, 10, 15]
#distances_kpc = [dm_to_kpc(dm) for dm in dm_values]
#print(distances_kpc)  # Output: [2.51188643150958, 10.0, 100.0]
print(calculate_dm(2.565))

def calculate_av(e_bv, rv=3.1):
    """
    Calculate total extinction (Av) from color excess (E(B-V)).

    :param e_bv: Color excess (E(B-V)) in magnitudes
    :param rv: Ratio of total to selective extinction (default is 3.1)
    :return: Total extinction (Av) in magnitudes
    """
    return rv * e_bv

# Example usage
e_bv = 0.4285
av = calculate_av(e_bv)
print(f"Av: {av:.2f} magnitudes")