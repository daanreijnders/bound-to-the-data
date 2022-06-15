import numpy as np
from scipy.optimize import fsolve


"""
------------------------
PHYSICS HELPER FUNCTIONS
------------------------
""" 


def FSI(T2m, Td2m, T850, W850):
    """
    Computes the Fog Stability Index.
        FSI < 31 indicates a high probability of fog formation, 
        31 < FSI < 55 implies moderate risk of fog
        FSI > 55 suggests low fog risk.
    See https://edepot.wur.nl/144635
        
    Parameters
    ----------
    T2m : int, float
        2 meter temperature
    
    Td2m : int, float
        2 meter dewpoint temperature
    
    T850 : int float
        temperature at 850 hPa
        
    W850 : int, float
        wind speed at 850 hPa
    """
    FSI = 2 * (T2m - Td2m) + 2 * (T2m - T850) + W850
    
    return FSI


def dispersion_relation(kappa, H):
    """
    Computes radial frequency based on wavenumber for capillary-gravity waves.
    
    Parameters
    ----------
    kappa : float
        Wavenumber (radial)
    
    H : float
        Locat water depth in meters
    
    Returns
    -------
    float
        frequency
    """
    
    g = 9.81 # m/s^2 gravitational acceleration
    tau = 0.08 # N/m surface tension
    rho = 1026 # kg / m^3 density
    
    sigma = np.sqrt((g * kappa + tau / rho * kappa**3) * np.tanh(kappa * H))
    
    return sigma


def wavelength_velocity(period, water_depth):
    """
    Computes wavelength and phase speed based on the wave period and water depth.
    Assumes free capillary gravity waves.
    
    Parameters
    ----------
    period : float
        Wave period in seconds
    
    water_depth : float
        Local water depth in meters
        
    Returns
    -------
    (float, float)
        (Wavelength [m], Phase Velocity [m/s]) tuple
    """
    g = 9.81 # m/s^2 gravitational acceleration
    tau = 0.08 # N/m surface tension
    rho = 1026 # kg / m^3 density
    
    f = 1/period # frequency
    omega = 2 * np.pi * f # radial frequency
    H = water_depth # meters
    
    # Define the function that needs to be optimized (omega - omega(kappa, H) = 0)
    def find_kappa(kappa):
        return dispersion_relation(kappa, H) - omega
    
    # Use scipy's zero finder (`fsolve`) to find the kappa related to omega.
    # We require positive wavenumbers, but based on the initial guess, negative
    # solutions may be found, so vary the initial guess in case the wavenumber 
    # is negative.
    for tries in range(1, 10):
        # Solve for kappa (wavenumber)
        wavenumber = fsolve(find_kappa, tries)[0]
        if wavenumber > 0:
            break
    
    wavelength = 2 * np.pi/wavenumber
    phase_velocity = omega/wavenumber
    
    return wavelength, phase_velocity

