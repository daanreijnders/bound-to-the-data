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
        2 meter temperature [degrees C]
    
    Td2m : int, float
        2 meter dewpoint temperature [degrees C]
    
    T850 : int float
        temperature at 850 hPa [degrees C]
        
    W850 : int, float
        wind speed at 850 hPa [m/s]
    """
    FSI = 2 * (T2m - Td2m) + 2 * (T2m - T850) + W850
    
    return FSI

def fogginess(fsi=None, T2m=None, Td2m=None, T850=None, W850=None):
    """
    Uses `FSI` to return a 'foginess' between 0 (no fog) and 1 (foggiest).

    Parameters
    ----------
    fsi : int, float
        FSI value

    T2m : int, float
        2 meter temperature [degrees C]
    
    Td2m : int, float
        2 meter dewpoint temperature [degrees C]
    
    T850 : int float
        temperature at 850 hPa [degrees C]
        
    W850 : int, float
        wind speed at 850 hPa [m/s]
    """
    if not fsi:
        fsi = FSI(T2m, Td2m, T850, W850)

    fogginess_val = (60 - fsi)/60
    if fogginess_val < 0: 
        fogginess_val = 0
    elif fogginess_val > 1:
        fogginess_val = 1

    return fogginess_val


def log_wind_profile(z_2, u_1, z_1 = 10.):
    """
    Assuming that the wind follows a logarithmic profile, 
    this function computes the strength of the wind at a requested height z_2
    based on the wind given (u_1) at a different height (z_1 = 10m by default).
    
    Parameters
    ----------
    z_2 : float, int
        height to request wind at [m]

    u_1 : float, int
        Wind speed at z_1 [m/s]

    z_1 : float, int
        Height that u_1 is specified at [m]
    
    Returns
    -------
    u_2 : float
        Wind speed at z_2 [m/s]
    """

    z_0 = 0.0002 # Roughness length for open sea [meters]
    d = 0.01     # Zero-plane displacement

    u_2  = u_1 * np.log((z_2 - d) / z_0) / np.log((z_1 - d) / z_0)

    return u_2


def dispersion_relation(kappa, H):
    """
    Computes radial frequency based on wavenumber for capillary-gravity waves.
    
    Parameters
    ----------
    kappa : float
        Wavenumber (radial) [rad/m]
    
    H : float
        Local water depth [meters]
    
    Returns
    -------
    float
        frequency [rad/s]
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
        Wave period [seconds]
    
    water_depth : float
        Local water depth [meters]
        
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

