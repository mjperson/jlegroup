    # -*- coding: utf-8 -*-
"""
Created on Sun Sep 9 2021
Edited on 13 Oct 2021 by William Saunders
Edited June/July 2026 by Michael J Person

@author: Wata Tubthong

Changes by William Saunders:
- Added inputs from T-r profiles and T-p profiles 
- Return theta and dtheta in the light curves class
- Allow mean molecular mass to be a vector --> make a function
Changes by Michael Person:
- Added top of atmosphere clamp to stabilize higher altitudes.
- Regularized constants and constant handling.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import numdifftools as nd
from . import physicalData
from . import ray_crossing as rc

class Atmosphere:
    
    '''Generate refractivity profile from temperature and pressure at 
    reference radius. Can also take into account of temperature gradient'''
    
    def __init__(self,  referencePressure, referenceTemperature,
                 referenceRadius, planetRadius, planetMass,
                 meanMolecularMass, polarizability,
                 refractivityAtSTP=None,
                 temperatureGradient=0, topOfGradient = np.inf,
                 topOfAtmosphere = np.inf, resolution = 1,
                 constants=None
                 ):
        
        '''
        referencePressure: float, pressure at reference radius in Pa
        referenceTemperature: float, temperature at reference radius in K
        referenceRadius: float, reference radius in km
        planetRadius: float, radius of the occulting body in km
        planetMass: float, mass of planet in kg
        meanMolecularMass: float, mean molecular mass of atmosphere in kg/mol
        polarizability: float, polarizability of the gas
        refractivityAtSTP (optional): refractivity of the gas at STP, default=None
        temperatureGradient (optional): float, temperature gradient at reference radius, default=0
        topOfGradient (optional): float, radius at which the gradient ends in km, default=inf
        topOfAtmosphere (optional): float, radius at which the atmosphere ends in km, default=inf
        resolution (optional): float, spatial resolution of the output atmosphere in km,default=1
        '''
        
        self.referencePressure = referencePressure
        self.referenceTemperature = referenceTemperature
        self.temperatureGradient = temperatureGradient
        self.referenceRadius = referenceRadius
        self.planetRadius = planetRadius
        self.planetMass = planetMass
        self.meanMolecularMass = meanMolecularMass
        self.polarizability = polarizability
        self.topOfGradient = topOfGradient
        self.refractivityAtSTP = refractivityAtSTP
        
        #Get the profile up to 5 times of planetary radius
        self.radialDistance = np.arange(self.planetRadius, np.min([5*self.planetRadius, topOfAtmosphere]), resolution)*1.0
        
        self.temperatureProfile = None
        self.temperature_func = None
        self.pressureProfile = None
        self.numberdensityProfile = None
        self.refractivityProfile = None
        self.atmosphericProfile = None
        
        #Constants: injected from jlegroup.physicalData (constants= keyword;
        #default is the package's current vintage, DEFAULT_CONSTANTS —
        #pass constants=physicalData.CODATA1986 for verification work).
        #Replaces the original implementation's hardcoded R/G/kB, 2026-07-15.
        if constants is None:
            constants = physicalData.DEFAULT_CONSTANTS
        self.constants = constants
        self.r0 = float(constants.gas_constant)
        self.G = float(constants.gravitational)
        self.kB = float(constants.boltzmann)
    
    def dP(self, p, r):
    
        '''Calculate dP for given
        p: float, pressure
        r: float, distance from planet centre'''

        #calculate scale height
        h = self.r0*self.temperature_func(r)*(r*1000)**2/(self.meanMolecularMass_func(r)*self.G*self.planetMass)/1000      

        return float(-p/h)

    def rk4(self, func, y, t, dt):
        
        '''Function for implementing Runge-Kutta method
        y: float
        t: float
        dt: float, step size'''
        
        k1 = func(y,t)
        k2 = func(y+k1/2*dt, t+dt/2)
        k3 = func(y+k2/2*dt, t+dt/2)
        k4 = func(y+k3*dt, t+dt)
    
        return y+dt*(k1+2*k2+2*k3+k4)/6
    
    def getTemperatureProfile(self):
        
        '''Generate temperature profile for atmosphere with temperature gradient
        '''
        
        #in case that r>topOfGradient, the profile would be isothermal
        #so T(r>r_top) = T(r_top)
        r = np.min([self.radialDistance, np.ones(len(self.radialDistance))*self.topOfGradient], axis=0)
        temperatureProfile = self.referenceTemperature*(r/self.referenceRadius)**self.temperatureGradient

        self.temperatureProfile = temperatureProfile
        self.temperature_func = interpolate.CubicSpline(self.radialDistance,self.temperatureProfile)

    def getMeanMolecularMassFunction(self):

        if np.array(self.meanMolecularMass).size == 1:
            self.meanMolecularMass = np.full(len(self.radialDistance), self.meanMolecularMass)
        self.meanMolecularMass_func = interpolate.CubicSpline(self.radialDistance, self.meanMolecularMass)

    def getPressureProfile(self):
        
        '''Generate pressure profile from reference pressure and (d/dr)P
        '''
    
        pressureProfile = [self.referencePressure]
        #find the index where r=r_ref
        k = np.argmin(np.square(self.radialDistance-self.referenceRadius))

        for i in range(0,k):
        
            #iterate down the atmosphere to the surface
            r = self.radialDistance[k-1-i]
            p = self.rk4(self.dP, pressureProfile[0], r, r-self.radialDistance[k-i])
            pressureProfile = [p]+pressureProfile
        
        for i in range(0,len(self.radialDistance)-k-1):
            
            #iterate up to the top of atmosphere
            r = self.radialDistance[k+1+i]
            p = self.rk4(self.dP, pressureProfile[-1], r, r-self.radialDistance[k+i])
            pressureProfile = pressureProfile + [p]
    
        self.pressureProfile = np.array(pressureProfile)
    
    def getNumberdensityProfile(self):
        
        '''Calculte number density profile from pressure and temperature profile'''
        
        self.numberdensityProfile = self.pressureProfile/(self.kB*self.temperatureProfile)

    def getRefractivityProfile(self):
        
        '''Calculate refractivity profile from number density profile'''
        
        if self.polarizability is None:
            #nu per molecule at the 1-atm Loschmidt reference state — the same
            #convention as physicalData.refractivity (unified 2026-07-15; the
            #original used a 1-bar state here, ~1.3% high).
            a = self.refractivityAtSTP/float(self.constants.loschmidt)
            self.refractivityProfile = a*self.numberdensityProfile
            
        else: #in case that the refractivity at STP is given
            k = 4/3*np.pi*self.polarizability*self.numberdensityProfile
            self.refractivityProfile = 1.5*k
            #approximated from Lorentz-Lorenz equation to avoid +1
    
    def main(self):
        
        '''Run all functions to get the atmospheric profile'''
        
        self.getTemperatureProfile()
        self.getMeanMolecularMassFunction()
        self.getPressureProfile()
        self.getNumberdensityProfile()
        self.getRefractivityProfile()
        self.atmosphericProfile = pd.DataFrame({'Radius': self.radialDistance,
                                                'Temperature':self.temperatureProfile,
                                                'Pressure':self.pressureProfile,
                                                'NumDensity':self.numberdensityProfile,
                                                'Refractivity':self.refractivityProfile})

class AtmospherefromTprofile:
    
    '''Generate refractivity profile from temperature-radius profile, given pressure at 
    reference radius.'''
    
    def __init__(self, referencePressure, referenceRadius,
                 temperatureProfile, radius,
                 planetRadius, planetMass,
                 meanMolecularMass, polarizability,
                 refractivityAtSTP=None,
                 constants=None,
                 ):
        '''
        referencePressure: float, pressure at reference radius in Pa
        referenceRadius: float, reference radius in km
        temperatureProfile: array, temperatures at given radius values in K
        radius: array, radius coordinates of temperature measurements in km
        planetRadius: float, radius of the occulting body in km
        planetMass: float, mass of planet in kg
        meanMolecularMass: float, mean molecular mass of atmosphere in kg/mol
        polarizability: float, polarizability of the gas
        refractivityAtSTP (optional): refractivity of the gas at STP, default=None
        '''
        
        self.referencePressure = referencePressure
        self.referenceRadius = referenceRadius
        self.temperatureProfile = temperatureProfile
        self.radialDistance = radius
        self.planetRadius = planetRadius
        self.planetMass = planetMass
        self.meanMolecularMass = meanMolecularMass
        self.polarizability = polarizability
        self.refractivityAtSTP = refractivityAtSTP
        
        self.temperature_func = None
        self.pressureProfile = None
        self.numberdensityProfile = None
        self.refractivityProfile = None
        self.atmosphericProfile = None
        
        #Constants: injected from jlegroup.physicalData (constants= keyword;
        #default is the package's current vintage, DEFAULT_CONSTANTS —
        #pass constants=physicalData.CODATA1986 for verification work).
        #Replaces the original implementation's hardcoded R/G/kB, 2026-07-15.
        if constants is None:
            constants = physicalData.DEFAULT_CONSTANTS
        self.constants = constants
        self.r0 = float(constants.gas_constant)
        self.G = float(constants.gravitational)
        self.kB = float(constants.boltzmann)
    
    def dP(self, p, r):
    
        '''Calculate dP for given
        p: float, pressure
        r: float, distance from planet centre'''

        #calculate scale height
        h = self.r0*self.temperature_func(r)*(r*1000)**2/(self.meanMolecularMass_func(r)*self.G*self.planetMass)/1000      

        return float(-p/h)

    def rk4(self, func, y, t, dt):
        
        '''Function for implementing Runge-Kutta method
        y: float
        t: float
        dt: float, step size'''
        
        k1 = func(y,t)
        k2 = func(y+k1/2*dt, t+dt/2)
        k3 = func(y+k2/2*dt, t+dt/2)
        k4 = func(y+k3*dt, t+dt)
    
        return y+dt*(k1+2*k2+2*k3+k4)/6
    
    def getTemperatureFunction(self):

        self.temperature_func = interpolate.CubicSpline(self.radialDistance,self.temperatureProfile)

    def getMeanMolecularMassFunction(self):

        if np.array(self.meanMolecularMass).size == 1:
            self.meanMolecularMass = np.full(len(self.radialDistance), self.meanMolecularMass)
        self.meanMolecularMass_func = interpolate.CubicSpline(self.radialDistance, self.meanMolecularMass)

    def getPressureProfile(self):
        
        '''Generate pressure profile from reference pressure and (d/dr)P
        '''
    
        pressureProfile = [self.referencePressure]
        #find the index where r=r_ref
        k = np.argmin(np.square(self.radialDistance-self.referenceRadius))

        for i in range(0,k):
        
            #iterate down the atmosphere to the surface
            r = self.radialDistance[k-1-i]
            p = self.rk4(self.dP, pressureProfile[0], r, r-self.radialDistance[k-i])
            pressureProfile = [p]+pressureProfile
        
        for i in range(0,len(self.radialDistance)-k-1):
            
            #iterate up to the top of atmosphere
            r = self.radialDistance[k+1+i]
            p = self.rk4(self.dP, pressureProfile[-1], r, r-self.radialDistance[k+i])
            pressureProfile = pressureProfile + [p]
    
        self.pressureProfile = np.array(pressureProfile)
    

    def getNumberdensityProfile(self):
        
        '''Calculte number density profile from pressure and temperature profile'''
        
        self.numberdensityProfile = self.pressureProfile/(self.kB*self.temperatureProfile)

    def getRefractivityProfile(self):
        
        '''Calculate refractivity profile from number density profile'''
        
        if self.polarizability is None:
            #nu per molecule at the 1-atm Loschmidt reference state — the same
            #convention as physicalData.refractivity (unified 2026-07-15; the
            #original used a 1-bar state here, ~1.3% high).
            a = self.refractivityAtSTP/float(self.constants.loschmidt)
            self.refractivityProfile = a*self.numberdensityProfile
            
        else: #in case that the refractivity at STP is not given
            k = 4/3*np.pi*self.polarizability*self.numberdensityProfile
            self.refractivityProfile = 1.5*k
            #approximated from Lorentz-Lorenz equation to avoid +1
    
    def main(self):
        
        '''Run all functions to get the atmospheric profile'''
        
        self.getTemperatureFunction()
        self.getMeanMolecularMassFunction()
        self.getPressureProfile()
        self.getNumberdensityProfile()
        self.getRefractivityProfile()
        self.atmosphericProfile = pd.DataFrame({'Radius': self.radialDistance,
                                                'Temperature':self.temperatureProfile,
                                                'Pressure':self.pressureProfile,
                                                'NumDensity':self.numberdensityProfile,
                                                'Refractivity':self.refractivityProfile})

class AtmospherefromTpProfile:
    
    '''Generate refractivity profile from temperature-pressure profile, given pressure at 
    reference radius.'''
    
    def __init__(self, referencePressure, referenceRadius,
                 temperatureProfile, pressureProfile,
                 planetRadius, planetMass,
                 meanMolecularMass, polarizability,
                 refractivityAtSTP=None,
                 constants=None
                 ):
        
        '''        
        referencePressure: float, pressure at reference radius in Pa
        referenceRadius: float, reference radius in km
        temperatureProfile: array, temperatures at given pressure values in K
        pressureProfile: array, measured pressures in Pa
        planetRadius: float, radius of the occulting body in km (not currently used) 
        planetMass: float, mass of planet in kg
        meanMolecularMass: float, mean molecular mass of atmosphere in kg/mol
        polarizability: float, polarizability of the gas
        refractivityAtSTP (optional): refractivity of the gas at STP, default=None
        '''
       
        self.referencePressure = referencePressure
        self.referenceRadius = referenceRadius
        self.temperatureProfile = temperatureProfile
        self.pressureProfile = pressureProfile
        self.planetRadius = planetRadius
        self.planetMass = planetMass
        self.meanMolecularMass = meanMolecularMass
        self.polarizability = polarizability
        self.refractivityAtSTP = refractivityAtSTP
        
        self.temperature_func = None
        self.numberdensityProfile = None
        self.refractivityProfile = None
        self.radialDistance = None
        self.atmosphericProfile = None
        
        #Constants: injected from jlegroup.physicalData (constants= keyword;
        #default is the package's current vintage, DEFAULT_CONSTANTS —
        #pass constants=physicalData.CODATA1986 for verification work).
        #Replaces the original implementation's hardcoded R/G/kB, 2026-07-15.
        if constants is None:
            constants = physicalData.DEFAULT_CONSTANTS
        self.constants = constants
        self.r0 = float(constants.gas_constant)
        self.G = float(constants.gravitational)
        self.kB = float(constants.boltzmann)

    def dr(self, r, p):
        '''Calculate dr (=dz) for given
        p: float, pressure in Pa
        r: float, distance from planet center in km'''
        
        #calculate scale height
        h = self.r0*self.temperature_func(p)*(r*1000)**2/(self.meanMolecularMass_func(p)*self.G*self.planetMass)/1000      
        
        return float(-h/p)

    def rk4(self, func, y, t, dt):
        
        '''Function for implementing Runge-Kutta method
        y: float
        t: float
        dt: float, step size'''
        
        k1 = func(y,t)
        k2 = func(y+k1/2*dt, t+dt/2)
        k3 = func(y+k2/2*dt, t+dt/2)
        k4 = func(y+k3*dt, t+dt)
    
        return y+dt*(k1+2*k2+2*k3+k4)/6
    
    def getTemperatureFunction(self):
        
        # temperature as a function of pressure
        self.temperature_func = interpolate.CubicSpline(self.pressureProfile[::-1], self.temperatureProfile[::-1])

    # def getMeanMolecularMassFunction(self):

    #     if np.array(self.meanMolecularMass).size == 1:
    #         self.meanMolecularMass = np.full(len(self.radialDistance), self.meanMolecularMass)
    #     self.meanMolecularMass_func = interpolate.CubicSpline(self.radialDistance, self.meanMolecularMass)
    #     print('1')
    #     print(self.meanMolecularMass_func)

    def getMeanMolecularMassFunction(self):

    	# mean molecular mass as a function of pressure

        if np.array(self.meanMolecularMass).size == 1:
            self.meanMolecularMass = np.full(len(self.pressureProfile), self.meanMolecularMass)
        self.meanMolecularMass_func = interpolate.CubicSpline(np.flip(self.pressureProfile), np.flip(self.meanMolecularMass))

    def getRadius(self):
        
        '''Generate radius from reference pressure and reference radius using (d/dp)
        NOTE: This subroutine looks similar to the one for an atmosphere from a T-radius profile but 
           you have to reverse the order b/c pressure is inverse to radius. 
        '''
    
        radius = [self.referenceRadius] 
        #find the index where p=p_ref
        k = np.argmin(np.square(self.pressureProfile-self.referencePressure))

        for i in range(0,k):
            
            p = self.pressureProfile[k-1-i]
            r = self.rk4(self.dr, radius[0], p, p-self.pressureProfile[k-i])
            radius = [r] + radius
        
        for i in range(0,len(self.pressureProfile)-k-1):
            
            p = self.pressureProfile[k+1+i]
            r = self.rk4(self.dr, radius[-1], p, p-self.pressureProfile[k+i])
            radius = radius + [r]
            
        self.radialDistance = np.array(radius)
                   
    def getNumberdensityProfile(self):
        
        '''Calculte number density profile from pressure and temperature profile'''
        
        self.numberdensityProfile = self.pressureProfile/(self.kB*self.temperatureProfile)

    def getRefractivityProfile(self):
        
        '''Calculate refractivity profile from number density profile'''
        
        if self.polarizability is None:
            #nu per molecule at the 1-atm Loschmidt reference state — the same
            #convention as physicalData.refractivity (unified 2026-07-15; the
            #original used a 1-bar state here, ~1.3% high).
            a = self.refractivityAtSTP/float(self.constants.loschmidt)
            self.refractivityProfile = a*self.numberdensityProfile
            
        else: #in case that the refractivity at STP is not given
            k = 4/3*np.pi*self.polarizability*self.numberdensityProfile
            self.refractivityProfile = 1.5*k
            #approximated from Lorentz-Lorenz equation to avoid +1
    
    def main(self):
        
        self.getTemperatureFunction()
        self.getMeanMolecularMassFunction()
        self.getRadius()
        self.getNumberdensityProfile()
        self.getRefractivityProfile()
        self.atmosphericProfile = pd.DataFrame({'Radius': self.radialDistance,
                                                'Temperature':self.temperatureProfile,
                                                'Pressure':self.pressureProfile,
                                                'NumDensity':self.numberdensityProfile,
                                                'Refractivity':self.refractivityProfile})

class ChamberlainElliot1997Model:

    '''Generate normalised light curve using Chamberlain & Elliot 1997.

    The above-atmosphere vacuum clamp (v0.11.0): rays landing beyond
    yTop = r_top + D*theta(r_top) passed entirely above the tabulated
    atmosphere, so their flux is exactly 1 (vacuum) and theta = 0; the
    model clamps them (they were previously y->r spline EXTRAPOLATION,
    capable of spurious caustic-like spikes). The boundary is exposed as
    the ``yTop`` attribute after main(). NB the deep end is NOT
    symmetric: positions below the mapped range have no unique physical
    answer (deeper atmosphere? surface?) and are left to the map —
    extend your table until the flux is negligible at its bottom, and
    well above the flux-recovery altitude at its top (a truncated top
    still leaves a ~1e-3 spline edge artifact within ~2 scale heights
    below yTop; the clamp does not cure that).'''
    
    def __init__(self, refractivityProfile, radialDistance, planetDistance, 
                 position, snrPerScaleHeight=0, scaleHeight=None,
                 observerPlaneSampling=1, lightCurve = None):
        
        '''refractivityProfile: array of float, refractivity profile of planetary atmosphere
        radius: array of float, radius corresponding to the points in refractivity profile in km
        planetDistance: float, distance between the occulting body and the observer in km
        position: array of float, radial coordinate in observer's plane in km (y in Ch&E 1997)
        snrPerScaleHeight (optional): float, snr per scale height of the observation, default=0
        scaleHeight (optional): float, pressure scale height of the atmosphere in km, default=None
        observerPlaneSampling (optional): float, resolution in observer's plane in km, default=1
        lightCurve (optional): table with index 'Radius' and 'Flux', position and flux from the light curve if user just want to add noise to the pre-calculated light curve, default=None'''
        
        self.refractivityProfile = refractivityProfile
        self.radialDistance = radialDistance
        self.planetDistance = planetDistance
        self.position = position
        self.scaleHeight = scaleHeight
        self.lightCurve = lightCurve
        
        if self.scaleHeight is not None:
            self.snr = snrPerScaleHeight*np.sqrt(observerPlaneSampling/self.scaleHeight)
        else:
            self.snr = np.inf
        
        self.d1Refractivity = None
        self.d2Refractivity = None
        self.yToR = None
        
        self.theta = None
        self.dtheta = None

        self.unfocusedFlux = None
        self.focusedFlux = None
        self.fluxWithNoiseAdded = None
        self.yTop = None #top of the mapped atmosphere, r_top + D*theta(r_top); set by main()

        self.splineResolution = 1.5 #resolution for the grid to make the spline that convert y to r
        self.integrationBin = 1.5
    
    def getDifferentialRefractivity(self):
        
        '''Calculate 1st and 2nd derivative of refractivity profile'''
        
        r = self.radialDistance
        
        csLogRefractivity = interpolate.CubicSpline(r, np.log(self.refractivityProfile))
        derivative = nd.Derivative(csLogRefractivity)
        d1Refractivity = derivative(r)*np.exp(csLogRefractivity(r))
        csD1Refractivity = interpolate.CubicSpline(r, d1Refractivity)

        self.d1Refractivity_arr = d1Refractivity
        
        csLogD1Refractivity = interpolate.CubicSpline(r, np.log(-1*d1Refractivity))
        derivative = nd.Derivative(csLogD1Refractivity)
        d2Refractivity = derivative(r)*np.exp(csLogD1Refractivity(r))*(-1)
        csD2Refractivity = interpolate.CubicSpline(r, d2Refractivity)
        
        self.d2Refractivity_arr = d2Refractivity

        self.d1Refractivity = csD1Refractivity
        self.d2Refractivity = csD2Refractivity
    
    def integrandTheta(self, x, r):
        
        '''Integrand of theta for numerical integration'''
        
        return r/np.sqrt(r**2+x**2)*self.d1Refractivity(np.sqrt(r**2+x**2))

    def integrandDTheta(self, x, r):
        
        '''Integrand of dTheta for numerical integration'''
        
        ans = x**2/(r**2+x**2)**(3./2.)*self.d1Refractivity(np.sqrt(r**2+x**2))
        ans = ans+r**2/(r**2+x**2)*self.d2Refractivity(np.sqrt(r**2+x**2))
        
        return ans

    def getRYRelation(self):
        
        '''Get the function to convert distance in observer's plane to distance
        in planet's plane'''
        
        r_array = np.arange(self.radialDistance[0], list(self.radialDistance)[-1], self.splineResolution) 
        x0 = np.arange(0, list(self.radialDistance)[-1],self.integrationBin) #array in x to compute the integral
        
        t_array = [] #array of bending angle
        
        #convert r to y to make spline
        for r in r_array:
            #use trapazoidal rule to do the integral
            t = np.trapezoid(self.integrandTheta(x0,r),x0)*2
            t_array.append(t)
        
        rayCrossing = rc.RayCrossing(r_array=r_array,
                                bendingAngle=np.array(t_array),
                                distance=self.planetDistance,
                                position=self.position)
        rayCrossing.main()
        
        self.yToR = rayCrossing.r
    
    def addNoise(self):
        
        '''Add noise to the light curve'''
        
        fluxWithNoise = []
        
        for e in self.focusedFlux:
            fluxWithNoise.append(np.random.normal(loc=e,scale=1/self.snr))
        
        self.fluxWithNoiseAdded = np.array(fluxWithNoise)
    
    def main(self):
        
        if self.lightCurve is None:
            
            #get the differentials of refractivity and the spline
            self.getDifferentialRefractivity()
            self.getRYRelation()
        
            x0 = np.arange(0, list(self.radialDistance)[-1],self.integrationBin)
        
            theta = []
            dtheta = []
            unfocusedFlux = []
            focusedFlux = []
            
            for i in range (0,len(self.position)):
                
                y = self.position[i]
                f_unf = 0
                f_f = 0
                
                for j in range(0,len(self.yToR[i])):
                    
                    r = self.yToR[i][j]
                    t = (y-r)/self.planetDistance
                    dt = np.trapezoid(self.integrandDTheta(x0,r),x0)*2
                    
                    f_unf = f_unf+1/np.abs(1+self.planetDistance*dt)
                    f_f = f_f+1/np.abs(1+self.planetDistance*dt)/np.abs(1+self.planetDistance*t/r)
                
                theta.append(t)
                dtheta.append(dt)
                focusedFlux.append(f_f)
                unfocusedFlux.append(f_unf)
            
            self.theta = np.array(theta)
            self.dtheta = np.array(dtheta)
            self.unfocusedFlux = np.array(unfocusedFlux)
            self.focusedFlux = np.array(focusedFlux)

            #The above-atmosphere vacuum clamp (v0.11.0): positions beyond
            #yTop passed above the tabulated atmosphere -> vacuum, flux = 1
            #exactly, no bending. See class docstring (the deep end is
            #deliberately NOT clamped).
            t_top = np.trapezoid(self.integrandTheta(x0, self.radialDistance[-1]), x0)*2
            self.yTop = float(self.radialDistance[-1] + self.planetDistance*t_top)
            above = np.asarray(self.position, dtype=float) > self.yTop
            self.theta[above] = 0.0
            self.dtheta[above] = 0.0
            self.unfocusedFlux[above] = 1.0
            self.focusedFlux[above] = 1.0

        else:
            #in case that the pre-calculated light curve is given as an input
            self.focusedFlux = self.lightCurve['Flux']
            self.position = self.lightCurve ['Radius']
        
        
        if self.scaleHeight is not None:
            #add noise if the parameters are given
            self.addNoise()

    def plot(self, noise=False):
        
        '''Plot the light curve
        noise (optional): bool, True if user want to plot the light curve with noise, default=False'''
        
        plt.figure()
        
        if self.lightCurve is not None:
            plt.plot(self.position, self.focusedFlux, 'b', label = 'Flux')
            plt.plot(self.position, self.fluxWithNoiseAdded, 'k.', label = 'Flux with noise')
        else:    
            plt.plot(self.position, self.focusedFlux, 'b', label = 'Focused flux')
            plt.plot(self.position, self.unfocusedFlux, 'r--', label = 'Unfocused flux')
            if self.fluxWithNoiseAdded is not None:
                if noise:
                    plt.plot(self.position, self.fluxWithNoiseAdded, 'k.', label = 'Flux with noise')
        plt.xlabel('Position')
        plt.ylim(bottom=-0.5,top=2)
        plt.legend()