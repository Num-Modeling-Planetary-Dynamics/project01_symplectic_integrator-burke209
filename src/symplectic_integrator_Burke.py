# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:14:52 2022

@author: angel
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from astroquery.jplhorizons import Horizons
import astropy.constants as const

def get_vectors(obj_val):
    #obj_name should be an integer between 1-9
    #location as @sun gets heliocentric positions and velocities
    obj_vec = {}
    obj = Horizons(id=obj_val,id_type='majorbody', location = '@sun', 
                   epochs={'start':'2022-10-01', 'stop':'2022-10-02', 'step': '1d'})
    vec_df = obj.vectors().to_pandas()
    el_df = obj.elements().to_pandas()
    
    #units = AU
    rvecnames = ['x','y','z']
    
    #units = AU/d
    vvecnames = ['vx','vy','vz']
    
    #units = AU, unitless
    elnames = ['a','e']
    
    rvec = vec_df[rvecnames].iloc[0].to_numpy()
    vvec = vec_df[vvecnames].iloc[0].to_numpy()
    el = el_df[elnames].iloc[0].to_numpy()
    
    #convert AU to meter
    rvec *= 1.496e+11
    #convert AU/day to m/s
    vvec *= 1.496e+11/86400
    #convert AU to meter
    el[0] *= 1.496e+11
    
    obj_vec = {'rvec':rvec, 'vvec':vvec, 'el':el}

    return obj_vec

def get_step(a,mu):
    P =2 * np.pi * np.sqrt(a**3/mu)
    dt = P / 30.0
    return dt

#calculate heliocentric to barycentric velocity
def vh2vb(vhvec, Gmass, Gmtot): 
    
    vbcb = -np.sum(Gmass * vhvec) / Gmtot
    vbvec = vhvec + vbcb
    return vbvec

#update heliocentric positions by barycentric velocity
def lindrift(vbvec,rhvec,Gmass,GMsun, dt):
    #vbvec = barycentric velocities
    pt = np.sum(Gmass * vbvec) / GMsun
    #rhvec is an nX3 array, lindrift is performed on all planets
    rhvec += pt *dt
    return rhvec

#calculate keplerian change in barycentric velocity and heliocentric velocity
def kep_drift(mu,rvec_in,vvec_in,a,ecc,dt):
    
    def danby(M,ecc):
        #initial guess
        k = 0.85
        E = [M +np.sin(M)* k * ecc]
        err_arr = [0.0]
        i=0
        while True :
            f_E = E[i]-ecc*np.sin(E[i])-M
            df_E = 1.0-ecc*np.cos(E[i])
            df2_E = ecc*np.sin(E[i])
            df3_E = ecc*np.cos(E[i])
            delta1 = 0.0-(f_E/df_E)
            delta2 = 0.0-(f_E/(df_E+(0.5*delta1*df2_E)))
            delta3 = 0.0-(f_E/(df_E+(0.5*delta2*df2_E)+((1./6.)*(delta2**2)*df3_E)))
            
            E_new =E[i] + delta3
            err = (E_new-E[i])/E_new #use this less than 10^-12 instead of accuracy in deg
            #acc = E_new-E[i]
            E = np.append(E,E_new)
            err_arr.append(err)
            i+=1
            
            if err<10e-12:
                break
        return(E[-1])
    
    for i in range(rvec_in.shape[0]):
        rvec0 = rvec_in[i,:]
        vvec0 = vvec_in[i,:]
        a0 = a[i]
        mu0 = mu[i]
        ecc0 = ecc[i]
    
        n0 = np.sqrt(mu0 / a0**3)  # Kepler's 3rd law to get the mean motion
        rmag0 = np.linalg.norm(rvec0)
        E0 = np.arccos(-(rmag0-a0)/(a0 * ecc0))
        if np.sign(np.vdot(rvec0,vvec0)) < 0.0:
            E0 = 2*np.pi - E0
        M0 = E0 - ecc0 * np.sin(E0)
        M = M0 + n0 + dt
        E = danby(M,ecc0)
        dE = E-E0
        
        #now implement f and g functions to advance cartesion vectors
        f = a0 / rmag0 * (np.cos(dE) - 1.0) + 1.0
        g = dt + 1.0 / n0 * (np.sin(dE)-dE)
        
        rvec = f * rvec0 +g *vvec0
        print(rvec)
        rmag = np.linalg.norm(rvec)
        
        fdot = -a0**2 / (rmag * rmag0) * n0 * np.sin(dE)
        gdot = a0 / rmag * (np.cos(dE) - 1.0) + 1.0
    
        vvec = fdot * rvec0 + gdot * vvec0
        print(vvec)
        
        rvec_in[i,:] = rvec
        vvec_in[i,:] = vvec
    
    return rvec_in,vvec_in

#update barycentric velocities
def kick(Gmass, rvec,vbvec,dt):
    n = rvec.shape[0]
    acc = []
    Gmass_flat = Gmass.flatten()
    for i in range(n):
        drvec = rvec - rvec[i,:]
        irij3 = np.linalg.norm(drvec, axis=1)**3
        irij3[i] = 1 #diagonal not included
        irij3 = Gmass_flat / irij3
        acc.append([np.sum(drvec.T * irij3)])
    
    #calculate each ith body separately
    vbvec += acc * dt
    return vbvec
    
#------------------------------------------------------------------------------
object_val_arr = [8,9]
#mass * 10^24kg
object_mass = {'1':0.330	,'2':4.87,'3':5.97, '4':	0.642,'5':1898,'6':568,'7':86.8,	'8':102,'9':	0.0130}
object_vec = {}
rhvec = []
vhvec = []
a = []
ecc = []
mass = []

G = np.float64(const.G.value)
AU2M = np.longdouble(const.au.value)
Msun = np.float64(const.M_sun.value)
Rsun = np.longdouble(const.R_sun.value)
JD2S = 86400
YR2S = np.longdouble(365.25 * JD2S)
#time unit should be years
# GMSunSI * time_unit **2 / distance unit ** 3

#data processing
for object_val in object_val_arr:
    object_vec[str(object_val)] = get_vectors(object_val)
    object_vec[str(object_val)]['mass'] = object_mass[str(object_val)]

for object_val in object_val_arr:
    rhvec.append(object_vec[str(object_val)]['rvec'])
    vhvec.append(object_vec[str(object_val)]['vvec'])
    a.append([np.float64(object_vec[str(object_val)]['el'][0])])
    ecc.append([object_vec[str(object_val)]['el'][1]])
    mass.append([object_vec[str(object_val)]['mass']])

rhvec = np.array(rhvec)
vhvec = np.array(vhvec)
a = np.array(a)
ecc = np.array(ecc)
mass = np.array(mass)

Gmass = G*(mass*10**24)
Gmtot = G*Msun + np.sum(Gmass)
mu = np.float64(G*(Msun+mass*10**24))

#convert heliocentric velocity to barycentric velocity
vbvec = vh2vb(vhvec,Gmass,Gmtot)

#get step size for neptune
dt = get_step(a[0],mu[0])
dth = 0.5*dt
tfinal = 1e5*(365.25 * 86400)
timestep = np.arange(0,tfinal,dt)
t_arr = []
rvec_arr = []
vvec_arr = []

for t in timestep:
    #drift heliocentric positions for each body by Sun
    rhvec = lindrift(vbvec,rhvec,Gmass,G*Msun,dth)
    print('Lin Drift 1 complete')
    #interaction kick for barycentric velocity
    vbvec = kick(Gmass, rhvec,vbvec,dth)
    print("Kick 1 complete")
    #keplerian drift heliocentric positions for each body
    rhvec,vbvec = kep_drift(mu,rhvec,vbvec,a,ecc,dt)
    print("Keplerian drift complete")
    #interaction kick for barycentric velocity
    vbvec = kick(Gmass, rhvec,vbvec,dt)
    print("Kick 2 complete")
    #drift heliocentric positions for each body by Sun
    rhvec = lindrift(vbvec,rhvec,Gmass,G*Msun,dth)
    print("Lin Drift 2 complete")
    #t_arr.append(t)
    rvec_arr.append(rhvec)
    vvec_arr.append(vbvec)
    print(t)
    

    

    




