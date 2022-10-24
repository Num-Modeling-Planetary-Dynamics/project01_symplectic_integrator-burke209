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

#kep_drift = drift_one (as in one body)

#need heliocentric positions and barycentric velocities

def get_vectors(obj_val):
    #obj_name should be an integer between 1-9
    #location as @sun gets heliocentric positions and velocities
    obj_vec = {}
    obj = Horizons(id=obj_val,id_type='majorbody', location = '@sun', 
                   epochs={'start':'2022-10-01', 'stop':'2022-10-02', 'step': '1d'})
    vec_df = obj.vectors().to_pandas()
    el_df = obj.elements().to_pandas()
    
    rvecnames = ['x','y','z']
    vvecnames = ['vx','vy','vz']
    elnames = ['a','e']
    
    rvec = vec_df[rvecnames].iloc[0].to_numpy()
    vvec = vec_df[vvecnames].iloc[0].to_numpy()
    el = el_df[elnames].iloc[0].to_numpy()
    
    obj_vec = {'rvec':rvec, 'vvec':vvec, 'el':el}
    
    #convert AUDay to AUYear
    return(obj_vec)

#calculate heliocentric to barycentric velocity
def vh2vb(vhvec, Gmass, Gmtot): 
    
    vbcb = -np.sum(Gmass * vhvec) / Gmtot
    vbvec = vhvec + vbcb
    return(vbvec)

#update heliocentric positions by barycentric velocity
def lindrift(vbvec,rhvec,dt):
    #vbvec = barycentric velocities
    pt = np.sum(G * vbvec) / GMcb
    #rhvec is an nX3 array, lindrift is performed on all planets
    rhvec += pt *dt
    return(rhvec)

#calculate keplerian change in barycentric velocity and heliocentric velocity
def kep_drift(mu,rvec_in,vvec_in,dt):
    
    def danby(M,ecc):
        #initial guess
        k = 0.85
        E = M +np.sign(np.sin(M)* k * ecc)
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
            err = (E_new-E[i])/(E_new) #use this less than 10^-12 instead of accuracy in deg
            #acc = E_new-E[i]
            E.append(E_new)
            err_arr.append(err)
            i+=1
            
            #if np.rad2deg(acc)<0.000001:
             #   break
        return(E[-1])
    
    rvec0 = rvec_in
    vve0 = vvec_in
    rmag0 = np.linalg.norm(rvec0)
    #a,ecc = xv2el(mu,x,y,z,vx,vy,vz)
    E0 = np.arccos(-(rmag0-a)/(a * ecc))
    if np.sign(np.vdot(rvec0,vvec0)) < 0.0:
        E0 = 2*np.pi - E0
    M0 = E0 - ecc * np.sin(E0)
    M = M0 + n + dt
    E = danby(M,ecc)
    dE = E-E0
    
    #now implement f and g functions to advance cartesion vectors
    f = a / rmag0 * (np.cos(dE) - 1.0) + 1.0
    g = dt + 1.0 / n * (np.sin(dE)-dE)
    
    rvec = f * rvec0 +g *vvec0
    rmag = np.linalg.norm(rvec)
    
    return()

#update barycentric velocities
def kick(rvec,vbvec,dt):
    drvec = rvec - rvec[i,:]
    irij3 = np.linalg.norm(drvec, axis=1)**3
    irij3[i] = 1 #diagonal not included
    irij3 = G / irij3
    acc = np.sum(drvec.T * irij3)
    
    #calculate each ith body separately
    vbvec += acc * dt
    return()

def step(dt):
    dth = 0.5 *dt
    lindrift(dth)
    kick(dth)
    kep_drift(dt)
    kick(dt)
    lindrift(dth)
    return()
    
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

G = np.longdouble(const.G.value)
AU2M = np.longdouble(const.au.value)
Msun = np.longdouble(const.M_sun.value)
Rsun = np.longdouble(const.R_sun.value)
JD2S = 86400
YR2S = np.longdouble(365.25 * JD2S)
#time unit should be years
# GMSunSI * time_unit **2 / distance unit ** 3

#data processing
for object_val in object_val_arr:
    object_vec[str(object_val)] = get_vectors(object_val)
    object_vec[str(object_val)]['mass'] = object_mass[object_val]

for object_val in object_val_arr:
    rhvec.append(object_vec[str(object_val)]['rvec'])
    vhvec.append(object_vec[str(object_val)]['vvec'])
    a.append(object_vec[str(object_val)]['el'][0])
    ecc.append(object_vec[str(object_val)]['el'][1])
    mass.append(object_vec[str(object_val)]['mass'])
    
Gmass = G*(mass*10**24)
Gmtot = G*Msun + np.sum(Gmass)

#convert heliocentric velocity to barycentric velocity
vbvec = vh2vb(vhvec,Gmass,Gmtot)

dt = 
for dt in timestep:
    #drift heliocentric positions for each body by Sun

    #interaction kick for barycentric velocity

    #keplerian drift heliocentric positions for each body

    #interaction kick for barycentric velocity
    
Mbody = object_mass[str(object_val)]
mu = G * (Msun+Mbody)
kick(dt)
kep_drift(dt)
kick(dt)
    




