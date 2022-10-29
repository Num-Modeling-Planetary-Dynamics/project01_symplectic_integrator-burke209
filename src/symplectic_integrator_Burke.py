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

#calculate function step size from orbit period
def get_step(a,mu):
#a is an integer
#mu is an integer
    #calculate period
    P =2 * np.pi * np.sqrt(a**3/mu)
    #1/30th of the period is adequate for step size
    dt = P / 30.0
    return dt

#calculate heliocentric to barycentric velocity
def vh2vb(vhvec, Gmass, Gmtot): 
#vhvec is an n x 3 array containing heliocentric velocity components for n bodies
#Gmass is an n-length array 
    vbcb = -np.sum(Gmass * vhvec) / Gmtot
    vbvec = vhvec + vbcb
    return vbvec

#calculate barycentric to heliocentric velocity
def vb2vh(vbvec,Gmass,Gmcb):
#vbvec is an n x 3 array containing barycentric velocity components for n bodies
#Gmass is an n-length array 
    vbcb = -np.sum(Gmass * vbvec) / Gmcb
    vhvec = vbvec - vbcb
    return vhvec

#update heliocentric positions by barycentric velocity
def lindrift(vbvec,rhvec,Gmass,GMsun, dt):
#rhvec is an n x 3 array containing heliocentric position components for n bodies
#vbvec is an n x 3 array containing barycentric velocity components for n bodies
#Gmass is an n-length array 
    #vbvec = barycentric velocities
    pt = np.sum(Gmass * vbvec,axis=0) / GMsun
    #rhvec is an nX3 array, lindrift is performed on all planets
    rhvec += pt *dt
    return rhvec

#calculate keplerian change in barycentric velocity and heliocentric position
def kep_drift(mu,rvec_in,vvec_in,dt):
#rvec_in is an n x 3 array containing heliocentric position components for n bodies
#vvec_in is an n x 3 array containing barycentric velocity components for n bodies
    def danby(M,ecc):
        #initial guess
        k = 0.85
        E = [M +np.sign(np.sin(M))* k * ecc]
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
            E.append(E_new)
            i+=1
            
            if err<10e-12:
                break
        return E[-1]
    
    for i in range(rvec_in.shape[0]):
        rvec0 = rvec_in[i,:]
        vvec0 = vvec_in[i,:]
        mu0 = mu[i]
        
        rmag0 = np.linalg.norm(rvec0)
        vmag2 = np.vdot(vvec0,vvec0)
        h = np.cross(rvec0,vvec0)
        hmag2 = np.vdot(h,h)
        a0 = 1.0/(2.0 / rmag0 - vmag2/mu0)
        ecc0 = np.sqrt(1 - hmag2 / (mu0 * a0))

    
        n0 = np.sqrt(mu0 / a0**3)  # Kepler's 3rd law to get the mean motion
        E0 = np.arccos(-(rmag0-a0)/(a0 * ecc0))
        if np.sign(np.vdot(rvec0,vvec0)) < 0.0:
            E0 = 2*np.pi - E0
        M0 = E0 - ecc0 * np.sin(E0)
        M = M0 + n0 * dt
        E = danby(M,ecc0)
        dE = E-E0
        
        #now implement f and g functions to advance cartesion vectors
        f = a0 / rmag0 * (np.cos(dE) - 1.0) + 1.0
        g = dt + (np.sin(dE) - dE) / n0
        
        rvec = f * rvec0 + g * vvec0
        rmag = np.linalg.norm(rvec)
        
        fdot = -a0**2 / (rmag * rmag0) * n0 * np.sin(dE)
        gdot = a0 / rmag * (np.cos(dE) - 1.0) + 1.0
    
        vvec = fdot * rvec0 + gdot * vvec0
        
        rvec_in[i,:] = rvec
        vvec_in[i,:] = vvec
    
    return rvec_in,vvec_in

#update barycentric velocities
def kick(Gmass, rvec,vbvec,dt):
#rvec is an n x 3 array containing heliocentric position components for n bodies
#vbvec is an n x 3 array containing barycentric velocity components for n bodies
    n = rvec.shape[0]
    acc = []
    Gmass_flat = Gmass.flatten()
    #calculate each ith body separately
    for i in range(n):
        drvec = rvec - rvec[i,:]
        irij3 = np.linalg.norm(drvec, axis=1)**3
        irij3[i] = 1 #diagonal not included
        irij3 = Gmass_flat / irij3
        acc.append([np.sum(drvec.T * irij3)])
    
    vbvec += acc * dt
    return vbvec

def xv2el(rhvec,vhvec,mu):
#rhvec = (timestep x 3 array containing all pos values for one body from simulation)
#vhvec = (timestep x 3 array containing all vel values for one body from simulation)
#mu = G * (Mcb * Mb) for one body
    Rmag = np.array([np.linalg.norm(rhvec[i]) for i in range(0,rhvec.shape[0])])
    V_sq = np.array([np.vdot(vhvec[i],vhvec[i]) for i in range(0,vhvec.shape[0])])
    h = np.array([np.cross(rhvec[i],vhvec[i]) for i in range(0,rhvec.shape[0])])
    hmag2 = np.array([np.vdot(h[i],h[i]) for i in range(0,h.shape[0])])
    R_dot = np.sign(np.array([np.vdot(rhvec[i],vhvec[i]) for i in range(0,rhvec.shape[0])]))*np.sqrt(V_sq - hmag2/(Rmag**2))
    
    a = 1.0/(2.0 / Rmag - V_sq/mu)
    ecc= np.sqrt(1 - hmag2 / (mu * a))
    I = np.arccos(h[:,2]/np.sqrt(hmag2))
    long_ascend= np.arcsin((rhvec[:,1]*vhvec[:,2] - rhvec[:,2]*vhvec[:,1])/(np.sqrt(hmag2)*np.sin(I)))

    true_anomaly = np.arcsin(((a*(1.0-ecc**2))/np.sqrt(hmag2)*ecc)*R_dot)
    arg_periapsis = np.arcsin(rhvec[:,2]/Rmag*np.sin(I)) - true_anomaly
    arg_periapsis = np.mod(arg_periapsis,2*np.pi)
    varpi = np.mod(long_ascend+arg_periapsis,2*np.pi)
    E = np.arccos(-(Rmag-a)/(a * ecc))
    check = np.array([np.vdot(rhvec[i],vhvec[i]) for i in range(0,vhvec.shape[0])])
    for i in range(0,check.shape[0]):
        if np.sign(check[i]) < 0.0:
            E[i] = 2*np.pi - E[i]
    M = E - ecc * np.sin(E)
    lam = np.mod(M + varpi,2*np.pi)
    
    orbit = {'a':a, 'ecc':ecc, 'I':I, 'lon_asc_node':long_ascend,
            'true_anom':true_anomaly, 'arg_peri':arg_periapsis, 
            'E':E,'M':M,'mean_long':lam, 'long_peri':varpi}

    return orbit

def get_energy(G, Gmcb, Gmass, Gmtot, rhvec_in, vhvec_in):
#Gmass is an n-length array
#rhvec_in is a timestep x n x 3 array containing all pos values for n bodies during simulation
#vhvec_in is a timestep x n x 3 array containing all vel values for n bodies during simulation
    tot_arr = []
    for i in range(0,rhvec_in.shape[0]):
        rhvec = rhvec_in[i,:]
        vbvec = vh2vb(vhvec_in[i,:],Gmass,Gmtot)
        vbcb = -np.sum(Gmass * vhvec_in[i,:]) / Gmtot
        vbmag2 = np.einsum("ij,ij->i", vbvec, vbvec)
        rmag_1 = 1.0 / np.linalg.norm(rhvec, axis=1)
        ke = 0.5 * (Gmcb * np.vdot(vbcb, vbcb) + np.sum(Gmass * vbmag2)) / G

        def pe_one(Gm, rvec, i):
            drvec = rvec[i + 1:, :] - rvec[i, :]
            irij = np.linalg.norm(drvec, axis=1)
            irij = Gm[i + 1:] * Gm[i] / irij
            return np.sum(drvec.T * irij, axis=1)
    
        n = rhvec.shape[0]
        pe = (-Gmcb * np.sum(Gmass * rmag_1) - np.sum([pe_one(Gmass.flatten(), rhvec, i) for i in range(n - 1)])) / G
        tot_arr.append(ke+pe)
    return tot_arr
#------------------------------------------------------------------------------
object_val_arr = [8,9]
#mass * 10^24kg
object_mass = {'1':0.330e24 ,'2':4.87e24,'3':5.97e24,'4':0.642e24,'5':1898e24,'6':568e24,'7':86.8e24,'8':102e24,'9':0.0130e24}
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

rhvec_in = np.array(rhvec)
vhvec_in = np.array(vhvec)
a = np.array(a)
ecc = np.array(ecc)
mass = np.array(mass)

Gmass = G*(mass)
Gmtot = G*Msun + np.sum(Gmass)
mu = np.float64(G*(Msun+mass))

#convert heliocentric velocity to barycentric velocity
vbvec_in = vh2vb(vhvec_in,Gmass,Gmtot)

#get step size for neptune
dt = get_step(a[0],mu[0])
#calcualte half step size for leap frog method
dth = 0.5*dt
tfinal = 1e5*(365.25 * 86400)
timestep = np.arange(0,tfinal,dt)

rhvec_arr_Pluto = []
vhvec_arr_Pluto = []
rhvec_arr_Neptune = []
vhvec_arr_Neptune = []
rhvec_arr_bodies = []
vhvec_arr_bodies = []

for t in timestep:
    #drift heliocentric positions for each body by Sun
    rhvec1 = lindrift(vbvec_in,rhvec_in,Gmass,G*Msun,dth)

    #interaction kick for barycentric velocity
    vbvec1 = kick(Gmass, rhvec1,vbvec_in,dth)

    #keplerian drift heliocentric positions for each body
    rhvec2,vbvec2 = kep_drift(mu,rhvec1,vbvec1,dt)

    #interaction kick for barycentric velocity
    vbvec3 = kick(Gmass, rhvec2,vbvec2,dth)

    #drift heliocentric positions for each body by Sun
    rhvec3= lindrift(vbvec3,rhvec2,Gmass,G*Msun,dth)
    

    rhvec_arr_bodies.append(rhvec3)
    rhvec_arr_Neptune.append(rhvec3[0,:])
    rhvec_arr_Pluto.append(rhvec3[1,:])
    
    vhvec = vb2vh(vbvec3, Gmass, G*Msun)
    vhvec_arr_bodies.append(vhvec)
    vhvec_arr_Neptune.append(vhvec[0,:])
    vhvec_arr_Pluto.append(vhvec[1,:])
    vbvec_in = vbvec3
    rhvec_in = rhvec3

rhvec_arr_bodies = np.asarray(rhvec_arr_bodies)
vhvec_arr_bodies = np.asarray(vhvec_arr_bodies)
rhvec_arr_Neptune = np.asarray(rhvec_arr_Neptune)
vhvec_arr_Neptune = np.asarray(vhvec_arr_Neptune)
rhvec_arr_Pluto= np.asarray(rhvec_arr_Pluto)
vhvec_arr_Pluto = np.asarray(vhvec_arr_Pluto)

#convert cartesian coordinates to orbital elements
orbit_els_Neptune = xv2el(rhvec_arr_Neptune,vhvec_arr_Neptune,mu[0])
orbit_els_Pluto = xv2el(rhvec_arr_Pluto,vhvec_arr_Pluto,mu[1])

#calculate resonance angle
Neptune_lam = orbit_els_Neptune['mean_long']
Pluto_lam = orbit_els_Pluto['mean_long']
Pluto_varpi = orbit_els_Pluto['long_peri']
phi = np.mod(3 * Pluto_lam - 2 * Neptune_lam - Pluto_varpi, 360)

#calculate energy error from total energy
energy = get_energy(G, G*Msun, Gmass, Gmtot, rhvec_arr_bodies, vhvec_arr_bodies)
dE_E0 = (energy - energy[0]) / energy[0]

#plot resonance angle
plt.plot(timestep[::50]/365.25/86400,phi[::50])
plt.savefig('../plots/phi_vs_t.png',dpi=300)
plt.clf()

#plot energy error
plt.plot(timestep[::50]/365.25/86400,dE_E0[::50])
plt.savefig('../plots/dE_E0_vs_t.png',dpi=300)
plt.clf()



    

    

    




