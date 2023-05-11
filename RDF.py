#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:22:35 2023

@author: audreycollins, (code writing aided by Zijie Wu)
"""
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
#%%

avg_gofr =[]
std_gofr =[]
folders=['0%HA', '10%HA', '20%HA', '30%HA', '40%HA', '50%HA', '60%HA']
prefixes = [['T0', 'T1', 'T2'],
            ['T0', 'T1', 'T2'],
            ['T0', 'T1', 'T2'],
            ['T0', 'T1', 'T2'],
            ['T0', 'T1', 'T2'],
            ['T0', 'T1', 'T2'],
            ['T1', 'T2', 'T3'], 
    ]
# for prefix_i in range(len(prefixes)):
for fi, folder_i in enumerate(folders): 
    gofr =[]
    for prefix_i in prefixes[fi]:
        fn = f'../ST_final/{folder_i}/{prefix_i}.xtc' #trajectory file to be analyzed
        ref_fn = f'../ST_final/{folder_i}/{prefix_i}.gro' #corresponding structure file
        t = md.load(fn, top=ref_fn, stride=3) #trajectory = trajectory file (variable), topology file, (structure file variable), every 4th frame
        
        atom_index = t.topology.select('resname PBL and symbol == O') # atoms or selections of interest
        lastframe = t.xyz[-3, atom_index, :] # assigning the last frame as a variable
        water_index = t.topology.select('water and name == O')
    
        
        atom_pairs = [] #creating an empty list named 'atom pairs' to compile (append) values into
        
        for i in range(len(atom_index)): # i corresponds to the number of selected elements in mc_index
            for j in range(i + 1, len(atom_index)): #j corresponds to every value not including itself 
                atom_pairs.append([atom_index[i],atom_index[j]]) #putting i in its respective list and j in its respective list
      
        dist = md.compute_distances(t, atom_pairs, periodic=True, opt=True) #calculate the distances between every atom not including itself (atom pairs), periodic istrue b/c we want nearest periodic neighbor)
         
        gr_sliced = dist[-20:] #only selecting the last 5 frames b/c system is not settled yet
        gr_binned = np.histogram(gr_sliced.flatten(), bins=30, range=(0, 5)) #flatten = consider everything as one dimension b/c frame dependance is not important for normalized g(r)
        #max range is equal to box length times the squareroot of 3 divided by 2
        r = (gr_binned[1][:-1]+gr_binned[1][1:])/2
        dr = r[1] - r[0]  
        x = r
        shell_volume = 4*3.14159*r**2*dr
        box_volume = t.unitcell_lengths[-1,-1]**3
        y = 2*(gr_binned[0]/len(atom_index)/len(atom_index)/shell_volume*box_volume/len(gr_sliced))
        gofr.append(y)
        
#%%
    gofr=np.array(gofr)
    avg_gofr_folder=np.mean(gofr, axis=0)
    avg_gofr.append(avg_gofr_folder)
    std_gofr_folder=np.std(gofr, axis=0)
    std_gofr.append(std_gofr_folder)
    
#%%
fig = plt.figure(figsize=(5,5))
ax = fig.gca()
col = pl.cm.inferno
for i in range(len(folders)):
    ax.errorbar(x, avg_gofr[i], yerr=std_gofr[i], ecolor= col(0.1*i), capsize=2, color= col(0.1*i), label=f'HPA-{90-10*i:.0f}%-HA') #plot the disances with r as the x value and y as the 0th index for the gr(binned)


ax.set_xlim(0,5)
ax.set_ylim(-20,650)

plt.title("g(r) PBL to PBL")
plt.xlabel("r (nm)")
plt.ylabel("g(r)")
plt.legend()
plt.savefig('/home/audreycollins/Desktop/all_wih_error.png',dpi=500,bbox_inches='tight')
#%%


