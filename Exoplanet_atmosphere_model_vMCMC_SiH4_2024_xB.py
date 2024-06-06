# Import our modules 

import matplotlib.pyplot as plt
import numpy as np
from numpy import log as ln
from numpy import log10 as log
from numpy import exp as exp
from scipy.optimize import dual_annealing
from scipy.optimize import minimize
from random import uniform, randint
from statistics import mean, stdev
import time
import os
import emcee
import corner
import math
import warnings
from multiprocessing.pool import Pool
from multiprocessing import get_start_method
from multiprocessing import get_context

print(" ")
print(" ")
print("-------------------------------------------------- ")
print(" ")
print("         MODEL FOR EXOPLANET ATMOSPHERE ")
print("           SIMULATED ANNEALING + MCMC")
print("   Solves for surface pressure simultaneously  ")
print("    ")
print("    VERSION USES HIGH T FOR RXNS 2, 4, 5, 7 TO ")
print("    SIMULATE CORE/MANTLE BOUNDARY TEMPERATURES")
print("    INCLUDES SiH4 AS AN ATMOSPHERE SPECIES")
print("         xB for water activity version")
print(" ")
#print("    NON-IDEAL H IN METAL!!")
#print("    NON-IDEAL H2 IN SILICATE (REGULAR SOLN)")
#print("    NON-IDEAL H2O IN SILICATE (REGULAR SOLN)")
print("--------------------------------------------------")
print("Read above! ")
time.sleep(5.0)

# Supress warnings
#warnings.simplefilter('ignore', RuntimeWarning)

# In this version T_max is used as the core-mantle equilibration temperature

#CLEAN UP FROM PREVIOUS RUN...
#Delte the progress.txt file containing intermediate values for the cost function during
#the simulated annealing algorithm
try:
    os.remove("progress.txt")
    print('Removed existing progress.txt file')
except OSError:
    pass

# Create array of total pressures
# P is in Pascal but limits initially given in bar for convenience
P_min=1.0      #bar
P_max=4000.0  #bar
P_min=P_min*1.0e5  #Pascal
P_max=P_max*1.0e5  #Pascal
num=200
P=np.linspace(P_min,P_max,num)
# Convert P back to bar for plotting etc.
Pbar=P/1.0e5
PGPa=P/1.0e9  #GPa units

#READ MODEL PARAMETERS FROM FILE, including inital mole fractions and moles
#First, setup arrays for mole fractions to be solved for.
#0-25 (i.e., 1-26) are the mole fractions, and the last three
#elements of array are the moles of the three phases, atmosphere, silicate, metal.
#The final value is the pressure
numvar=30  # ORIGINALLY 2
var=np.zeros(numvar)

# VALUES CONTAIN ALL MODEL INPUT PARAMETERS IN ORDER
print('Opening initial_atm_vMCMC_2024.txt...')
name="initial_atm_vMCMC_2024.txt"
count = len(open(name).readlines())
print('  Input file length = ', count)
values=np.genfromtxt(name,'float')  # Read from file into values
print('')
for i in range(0,numvar-1):
    var[i]=values[i]
var[numvar-1]=values[31]
    
# Specify mass of planet
Mplanet_Mearth=values[29]  #Planet mass in Earth masses

# Create array of temperatures
T_min = 1300.0  #K
T_max = values[70] #K THIS DEFINES CORE-MANTLE BOUNDARY T, as T(num-1) or as T_max
TK=np.linspace(T_min,T_max,num)

#select temperature for optimization and find nearest value in array TK
temp=values[30]
print('Selected surface temperature=',temp,'K')
for i in range(0,num-1):
    if TK[i] < temp:
        nT=i
print('Temperature used =',TK[nT],'K')
print('Core temperature for rxns 2, 4, 5, and 7 = ',T_max)
print('')

#Take initial pressure from file
P_penalty=values[31]

#Assign weight for mass balance equations
wt_massbalance=values[32]
print('Weight for mass balance equations =', wt_massbalance)

#Assign weight for summing constraints
wt_summing=values[33]
print('Weight for summing constraints =', wt_summing)

#Assign weights to groups of reactions
watm_m=values[34]  #Atmosphere rxns
wsolub_m=values[35] #solubility
wmelt_m=values[36]  #intra-melt reactions
wevap_m=values[37] #evaporation reactions
print('Weight for atmosphere rxns =',watm_m)
print('Weight for solubility rxns =',wsolub_m)
print('Weight for intra-melt rxns =',wmelt_m)
print('Weight for evaporation rxns =',wevap_m)

#Reaction identifications useful for outputs
rxn_names=[' ','Na2SiO3 = Na2O + SiO2','1/2SiO2 + Fe = FeO + 1/2Si','MgSiO3 = MgO + SiO2','O + 1/2Si = 1/2SiO2',\
    '2Hmetal=H2,sil','FeSiO3 = FeO + SiO2','2H2O,sil + Si=SiO2 + 2H2,sil','CO,g + 1/2O2 = CO2,g','CH4,g + 1/2O2 = 2H2,g + CO,g',\
    'H2,g + 1/2O2 = H2O,g','FeO = Fe,g + 1/2O2','MgO = Mg,g + 1/2O2','SiO2 = SiO,g + 1/2O2','Na2O = 2Na,g + 1/2O2',\
    'H2,g = H2,sil','H2O,g = H2O,sil','CO,g = CO,sil','CO2,g = CO2,sil','SiO + 2H2 = SiH4 + 1/2O2',\
    'Si','Mg','O','Fe','H','Na','C','sum xi melt',\
    'sum xi metal','sum xi atm']

#Assign individual weights for each part of the objective function
k=1
fm=np.ones(31)
while k < 30:
    fm[k]=values[k+37]
    print('weight for f',k,'= ',fm[k],'  ', rxn_names[k])
    k=k+1

#Input seed value from file
nseed_prov=int(values[67])

#Assign iterations per temperature for sim annealing
iter=int(values[68])
print('')
print('Number of iterations =',iter)

#Assign a value for the offset between walkers for the MCMC search, e.g. 1e-9 is a common value
ranoffset=float(values[69])
print('')
print('Offset for walkers = %10.3e' %ranoffset)
    
#var[0]=0.001 # MgO melt
#var[1]=0.001 # SiO2 melt
#var[2]=0.903 #MgSiO3 melt
#var[3]=0.001 #FeO melt, 0.055 total Fe as FeO gives DIW =-2.43 with XFe = 0.9
#var[4]=0.055 #FeSiO3 melt
#var[5]=0.007 #Na2O melt
#var[6]=0.001 #Na2SiO3 melt
#var[7]=0.015 #H2 melt
#var[8]=0.016 #H2O melt
#var[9]=1.0e-7 #CO melt
#var[10]=1.0e-7 #CO2 melt
#var[11]=0.90 #Fe metal
#var[12]=0.05 #Si metal
#var[13]=0.025 #O metal
#var[14]=0.025 #H metal
#var[15]=0.90  #H2 gas
#var[16]=0.08  #CO gas
#var[17]=1.0e-7  #CO2 gas
#var[18]=1.0e-7  #CH4 gas
#var[19]=0.016  #O2 gas
#var[20]=1.0e-7  #H2O gas
#var[21]=0.001 #Fe gas
#var[22]=0.001 #Mg gas
#var[23]=0.001 #SiO gas
#var[24]=0.001 #Na gas
#var[25]=0.001 #SiH4 gas

var_names=['xMgO melt','xSiO2 melt','xMgSiO3 melt','xFeO melt','xFeSiO3 melt','xNa2O melt','xNa2SiO3 melt','xH2 melt','xH2O melt', \
    'xCO melt','xCO2 melt','xFe metal','xSi metal','xO metal','xH metal','xH2 gas','xCO gas','xCO2 gas','xCH4 gas','xO2 gas', \
    'xH2O gas','xFe gas','xMg gas','xSiO gas','xNa gas','xSiH4 gas','Moles atm','Moles silicate','Moles metal','Pressure weighting']
print('')
print('Input mole fractions and moles:')
for i in range (0,numvar):
    print(var_names[i],'=', var[i])
    
#Initial compositions in mass:
#
#Molecular weights for conversions to weight fractions, in order of var[]
mol_wts=np.zeros(26)
mol_wts[0]=40.3044
mol_wts[1]=60.08
mol_wts[2]=100.39
mol_wts[3]=71.844
mol_wts[4]=131.9287
mol_wts[5]=61.9789
mol_wts[6]=122.063
mol_wts[7]=2.016
mol_wts[8]=18.01528
mol_wts[9]=28.01
mol_wts[10]=44.0095
mol_wts[11]=55.847
mol_wts[12]=28.0855
mol_wts[13]=15.9994
mol_wts[14]=1.00794
mol_wts[15]=2.016
mol_wts[16]=28.01
mol_wts[17]=44.0095
mol_wts[18]=16.04
mol_wts[19]=31.9988
mol_wts[20]=18.01528
mol_wts[21]=55.847
mol_wts[22]=24.305
mol_wts[23]=60.08
mol_wts[24]=22.98977
mol_wts[25]=32.12

#Approximate molar volumes of condensed species for correcting molar G for P, J/bar (=0.1 cc/mole)
#Dividing by a factor in some cases (SiO2) to get approximations for tens of GPa values
v_Jbar=np.zeros(25)
v_Jbar[0]=1.246 #molar V for MgO
v_Jbar[1]=2.71/2.0 #molar V for SiO2
v_Jbar[2]=3.958 #molar V for MgSiO3
v_Jbar[3]=1.319 #molar V for FeO
v_Jbar[4]=4.031 #molar V for FeSiO3
v_Jbar[5]=2.997 #molar V for Na2O
v_Jbar[6]=5.709 #molar V for Na2SiO3
v_Jbar[7]=1.10 #H2 melt Hirschmann + (2012, EPSL)
v_Jbar[8]=1.92 #H2O melt Malfait+ (2014, EPSL)
v_Jbar[9]=3.958 #CO melt, set equal to MgSiO3 for now
v_Jbar[10]=2.3 #CO2 melt, Duncan and Agee (2011, EPSL)
v_Jbar[11]=0.776 #molten Fe metal ~ as solid Fe
v_Jbar[12]=1.0886 #Si in metal, Mizuna + (2014, ISIJ Int., v. 54)
v_Jbar[13]=0.4 #molten O metal (estimated from carbon in the literature)
v_Jbar[14]=0.266 #molten H metal (Bockris 1971, Acta Metallurgica v 19)


#Grams per mole initial silicate
grams_per_mole_silicate=0.000
for i in range(0,11):
    grams_per_mole_silicate=grams_per_mole_silicate+var[i]*mol_wts[i]
    
#Grams per mole initial atmosphere
grams_per_mole_atm=var[15]*mol_wts[15]+var[16]*mol_wts[16]+\
    var[17]*mol_wts[17]+var[18]*mol_wts[18]+var[19]*mol_wts[19]+ \
    var[20]*mol_wts[20]+var[21]*mol_wts[21]+var[22]*mol_wts[22]+ \
    var[23]*mol_wts[23]+var[24]*mol_wts[24]+var[25]*mol_wts[25]

#Grams per mole initial metal
grams_per_mole_metal=var[11]*55.847+var[12]*28.0855+var[13]*15.9994+var[14]*1.00794

print('')
print("grams per mole silicate   =",grams_per_mole_silicate)
print('grams per mole atmosphere =',grams_per_mole_atm)
print('grams per mole metal      =',grams_per_mole_metal)

#Convert silicate to wt% oxides
print('')
print('Initial silicate composition in wt%:')
wtpercentSiO2=100.0*(var[1]+var[2]+var[4])*mol_wts[1]/grams_per_mole_silicate
print('Wt_%_SiO2_(total Si)_silicate =',wtpercentSiO2)
wtpercentMgO=100.0*(var[0]+var[2])*mol_wts[0]/grams_per_mole_silicate
print('Wt_%_MgO_(total Mg)_silicate =',wtpercentMgO)
wtpercentFeO=100.0*(var[3]+var[4])*mol_wts[3]/grams_per_mole_silicate
print('Wt_%_FeO_(total Fe)_silicate =',wtpercentFeO)
wtpercentNa2O=100.0*(var[5]+var[6])*mol_wts[5]/grams_per_mole_silicate
print('Wt_%_Na2O_(total Na)_silicate =',wtpercentNa2O)

print('')
print('Initial planet composition:')
moles_atm=var[26]
print('  Moles atm =',moles_atm)
moles_silicate=var[27]
print('  Moles silicate = ',moles_silicate)
moles_metal=var[28]
print('  Moles metal =',moles_metal)
molefrac_atm=moles_atm/(moles_atm+moles_silicate+moles_metal)
molefrac_silicate=moles_silicate/(moles_atm+moles_silicate+moles_metal)
molefrac_metal=1.0-molefrac_atm-molefrac_silicate
print('')
print('  Mole fraction atmosphere =',molefrac_atm)
print('  Mole fraction silicate =',molefrac_silicate)
print('  Mole fraction metal =',molefrac_metal)
print(' ')
grams_atm=molefrac_atm*grams_per_mole_atm  #actually grams_i/mole planet
grams_silicate=molefrac_silicate*grams_per_mole_silicate
grams_metal=molefrac_metal*grams_per_mole_metal
totalmass=grams_atm+grams_silicate+grams_metal
massfrac_atm=grams_atm/totalmass
massfrac_silicate=grams_silicate/totalmass
massfrac_metal=grams_metal/totalmass
print('  Mass fraction atmosphere =',massfrac_atm)
print('  Mass fraction silicate =',massfrac_silicate)
print('  Mass fraction metal =',massfrac_metal)

#Estimate atmospheric pressure at the surface of the planet: fratio is the Matm/Mplanet mass ratio
fratio=massfrac_atm/(1.0-massfrac_atm)
P_initial=1.2e6*fratio*(Mplanet_Mearth)**(2.0/3.0)
Pressure_GPa=P_initial*0.0001
print('')
print('  Mass of planet =',Mplanet_Mearth,'Earth masses')
print('  Estimated surface pressure initial =',P_initial,' bar')
print('  Initial pressure in GPa =',Pressure_GPa,' GPa')

#Reassign pressure for start of search to the physical estimate
var[numvar-1]=P_initial


#THERMODYNAMICS OF MELT REACTIONS
#
log_to_ln = 2.302585093
Rgas=8.314462618153
TKlength=len(TK)

#MgO melt
# from NIST
ti=TK/1000.0
a=66.944
b=0.00
c=0.00
d=0.00
e=0.00
f=-580.9944
g=93.74712
h=-532.6106
HmeltMgO=-532.61+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HmeltMgO=HmeltMgO*1000.0 #convert kJ to J
SmeltMgO=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GmeltMgO=HmeltMgO-TK*SmeltMgO  #apparent G of formation of MgO liquid at T and 1 bar

#MgSiO3 melt
# from NIST
ti=TK/1000.0
a=146.440
b=-1.499926e-7
c=6.220145e-8
d=-8.733222e-09
e=-3.144171e-8
f=-1563.306
g=220.6679
h=-1494.864
HmeltMgSiO3=-1494.86+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HmeltMgSiO3=HmeltMgSiO3*1000.0 #convert kJ to J
SmeltMgSiO3=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GmeltMgSiO3=HmeltMgSiO3-TK*SmeltMgSiO3  #apparent G of formation of MgO liquid at T and 1 bar

#G SiO2 melt std state
# from NIST
ti=TK/1000.0
a=85.772
b=-0.000016
c=0.000004
d=-3.809081e-7
e=-0.000017
f=-952.87
g=113.344
h=-902.6610
HmeltSiO2=-902.661+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HmeltSiO2=HmeltSiO2*1000.0 #convert kJ to J
SmeltSiO2=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GmeltSiO2=HmeltSiO2-TK*SmeltSiO2  #apparent G of formation of SiO2 melt at T and 1 bar

#G FeO melt std state
# from NIST
ti=TK/1000.0
a=68.19920
b=-4.501232e-10
c=1.195227e-10
d=-1.064302e-11
e=-3.09268e-10
f=-281.4326
g=137.8377
h=-249.5321
HmeltFeO=-249.5321+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HmeltFeO=HmeltFeO*1000.0 #convert kJ to J
SmeltFeO=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GmeltFeO=HmeltFeO-TK*SmeltFeO  #apparent G of formation for Fe metal at T and 1 bar

#G Fe metal std state
# from NIST
ti=TK/1000.0
a=46.024
b=-1.884667e-8
c=6.094750e-9
d=-6.640301e-10
e=-8.246121e-9
f=-10.80543
g=72.54094
h=12.39602
HmetalFe=12.40+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HmetalFe=HmetalFe*1000.0 #convert kJ to J
SmetalFe=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GmetalFe=HmetalFe-TK*SmetalFe  #apparent G of formation for Fe metal at T and 1 bar

#REACTION 1: Na2SiO3 = Na2O + SiO2 in melt
G1=-(log_to_ln*(-1.33+13870.0/TK))*Rgas*TK  #Magma code line 809
G1=-G1  #our reaction is reverse of that on line 809 of Magma code
logK1=-G1/(Rgas*TK*log_to_ln)
GRT1=np.zeros(num)
for i in range(0,TKlength):
    GRT1[i]=G1[i]/(Rgas*TK[i])
#print('G1/RT =',GRT1)
#logK1=-(-1.33+13870.0/TK)  #alternate, just negative of Magma code line 809
#logK818=-(1.29+8788/TK) #for comparison, feldspar component in melt from Magma code line 818
# Create the plot of logK for reaction
#plt.figure(1)
#plt.plot(TK,logK1,label='Na2SiO3 = Na2O + SiO2 melt')
##plt.plot(TK,logK818,label='NaAlSi3O8=Na2O+1/2Al2O3+SiO2 melt')
#plt.xlabel('Temperature (K)',fontsize=14)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 1}}$ melt',fontsize=12)
#plt.legend()


#REACTION 2: 1/2SiO2 + Fe_metal = FeO + 1/2Si metal, in melt
G_Corgne=(-log_to_ln*(2.97-21800.0/TK))*Rgas*TK  #Corgne et al. (2008)
GmetalSi=G_Corgne-2.0*GmeltFeO+2.0*GmetalFe+GmeltSiO2
G2=0.5*GmetalSi+GmeltFeO-GmetalFe-0.5*GmeltSiO2
# Create the plot of logK for reaction
logK2=-G2/(Rgas*TK*log_to_ln)
GRT2=np.zeros(num)
for i in range(0,TKlength):
    GRT2[i]=G2[i]/(Rgas*TK[i])
#print('G2/RT =',GRT2)
#plt.figure(2)
#plt.plot(TK,logK2,label='1/2SiO2 + Fe metal = FeO + 1/2Si metal')
#plt.xlabel('Temperature (K)',fontsize=14)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 2}}$ melt',fontsize=12)
#plt.legend()


#REACTION 3: MgSiO3 = MgO + SiO2 melt
G3=-(log_to_ln*(0.42+2329.0/TK))*Rgas*TK
G3=GmeltSiO2+GmeltMgO-GmeltMgSiO3
# Create the plot of logK for reaction
logK3=-G3/(Rgas*TK*log_to_ln)
GRT3=np.zeros(num)
for i in range(0,TKlength):
    GRT3[i]=G3[i]/(Rgas*TK[i])
#print('G3/RT =',GRT3)
#plt.figure(3)
#plt.plot(TK,logK3,label='MgSiO3 = MgO + SiO2 melt')
#plt.xlabel('Temperature (K)',fontsize=14)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 3}}$ melt',fontsize=12)
#plt.legend()


#REACTION 4: O metal + 1/2 Si metal = 1/2 SiO2
#G for FeO=Fe+O Badro et al. 2015 with correction for typo sign error
# for the H/R term confirmed by Julien Siebert (Pers. comm.)
G_ox_metal=-log_to_ln*(2.736-11439.0/TK)*Rgas*TK
G4=-(G_ox_metal+G2) #negative sum of Gs for rxn 2 and FeO=Fe+O in Badro et al. 2015
# Create the plot of logK for reaction
logK4=-G4/(Rgas*TK*log_to_ln)
GRT4=np.zeros(num)
for i in range(0,TKlength):
    GRT4[i]=G4[i]/(Rgas*TK[i])
#print('G4/RT =',GRT4)
#plt.figure(4)
#plt.plot(TK,logK4,label='O metal + 1/2 Si metal = 1/2 SiO2 silicate')
#plt.xlabel('Temperature (K)',fontsize=14)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 4}}$ melt',fontsize=12)
#plt.legend()


#REACTION 5: 2H metal = H2,silicate
#
#First extract G of formation of H2 in melt from the G of reaction
# from Hirschmann et al. (2012) for the reaction H2 gas = H2 melt:
#G_formation H2 in melt is Grxn+GH2_gas (because Grxn=G_melt - G_gas)
#lnk=-12.5-0.76*1.0e-4  #lnk for a standard state of 1 bar, or 1e-4 GPa
lnk=-12.5-0.76*1.0e-4*1.0  #Hirschmann with P in bar, his in GPa, here at 1 bar
G_meltH2_gasH2=-Rgas*TK*lnk   #Delta G of the Hirschmann reaction at 1 bar
#
#Next obtain G of formation of H2 gas:
#Thermo data for H2 gas in J/mole
ti=TK/1000.0
a=18.563083
b=12.257357
c=-2.859786
d=0.268238
e=1.977990
f=-1.147438
g=156.288133
h=0.000
HgasH2=a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HgasH2=HgasH2*1000.0 #convert kJ to J
SgasH2=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GgasH2=HgasH2-TK*SgasH2  #apparent G of formation for H2 gas at T and 1 bar
#
GmeltH2=G_meltH2_gasH2+GgasH2  #apparent free energy of formation of H2 in melt by difference
#
#Now we require G of formation of H metal...
#G for reaction Fe + H2O melt = FeO +2H metal is obtained from
# data in Okuchi (1997, Science)
G_Okuchi97=143589.7-TK*69.1  #regression of lnk vs 1/T data in reference
#combine above with data for FeO, H2O and Fe metal
#
#G H2O in melt std state:
# obtained by difference from solubility calibration, e.g., Moore et al. (1998)
# rxn is H2O gas = H2O melt, Grxn = GmeltH2O-GgasH2O
xH2Omelt = exp((2565.0/1500.0-14.21+1.17*ln(Pbar))/2.0) # Testing, 2.0 denominator here is for aH2O = xH2O^2 used in paper
wtpercentH2O=xH2Omelt/0.033  #Plotting vs pressure shows a match with data in Moore
#std state values are -Hrxn/R = 2565+/- 362, Srxn/R = -14.21+/- 0.54, lnKeq=2565/T -14.21
G_meltH2O_vaporH2O = -Rgas*TK*(2565.0/TK -14.21)  #Rxn G for H2O vapor = H2O melt for xH2O
#
#G H2O gas std state
# from NIST: 500 to 1700K and 1700 to 6000K
HgasH2O=np.zeros(TKlength)
SgasH2O=np.zeros(TKlength)
GgasH2O=np.zeros(TKlength)
i=0
while i < TKlength:
    if TK[i] < 1700.0:
        ti=TK[i]/1000.0
        a=30.09200
        b=6.832514
        c=6.793435
        d=-2.534480
        e=0.082139
        f=-250.8810
        g=223.3967
        h=-241.8264
        HgasH2O[i]=-241.83+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasH2O[i]=HgasH2O[i]*1000.0 #convert kJ to J
        SgasH2O[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasH2O[i]=HgasH2O[i]-TK[i]*SgasH2O[i]  #apparent G of formation for H2O gas at T and 1 bar
    else:
        ti=TK[i]/1000.0
        a=41.96426
        b=8.622053
        c=-1.49978
        d=0.098119
        e=-11.15764
        f=-272.1797
        g=219.7809
        h=-241.8264
        HgasH2O[i]=-241.83+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasH2O[i]=HgasH2O[i]*1000.0 #convert kJ to J
        SgasH2O[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasH2O[i]=HgasH2O[i]-TK[i]*SgasH2O[i]  #apparent G of formation for H2O gas at T and 1 bar
    i=i+1
#
#G H2O std state in oxide/silicate melt by difference
GmeltH2O=G_meltH2O_vaporH2O + GgasH2O
#
#G of formation of H in Fe metal std state
# by difference
GmetalH=0.5*(G_Okuchi97-GmeltFeO+GmetalFe+GmeltH2O)
#
#Reaction 5 Delta G reaction std state
# by difference
G5=GmeltH2-2.0*GmetalH
logK5=-G5/(Rgas*TK*log_to_ln)
GRT5=np.zeros(num)
for i in range(0,TKlength):
    GRT5[i]=G5[i]/(Rgas*TK[i])
#print('G5/RT =',GRT5)
#plt.figure(5)
#plt.plot(TK,logK5,label='2H metal = H2 silicate melt')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 5}}$ melt',fontsize=12)
#plt.legend()


#REACTION 6: FeSiO3 = FeO + SiO2 in melt
#Nearest model in Magma code for comparison: 2FeO+SiO2=Fe2SiO4 melt
#Oxides sum to formula units in Gf to a few percent (e.g., 3%), so reasonable
#to use this approximation, since FeO change in stoichiometry on either
#side of reaciton will tend to cancel.
G6magma=-log_to_ln*Rgas*TK*(-0.63+3103.0/TK)  #Magma code line 653
G6magma=-G6magma  #reverse reaction given on line 653 of Magma code
logK6magma=-G6magma/(Rgas*TK*log_to_ln)
#
#G FeO melt std state
# from NIST
ti=TK/1000.0
a=68.19920
b=-4.501232e-10
c=1.195227e-10
d=-1.064302e-11
e=-3.09268e-10
f=-281.4326
g=137.8377
h=-249.5321
HmeltFeO=-249.5321+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HmeltFeO=HmeltFeO*1000.0 #convert kJ to J
SmeltFeO=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GmeltFeO=HmeltFeO-TK*SmeltFeO  #apparent G of formation for Fe metal at T and 1 bar
#
#G solid FeSiO3 std state
# from Holland Powell (1998, J. Met. Geol.), using Joules
ko=0.3987*1000.0
k1=-0.6579e-5*1000.0
k2=-4.058*1000.0
k3=129.01*1000.0
GFerrosilite=-2388750.0+ko*(TK-298.15)+0.5*k1*(TK**2.0-298.015**2.0)+2.0*k2*(np.sqrt(TK)-np.sqrt(298.15))-k3*(1.0/TK-1.0/298.15)
GFerrosilite=GFerrosilite-TK*(190.6+1.0/(2.0*TK**3.0)*ko*(ko*TK**2.0+2.0*(k1*TK**3.0+k2*TK**(3.0/2.0)+k3)))
#
# G FeSiO3 melt std state
# from fusion data of Ueki and Imamori (2013, G^3)
dHfus=66.48*1000.0
dSfus=67.73
dCpfus=-94.19
Tfus=904.49
GmeltFeSiO3=GFerrosilite+dHfus+dCpfus*(TK-Tfus)-TK*(dSfus+dCpfus*ln(TK/Tfus))
#GmeltFeSiO3=1.5*GmeltFeSiO3
#
#Reaction 6 delta G is the difference
#G6=GmeltSiO2+GmeltFeO-GmeltFeSiO3
G6=G6magma  #on this reaction Magma code is more stable
#
logK6=-G6/(Rgas*TK*log_to_ln)
GRT6=np.zeros(num)
for i in range(0,TKlength):
    GRT6[i]=G6[i]/(Rgas*TK[i])
#print('G6/RT =',GRT6)
#plt.figure(6)
#plt.plot(TK,logK6,label='FeSiO3=FeO+SiO2 melt')
##plt.plot(TK,logK6magma,label='Magma code Fe2SiO4=2Feo+SiO2')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 6}}$ melt',fontsize=12)
#plt.legend()


#REACTION 7: 2H2O melt + Si metal = SiO2 melt + 2H2 melt
# have GmeltH2, GmeltH2O, GmeltSiO2, require GmetalSi.
# Get by difference with reaction 2 above:
#
#G of formation of molten Fe:
# from NIST
ti=TK/1000.0
a=46.024
b=-1884667e-8
c=6.094750e-9
d=-6.640301e-10
e=-8.246121e-9
f=-10.80543
g=72.54094
h=12.39502
HmetalFe=12.40+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HmetalFe=HmetalFe*1000.0 #convert kJ to J
SmetalFe=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GmetalFe=HmetalFe-TK*SmetalFe  #apparent G of formation of Fe liquid at T and 1 bar
#
#G for Si in Fe metal at T, 1 bar, pure standard state by difference
GmetalSi=2.0*(G2-GmeltFeO+0.5*GmeltSiO2+GmetalFe)
#
#G reaction 7 by difference
G7=2.0*GmeltH2+GmeltSiO2-GmetalSi-2.0*GmeltH2O
logK7=-G7/(Rgas*TK*log_to_ln)
GRT7=np.zeros(num)
for i in range(0,TKlength):
    GRT7[i]=G7[i]/(Rgas*TK[i])
#print('G7/RT =',GRT7)
#plt.figure(7)
#plt.plot(TK,logK7,label='2H2O melt + Si metal = SiO2 melt + 2H2 melt')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 7}}$ melt',fontsize=12)
#plt.legend()


#THERMODYNAMICS OF GAS REACTIONS
#
#REACTION 8: COgas + 1/2O2,gas = CO2,gas
#Data from NIST
#CO2 gas std state of 1 bar over two T ranges from 298 to 1200 and 1200 to 6000K
HgasCO2=np.zeros(TKlength)
SgasCO2=np.zeros(TKlength)
GgasCO2=np.zeros(TKlength)
i=0
while i < TKlength:
    if TK[i] < 1200.0:
        ti=TK[i]/1000.0
        a=24.99735
        b=55.18696
        c=-33.69137
        d=7.948387
        e=-0.136638
        f=-403.6075
        g=228.2431
        h=-393.5224
        HgasCO2[i]=-393.51+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasCO2[i]=HgasCO2[i]*1000.0 #convert kJ to J
        SgasCO2[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasCO2[i]=HgasCO2[i]-TK[i]*SgasCO2[i] #apparent G of formation of CO2 gas at T and 1 bar
    else:
        ti=TK[i]/1000.0
        a=58.16639
        b=2.720074
        c=-0.492289
        d=0.038844
        e=-6.447293
        f=-425.9186
        g=263.6125
        h=-393.5224
        HgasCO2[i]=-393.51+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasCO2[i]=HgasCO2[i]*1000.0 #convert kJ to J
        SgasCO2[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasCO2[i]=HgasCO2[i]-TK[i]*SgasCO2[i] #apparent G of formation of CO2 gas at T and 1 bar
    i=i+1
#
#O2 gas std state of 1 bar 700 to 2000K
ti=TK/1000.0
a=30.03235
b=8.772972
c=-3.988133
d=0.788313
e=-0.741599
f=-11.32468
g=236.1663
h=0.00
HgasO2=0.000+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HgasO2=HgasO2*1000.0 #convert kJ to J
SgasO2=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GgasO2=HgasO2-TK*SgasO2  #apparent G of formation of O2 gas at T and 1 bar
#
#CO gas std state of 1 bar at 298-1300 and 1300-6000K
HgasCO=np.zeros(TKlength)
SgasCO=np.zeros(TKlength)
GgasCO=np.zeros(TKlength)
i=0
while i < TKlength:
    if TK[i] < 1300.0:
        ti=TK[i]/1000.0
        a=25.56759
        b=6.096130
        c=4.054656
        d=-2.671301
        e=0.131021
        f=-118.0089
        g=227.3665
        h=-110.5271
        HgasCO[i]=-110.53+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasCO[i]=HgasCO[i]*1000.0 #convert kJ to J
        SgasCO[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasCO[i]=HgasCO[i]-TK[i]*SgasCO[i] #apparent G of formation of CO gas at T and 1 bar
    else:
        ti=TK[i]/1000.0
        a=35.15070
        b=1.300095
        c=-0.205921
        d=0.013550
        e=-3.282780
        f=-127.8375
        g=231.7120
        h=-110.5271
        HgasCO[i]=-110.53+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasCO[i]=HgasCO[i]*1000.0 #convert kJ to J
        SgasCO[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasCO[i]=HgasCO[i]-TK[i]*SgasCO[i]  #apparent G of formation of CO gas at T and 1 bar
    i=i+1
#
#G for reaction 8 at standard state of 1 bar and pure at T
G8=GgasCO2-GgasCO-0.5*GgasO2
logK8=-G8/(Rgas*TK*log_to_ln)
GRT8=np.zeros(num)
for i in range(0,TKlength):
    GRT8[i]=G8[i]/(Rgas*TK[i])
#print('G8/RT =',GRT8)
#plt.figure(8)
#plt.plot(TK,logK8,label='COgas + 1/2O2,gas = CO2,gas')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 8}}$ gas',fontsize=12)
#plt.legend()


#REACTION 9: CH4,gas + 1/2O2,gas = 2H2,gas + COgas
#CH4 std state data
# from NIST
HgasCH4=np.zeros(TKlength)
SgasCH4=np.zeros(TKlength)
GgasCH4=np.zeros(TKlength)
i=0
while i < TKlength:
    if TK[i] < 1300.0:
        ti=TK[i]/1000.0
        a=-0.703029
        b=108.4773
        c=-42.52157
        d=5.862788
        e=0.678565
        f=-76.84376
        g=158.7163
        h=-74.87310
        HgasCH4[i]=-74.873+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasCH4[i]=HgasCH4[i]*1000.0 #convert kJ to J
        SgasCH4[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasCH4[i]=HgasCH4[i]-TK[i]*SgasCH4[i] #apparent G of formation of CO gas at T and 1 bar
    else:
        ti=TK[i]/1000.0
        a=85.81217
        b=11.26467
        c=-2.114146
        d=0.138190
        e=-26.42221
        f=-153.5327
        g=224.4143
        h=-74.87310
        HgasCH4[i]=-74.873+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasCH4[i]=HgasCH4[i]*1000.0 #convert kJ to J
        SgasCH4[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasCH4[i]=HgasCH4[i]-TK[i]*SgasCH4[i]  #apparent G of formation of CO gas at T and 1 bar
    i=i+1
#
#H2 std state data
# from NIST
HgasH2=np.zeros(TKlength)
SgasH2=np.zeros(TKlength)
GgasH2=np.zeros(TKlength)
i=0
while i < TKlength:
    if TK[i] < 6000.0:
        ti=TK[i]/1000.0
        a=43.41356
        b=-4.293079
        c=1.272428
        d=-0.096876
        e=-20.533862
        f=-38.515158
        g=162.081354
        h=0.000
        HgasH2[i]=0.000+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasH2[i]=HgasH2[i]*1000.0 #convert kJ to J
        SgasH2[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasH2[i]=HgasH2[i]-TK[i]*SgasH2[i] #apparent G of formation of H2 gas at T and 1 bar
    if TK[i] < 2500.0:
        ti=TK[i]/1000.0
        a=18.563083
        b=12.257357
        c=-2.859786
        d=0.268238
        e=1.977990
        f=-1.147438
        g=156.288133
        h=0.00
        HgasH2[i]=0.000+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasH2[i]=HgasH2[i]*1000.0 #convert kJ to J
        SgasH2[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasH2[i]=HgasH2[i]-TK[i]*SgasH2[i] #apparent G of formation of H2 gas at T and 1 bar
    if TK[i] < 1000.0:
        ti=TK[i]/1000.0
        a=33.066178
        b=-11.363417
        c=11.432816
        d=-2.772874
        e=-0.158558
        f=-9.980797
        g=172.707974
        h=0.00
        HgasH2[i]=0.000+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasH2[i]=HgasH2[i]*1000.0 #convert kJ to J
        SgasH2[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasH2[i]=HgasH2[i]-TK[i]*SgasH2[i] #apparent G of formation of H2 gas at T and 1 bar
    i=i+1
#
#G for reaction at standard state of 1 bar, pure, and T
G9=2.0*GgasH2+GgasCO-GgasCH4-0.5*GgasO2
logK9=-G9/(Rgas*TK*log_to_ln)
GRT9=np.zeros(num)
for i in range(0,TKlength):
    GRT9[i]=G9[i]/(Rgas*TK[i])
#print('G9/RT =',GRT9)
#plt.figure(9)
#plt.plot(TK,logK9,label='CH4,gas + 1/2O2,gas = 2H2,gas + COgas')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 9}}$ gas',fontsize=12)
#plt.legend()


#REACTION 10: H2,gas + 1/2O2,gas = H2Ogas
G10=GgasH2O-0.5*GgasO2-GgasH2
logK10=-G10/(Rgas*TK*log_to_ln)
GRT10=np.zeros(num)
for i in range(0,TKlength):
    GRT10[i]=G10[i]/(Rgas*TK[i])
#print('G10/RT =',GRT10)
#plt.figure(10)
#plt.plot(TK,logK10,label='H2,gas + 1/2O2,gas = H2Ogas')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 10}}$ gas',fontsize=12)
#plt.legend()


#Collect additional thermodynamic data for reactions 11 through 18 involving
# evaporation of oxides from the melt:
# First, the gases produced by SiO2 and FeO melt already defined above:
#
#SiO gas std state of 1 bar, T, pure
# from NIST from 298-1100 and 1100-6000K
HgasSiO=np.zeros(TKlength)
SgasSiO=np.zeros(TKlength)
GgasSiO=np.zeros(TKlength)
i=0
while i < TKlength:
    if TK[i] < 1100.0:
        ti=TK[i]/1000.0
        a=19.52413
        b=37.46370
        c=-30.51805
        d=9.094050
        e=0.148934
        f=-107.1514
        g=226.1506
        h=-100.4160
        HgasSiO[i]=-100.42+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasSiO[i]=HgasSiO[i]*1000.0 #convert kJ to J
        SgasSiO[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasSiO[i]=HgasSiO[i]-TK[i]*SgasSiO[i] #apparent G of formation of SiO gas at T and 1 bar
    else:
        ti=TK[i]/1000.0
        a=35.69893
        b=1.731252
        c=-0.509348
        d=0.059404
        e=-1.248055
        f=-114.6019
        g=249.1911
        h=-100.416
        HgasSiO[i]=-100.42+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasSiO[i]=HgasSiO[i]*1000.0 #convert kJ to J
        SgasSiO[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasSiO[i]=HgasSiO[i]-TK[i]*SgasSiO[i] #apparent G of formation of SiO gas at T and 1 bar
    i=i+1
#
#Fe gas
# from NIST 3000 to 6000K
ti=TK/1000.0
a=11.29253
b=6.989707
c=-1.110305
d=0.122354
e=5.689278
f=423.5380
g=206.3591
h=415.4716
HgasFe=415.47+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HgasFe=HgasFe*1000.0 #convert kJ to J
SgasFe=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GgasFe=HgasFe-TK*SgasFe  #apparent G of formation of Na gas at T and 1 bar
#
#Mg gas
# from NIST
HgasMg=np.zeros(TKlength)
SgasMg=np.zeros(TKlength)
GgasMg=np.zeros(TKlength)
i=0
while i < TKlength:
    if TK[i] < 2200.0:
        ti=TK[i]/1000.0
        a=20.77306
        b=0.035592
        c=-0.031917
        d=0.009109
        e=0.000461
        f=140.9071
        g=173.7799
        h=147.1002
        HgasMg[i]=147.1+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasMg[i]=HgasMg[i]*1000.0 #convert kJ to J
        SgasMg[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasMg[i]=HgasMg[i]-TK[i]*SgasMg[i] #apparent G of formation of Mg gas at T and 1 bar
    else:
        ti=TK[i]/1000.0
        a=47.60848
        b=-15.40875
        c=2.875965
        d=-0.120806
        e=-27.01764
        f=97.40017
        g=177.2305
        h=147.1002
        HgasMg[i]=147.1+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasMg[i]=HgasMg[i]*1000.0 #convert kJ to J
        SgasMg[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasMg[i]=HgasMg[i]-TK[i]*SgasMg[i]  #apparent G of formation of Mg gas at T and 1 bar
    i=i+1
#
#Na2O melt
# from NIST for 1400 to 3000K
ti=TK/1000.0
a=104.600
b=9.909135e-10
c=-6.022074e-10
d=1.113058e-10
e=2.362827e-11
f=-404.0296
g=218.1902
h=-372.8434
HmeltNa2O=-372.84+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HmeltNa2O=HmeltNa2O*1000.0 #convert kJ to J
SmeltNa2O=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GmeltNa2O=HmeltNa2O-TK*SmeltNa2O  #apparent G of formation of Na2O liquid at T and 1 bar
#
#Na gas
# from NIST, 1170 to 6000K
ti=TK/1000.0
a=20.80573
b=0.277206
c=-0.392086
d=0.119634
e=-0.008879
f=101.0386
g=178.7095
h=107.2999
HgasNa=107.3+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
HgasNa=HgasNa*1000.0 #convert kJ to J
SgasNa=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
GgasNa=HgasNa-TK*SgasNa  #apparent G of formation of Na gas at T and 1 bar


#THERMODYNAMICS OF EVAPORATION AND CONDENSATION
#
#REACTION 11:  FeO = Fegas + 1/2O2,gas
G11=0.5*GgasO2+GgasFe-GmeltFeO
logK11=-G11/(Rgas*TK*log_to_ln)
logK11magma=12.06-44992.0/TK  #Magma code version for comparison
GRT11=np.zeros(num)
for i in range(0,TKlength):
    GRT11[i]=G11[i]/(Rgas*TK[i])
#print('G11/RT =',GRT11)
#plt.figure(11)
#plt.plot(TK,logK11,label='FeO = Fegas + 1/2O2,gas')
#plt.plot(TK,logK11magma,label='Magma code',linestyle='--')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 11}}$ gas',fontsize=12)
#plt.legend()


#REACTION 12: MgO = Mg,gas + 1/2O2,gas
G12=0.5*GgasO2+GgasMg-GmeltMgO
logK12=-G12/(Rgas*TK*log_to_ln)
logK12magma=12.56-46992/TK  #Magma code version for comparison
GRT12=np.zeros(num)
for i in range(0,TKlength):
    GRT12[i]=G12[i]/(Rgas*TK[i])
#print('G12/RT =',GRT12)
#plt.figure(12)
#plt.plot(TK,logK12,label='MgO = Mg,gas + 1/2O2,gas')
#plt.plot(TK,logK12magma,label='Magma code',linestyle='--')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 12}}$ gas',fontsize=12)
#plt.legend()


#REACTION 13: SiO2 = SiO,gas +1/2O2
G13=0.5*GgasO2+GgasSiO-GmeltSiO2
logK13=-G13/(Rgas*TK*log_to_ln)
GRT13=np.zeros(num)
for i in range(0,num-1):
    GRT13[i]=G13[i]/(Rgas*TK[i])
#print('G13/RT =',GRT13)
#plt.figure(13)
#plt.plot(TK,logK13,label='SiO2 melt = SiO,gas +1/2O2,gas')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 13}}$ gas',fontsize=12)
#plt.legend()


#REACTION 14: Na2O = 2Na gas + 1/2O2
G14=0.5*GgasO2+2.0*GgasNa-GmeltNa2O
logK14=-G14/(Rgas*TK*log_to_ln)
logK14magma=-(-15.56+40286.0/TK)
GRT14=np.zeros(num)
for i in range(0,num-1):
    GRT14=G14/(Rgas*TK)
#print('G14/RT =',GRT14)
#plt.figure(14)
#plt.plot(TK,logK14,label='Na2O = 2Na gas + 1/2O2')
#plt.plot(TK,logK14magma,label='Magma code',linestyle='--')  #Magma code version for comparison
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 14}}$ gas',fontsize=12)
#plt.legend()


#REACTION 15: H2,gas = H2,silicate
# from Hirschmann et al. (2012) for the reaction H2 gas = H2 melt, no T dependence given
#lnk=-12.5-0.76*1.0e-4  #lnk for a standard state of 1 bar, or 1e-4 GPa
#G15=-Rgas*TK*lnk   #Delta G of the Hirschmann reaction at 1 bar
G15=GmeltH2-GgasH2  #Self consistent with above
logK15=-G15/(Rgas*TK*log_to_ln)   #at 1 bar
#G15prov=GmeltH2-GgasH2
#logK15prov=-G15prov/(Rgas*TK*log_to_ln)
GRT15=np.zeros(num)
for i in range(0,TKlength):
    GRT15[i]=G15[i]/(Rgas*TK[i])
#print('G15/RT =',GRT15)
#plt.figure(15)
#plt.plot(TK,logK15,label='H2,gas = H2,silicate')
#plt.plot(TK,logK15prov,label='using thermo data')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 15}}$ gas',fontsize=12)
#plt.legend()
#plt.show()


#REACTION 16: H2Ogas = H2Osilicate
#G16 = -Rgas*TK*(2565.0/TK -14.21)  #Rxn G for H2O vapor = H2O melt from Moore et al. (1998)
G16=GmeltH2O-GgasH2O  #Self consistent with above
logK16=-G16/(Rgas*TK*log_to_ln)
#G16prov=GmeltH2O-GgasH2O  #Using extracted melt H2O thermodynamic data with NIST H2Ogas data
#logK16prov=-G16/(Rgas*TK*log_to_ln)
GRT16=np.zeros(num)
for i in range(0,TKlength):
    GRT16[i]=G16[i]/(Rgas*TK[i])
#print('G16/RT =',GRT16)
#plt.figure(16)
#plt.plot(TK,logK16,label='H2Ogas = H2Osilicate')
##plt.plot(TK,logK16prov,label='Extracted silicate H2O data',linestyle='--')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 16}}$ gas',fontsize=12)
#plt.legend()


#REACTION 17: COgas = CO melt
# data are scarce, so we use Table 2 of Hirschmann (2016, Am Min) to suggest
# that CO solubility is about 1/3 that of CO2 (see below for G18)
G18=5200.0-TK*(-119.77)
logK18=-G18/(Rgas*TK*log_to_ln)
logK17=logK18-log(3.0)
G17=-Rgas*TK*log_to_ln*logK17
GRT17=np.zeros(num)
for i in range(0,TKlength):
    GRT17[i]=G17[i]/(Rgas*TK[i])
#print('G17/RT =',GRT17)
#plt.figure(17)
#plt.plot(TK,logK17,label='CO = COmelt')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 17}}$ gas',fontsize=12)
#plt.legend()


#REACTION 18: CO2,gas = CO2,melt
# Hrxn CO2 gas = CO2 melt = 5200 J/mole (Pan et al., 1991) at 1473K
# lnK for this reaction at 1473K = -14.83 (Pan et al., 1991)
# Grxn =-R*1473K*lnk = 181626.027 J/mole
# Grxn=5200-T*Srxn, solve for Srxn = -119.77 J/mole
# Grxn = 5200 -T*(-119.77)
G18=5200.0-TK*(-119.77)
logK18=-G18/(Rgas*TK*log_to_ln)
GRT18=np.zeros(num)
for i in range(0,TKlength):
    GRT18[i]=G18[i]/(Rgas*TK[i])
#print('G18/RT =',GRT18)
#plt.figure(18)
#plt.plot(TK,logK18,label='CO2 = CO2melt')
# Add X and y Label
#plt.xlabel('T (K)', fontsize=12)
#plt.ylabel('$\mathrm{Log(K)}_{\mathrm{rxn 18}}$ gas',fontsize=12)
#plt.legend()

#REACTION 19: SiO + 2H2 = SiH4 + 1/2O2 in vapor phase
#G SiH4 gas from NIST: 298 to 1300 K and 1300 to 6000K
HgasSiH4=np.zeros(TKlength)
SgasSiH4=np.zeros(TKlength)
GgasSiH4=np.zeros(TKlength)
i=0
while i < TKlength:
    if TK[i] < 1300.0:
        ti=TK[i]/1000.0
        a=6.060189
        b=139.9632
        c=-77.88474
        d=16.24095
        e=0.135509
        f=27.39081
        g=174.3351
        h=34.30905
        HgasSiH4[i]=34.30905+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasSiH4[i]=HgasSiH4[i]*1000.0 #convert kJ to J
        SgasSiH4[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasSiH4[i]=HgasSiH4[i]-TK[i]*SgasSiH4[i]  #apparent G of formation for H2O gas at T and 1 bar
    else:
        ti=TK[i]/1000.0
        a=99.84949
        b=4.251530
        c=-0.809269
        d=0.053437
        e=-20.39005
        f=-40.54016
        g=266.8015
        h=34.30905
        HgasSiH4[i]=34.30905+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasSiH4[i]=HgasSiH4[i]*1000.0 #convert kJ to J
        SgasSiH4[i]=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GgasSiH4[i]=HgasSiH4[i]-TK[i]*SgasSiH4[i]  #apparent G of formation for H2O gas at T and 1 bar
    i=i+1
#
G19=0.5*GgasO2 + GgasSiH4 -2.0*GgasH2 - GgasSiO  #Self consistent with above
logK19=-G19/(Rgas*TK*log_to_ln)   #at 1 bar
GRT19=np.zeros(num)
for i in range(0,TKlength):
    GRT19[i]=G19[i]/(Rgas*TK[i])

#--------------------------------------------------------------------------------------------------------
#PRINT RESULTING G/(RT) TERMS TO FILES:
#
a_file = open('G1_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G1/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT1[i])
a_file.close()

a_file = open('G2_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G2/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT2[i])
a_file.close()

a_file = open('G3_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G3/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT3[i])
a_file.close()

a_file = open('G4_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G4/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT4[i])
a_file.close()

a_file = open('G5_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G5/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT5[i])
a_file.close()

a_file = open('G6_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G6/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT6[i])
a_file.close()

a_file = open('G7_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G7/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT7[i])
a_file.close()

a_file = open('G8_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G8/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT8[i])
a_file.close()

a_file = open('G9_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G9/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT9[i])
a_file.close()

a_file = open('G10_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G10/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT10[i])
a_file.close()

a_file = open('G11_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G11/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT11[i])
a_file.close()

a_file = open('G12_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G12/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT12[i])
a_file.close()

a_file = open('G13_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G13/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT13[i])
a_file.close()

a_file = open('G14_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G14/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT14[i])
a_file.close()

a_file = open('G15_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G15/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT15[i])
a_file.close()

a_file = open('G16_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G16/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT16[i])
a_file.close()

a_file = open('G17_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G17/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT17[i])
a_file.close()

a_file = open('G18_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G18/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT18[i])
a_file.close()

a_file = open('G19_RT.txt', 'w')
a_file.write("T(K)      ")
a_file.write("")
a_file.write("  G19/(RT)\n")
for i in range(0,num):
    a_file.write("%10.5e " % TK[i])
    a_file.write("")
    a_file.write("%13.8e\n" % GRT19[i])
a_file.close()

print('')
print('Thermodynamic data in Gi_RT.txt files')

# SHOW PLOTS OF LOGK_EQ VS TEMPERATURE FOR REACTIONS
##plt.show()

#--------------------------------------------------------------------------------------------------------
#EQUATIONS TO BE SOLVED SIMULTANEOUSLY BY OPTIMIZATION ARE DEFINED HERE
#
#Calculate total moles of chemical constituents, these are the extensive constraints on the system
#
nSi=var[27]*(var[1]+var[2]+var[4]+var[6])+var[12]*var[28]+(var[23]+var[25])*var[26]
nMg=var[27]*(var[0]+var[2])+var[22]*var[26]
nO=var[27]*(var[0]+2.0*var[1]+3.0*var[2]+var[3]+3.0*var[4]+var[5]+3.0*var[6]+var[8]+var[9]+2.0*var[10]) \
    + var[28]*var[13]+var[26]*(var[23]+2.0*var[17]+2.0*var[19]+var[20]+var[16])
nFe=var[27]*(var[3]+var[4])+var[28]*var[11]+var[26]*var[21]
nH=var[28]*var[14]+var[27]*(2.0*var[7]+2.0*var[8])+var[26]*(2.0*var[15]+2.0*var[20]+4.0*var[18]+4.0*var[25])
nNa=var[27]*(2.0*var[5]+2.0*var[6])+var[26]*var[24]
nC=var[27]*(var[9]+var[10])+var[26]*(var[18]+var[17]+var[16])

#Auxiliary manual method for specifying moles of elements comprising system
#nSi=4.79
#nMg=4.7655
#nH=19.98
#nO=14.5665
#nFe=3.15
#nNa=0.08
#nC=0.01

#make a 2D array containing the min and max for each variable listed above.
#
bounds=np.zeros((numvar,2))

#assign ranges, assume all variables are mole fractions here
# Use first two for normal usage, second two for reprocessing previous solutions
for i in range(0,numvar):
    bounds[i,0]=1.0e-20
    bounds[i,1]=0.99999
    #bounds[i,0]=0.8*var[i]
    #bounds[i,1]=1.2*var[i]

#...then modify range for the moles of the three phases comprising the planet
melt_min=0.5 #0.9
melt_max=2.0  #1.1
bounds[26,0]=var[26]*1.0e-20
bounds[26,1]=var[26]*50.0
bounds[27,0]=var[27]*melt_min
bounds[27,1]=var[27]*melt_max
bounds[28,0]=var[28]*0.5
bounds[28,1]=var[28]*2.0

#Modify range for pressures
bounds[numvar-1,0]=1.0e-03
bounds[numvar-1,1]=900000.0


#Additional constraints to make the problem simpler...
#
#bounds[0,0]=1.0e-12  #MgO
#bounds[0,1]=0.35
#bounds[1,0]=0.15  #SiO2
#bounds[1,1]=0.03
#bounds[2,0]=1.0e-12  #MgSiO3
#bounds[2,1]=0.5
#bounds[3,0]=1.0e-12 #FeO
#bounds[3,1]=0.01
#bounds[4,0]=1.0e-12 #FeSiO3
#bounds[4,1]=0.05
#bounds[5,0]=1.0e-10 #Na2O in melt
#bounds[5,1]=0.01
#bounds[6,0]=1.0e-16 #Na2SiO3
#bounds[6,1]=0.005
bounds[7,0]=1.0e-15  #H2 melt
bounds[7,1]=0.4  #This helps push rxn 15 to equilibrium, usually use 0.2
bounds[8,0]=1.0e-12  #H2O
bounds[8,1]=0.4      #Limit this value to account for its use in calculating xB
#bounds[11,0]=0.10  #metal Fe
#bounds[11,1]=0.99
#bounds[9,0]=1.0e-12  #CO melt
#bounds[9,1]=0.1
#bounds[10,0]=1.0e-12  #CO2 melt
#bounds[10,1]=0.1
#bounds[19,0]=1.0e-15
#bounds[19,1]=1.0e-2
bounds[21,0]=1.0e-15  #Fe gas, 1e-15 is usual
bounds[21,1]=1.0
bounds[22,0]=1.0e-15 #Mg gas
bounds[22,1]=1.0
bounds[23,0]=1.0e-15 #SiO gas, usually 1e-15
bounds[23,1]=1.0  #0.2 to 0.3 is reasonable for general cases with H2

#write the cost function to be minimized.  Each individual fi should be zero at equilibrium.
#
#incude a tuple with: pressure(bar), and Reaction Gibbs free energies/(RT) for all reactions
GRT1_T=GRT1[nT]
#GRT2_T=GRT2[nT]
GRT2_T=GRT2[TKlength-1] # proxy for core-mantle boundary using maximum T as T
GRT3_T=GRT3[nT]
#GRT4_T=GRT4[nT]
GRT4_T=GRT4[TKlength-1] # proxy for core-mantle boundary using maximum T as T
#GRT5_T=GRT5[nT]
GRT5_T=GRT5[TKlength-1] # proxy for core-mantle boundary using maximum T as T
GRT6_T=GRT6[nT]
#GRT7_T=GRT7[nT]
GRT7_T=GRT7[TKlength-1] # proxy for core-mantle boundary using maximum T as T
GRT8_T=GRT8[nT]
GRT9_T=GRT9[nT]
GRT10_T=GRT10[nT]
GRT11_T=GRT11[nT]
GRT12_T=GRT12[nT]
GRT13_T=GRT13[nT]
GRT14_T=GRT14[nT]
GRT15_T=GRT15[nT]
GRT16_T=GRT16[nT]
GRT17_T=GRT17[nT]
GRT18_T=GRT18[nT]
GRT19_T=GRT19[nT]

#Add pressure correction for condensed phases for the Delta G of reactions:
#Pressure is in bar, and corrections are Joules/bar
#DeltaVi refers to delta V of condensed phases only

#Intramelt reactions
#DeltaV1=(v_Jbar[1]+v_Jbar[5]-v_Jbar[6])
#GRT1_T=GRT1_T+(DeltaV1*P_initial)/(Rgas*TK[nT])
#
#DeltaV2=0.5*v_Jbar[12]+v_Jbar[3]-v_Jbar[11]-0.5*v_Jbar[1]
#GRT2_T=GRT2_T+(DeltaV2*P_initial)/(Rgas*TK[nT])
#
#DeltaV3=v_Jbar[1]+v_Jbar[0]-v_Jbar[2]
#GRT3_T=GRT3_T+(DeltaV3*P_initial)/(Rgas*TK[nT])
#
#DeltaV4=0.5*v_Jbar[1]-0.5*v_Jbar[12]-v_Jbar[13]
#GRT4_T=GRT4_T+(DeltaV4*P_initial)/(Rgas*TK[nT])
#
#DeltaV5=0.0
#GRT5_T=GRT5_T+(DeltaV5*P_initial)/(Rgas*TK[nT])
#
#DeltaV6=v_Jbar[1]+v_Jbar[3]-v_Jbar[4]
#GRT6_T=GRT6_T+(DeltaV6*P_initial)/(Rgas*TK[nT])
#
#DeltaV7=0.0
#GRT7_T=GRT7_T+(DeltaV7*P_initial)/(Rgas*TK[nT])

#Pressure corrections for evaporation reactions, negative from nu_i being for reactant oxide,
#and so the chemical potential for the oxide reactant is being subracted from the delta G std state.
#DeltaV11=-v_Jbar[3] #J/bar, -FeO melt partial molar volume
#DeltaV12=-v_Jbar[0] #J/bar, -MgO melt partial molar volume
#DeltaV13=-v_Jbar[1] #J/bar, -SiO2 melt partial molar volume
#DeltaV14=-v_Jbar[5] #J/bar, -Na2O melt partial molar volume
# Use these for no pressure corrections for evaporation
DeltaV11=0.0 #J/bar
DeltaV12=0.0 #J/bar
DeltaV13=0.0 #J/bar
DeltaV14=0.0 #J/bar


#***********SET Fe SPECIATION EQUAL TO Mg SPECIATION***************************
#GRT6_T=GRT3_T

#Pressure (bar) is the first element of this truple, with the rest being the DeltaG/(RT) terms
#
const=(P_initial,GRT1_T,GRT2_T,GRT3_T,GRT4_T,GRT5_T,GRT6_T,GRT7_T,GRT8_T,GRT9_T,GRT10_T,GRT11_T, \
GRT12_T,GRT13_T,GRT14_T,GRT15_T,GRT16_T,GRT17_T,GRT18_T,GRT19_T,nSi,nMg,nO,nFe,nH,nNa,nC)

#--------------------------------------------------------------------------------------------------------
#PRE-OPTIMIZATION - print deviations from equilibrium and mass balance before optimization.  The terms
#after the G/(RT) terms are the total pressure corrections, so the sum of natural logs represent
#the equilibrium KD values, rather than Keq (i.e., products of mole fractions).
#
P=const[0]
Pstd=1.0
lngSi=-6.65*1873.0/T_max-(12.41*1873.0/T_max)*ln(1.0-var[12])
lngSi=lngSi-((-5.0*1873.0/T_max)*var[13]*(1.0+ln(1-var[13])/var[13]-1.0/(1.0-var[12])))
lngSi=lngSi+(-5.0*1873.0/T_max)*var[13]**2.0*var[12]*(1.0/(1.0-var[12])+1.0/(1.0-var[13])+var[12]/(2.0*(1.0-var[12])**2.0)-1.0)
lngO=(4.29-16500.0/T_max)-(-1.0*1873.0/T_max)*ln(1.0-var[13])
lngO=lngO-((-5.0*1873.0/T_max)*var[12]*(1.0+ln(1-var[12])/var[12]-1.0/(1.0-var[13])))
lngO=lngO+(-5.0*1873.0/T_max)*var[12]**2.0*var[13]*(1.0/(1.0-var[13])+1.0/(1.0-var[12])+var[13]/(2.0*(1.0-var[13])**2.0)-1.0)
#lngH2=(1.0-var[7])**2.0*74829.6/(8.3144*T_max) # Regular solution model for xH2 in melt
lngH2=0.0
#lngH2Omelt=(1.0-var[8])**2.0*74826.0/(8.3144*T_max) # Regulat solution model for xH2O in melt
lngH2Omelt=0.0
#lngHmetal=(1.0-var[14])**2.0*(-1.9) # Regular solution model for H in metal with epsilon independent of T
x_light=var[12]+var[13] # Si + O mole fractions in metal, for psuedo-ternary mixing of H and (Si+O) with Fe below
#lngHmetal=-3.8*x_light*(1.0+ln(1.0-x_light)/x_light-1.0/(1.0-var[14]))
#lngHmetal=lngHmetal+3.8*x_light**2.0*var[14]*(1.0/(1.0-var[14])+1.0/(1.0-x_light)+var[14]/(2.0*(1.0-var[14])**2.0)-1.0)
lngHmetal=0.0
#print("Initial ln gamma H metal=",lngHmetal)
# Calculate H2O mole fraction on single-oxygen basis
r_H2Omelt=var[8]/(1.0-var[8])*(1.0/3.0) # assumes oxygen pfu anhydrous = 3, xB=r_H2Omelt/(1+r_H2Omelt)
xB=r_H2Omelt/(1.0+r_H2Omelt)
f1_ini=ln(var[5])+ln(var[1])-ln(var[6])+GRT1_T
f2_ini=0.5*ln(var[12])+0.5*lngSi+ln(var[3])-0.5*ln(var[1])-ln(var[11])+GRT2_T
f3_ini=ln(var[0])+ln(var[1])-ln(var[2])+GRT3_T
f4_ini=0.5*ln(var[1])-ln(var[13])-lngO-0.5*ln(var[12])-0.5*lngSi+GRT4_T
f5_ini=ln(var[7]) + lngH2 -2.0*ln(var[14])-2.0*lngHmetal +GRT5_T

f6_ini=ln(var[3])+ln(var[1])-ln(var[4])+GRT6_T
#f7_ini=ln(var[1])+2.0*ln(var[7])+2.0*lngH2-2.0*ln(var[8])-2.0*lngH2Omelt-ln(var[12])-lngSi+GRT7_T # aH2O = xH2O
f7_ini=ln(var[1])+2.0*ln(var[7])+2.0*lngH2-4.0*ln(xB)-2.0*lngH2Omelt-ln(var[12])-lngSi+GRT7_T # aH2O = xB^2
f8_ini=ln(var[17])-ln(var[16])-0.5*ln(var[19])+GRT8_T+ln(P/Pstd)-ln(P/Pstd)-0.5*ln(P/Pstd)
f9_ini=2.0*ln(var[15])+ln(var[16])-ln(var[18])-0.5*ln(var[19])+GRT9_T+2.0*ln(P/Pstd)+ln(P/Pstd)-ln(P/Pstd)-0.5*ln(P/Pstd)
f10_ini=ln(var[20])-0.5*ln(var[19])-ln(var[15])+GRT10_T+ln(P/Pstd)-0.5*ln(P/Pstd)-ln(P/Pstd)
f11_ini=0.5*ln(var[19])+ln(var[21])-ln(var[3])+GRT11_T+0.5*ln(P/Pstd)+ln(P/Pstd)
f12_ini=0.5*ln(var[19])+ln(var[22])-ln(var[0])+GRT12_T+0.5*ln(P/Pstd)+ln(P/Pstd)
f13_ini=0.5*ln(var[19])+ln(var[23])-ln(var[1])+GRT13_T+0.5*ln(P/Pstd)+ln(P/Pstd)
f14_ini=0.5*ln(var[19])+2.0*ln(var[24])-ln(var[5])+GRT14_T+0.5*ln(P/Pstd)+2.0*ln(P/Pstd)
#f15_ini=ln(var[7])-ln(var[15])+GRT15_T #fixed KD version
#f15_ini=ln(var[7])-ln(var[15])+GRT15_T-ln(P/Pstd)
f15_ini=ln(var[7])+lngH2-ln(var[15])+GRT15_T-ln(1.0e4/Pstd) #Fix KD at 3GPa
f16_ini=2.0*ln(xB)-ln(var[20])+GRT16_T-ln(P/Pstd) # aH2O = xB^2
#f16_ini=ln(var[8])+lngH2Omelt-ln(var[20])+GRT16_T-ln(P/Pstd) # aH2O =xH2O
f17_ini=ln(var[9])-ln(var[16])+GRT17_T-ln(P/Pstd)
f18_ini=ln(var[10])-ln(var[17])+GRT18_T-ln(P/Pstd)
f19_ini=ln(var[25])+0.5*ln(var[19])-ln(var[23])-2.0*ln(var[15])+GRT19_T+ln(P/Pstd)+0.5*ln(P/Pstd)-ln(P/Pstd)-2.0*ln(P/Pstd) # SiO+2H2=SiH4+1/2O2

#Mass balance for elements
f20_ini=nSi-(var[27]*(var[1]+var[2]+var[4]+var[6])+var[12]*var[28]+(var[23]+var[25])*var[26])
f21_ini=nMg-(var[27]*(var[0]+var[2])+var[22]*var[26])
f22_ini=nO-(var[27]*(var[0]+2.0*var[1]+3.0*var[2]+var[3]+3.0*var[4]+var[5]+3.0*var[6]+var[8]+var[9]+2.0*var[10]) \
        + var[28]*var[13]+var[26]*(var[23]+2.0*var[17]+2.0*var[19]+var[20]+var[16]) )
f23_ini=nFe-(var[27]*(var[3]+var[4])+var[28]*var[11]+var[26]*var[21])
f24_ini=nH-(var[28]*var[14]+var[27]*(2.0*var[7]+2.0*var[8])+var[26]*(2.0*var[15]+2.0*var[20]+4.0*var[18]+4.0*var[25]))
f25_ini=nNa-(var[27]*(2.0*var[5]+2.0*var[6])+var[26]*var[24])
f26_ini=nC-(var[27]*(var[9]+var[10])+var[26]*(var[18]+var[17]+var[16]))

#Summing constraint on mole fractions
f27_ini=1.0-var[0]-var[1]-var[2]-var[3]-var[4]-var[5]-var[6]-var[7]-var[8]-var[9]-var[10]
f28_ini=1.0-var[11]-var[12]-var[13]-var[14]
f29_ini=1.0-var[15]-var[16]-var[17]-var[18]-var[19]-var[20]-var[21]-var[22]-var[23]-var[24]-var[25]

# Printing the elements of the objective function first is used for troubleshooting
#print('')
#print('Initial deviations from equilibrium:')
#print('f1 =', (f1_ini))
#print('f2 =', (f2_ini))
#print('f3 =', (f3_ini))
#print('f4 =', (f4_ini))
#print('f5 =', (f5_ini))
#print('f6 =', (f6_ini))
#print('f7 =', (f7_ini))
#print('f8=', (f8_ini))
#print('f9=', (f9_ini))
#print('f10 =', (f10_ini))
#print('f11 =', (f11_ini))
#print('f12 =', (f12_ini))
#print('f13 =', (f13_ini))
#print('f14 =', (f14_ini))
#print('f15 =', (f15_ini))
#print('f16 =', (f16_ini))
#print('f17 =', (f17_ini))
#print('f18 =', (f18_ini))
sum_thermo_i=abs(f1_ini)+abs(f2_ini)+abs(f3_ini)+abs(f4_ini)+abs(f5_ini)+abs(f6_ini)+abs(f7_ini)+abs(f8_ini)+abs(f9_ini)+abs(f10_ini)
sum_thermo_i=sum_thermo_i+abs(f11_ini)+abs(f12_ini)+abs(f13_ini)+abs(f14_ini)+abs(f15_ini)+abs(f16_ini)+abs(f17_ini)+abs(f18_ini)+abs(f19_ini)
#print('Sum abs(f1) to abs(f18) =',sum_thermo_i)
#print('')
#print('Initial deviations from mass balance:')
#print('f19=',abs(f19_ini))
#print('f20=',abs(f20_ini))
#print('f21=',abs(f21_ini))
#print('f22=',abs(f22_ini))
#print('f23=',abs(f23_ini))
#print('f24=',abs(f24_ini))
#print('f25=',abs(f25_ini))
#print('')
#print('Initial deviations from sum phase mole fractions = 1:')
#print('f26 =',abs(f26_ini))
#print('f27 =',abs(f27_ini))
#print('f28 =',abs(f28_ini))

#Find maximum among f1 through f18 for scaling factors
max_f=f1_ini
if abs(f2_ini) > max_f:
    max_f=abs(f2_ini)
if abs(f3_ini) > max_f:
    max_f=abs(f3_ini)
if abs(f4_ini) > max_f:
    max_f=abs(f4_ini)
if abs(f5_ini) > max_f:
    max_f=abs(f5_ini)
if abs(f6_ini) > max_f:
    max_f=abs(f6_ini)
if abs(f7_ini) > max_f:
    max_f=abs(f7_ini)
if abs(f8_ini) > max_f:
    max_f=abs(f8_ini)
if abs(f9_ini) > max_f:
    max_f=abs(f9_ini)
if abs(f10_ini) > max_f:
    max_f=abs(f10_ini)
if abs(f11_ini) > max_f:
    max_f=abs(f11_ini)
if abs(f12_ini) > max_f:
    max_f=abs(f12_ini)
if abs(f13_ini) > max_f:
    max_f=abs(f13_ini)
if abs(f14_ini) > max_f:
    max_f=abs(f14_ini)
if abs(f15_ini) > max_f:
    max_f=abs(f15_ini)
if abs(f16_ini) > max_f:
    max_f=abs(f16_ini)
if abs(f17_ini) > max_f:
    max_f=abs(f17_ini)
if abs(f18_ini) > max_f:
    max_f=abs(f18_ini)
if abs(f19_ini) > max_f:
    max_f=abs(f19_ini)
print('')
print('Maximum abs value for objective function f values =',max_f)

wgas=1.0/max_f  #weight for thermodynamic equations (0.2)
wtn=wt_massbalance*wgas   #weight for mass balance (0.005)
wtx=wt_summing*wgas  #weight for summing constraints on mole fractions (0.85)

print('Scaling for f1 through f19 =',wgas)
print('Scaling for f20 through f26 =',wtn)
print('Scaling for f27 through f29 =',wtx)

#--------------------------------------------------------------------------------------------------------
#Function to minimize in order to solve for mole fractions and phase abundances at equilibrium
def func(var):
    P=var[29]
    Pstd=1.0
    
    #W's are tuning factors for groups of reactions, gas, solubility, melts
    watm=watm_m*wgas
    wsolub=wsolub_m*wgas
    wmelt=wmelt_m*wgas
    wevap=wevap_m*wgas
    lngSi=-6.65*1873.0/T_max-(12.41*1873.0/T_max)*ln(1.0-var[12])
    lngSi=lngSi-((-5.0*1873.0/T_max)*var[13]*(1.0+ln(1-var[13])/var[13]-1.0/(1.0-var[12])))
    lngSi=lngSi+(-5.0*1873.0/T_max)*var[13]**2.0*var[12]*(1.0/(1.0-var[12])+1.0/(1.0-var[13])+var[12]/(2.0*(1.0-var[12])**2.0)-1.0)
    lngO=(4.29-16500.0/T_max)-(-1.0*1873.0/T_max)*ln(1.0-var[13])
    lngO=lngO-((-5.0*1873.0/T_max)*var[12]*(1.0+ln(1-var[12])/var[12]-1.0/(1.0-var[13])))
    lngO=lngO+(-5.0*1873.0/T_max)*var[12]**2.0*var[13]*(1.0/(1.0-var[13])+1.0/(1.0-var[12])+var[13]/(2.0*(1.0-var[13])**2.0)-1.0)
    #lngH2=(1.0-var[7])**2.0*74829.6/(8.3144*T_max) # Regular solution model for xH2 in melt
    lngH2=0.0
    # Start with a simple, constant lngH2Omelt value
    lngH2Omelt=0.0
    #lngH2Omelt=(1.0-var[8])**2.0*74826.0/(8.3144*T_max) # Regular solution model for xH2O in melt
    # Have to start with the simpler mixing model for H metal to kick things off, more complicated in MCMC section
    #lngHmetal=(1.0-var[14])**2.0*(-1.9) # Regular solution model for H in metal with epsilon independent of T
    #x_light=var[12]+var[13] # Si + O mole fractions in metal, for psuedo-ternary mixing of H and (Si+O) with Fe below
    #lngHmetal=-3.8*x_light*(1.0+ln(1.0-x_light)/x_light-1.0/(1.0-var[14]))
    #lngHmetal=lngHmetal+3.8*x_light**2.0*var[14]*(1.0/(1.0-var[14])+1.0/(1.0-x_light)+var[14]/(2.0*(1.0-var[14])**2.0)-1.0)
    lngHmetal=0.0
    # Calculate H2O mole fraction on single-oxygen basis
    r_H2Omelt=var[8]/(1.0-var[8])*(1.0/3.0) # assumes oxygen pfu anhydrous = 3, xB=r_H2Omelt/(1+r_H2Omelt)
    xB=r_H2Omelt/(1.0+r_H2Omelt)
    f1=fm[1]*wmelt*( ln(var[5])+ln(var[1])-ln(var[6])+GRT1_T )
    f2=fm[2]*wmelt*( 0.5*ln(var[12])+0.5*lngSi+ln(var[3])-0.5*ln(var[1])-ln(var[11])+GRT2_T )
    f3=fm[3]*wmelt*( ln(var[0])+ln(var[1])-ln(var[2])+GRT3_T )
    f4=fm[4]*wmelt*( 0.5*ln(var[1])-ln(var[13])-lngO-0.5*ln(var[12])-0.5*lngSi+GRT4_T )
    f5=fm[5]*wmelt*( ln(var[7])+lngH2-2.0*ln(var[14])-2.0*lngHmetal +GRT5_T)
    f6=fm[6]*wmelt*( ln(var[3])+ln(var[1])-ln(var[4])+GRT6_T )
    #f7=fm[7]*wmelt*( ln(var[1])+2.0*ln(var[7])+2.0*lngH2-2.0*ln(var[8])-2.0*lngH2Omelt-ln(var[12])-lngSi+GRT7_T ) #Includes aH2O=xH2O
    f7=fm[7]*wmelt*( ln(var[1])+2.0*ln(var[7])+2.0*lngH2-4.0*ln(xB)-2.0*lngH2Omelt-ln(var[12])-lngSi+GRT7_T ) #Includes aH2O= xB^2
    f8=fm[8]*watm*( ln(var[17])-ln(var[16])-0.5*ln(var[19])+GRT8_T+ln(P/Pstd)-ln(P/Pstd)-0.5*ln(P/Pstd) ) # Fixed this
    f9=fm[9]*watm*( 2.0*ln(var[15])+ln(var[16])-ln(var[18])-0.5*ln(var[19])+GRT9_T+2.0*ln(P/Pstd)+ln(P/Pstd)-ln(P/Pstd)-0.5*ln(P/Pstd) )
    f10=fm[10]*watm*(ln(var[20])-0.5*ln(var[19])-ln(var[15])+GRT10_T+ln(P/Pstd)-0.5*ln(P/Pstd)-ln(P/Pstd) )
    f11=fm[11]*wevap*(0.5*ln(var[19])+ln(var[21])-ln(var[3])+(GRT11_T+(DeltaV11/(Rgas*temp))*(P-Pstd))+0.5*ln(P/Pstd)+ln(P/Pstd))
    f12=fm[12]*wevap*(0.5*ln(var[19])+ln(var[22])-ln(var[0])+(GRT12_T+(DeltaV12/(Rgas*temp))*(P-Pstd))+0.5*ln(P/Pstd)+ln(P/Pstd))
    f13=fm[13]*wevap*(0.5*ln(var[19])+ln(var[23])-ln(var[1])+(GRT13_T+(DeltaV13/(Rgas*temp))*(P-Pstd))+0.5*ln(P/Pstd)+ln(P/Pstd))
    f14=fm[14]*wevap*(0.5*ln(var[19])+2.0*ln(var[24])-ln(var[5])+(GRT14_T+(DeltaV14/(Rgas*temp))*(P-Pstd))+0.5*ln(P/Pstd)+2.0*ln(P/Pstd) )
    #f15=fm[15]*wsolub*( ln(var[7])-ln(var[15])+GRT15_T-ln(P/Pstd) ) #Keq 1bar adjusted for pressure as usual
    #f15=fm[15]*wsolub*(ln(var[7])-ln(var[15])+GRT15_T)  #fixed KD rather than Keq with pressure
    f15=fm[15]*wsolub*( ln(var[7])+lngH2-ln(var[15])+GRT15_T-ln(1.0e4/Pstd) ) #Fix KD at 1 GPa
    f16=fm[16]*wsolub*( 2.0*ln(xB)-ln(var[20])+GRT16_T-ln(P/Pstd) ) # aH2O = xB^2
    #f16=fm[16]*wsolub*(ln(var[8])+lngH2Omelt-ln(var[20])+GRT16_T-ln(P/Pstd) ) # aH2O =xH2O
    f17=fm[17]*wsolub*( ln(var[9])-ln(var[16])+GRT17_T-ln(P/Pstd) )
    f18=fm[18]*wsolub*( ln(var[10])-ln(var[17])+GRT18_T-ln(P/Pstd) )
    f19=fm[19]*watm*( ln(var[25])+0.5*ln(var[19])-ln(var[23])-2.0*ln(var[15])+GRT19_T+ln(P/Pstd)+0.5*ln(P/Pstd)-ln(P/Pstd)-2.0*ln(P/Pstd) )# SiO+2H2=SiH4+1/2O2
    
    # Add a 2-sided sigmoidal penalty function outside of 0 +/- toler using the logistic function
    val=0.0 #expected values for f's are zero
    sharp=5.0 #sharpness of sigmoidal edges, sharpness of the walls of the well
    toler=1.0 # 0.5 for sharp=5 gives a flat bottomed well with a width of about 1/2
    mult=10000.0 #magnitude of penalty 10000
#    mult=max_f
    f1_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f1))) + 1.0/(1.0+exp(-sharp*((val-toler)-f1)))
    f1=f1*(1.0+f1_penalty*mult) #preserves the shape

    f2_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f2))) + 1.0/(1.0+exp(-sharp*((val-toler)-f2)))
    f2=f2*(1.0+f2_penalty*mult) #preserves the shape outside of the narrow well

    f3_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f3))) + 1.0/(1.0+exp(-sharp*((val-toler)-f3)))
    f3=f3*(1.0+f3_penalty*mult) #preserves the shape outside of the narrow well

    f4_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f4))) + 1.0/(1.0+exp(-sharp*((val-toler)-f4)))
    f4=f4*(1.0+f4_penalty*mult) #preserves the shape outside of the narrow well

    f5_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f5))) + 1.0/(1.0+exp(-sharp*((val-toler)-f5)))
    f5=f5*(1.0+f5_penalty*mult) #preserves the shape outside of the narrow well

    f6_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f6))) + 1.0/(1.0+exp(-sharp*((val-toler)-f6)))
    f6=f6*(1.0+f6_penalty*mult) #preserves the shape outside of the narrow well

    f7_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f7))) + 1.0/(1.0+exp(-sharp*((val-toler)-f7)))
    f7=f7*(1.0+f7_penalty*mult) #preserves the shape outside of the narrow well

    f8_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f8))) + 1.0/(1.0+exp(-sharp*((val-toler)-f8)))
    f8=f8*(1.0+f8_penalty*mult) #preserves the shape outside of the narrow well

    f9_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f9))) + 1.0/(1.0+exp(-sharp*((val-toler)-f9)))
    f9=f9*(1.0+f9_penalty*mult) #preserves the shape outside of the narrow well

    f10_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f10))) + 1.0/(1.0+exp(-sharp*((val-toler)-f10)))
    f10=f10*(1.0+f10_penalty*mult) #preserves the shape outside of the narrow well

    f11_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f11))) + 1.0/(1.0+exp(-sharp*((val-toler)-f11)))
    f11=f11*(1.0+f11_penalty*mult) #preserves the shape outside of the narrow well

    f12_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f12))) + 1.0/(1.0+exp(-sharp*((val-toler)-f12)))
    f12=f12*(1.0+f12_penalty*mult) #preserves the shape outside of the narrow well

    f13_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f13))) + 1.0/(1.0+exp(-sharp*((val-toler)-f13)))
    f13=f13*(1.0+f13_penalty*mult) #preserves the shape outside of the narrow well

    f14_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f14))) + 1.0/(1.0+exp(-sharp*((val-toler)-f14)))
    f14=f14*(1.0+f14_penalty*mult) #preserves the shape outside of the narrow well

    f15_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f15))) + 1.0/(1.0+exp(-sharp*((val-toler)-f15)))
    f15=f15*(1.0+f15_penalty*mult) #preserves the shape outside of the narrow well

    f16_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f16))) + 1.0/(1.0+exp(-sharp*((val-toler)-f16)))
    f16=f16*(1.0+f16_penalty*mult) #preserves the shape outside of the narrow well

    f17_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f17))) + 1.0/(1.0+exp(-sharp*((val-toler)-f17)))
    f17=f17*(1.0+f17_penalty*mult) #preserves the shape outside of the narrow well

    f18_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f18))) + 1.0/(1.0+exp(-sharp*((val-toler)-f18)))
    f18=f18*(1.0+f18_penalty*mult) #preserves the shape
    
    f19_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f19))) + 1.0/(1.0+exp(-sharp*((val-toler)-f19)))
    f19=f19*(1.0+f19_penalty*mult) #preserves the shape

    
    #Mass balance for elements, with weighting factors
    #wtm=5.0*nH  # Scaling to most abundant element on a molar basis
    wtm=5.0
    f20=fm[20]*(wtm/nSi)*wtn*(nSi-(var[27]*(var[1]+var[2]+var[4]+var[6])+var[12]*var[28]+(var[23]+var[25])*var[26]))
    f21=fm[21]*(wtm/nMg)*wtn*(nMg-(var[27]*(var[0]+var[2])+var[22]*var[26]))
    f22=fm[22]*(wtm/nO)*wtn*(nO-(var[27]*(var[0]+2.0*var[1]+3.0*var[2]+var[3]+3.0*var[4]+var[5]+3.0*var[6]+var[8]+var[9]+2.0*var[10]) \
            + var[28]*var[13]+var[26]*(var[23]+2.0*var[17]+2.0*var[19]+var[20]+var[16]) ))
    f23=fm[23]*(wtm/nFe)*wtn*(nFe-(var[27]*(var[3]+var[4])+var[28]*var[11]+var[26]*var[21]))
    f24=fm[24]*(wtm/nH)*wtn*(nH-(var[28]*var[14]+var[27]*(2.0*var[7]+2.0*var[8])+var[26]*(2.0*var[15]+2.0*var[20]+4.0*var[18]+4.0*var[25])))
    f25=fm[25]*(5.0/nNa)*wtn*(nNa-(var[27]*(2.0*var[5]+2.0*var[6])+var[26]*var[24]))
    f26=fm[26]*(5.0/nC)*wtn*(nC-(var[27]*(var[9]+var[10])+var[26]*(var[18]+var[17]+var[16])))
    
    # Add a 2-sided sigmoidal penalty function for mass balance equations outside of 0 +/- toler using the logistic function
    val=0.0 #expected values for f are zero
    sharp=1.0 #sharpness of sigmoidal edges, sharpness of the walls of the well
    toler=0.01 # 0.007 goes with sharpness of 500.0, or 0.07 with sharp=100, or 0.2 for sharp=10
    mult=1000.0 #magnitude of penalty

    f20_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f20))) + 1.0/(1.0+exp(-sharp*((val-toler)-f20)))
    f20=f20*(1.0+f20_penalty*mult)

    f21_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f21))) + 1.0/(1.0+exp(-sharp*((val-toler)-f21)))
    f21=f21*(1.0+f21_penalty*mult)

    f22_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f22))) + 1.0/(1.0+exp(-sharp*((val-toler)-f22)))
    f22=f22*(1.0+f22_penalty*mult)

    f23_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f23))) + 1.0/(1.0+exp(-sharp*((val-toler)-f23)))
    f23=f23*(1.0+f23_penalty*mult)

    f24_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f24))) + 1.0/(1.0+exp(-sharp*((val-toler)-f24)))
    f24=f24*(1.0+f24_penalty*mult)

    f25_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f25))) + 1.0/(1.0+exp(-sharp*((val-toler)-f25)))
    f25=f25*(1.0+f25_penalty*mult)
    
    f26_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f26))) + 1.0/(1.0+exp(-sharp*((val-toler)-f26)))
    f26=f26*(1.0+f26_penalty*mult)
    
    #Summing constraint on mole fractions, with weighting factor
    f27=fm[27]*wtx*(1.0-var[0]-var[1]-var[2]-var[3]-var[4]-var[5]-var[6]-var[7]-var[8]-var[9]-var[10])
    f28=fm[28]*wtx*(1.0-var[11]-var[12]-var[13]-var[14])
    f29=fm[29]*wtx*(1.0-var[15]-var[16]-var[17]-var[18]-var[19]-var[20]-var[21]-var[22]-var[23]-var[24]-var[25])
    
    # Add a 2-sided sigmoidal penalty function for sum of mole fractions -1 being outside of 0 +/- toler
    val=0.0 #expected values for f are zero
    sharp=1.0 #sharpness of sigmoidal edges, sharpness of the walls of the well
    toler=0.005 # 0.007 goes with sharpness of 500.0, or 0.07 with sharp=100, or 0.2 for sharp=10
    mult=100000.0
    f27_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f27))) + 1.0/(1.0+exp(-sharp*((val-toler)-f27)))
    f27=f27*(1.0+f27_penalty*mult)
    
    f28_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f28))) + 1.0/(1.0+exp(-sharp*((val-toler)-f28)))
    f28=f28*(1.0+f28_penalty*mult)
    
    f29_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f29))) + 1.0/(1.0+exp(-sharp*((val-toler)-f29)))
    f29=f29*(1.0+f29_penalty*mult)
    
# Solve for the pressure to match the 'weight' of the atmosphere
# Grams per mole initial silicate
    grams_per_mole_silicate=0.000
    for i in range(0,11):
        grams_per_mole_silicate=grams_per_mole_silicate+var[i]*mol_wts[i]
    
# Grams per mole initial atmosphere
    grams_per_mole_atm=var[15]*mol_wts[15]+var[16]*mol_wts[16]+\
        var[17]*mol_wts[17]+var[18]*mol_wts[18]+var[19]*mol_wts[19]+ \
        var[20]*mol_wts[20]+var[21]*mol_wts[21]+var[22]*mol_wts[22]+ \
        var[23]*mol_wts[23]+var[24]*mol_wts[24]+var[25]*mol_wts[25]

# Grams per mole initial metal
    grams_per_mole_metal=var[11]*55.847+var[12]*28.0855+var[13]*15.9994+var[14]*1.00794
    moles_atm=var[26]
    moles_silicate=var[27]
    moles_metal=var[28]
    molefrac_atm=moles_atm/(moles_atm+moles_silicate+moles_metal)
    molefrac_silicate=moles_silicate/(moles_atm+moles_silicate+moles_metal)
    molefrac_metal=1.0-molefrac_atm-molefrac_silicate

    grams_atm=molefrac_atm*grams_per_mole_atm  #actually grams_i/mole planet
    grams_silicate=molefrac_silicate*grams_per_mole_silicate
    grams_metal=molefrac_metal*grams_per_mole_metal
    totalmass=grams_atm+grams_silicate+grams_metal
    massfrac_atm=grams_atm/totalmass
    massfrac_silicate=grams_silicate/totalmass
    massfrac_metal=grams_metal/totalmass

# Estimate atmospheric pressure at the surface of the planet: fratio is the Matm/Mplanet mass ratio
    fratio=massfrac_atm/(1.0-massfrac_atm)
    P_guess=1.2e6*fratio*(Mplanet_Mearth)**(2.0/3.0) #bar
# Error in pressure estimate
    f30=(P_guess-var[29])/P_guess
# Add a 2-sided sigmoidal penalty function outside of 0 +/- toler using the logistic function
    val=0.0 #expected values for f are zero
    sharp=1.0 #sharpness of sigmoidal edges, sharpness of the walls of the well
    toler=0.2 # 0.007 goes with sharpness of 500.0, or 0.07 with sharp=100, or 0.2 for sharp=10
    mult=P_penalty
    f30_penalty=1.0-1.0/(1.0+exp(-sharp*((val+toler)-f30))) + 1.0/(1.0+exp(-sharp*((val-toler)-f30)))
    f30=f30*(1.0+f30_penalty*mult)

#Sum of squared errors
    sum1=f1**2.0+f2**2.0+f3**2.0+f4**2.0+f5**2.0+f6**2.0+f7**2.0+f8**2.0+f9**2.0+f10**2.0
    sum1=sum1+f11**2.0+f12**2.0+f13**2.0+f14**2.0+f15**2.0+f16**2.0+f17**2.0+f18**2.0+f19**2.0
    sum2=f20**2.0+f21**2.0+f22**2.0+f23**2.0+f24**2.0+f25**2.0+f26**2.0
    sum3=(f27**2.0+f28**2.0+f29**2.0)
    sum4=f30**2.0

    sum1=1.0*sum1
    sum2=1.0*sum2
    sum3=1.0*sum3
    sum4=1.0*sum4
    sum=sum1+sum2+sum3+sum4
    return sum
    
#--------------------------------------------------------------------------------------------------------

#Create a progress callback function for dual_annealing to report the cost function values
#at each minium found and save values to file progress.txt
def progressF(x, fun, context):
    a_file = open('progress.txt', 'a')
    a_file.write("%10.5e \n" % fun)
    a_file.close()

#Initial value for cost (or objective) function, can use 1.2x this value for the initial_temp in dual_annealing below
cost=func(var)
print('')
print('Initial objective function =',cost)
print('Initial gamma H metal = ',lngHmetal)

#Estimate the mean cost function by sampling values
num_test=500
cost_estimates=np.zeros(num_test)
var_random=np.zeros(numvar)
for k in range(0,num_test):
    for i in range(0,numvar):
        var_random[i]=uniform(bounds[i,0],bounds[i,1])
#        print('var', i, '=',var_random[i])
    cost_estimates[k]=func(var_random)
#    print('test random cost values =',cost_estimates)

#Smooth random draws
def smoothTriangle(data, degree ):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]
    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
# Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed
cost_smoothed=smoothTriangle(cost_estimates, 5)

mean_cost=mean(cost_smoothed)
sigma_cost=stdev(cost_smoothed)

print('Mean Objective function at start =',mean_cost)
print('Std deviation of smoothed Objective function at start =',sigma_cost)

#CALCULATE INITIAL SEARCH TEMPERATURE based on % probability of accepting uphill;
#The probability for accepting uphill results is the argument to the ln ... usually 0.8
T_estimate=-sigma_cost/ln(0.98)
print('Estimated initial search T =',T_estimate)

plt.plot(cost_estimates,label='Random',linewidth=0.5)
plt.plot(cost_smoothed,label='Smoothed')
plt.ylabel('Objective func', fontsize=12)
plt.xlabel('Random draw',fontsize=12)
plt.legend()
plt.show(block=False)  #Permit execution to continue with plot open
plt.pause(7) #Pause x seconds before moving on
plt.close()

Tini=T_estimate
#Tini=100.0
print('Initial annealing T =',Tini)
print('Working..........')
print('')

#--------------------------------------------------------------------------------------------------------
#PERFORM MINIMIZATION OF func(var) TO OBTAIN SOLUTION
#
#INVOKE SIMULATED ANNEALING...visit parameter controls cooling, default visit=2.62, I use 2.985,
#larger values cool fast early (max=3), then flattens;
# restart_temp_ratio is the temperature value that triggers reannealing, 0 to 1;
# default initial_temp=5230.0, max initial temp is 5e4, should be 4x variance in cost function;
# max number of function calls, maxfun (default = 1e7), overrides number of iterations if breached;
# accept ranges from -5 to -10000, -500 works well, lower means lower prob of acceptance;
# seed selects a random number for the seed and reports the seed to the user;
# no_local_search=True causes classical simulated annealing with no local search strategy.
nseed=nseed_prov
if nseed_prov == 0:
    nseed=randint(1,500)
print('Seed for search =',nseed)
soln=dual_annealing(func,bounds,maxiter=iter,initial_temp=Tini,visit=2.98,maxfun=100000000, seed=nseed,\
    accept=-500.0,restart_temp_ratio=1.0e-9,callback=progressF)
print('')
print("solution for parameters x =",soln)
#solution contained in soln.x

quality=soln.fun/mean_cost
print(" Objective function end/start =",quality)

#Plot cost function at minima found during simulated annealing...
name='progress.txt'
values=np.genfromtxt(name,'float',delimiter="")  #Very useful function in numpy
cost_values=values

plt.plot(cost_values,linewidth=0.5,marker='o',markersize=1,markerfacecolor='white')
plt.title('Simulated Annealing Search')
plt.ylabel('Objective func', fontsize=12)
plt.yscale("log")
plt.xlabel('Succession of minima found',fontsize=12)
plt.show(block=False)  #Permit execution to continue with plot open
plt.pause(10) #Pause x seconds before moving on
plt.savefig('progress.png')
plt.close()

print('Plot of simulated annealing minima saved as progress.png')
print('')

#--------------------------------------------------------------------------------------------------------
# MCMC SEARCH STARTING WITH SIMULATED ANNEALING RESULT IN soln.x
# In this section, the parameters, mole fractions etc., comprising the model are stored in vector theta, while
# the model itself is considered to be the equilibrium constants and mass-balance constraints. They have the same
# dimensions since we are solving a system of equations equal to the number of unknowns.
print('')
print('MCMC Search beginning....')
print('')
theta=np.zeros(numvar)
theta=soln.x #Current estimate of best parameters from simulated annealing is assigned to model vector theta
y_model=np.zeros(numvar) # These equilibrium constants and mass-balance constraints comprise our model
yerr=np.zeros(numvar) # Will use this for estimates of uncertainties in equilibrium constants and mass-balance constraints

# Our 'data' are the actual expected values for the equilibrium constants and mass-balance constraints.
# For historical reasons we assign this to vector y
y=np.zeros(numvar)
# Assign actual natural logs of equilibrium constants with pressure corrections, although note, these are actually
# natural logs of KD values since we have coupled the total pressure corrections with the delta GRT (DG/(RT)) terms.
y[0]=-(GRT1_T )
y[1]=-(GRT2_T )
y[2]=-(GRT3_T )
y[3]=-(GRT4_T )
y[4]=-(GRT5_T)
y[5]=-(GRT6_T )
y[6]=-(GRT7_T )
y[7]=-(GRT8_T)
y[8]=-(GRT9_T)
y[9]=-(GRT10_T)
y[10]=-((GRT11_T+(DeltaV11/(Rgas*temp))*(P-Pstd))) # now excludes P correction for evaporation if DeltaV11 ne zero
y[11]=-((GRT12_T+(DeltaV12/(Rgas*temp))*(P-Pstd))) # now excludes P correction for evaporation ...
y[12]=-((GRT13_T+(DeltaV13/(Rgas*temp))*(P-Pstd))) # now excludes P correction for evaporation ...
y[13]=-((GRT14_T+(DeltaV14/(Rgas*temp))*(P-Pstd))) # now excludes P correction for evaporation ...
#y[14]=-(GRT15_T)  # Rxn 15 fixed KD with P rather than fixed Keq with pressure, thus cancelling -ln(P) term
#y[14]=-(GRT15_T-ln(P/Pstd) )  # Rxn 15 fixed KD with P rather than fixed Keq with pressure, thus cancelling -ln(P) term
y[14]=-(GRT15_T-ln(1.0e4/Pstd) )  # Rxn 15 fixed KD for value at 1GPa
y[15]=-(GRT16_T) # excludes P corr
y[16]=-(GRT17_T) # excludes P corr
y[17]=-(GRT18_T) # excludes P corr
y[18]=-(GRT19_T) # excludes P corr
# Assign actual moles of components
y[19]=nSi
y[20]=nMg
y[21]=nO
y[22]=nFe
y[23]=nH
y[24]=nNa
y[25]=nC
# Assign actual mole fractions sums, which must be unity for each of the three phases.
y[26]=1.000
y[27]=1.000
y[28]=1.000
# Pressure should exactly match physics
y[29]=0.0

# Create a vector yerr that contains errors associated with each aspect of the model
# Blanket relative uncertainties for all lnKeq values
lnk_err=0.005 # 0.005 for aH2O = xB^2 for walkers = 1e-7, and aH2O = xH2O for walkers =1e-9
yerr[0]=abs(lnk_err*y[0])
yerr[1]=abs(lnk_err*y[1])
yerr[2]=abs(lnk_err*y[2])
yerr[3]=abs(lnk_err*y[3])
yerr[4]=abs(lnk_err*y[4])
yerr[5]=abs(lnk_err*y[5])
yerr[6]=abs(lnk_err*y[6])
yerr[7]=abs(lnk_err*y[7])
yerr[8]=abs(lnk_err*y[8])
yerr[9]=abs(lnk_err*y[9])
yerr[10]=abs(lnk_err*y[10])
yerr[11]=abs(lnk_err*y[11])
yerr[12]=abs(lnk_err*y[12])
#yerr[13]=abs(lnk_err*y[13])
yerr[13]=abs((lnk_err/5.0)*y[13])
yerr[14]=abs(lnk_err*y[14])
yerr[15]=abs(lnk_err*y[15])
#yerr[16]=abs(lnk_err*y[16])
yerr[16]=abs((lnk_err/5.0)*y[16])
yerr[17]=abs(lnk_err*y[17])
#yerr[18]=abs(lnk_err*y[18])
yerr[18]=abs((lnk_err/5.0)*y[18])

# Blanket relative uncertainty for mass balance
moles_err=0.0001
yerr[19]=abs(moles_err*y[19])
yerr[20]=abs(moles_err*y[20])
yerr[21]=abs(moles_err*y[21])
yerr[22]=abs(moles_err*y[22])
yerr[23]=abs(moles_err*y[23])
yerr[24]=abs(moles_err*y[24])
yerr[25]=abs(moles_err*y[25])
# Blanket relative uncertainty for mole fraction sums
x_err=1.0e-5
yerr[26]=x_err
yerr[27]=x_err
yerr[28]=x_err
# Pressure fractional error
yerr[29]=0.1

print('y(actual model)= \n ',y)
print('')


#--------------------------------------------------------------------------------------------------------
# Our model function
def model(theta):
    P=theta[29]
    Pstd=1.0
    # Model natural log of equilibrium KDs (no pressures in these quotients)
    lngSi=-6.65*1873.0/T_max-(12.41*1873.0/T_max)*ln(1.0-theta[12])
    lngSi=lngSi-((-5.0*1873.0/T_max)*theta[13]*(1.0+ln(1-theta[13])/theta[13]-1.0/(1.0-theta[12])))
    lngSi=lngSi+(-5.0*1873.0/T_max)*theta[13]**2.0*theta[12]*(1.0/(1.0-theta[12])+1.0/(1.0-theta[13])+theta[12]/(2.0*(1.0-theta[12])**2.0)-1.0)
    lngO=(4.29-16500.0/T_max)-(-1.0*1873.0/T_max)*ln(1.0-theta[13])
    lngO=lngO-((-5.0*1873.0/T_max)*theta[12]*(1.0+ln(1-theta[12])/theta[12]-1.0/(1.0-theta[13])))
    lngO=lngO+(-5.0*1873.0/T_max)*theta[12]**2.0*theta[13]*(1.0/(1.0-theta[13])+1.0/(1.0-theta[12])+theta[13]/(2.0*(1.0-theta[13])**2.0)-1.0)
    #lngH2=(1.0-theta[7])**2.0*74829.6/(8.3144*T_max) # Regular solution model for xH2 in melt
    lngH2=0.0
    #lngH2Omelt=(1.0-theta[8])**2.0*74826.0/(8.3144*T_max) # Regular solution model for xH2O in melt
    lngH2Omelt=0.0
    #lngHmetal=(1.0-theta[14])**2.0*(-1.9) # Regular solution model for H in metal with epsilon independent of T
    x_light=theta[12]+theta[13] # Si + O mole fractions in metal, for psuedo-ternary mixing of H and (Si+O) with Fe below
    #lngHmetal=-3.8*x_light*(1.0+ln(1.0-x_light)/x_light-1.0/(1.0-theta[14]))
    #lngHmetal=lngHmetal+3.8*x_light**2.0*theta[14]*(1.0/(1.0-theta[14])+1.0/(1.0-x_light)+theta[14]/(2.0*(1.0-theta[14])**2.0)-1.0)
    lngHmetal=0.0
    # Calculate H2O mole fraction on single-oxygen basis
    r_H2Omelt=theta[8]/(1.0-theta[8])*(1.0/3.0) # assumes oxygen pfu anhydrous = 3, xB=r_H2Omelt/(1+r_H2Omelt)
    xB=r_H2Omelt/(1.0+r_H2Omelt)
    y_model[0]=ln(theta[5])+ln(theta[1])-ln(theta[6]) #Rxn 1
    y_model[1]=0.5*ln(theta[12])+0.5*lngSi+ln(theta[3])-0.5*ln(theta[1])-ln(theta[11])  #Rxn 2
    y_model[2]=ln(theta[0])+ln(theta[1])-ln(theta[2]) #Rxn 3
    y_model[3]=0.5*ln(theta[1])-ln(theta[13])-lngO - 0.5*ln(theta[12])-0.5*lngSi #Rxn 4
    y_model[4]=ln(theta[7])+lngH2-2.0*ln(theta[14])-2.0*lngHmetal  #Rxn 5
    y_model[5]=ln(theta[3])+ln(theta[1])-ln(theta[4]) #Rxn 6
    #y_model[6]=ln(theta[1])+2.0*ln(theta[7])+2.0*lngH2-2.0*ln(theta[8])-2.0*lngH2Omelt-ln(theta[12])-lngSi #Rxn 7, here aH2O =xH2O
    y_model[6]=ln(theta[1])+2.0*ln(theta[7])+2.0*lngH2-4.0*ln(xB)-2.0*lngH2Omelt-ln(theta[12])-lngSi #Rxn 7, here aH2O = xB^2
    y_model[7]=ln(theta[17])-ln(theta[16])-0.5*ln(theta[19])+ln(P/Pstd)-ln(P/Pstd)-0.5*ln(P/Pstd)
    y_model[8]=2.0*ln(theta[15])+ln(theta[16])-ln(theta[18])-0.5*ln(theta[19])+2.0*ln(P/Pstd)+ln(P/Pstd)-ln(P/Pstd)-0.5*ln(P/Pstd)
    y_model[9]=ln(theta[20])-0.5*ln(theta[19])-ln(theta[15])+ln(P/Pstd)-0.5*ln(P/Pstd)-ln(P/Pstd)
    y_model[10]=0.5*ln(theta[19])+ln(theta[21])-ln(theta[3])+0.5*ln(P/Pstd)+ln(P/Pstd)
    y_model[11]=0.5*ln(theta[19])+ln(theta[22])-ln(theta[0])+0.5*ln(P/Pstd)+ln(P/Pstd)
    y_model[12]=0.5*ln(theta[19])+ln(theta[23])-ln(theta[1])+0.5*ln(P/Pstd)+ln(P/Pstd)
    y_model[13]=0.5*ln(theta[19])+2.0*ln(theta[24])-ln(theta[5])+0.5*ln(P/Pstd)+2.0*ln(P/Pstd)
    y_model[14]=ln(theta[7])+lngH2-ln(theta[15]) # not including Ptotal in constant, kD is fixed by DG/RT
    y_model[15]=2.0*ln(xB)-ln(theta[20])-ln(P/Pstd) #Accounting for aH2O =xB^2
    #y_model[15]=ln(theta[8])+lngH2Omelt-ln(theta[20])-ln(P/Pstd) #Accounting for aH2O =xH2O
    y_model[16]=ln(theta[9])-ln(theta[16])-ln(P/Pstd)
    y_model[17]=ln(theta[10])-ln(theta[17])-ln(P/Pstd)  #Rxn 18
    y_model[18]=ln(theta[25])+0.5*ln(theta[19])-ln(theta[23])-2.0*ln(theta[15])+ln(P/Pstd)+0.5*ln(P/Pstd)-ln(P/Pstd)-2.0*ln(P/Pstd)# Rxn 19: SiO+2H2=SiH4+1/2O
    
    #Model mass balance for elements
    y_model[19]=(theta[27]*(theta[1]+theta[2]+theta[4]+theta[6])+theta[12]*theta[28]+(theta[23]+theta[25])*theta[26]) #Si
    y_model[20]=(theta[27]*(theta[0]+theta[2])+theta[22]*theta[26]) #Mg
    y_model[21]=(theta[27]*(theta[0]+2.0*theta[1]+3.0*theta[2]+theta[3]+3.0*theta[4]+theta[5]+3.0*theta[6]+theta[8]+theta[9]+2.0*theta[10]) \
        + theta[28]*theta[13]+theta[26]*(theta[23]+2.0*theta[17]+2.0*theta[19]+theta[20]+theta[16]) ) #O
    y_model[22]=(theta[27]*(theta[3]+theta[4])+theta[28]*theta[11]+theta[26]*theta[21]) #Fe
    y_model[23]=(theta[28]*theta[14]+theta[27]*(2.0*theta[7]+2.0*theta[8])+theta[26]*(2.0*theta[15]+2.0*theta[20]+4.0*theta[18]+4.0*theta[25])) #H
    y_model[24]=(theta[27]*(2.0*theta[5]+2.0*theta[6])+theta[26]*theta[24]) #Na
    y_model[25]=(theta[27]*(theta[9]+theta[10])+theta[26]*(theta[18]+theta[17]+theta[16])) #C
    
    #Model summing constraint on mole fractions, should sum to unity
    y_model[26]=(theta[15]+theta[16]+theta[17]+theta[18]+theta[19]+theta[20]+theta[21]+theta[22]+theta[23]+theta[24]+theta[25]) #gas
    y_model[27]=(theta[0]+theta[1]+theta[2]+theta[3]+theta[4]+theta[5]+theta[6]+theta[7]+theta[8]+theta[9]+theta[10]) #melt
    y_model[28]=(theta[11]+theta[12]+theta[13]+theta[14]) #metal
    
    # Solve for the pressure to match the 'weight' of the atmosphere
    # Grams per mole initial silicate
    grams_per_mole_silicate=0.000
    for i in range(0,11):
        grams_per_mole_silicate=grams_per_mole_silicate+theta[i]*mol_wts[i]
    # Grams per mole initial atmosphere
    grams_per_mole_atm=theta[15]*mol_wts[15]+theta[16]*mol_wts[16]+\
        theta[17]*mol_wts[17]+theta[18]*mol_wts[18]+theta[19]*mol_wts[19]+ \
        theta[20]*mol_wts[20]+theta[21]*mol_wts[21]+theta[22]*mol_wts[22]+ \
        theta[23]*mol_wts[23]+theta[24]*mol_wts[24]+theta[25]*mol_wts[25]
    # Grams per mole initial metal
    grams_per_mole_metal=theta[11]*55.847+theta[12]*28.0855+theta[13]*15.9994+theta[14]*1.00794
    moles_atm=theta[26]
    moles_silicate=theta[27]
    moles_metal=theta[28]
    molefrac_atm=moles_atm/(moles_atm+moles_silicate+moles_metal)
    molefrac_silicate=moles_silicate/(moles_atm+moles_silicate+moles_metal)
    molefrac_metal=1.0-molefrac_atm-molefrac_silicate
    grams_atm=molefrac_atm*grams_per_mole_atm  #actually grams_i/mole planet
    grams_silicate=molefrac_silicate*grams_per_mole_silicate
    grams_metal=molefrac_metal*grams_per_mole_metal
    totalmass=grams_atm+grams_silicate+grams_metal
    massfrac_atm=grams_atm/totalmass
    massfrac_silicate=grams_silicate/totalmass
    massfrac_metal=grams_metal/totalmass

    # Estimate atmospheric pressure at the surface of the planet: fratio is the Matm/Mplanet mass ratio
    fratio=massfrac_atm/(1.0-massfrac_atm)
    P_guess=1.2e6*fratio*(Mplanet_Mearth)**(2.0/3.0) #bar
    # Error in pressure estimate relative to weight of atmosphere, should be 0
    y_model[29]=100.0*(P_guess-theta[29])/P_guess #Added a multiplier here to force better pressure solutions
    
    return y_model
#--------------------------------------------------------------------------------------------------------
# DEFINE LIKELIHOOD FUNCTION, making use of model function above.
# Notice by using the sum, this is the logarithm of the likelihood probability.
# The total probability would be the product of exponentials of the square differences etc.
def lnlike(theta,y,yerr):
    y_model=model(theta)
    diffs=((y-y_model)**2.0)/yerr**2.0
    lnlike=-0.5*sum(diffs)
    return lnlike
#--------------------------------------------------------------------------------------------------------
# DEFINE PRIORS using simple square function for each variable
# This function output is prescribed by what emcee uses for the prior
# probabilities, e.g., 0 for in range, -infinity if out the range for the priors.
# Formulated this way, these are uninformative priors.
def lnprior(theta):
    # Priors on mole fractions require them to be between 0 and 1 else ln(probability) is -infinity
    # This structure relies on first encountered return applies
    for i in range(0,25):
        if theta[i] < 0.0:
            return -np.inf
        if theta[i] > 1.00:
            return -np.inf
    # Special prior for water in melt using aH2O = xB^2
    #if theta[8] > 0.40:
    #    return -np.inf
    # Priors for moles of phases, must be positive
    if theta[26] < 0.0:
        return -np.inf
    if theta[27] < 0.0:
        return -np.inf
    if theta[28] < 0.0:
        return -np.inf
    return 0.0
#--------------------------------------------------------------------------------------------------------
# POSTERIOR PROBABILITIES as product of likelihood and priors (i.e., sum of logs).
# We add the output from the functions above because they return logs of
# probabilities. If outside of priors, return infinitely bad probability
# as -inf since this would be the log of a very tiny number.
def lnprob(theta,y,yerr):
    lp=lnprior(theta)
    if lp == -np.inf:
        return -np.inf
    like=lnlike(theta,y,yerr)
    if math.isnan(like):
        return -np.inf
    return lp+like
#--------------------------------------------------------------------------------------------------------
# Make a tuple with the data to be used by emcee, this will be a list of
# arguments that tells emcee the list required to call the probability function,
# so this should match the lnprob argument list
data = (y,yerr)

# Print current model (not parameters) before starting MCMC
print('y(current model)= \n',model(soln.x))
print('')
print('yerr(errors for model)= \n ',yerr)
print('')

# INPUT MCMC PARAMETERS: Set number of independent Markov chain walkers and iterations
nwalkers=200 #100

# Utilize thin for emcee to return every thin'th sample, add thin_by=thin to smapler.run_mcmc options
thin=10
niter=1000000 #2000000 works
niter_eff = int(niter/thin) # emcee does niter*thin iteractions, so correct for this to save time if using thin
        
# p0 is the array of initial positions for each walker, i.e., each separate
# Markov chain operating slightly displaced at random from one another.
# The code below creates an array of arrays with slightly purturbed locations
# in parameter space based on random draws from a multivariate Gaussian for
# these variables. Increase the prefix multiplier for random position offsets if the error
# "Initial state has a large condition number" is returned from emcee, indicating
# walkers are not sufficiently independent.
n=numvar
p0=[(theta)+ranoffset*np.random.randn(n) for i in range(nwalkers)]  #+1.0e-9*np.random.randn(n)

# DEFINE A FUNCTION THAT RUNS MCMC SEARCH.  Start by instantiating the EnsembleSampler.
# for emcee.
def main(p0,nwalkers,niter,n,lnprob,data):
    ctx = get_context('fork')
    with Pool(6,context=ctx) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, n, lnprob, args=data, pool=pool)
    
        print("Initial burn in running...")
        pos,_, _ = sampler.run_mcmc(p0,500)  # initial run, save position (i.e., model parameters)
    
        sampler.reset()  # reset results from sampler for actual search
    
        print("Running full MCMC search...")
        pos, prob, state = sampler.run_mcmc(pos, niter_eff, skip_initial_state_check=False, progress=True, thin_by=thin) # run actual search starting at burn-in position
    
    return sampler, pos, prob, state

# RUN THE SEARCH....this one line of code initiates the search
sampler, pos, prob, state = main(p0,nwalkers,niter,n,lnprob,data)
print('')
print('...MCMC search completed.')
# EXPLORE results...
#position contains the parameter position (i.e, values) for each walker at the end of search
#print("Position of each sampler = ",pos)

# Concatenate all walker results into a single chain, array is (nwalkers*iterations x n(variables))
# or said another way, each row is a test position, each column is a variable.
samples=sampler.flatchain
posteriors=sampler.flatlnprobability

print('memory required for chain = ', samples.size * samples.itemsize)

# Find dimensions of the returned data for the chains
print('sampler.flatchain shape = ',np.shape(samples))
print('sampler.flatlnprobability shape = ',np.shape(posteriors))

# Select as your best set of parameters theta in the sampler that has the greatest posterior probability
# by interrogating the ln probability for each sample, also concatenated from all walkers
result=samples[np.argmax(sampler.flatlnprobability)]

# Best-fit final model consisting of equilibrium constants, total moles of components, and mole fraction sums
best_fit_model = model(result)

# Calculate goodness of fit for this best-fit model
chi_square=sum(((y-best_fit_model)**2.0)/yerr**2.0)
red_chi_square=chi_square/(float(count)-float(n))
print('')
print('[Reduced chi-squared for fit] = %10.6e' %red_chi_square)
print('')
print('New y (new calculated model)= \n ',best_fit_model)
print('')

#COMPARE parameters from simulated annealing with MCMC, latter stored in result
refinement=result-soln.x
print('')
print('Differences in model parameters =',refinement)

grams_per_mole_silicate=0.000
for i in range(0,11):
    grams_per_mole_silicate=grams_per_mole_silicate+result[i]*mol_wts[i]

New_Press=result[29]

#print(result)  #reprint entire solution array
print('')
xMgO_melt=result[0]
print('xMgO_silicate =',xMgO_melt)
xSiO2_melt=result[1]
print('xSiO2_silicate =',xSiO2_melt)
xMgSiO3_melt=result[2]
print('xMgSiO3_silicate =',xMgSiO3_melt)
xFeO_melt=result[3]
print('xFeO_silicate =',xFeO_melt)
xFeSiO3_melt=result[4]
print('xFeSiO3_silicate =',xFeSiO3_melt)
xNa2O_melt=result[5]
print('xNa2O_silicate =',xNa2O_melt)
xNa2SiO3_melt=result[6]
print('xNa2SiO3_silicate =',xNa2SiO3_melt)
xH2_melt=result[7]
print('xH2_silicate =',xH2_melt)
xH2O_melt=result[8]
print('xH2O_silicate =', xH2O_melt)
xCO_melt=result[9]
print('xCO_silicate =', xCO_melt)
xCO2_melt=result[10]
print('xCO2_silicate =', xCO2_melt)
melt_sum=0.0
for i in range(0,11):
    melt_sum=melt_sum+result[i]
print('Mole_fraction_sum_for_melt = ',melt_sum)

#H2/H2O molecular for silicate and apparent DIW fO2
xH2_xH2O_silicate=result[7]/result[8]
print('')
print('xH2/xH2O silicate =',xH2_xH2O_silicate)
DIW_apparent=2.0*log((result[3]+result[4])/0.85)
DIW_actual=2.0*log((result[3]+result[4]/result[11]))
print('DIW_silicate_for_xFe_metal=0.85 =',DIW_apparent)
print('DIW_actual = ',DIW_actual)

#Convert silicate to wt% oxides etc.
print('')
print("grams_per_mole_silicate =", grams_per_mole_silicate)
wtpercentSiO2=100.0*(result[1]+result[2]+result[4])*mol_wts[1]/grams_per_mole_silicate
print('Wt_%_SiO2_(total Si)_silicate =',wtpercentSiO2)
wtpercentMgO=100.0*(result[0]+result[2])*mol_wts[0]/grams_per_mole_silicate
print('Wt_%_MgO_(total Mg)_silicate =',wtpercentMgO)
wtpercentFeO=100.0*(result[3]+result[4])*mol_wts[3]/grams_per_mole_silicate
print('Wt_%_FeO_(total Fe)_silicate =',wtpercentFeO)
wtpercentNa2O=100.0*(result[5]+result[6])*mol_wts[5]/grams_per_mole_silicate
print('Wt_%_Na2O_(total Na)_silicate =',wtpercentNa2O)
wtpercentHmelt=100.0*(result[7])*mol_wts[7]/grams_per_mole_silicate
print('Wt_%_H2_melt =',wtpercentHmelt)
wtpercentH2Omelt=100.0*(result[8])*mol_wts[8]/grams_per_mole_silicate
print('Wt_%_H2O_melt =',wtpercentH2Omelt)
wtpercentCOmelt=100.0*(result[9])*mol_wts[9]/grams_per_mole_silicate
print('Wt_%_CO_melt =',wtpercentCOmelt)
wtpercentCO2melt=100.0*(result[10])*mol_wts[10]/grams_per_mole_silicate
print('Wt_%_CO2_melt =',wtpercentCO2melt)

#Density of silicate mantle
density_mantle=0.00
for i in range(0,11):
    density_mantle=density_mantle+result[i]*(mol_wts[i]/(10.0*v_Jbar[i]))
print('')
print('Uncompressed_density_of_silicate =',density_mantle,'g/cc' )

print('')
xFe_metal=result[11]
print('xFe_metal =',xFe_metal)
xSi_metal=result[12]
print('xSi_metal =',xSi_metal)
xO_metal=result[13]
print('xO_metal =', xO_metal)
xH_metal=result[14]
print('xH_metal =',xH_metal)
lngSi=-6.65*1873.0/T_max-(12.41*1873.0/T_max)*ln(1.0-result[12])
lngSi=lngSi-((-5.0*1873.0/T_max)*result[13]*(1.0+ln(1-result[13])/result[13]-1.0/(1.0-result[12])))
lngSi=lngSi+(-5.0*1873.0/T_max)*result[13]**2.0*result[12]*(1.0/(1.0-result[12])+1.0/(1.0-result[13])+result[12]/(2.0*(1.0-result[12])**2.0)-1.0)
lngO=(4.29-16500.0/T_max)-(-1.0*1873.0/T_max)*ln(1.0-result[13])
lngO=lngO-((-5.0*1873.0/T_max)*result[12]*(1.0+ln(1-result[12])/result[12]-1.0/(1.0-result[13])))
lngO=lngO+(-5.0*1873.0/T_max)*result[12]**2.0*result[13]*(1.0/(1.0-result[13])+1.0/(1.0-result[12])+result[13]/(2.0*(1.0-result[13])**2.0)-1.0)
#lngHmetal=(1.0-result[14])**2.0*(-1.9) # Regular solution model for H in metal with epsilon independent of T
x_light=result[12]+result[13] # Si + O mole fractions in metal, for psuedo-ternary mixing of H and (Si+O) with Fe below
#lngHmetal=-3.8*x_light*(1.0+ln(1.0-x_light)/x_light-1.0/(1.0-result[14]))
#lngHmetal=lngHmetal+3.8*x_light**2.0*result[14]*(1.0/(1.0-result[14])+1.0/(1.0-x_light)+result[14]/(2.0*(1.0-result[14])**2.0)-1.0)
lngHmetal=0.0
#lngH2=(1.0-result[7])**2.0*74829.6/(8.3144*T_max) # Regular solution model for xH2 in melt
lngH2=0.0
#lngH2Omelt=(1.0-result[8])**2.0*74826.0/(8.3144*T_max) # Regular solution model for xH2O in melt
lngH2Omelt=0.0
print('ln(gamma H2 silicate) =',lngH2)
print('ln(gamma H2O silicate) =',lngH2Omelt)
print('ln(gamma H metal) =',lngHmetal)
print('ln(gamma Si metal) =', lngSi)
print('ln(gamma O metal) = ',lngO)
metal_sum=xFe_metal+xSi_metal+xO_metal+xH_metal
print('Mole_fraction_sum_for_metal = ',metal_sum)

#Convert metal to weight %
print('')
grams_per_mole_metal=result[11]*55.847+result[12]*28.0855+result[13]*15.9994+result[14]*1.00794
wtpercentFe=result[11]*55.847/grams_per_mole_metal
wtpercentFe=wtpercentFe*100.0
print('grams_per_mole_metal =',grams_per_mole_metal)
print('Wt_%_Fe_metal =', wtpercentFe)
wtpercentSi=result[12]*28.0855/grams_per_mole_metal
wtpercentSi=wtpercentSi*100.0
print('Wt_%_Si_metal =', wtpercentSi)
wtpercentO=result[13]*15.9994/grams_per_mole_metal
wtpercentO=wtpercentO*100.0
print('Wt_%_O_metal =', wtpercentO)
wtpercentH=result[14]*1.00794/grams_per_mole_metal
wtpercentH=wtpercentH*100.0
print('Wt_%_H_metal =', wtpercentH)

#Density of metal core
density_core=0.00
for i in range(11,15):
    density_core=density_core+result[i]*(mol_wts[i]/(10.0*v_Jbar[i]))
print('')
print('Uncompressed_density_of_metal =',density_core,'g/cc' )
density_pure_iron=mol_wts[11]/(10.0*v_Jbar[11])
#density_deficit=(density_pure_iron-density_core)/density_pure_iron
density_deficit=8.7*wtpercentH+1.2*wtpercentO+0.8*wtpercentSi
print('Uncompressed_density_of_iron =', density_pure_iron)
print('Core density deficit based on experimental values  = ',density_deficit,'%')

print('')
xH2_gas=result[15]
print('xH2_atm =',xH2_gas)
xCO_gas=result[16]
print('xCO_atm =',xCO_gas)
xCO2_gas=result[17]
print('xCO2_atm =',xCO2_gas)
xCH4_gas=result[18]
print('xCH4_atm =',xCH4_gas)
xO2_gas=result[19]
print('xO2_atm =',xO2_gas)
xH2O_gas=result[20]
print('xH2O_atm =',xH2O_gas)
xFe_gas=result[21]
print('xFe_atm =',xFe_gas)
xMg_gas=result[22]
print('xMg_atm =',xMg_gas)
xSiO_gas=result[23]
print('xSiO_atm =',xSiO_gas)
xNa_gas=result[24]
print('xNa_atm =',xNa_gas)
xSiH4_gas=result[25]
print('xSiH4_atm =',xSiH4_gas)
gas_sum=0.0
for i in range(15,26):
    gas_sum=gas_sum+result[i]
print('Mole_fraction_sum_for_atm = ',gas_sum)

#H2/H2O molecular for atmosphere
xH2_xH2O_atm=result[15]/result[20]
print('xH2/xH2O atmosphere =',xH2_xH2O_atm)
##IW fO2 buffer curve using molten FeO and Fe so that the activities
## must refer to activities in the molten phases relative to the pure
## molten phases.
GIW=0.5*GgasO2[nT]+GmetalFe[nT]-GmeltFeO[nT]
logKIW=-GIW/(Rgas*TK[nT]*log_to_ln)
#For unit activities in the oxide and metal melts, the KIW defines an
#an equilibrium pressure of O2 such that (PO2/Pstd)=Keq^2.  The ratio
#of pressures is the correction from actual PO2 to pure O2 at P std.
#If Pstd state is 1 bar, then this PO2 is pressure of pure O2 in bar.
KIW=10.0**logKIW
PO2IW=KIW**2.0
logPO2IW=log(PO2IW)
logPO2atm=log(result[19]*New_Press) #PO2 in bar relative to std state of 1 bar
DIW_atm=logPO2atm-logPO2IW
print('DIW_atmosphere =',DIW_atm)
#Grams per mole atmosphere
grams_per_mole_atm=result[15]*mol_wts[15]+result[16]*mol_wts[16]+\
    result[17]*mol_wts[17]+result[18]*mol_wts[18]+result[19]*mol_wts[19]+ \
    result[20]*mol_wts[20]+result[21]*mol_wts[21]+result[22]*mol_wts[22]+ \
    result[23]*mol_wts[23]+result[24]*mol_wts[24]+result[25]*mol_wts[25]
print('')
print('grams_per_mole_atmosphere = ',grams_per_mole_atm)
wtpercentH2atm=100.0*result[15]*mol_wts[15]/grams_per_mole_atm
print('Wt_%_H2_atmosphere =', wtpercentH2atm)
wtpercentCOatm=100.0*result[16]*mol_wts[16]/grams_per_mole_atm
print('Wt_%_CO_atmosphere =', wtpercentCOatm)
wtpercentCO2atm=100.0*result[17]*mol_wts[17]/grams_per_mole_atm
print('Wt_%_CO2_atmosphere =', wtpercentCO2atm)
wtpercentCH4atm=100.0*result[18]*mol_wts[18]/grams_per_mole_atm
print('Wt_%_CH4_atmosphere =', wtpercentCH4atm)
wtpercentO2atm=100.0*result[19]*mol_wts[19]/grams_per_mole_atm
print('Wt_%_O2_atmosphere =', wtpercentO2atm)
wtpercentH2Oatm=100.0*result[20]*mol_wts[20]/grams_per_mole_atm
print('Wt_%_H2O_atmosphere =', wtpercentH2Oatm)
wtpercentFeatm=100.0*result[21]*mol_wts[21]/grams_per_mole_atm
print('Wt_%_Fe_atmosphere =', wtpercentFeatm)
wtpercentMgatm=100.0*result[22]*mol_wts[22]/grams_per_mole_atm
print('Wt_%_Mg_atmosphere =', wtpercentMgatm )
wtpercentSiOatm=100.0*result[23]*mol_wts[23]/grams_per_mole_atm
print('Wt_%_SiO_atmosphere =', wtpercentSiOatm)
wtpercentNaatm=100.0*result[24]*mol_wts[24]/grams_per_mole_atm
print('Wt_%_Na_atmosphere =', wtpercentNaatm)
wtpercentSiH4atm=100.0*result[25]*mol_wts[25]/grams_per_mole_atm
print('Wt_%_SiH4_atmosphere =', wtpercentSiH4atm)

density_atm=grams_per_mole_atm/(10.0*Rgas*TK[nT]/New_Press) #factor 10 converts J/bar to cc/mole
print('')
print('Compressed_density_of_atmosphere =',density_atm,'g/cc' )

print('')
print('Equilibrium_planet_composition:')
moles_atm=result[26]
print('  Moles_atm =',moles_atm)
moles_silicate=result[27]
print('  Moles_silicate = ',moles_silicate)
moles_metal=result[28]
print('  Moles_metal =',moles_metal)
molefrac_atm=moles_atm/(moles_atm+moles_silicate+moles_metal)
molefrac_silicate=moles_silicate/(moles_atm+moles_silicate+moles_metal)
molefrac_metal=1.0-molefrac_atm-molefrac_silicate
print('')
print('  Mole_fraction_atmosphere =',molefrac_atm)
print('  Mole_fraction_silicate =',molefrac_silicate)
print('  Mole_fraction_metal =',molefrac_metal)
print(' ')
grams_atm=molefrac_atm*grams_per_mole_atm  #actually grams_i/mole planet
grams_silicate=molefrac_silicate*grams_per_mole_silicate
grams_metal=molefrac_metal*grams_per_mole_metal
totalmass=grams_atm+grams_silicate+grams_metal
massfrac_atm=grams_atm/totalmass
massfrac_silicate=grams_silicate/totalmass
massfrac_metal=grams_metal/totalmass
print('  Mass_fraction_atmosphere =',massfrac_atm)
print('  Mass_fraction_silicate =',massfrac_silicate)
print('  Mass_fraction_metal =',massfrac_metal)

#Uncompressed density of planet
density_planet=massfrac_metal*density_core+massfrac_silicate*density_mantle+massfrac_atm*density_atm
print('  Density_of_planet =',density_planet,'g/cc')

#Estimate atmospheric pressure at the surface of the planet: fratio is the Matm/Mplanet mass ratio
#fratio=massfrac_atm/(1.0-massfrac_atm)
#New_Press=1.2e6*fratio*(Mplanet_Mearth)**(2.0/3.0)
print('')
print('  Mass_of_planet =',Mplanet_Mearth,'Earth masses')
print('  Estimated_surface_pressure =',New_Press,' bar')

#Estimate nominal atmospheric pressure at the surface of the planet: fratio is the Matm/Mplanet mass ratio
fratio=massfrac_atm/(1.0-massfrac_atm)
Nominal_Press=1.2e6*fratio*(Mplanet_Mearth)**(2.0/3.0)
print('  Nominal_surface_pressure =',Nominal_Press,' bar')

# Alternative calculation of pressure at surface of planet: Matm*g/Area_surface
#G=6.67408e-11 #m^3/(kg s^2)
#pi=3.141592
#Mearth= 5.972e24  #kg
#Mcorekg=Mplanet_Mearth*Mearth
#Rearth=6371.0*1000.0 #m
#Rcore=Rearth*(Mcorekg/Mearth)**(0.25)  #Valencia et al. 2006
#Alt_Press=1.0e-5*(G*Mcorekg/(Rcore**2.0))*(fratio*Mcorekg/(4.0*pi*Rcore**2.0))  #in bar
#print('  Alternate_pressure_estimate =',Alt_Press,' bar')

print('')
nSi_final=result[27]*(result[1]+result[2]+result[4]+result[6])+result[12]*result[28]+(result[23]+result[25])*result[26] #Si
nMg_final=result[27]*(result[0]+result[2])+result[22]*result[26] #Mg
nO_final=(result[27]*(result[0]+2.0*result[1]+3.0*result[2]+result[3]+3.0*result[4]+result[5]+3.0*result[6]+result[8]+result[9]+2.0*result[10]) \
    + result[28]*result[13]+result[26]*(result[23]+2.0*result[17]+2.0*result[19]+result[20]+result[16]) ) #O
nFe_final=result[27]*(result[3]+result[4])+result[28]*result[11]+result[26]*result[21] #Fe
nH_final=result[28]*result[14]+result[27]*(2.0*result[7]+2.0*result[8])+result[26]*(2.0*result[15]+2.0*result[20]+4.0*result[18]+4.0*result[25]) #H
nNa_final=result[27]*(2.0*result[5]+2.0*result[6])+result[26]*result[24] #Na
nC_final=result[27]*(result[9]+result[10])+result[26]*(result[18]+result[17]+result[16]) #C

print('Mass balance checks:')
print('  Initial moles Si =',nSi)
print('  Final moles Si   =', nSi_final)
print('')
print('  Initial moles Mg =',nMg)
print('  Final moles Mg   =', nMg_final)
print('')
print('  Initial moles H  =',nH)
print('  Final moles H    =',nH_final)
print('')
print('  Initial moles O  =',nO)
print('  Final moles O    =',nO_final)
print('')
print('  Initial moles Fe =',nFe)
print('  Final moles Fe   = ',nFe_final)
print('')
print('  Initial moles Na =',nNa)
print('  Final moles Na   = ',nNa_final)
print('')
print('  Initial moles C =',nC)
print('  Final moles C   = ',nC_final)
print('')
print('Finished: Mole fractions and moles stored in final_atm.txt.')


# POST-OPTIMIZATION TRIAGE - print deviations from equilibrium and mass balance
# Here KEQ includes mole fractions and pressures and so these are actual equilibrium constants....
KEQ=np.zeros(19)
K=np.zeros(19)
for i in range(0,19):
    KEQ[i]=exp(y[i])  # Thermodynamic equilibrium constants
    K[i]=exp(best_fit_model[i])  # Calculated equilibrium constants from mole fractions and pressures

# Mass balance for elements

f20_f=(nSi-(result[27]*(result[1]+result[2]+result[4]+result[6])+result[12]*result[28]+(result[23]+result[25])*result[26]))
f21_f=(nMg-(result[27]*(result[0]+result[2])+result[22]*result[26]))
f22_f=(nO-(result[27]*(result[0]+2.0*result[1]+3.0*result[2]+result[3]+3.0*result[4]+result[5]+3.0*result[6]+result[8]+result[9]+2.0*result[10]) \
        + result[28]*result[13]+result[26]*(result[23]+2.0*result[17]+2.0*result[19]+result[20]+result[16]) ))
f23_f=(nFe-(result[27]*(result[3]+result[4])+result[28]*result[11]+result[26]*result[21]))
f24_f=(nH-(result[28]*result[14]+result[27]*(2.0*result[7]+2.0*result[8])+result[26]*(2.0*result[15]+2.0*result[20]+4.0*result[18]+4.0*result[25])))
f25_f=(nNa-(result[27]*(2.0*result[5]+2.0*result[6])+result[26]*result[24]))
f26_f=(nC-(result[27]*(result[9]+result[10])+result[26]*(result[18]+result[17]+result[16])))
    
# Summing constraint on mole fractions
f27_f=1.0-result[0]-result[1]-result[2]-result[3]-result[4]-result[5]-result[6]-result[7]-result[8]-result[9]-result[10]
f28_f=1.0-result[11]-result[12]-result[13]-result[14]
f29_f=1.0-result[15]-result[16]-result[17]-result[18]-result[19]-result[20]-result[21]-result[22]-result[23]-result[24]-result[25]
print('')
print('Calculated KEQ vs. actual KEQ values:')
sum_thermo=0
for i in range(0,19):
    print("Rxn %.d KEQmodel = %10.3e vs KEQ = %10.3e " % ((i+1),K[i],KEQ[i]))
    sum_thermo=sum_thermo+abs(ln(K[i])-ln(KEQ[i]))
print('')
print('Sum abs(f1) to abs(f19), i.e. abs(ln(keq)-model values) =',sum_thermo)
print('')
print('Final deviations from mass balance:')
percentf20=100.0*f20_f/nSi
print("f20 = %10.6f (%6.3f %%) " %(f20_f, percentf20))

percentf21=100.0*f21_f/nMg
print("f21 = %10.6f (%6.3f %%) " %(f21_f, percentf21))

percentf22=100.0*f22_f/nO
print("f22 = %10.6f (%6.3f %%) " %(f22_f, percentf22))

percentf23=100.0*f23_f/nFe
print("f23 = %10.6f (%6.3f %%) " %(f23_f, percentf23))

percentf24=100.0*f24_f/nH
print("f24 = %10.6f (%6.3f %%) " %(f24_f, percentf24))

percentf25=100.0*f25_f/nNa
print("f25 = %10.6f (%6.3f %%) " %(f25_f, percentf25))

percentf26=100.0*f26_f/nC
print("f26 = %10.6f (%6.3f %%) " %(f26_f, percentf26))

sum_massbalance=abs(f20_f)+abs(f21_f)+abs(f22_f)+abs(f23_f)+abs(f24_f)+abs(f25_f)+abs(f25_f)
sum_moles=nSi+nMg+nO+nFe+nH+nNa+nC
percent_tot_moles=100.0*sum_massbalance/sum_moles
print("Sum abs(f20) to abs(f26) = %10.6f (%6.3f %%) " %(sum_massbalance, percent_tot_moles))
print('')

print('Final deviations from sum phase mole fractions = 1:')
percentf27=f27_f*100.0
print("f27 (mole frac silicate dev from 1) = %10.6f (%6.3f %%) " %(f27_f,percentf27))

percentf28=f28_f*100.0
print("f28 (mole frac metal dev from 1) = %10.6f (%6.3f %%) " %(f28_f,percentf28))

percentf29=f29_f*100.0
print("f29 (mole frac atm dev from 1) = %10.6f (%6.3f %%) " %(f29_f,percentf29))

sum_summing=abs(f27_f)+abs(f28_f)+abs(f29_f)
percent_summing=100.0*sum_summing/3.0
print("Sum abs(f27) to abs(f29) = %10.6f (%6.3f %%) " %(sum_summing, percent_summing))
print('')
print('')

#Write summary of results to a file
a_file = open('output_summary_atm_SiH4_xB.txt', 'w')

a_file.write("%10.5e " %red_chi_square)
a_file.write('# Red chi-square of fit\n')

a_file.write("%10.5e " %Mplanet_Mearth)
a_file.write('# Planet mass in Earth masses\n')
a_file.write("%10.5e " %temp)
a_file.write('# Temperature K\n')
a_file.write("%10.5e " %New_Press)
a_file.write('# Pressure in bar\n')

a_file.write("%10.5e " %massfrac_atm)
a_file.write('# Mass fraction atmosphere\n')

a_file.write("%10.5e " %massfrac_silicate)
a_file.write('# Mass fraction silicate mantle\n')

a_file.write("%10.5e " %massfrac_metal)
a_file.write('# Mass fraction metal\n')

for i in range(0,numvar-1):
    a_file.write("%10.5e " % result[i])
    a_file.write("")
    a_file.write('# %s\n' %var_names[i])

a_file.write("%10.5e " %DIW_actual)
a_file.write('# DIW condensed planet rel to IW\n')

a_file.write("%10.5e " %wtpercentSiO2)
a_file.write('# wt percent SiO2 melt\n')
a_file.write("%10.5e " %wtpercentMgO)
a_file.write('# wt percent MgO melt\n')
a_file.write("%10.5e " %wtpercentFeO)
a_file.write('# wt percent FeO melt\n')
a_file.write("%10.5e " %wtpercentNa2O)
a_file.write('# wt percent Na2O melt\n')
a_file.write("%10.5e " %wtpercentHmelt)
a_file.write('# wt percent H2 melt\n')
a_file.write("%10.5e " %wtpercentH2Omelt)
a_file.write('# wt percent H2O melt\n')
a_file.write("%10.5e " %wtpercentCOmelt)
a_file.write('# wt percent CO melt\n')
a_file.write("%10.5e " %wtpercentCO2melt)
a_file.write('# wt percent CO2 melt\n')

a_file.write("%10.5e " %wtpercentFe)
a_file.write('# wt percent Fe metal\n')
a_file.write("%10.5e " %wtpercentSi)
a_file.write('# wt percent Si metal\n')
a_file.write("%10.5e " %wtpercentO)
a_file.write('# wt percent O metal\n')
a_file.write("%10.5e " %wtpercentH)
a_file.write('# wt percent H metal\n')
a_file.write("%10.5e " %density_core)
a_file.write('# Uncompressed density of metal g/cc\n')
a_file.write("%10.5e " %density_pure_iron)
a_file.write('# Uncompressed density of pure Fe g/cc\n')
a_file.write("%10.5e " %density_deficit)
a_file.write('# Experimental metal percent density deficit\n')

a_file.write("%10.5e " %grams_per_mole_atm)
a_file.write('# grams per mole atmosphere\n')
a_file.write("%10.5e " %wtpercentH2atm)
a_file.write('# wt percent H2 atm\n')
a_file.write("%10.5e " %wtpercentH2Oatm)
a_file.write('# wt percent H2O atm\n')
a_file.write("%10.5e " %wtpercentSiOatm)
a_file.write('# wt percent SiO atm\n')
a_file.write("%10.5e " %wtpercentMgatm)
a_file.write('# wt percent Mg atm\n')
a_file.write("%10.5e " %wtpercentFeatm)
a_file.write('# wt percent Fe atm\n')
a_file.write("%10.5e " %wtpercentNaatm)
a_file.write('# wt percent Na atm\n')
a_file.write("%10.5e " %wtpercentCOatm)
a_file.write('# wt percent CO atm\n')
a_file.write("%10.5e " %wtpercentCO2atm)
a_file.write('# wt percent CO2 atm\n')
a_file.write("%10.5e " %wtpercentCH4atm)
a_file.write('# wt percent CH4 atm\n')
a_file.write("%10.5e " %wtpercentO2atm)
a_file.write('# wt percent O2 atm\n')
a_file.write("%10.5e  " %DIW_atm)
a_file.write('# DIW atmosphere rel to IW\n')

for i in range(0,19):
    if i == 14:
        K14temp=K[i]/New_Press
        a_file.write("%10.5e  " %K14temp)
    else:
        a_file.write("%10.5e  " %K[i])
    a_file.write('# Model KEQ for rxn %d : %s \n' %((i+1),rxn_names[i+1]))
a_file.close()
print('')
# APPEND input file to the output_summary file
name_initial='initial_atm_vMCMC_2024.txt'
def copy_inputs(name_initial):
    count = len(open(name_initial).readlines())
    file_input=open(name_initial, 'r')
    Lines =file_input.readlines()
    afile = open('output_summary_atm_SiH4_xB.txt', 'a+')
    afile.write('\nINPUT FILE:\n')
    i=0
    while i < count:
        Newline=Lines[i].strip('\n')
    #    print(Newline)
        afile.write('%s \n' %Newline)
        i=i+1
    afile.close()
copy_inputs(name_initial)



# plot histogram of posteriors
#plt.hist(posteriors, 100,histtype="step")
#plt.xlabel('ln(Posteriors)')
#plt.show(block=False)
#plt.pause(10)
#plt.savefig('ln_posteriors_hist.png')
#plt.close()
#
#print('histogram of ln(posteriors) saved as ln_posteriors_hist.png')
#print('')

#print('Making corner plot, this takes a very long while...')
#print('')
#
## CORNERS allows us to make corner plots of the samplings for the search
#labels=['xMgO','xSiO2','xMgSiO3','xFeO','xFeSiO3','xNa2O','xNa2SiO3','xH2m','xH2Om', \
#'xCOm','xCO2m','xFe','xSi','xO','xH','xH2g','xCOg','xCO2g','xCH4g','xO2g', \
#'xH2Og','xFeg','xMg','xSiOg','xNag','xSiH4','Matm','Msil','Mmetal',' Press']
#labels_short=labels[12:26] #replaces labels
#smpl_test=samples[:,12:26] #replaces samples
#figure = corner.corner(smpl_test,show_titles=True,labels=labels_short,plot_datapoints=True) #plot_datapoints=True
##plt.show() # necessary to show the actual plot
#figure.savefig("corner_atmosphere.png")
#plt.close()


print('End.')





