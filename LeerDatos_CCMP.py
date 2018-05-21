
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.basemap 
from mpl_toolkits.basemap import Basemap
import datetime
import matplotlib.colors
import pandas as pd
import math
from windrose import WindroseAxes
import os 
import netCDF4 as nc


Wind=Dataset('CCMP_Wind_Analysis_19930201_V02.0_L3.0_RSS.nc','r')

print Wind.variables
print Wind.variables.keys()

Uwind= np.array(Wind.variables['uwnd'][:])
Vwind= np.array(Wind.variables['vwnd'][:])
lat= np.array(Wind.variables['latitude'][:])
lon= np.array(Wind.variables['longitude'][:])
time= np.array(Wind.variables['time'][:]).astype(np.float)


#se recorta lat y lon 

lat1=np.where((lat>10.875) & (lat<15.125))[0]
lon1=np.where((lon>277.625) & (lon<281.875))[0]
#se recort la info (time, lat , lon)

VientoU=Uwind[:,:,lon1]
VientoU=VientoU[:,lat1,:]

VientoV=Vwind[:,:,lon1]
VientoV=VientoV[:,lat1,:]



############Ciclo para leer todos los datos########################
rango = pd.date_range('20010101', '20121231', freq='D')

fechas= np.array([rango[i].strftime('%Y%m%d') for i in range(len(rango))])

for k in range (2001,2013):
    for i in fechas:
    	if os.path.isfile('C:/Users/Unalmed/Documents/Codigos/CCMP/'+str(k)+'/CCMP_Wind_Analysis_'+i+'_V02.0_L3.0_RSS.nc')==True:               ###para verificar que el archivo exista porque hay algunos días que no están
    	
    	    	Wind=Dataset('C:/Users/Unalmed/Documents/Codigos/CCMP/'+str(k)+'/CCMP_Wind_Analysis_'+i+'_V02.0_L3.0_RSS.nc','r')
    	
    		Uwind= np.array(Wind.variables['uwnd'][:,lat1,lon1])
    		Vwind= np.array(Wind.variables['vwnd'][:,lat1,lon1])
    		lat= np.array(Wind.variables['latitude'][:])
    		lon= np.array(Wind.variables['longitude'][:])
    		time= np.array(Wind.variables['time'][:]).astype(np.float)
    
    		if i == fechas[0]:
    			Uwind_nc = Uwind
    			Vwind_nc = Vwind
    			time_nc = time
                else:
        		
        			Uwind_nc = np.concatenate((Uwind_nc,Uwind), axis=0)
        			Vwind_nc = np.concatenate((Vwind_nc,Vwind), axis=0)
        			time_nc = np.concatenate((time_nc,time), axis=0)
    	else:
    		fecha_faltante=np.zeros([4,len(lat1),len(lon1)])*np.nan
    		Uwind_nc = np.concatenate((Uwind_nc,fecha_faltante), axis=0)
    		Vwind_nc = np.concatenate((Vwind_nc,fecha_faltante), axis=0)
    		time_nc = np.concatenate((time_nc,time), axis=0)
    		  
####crear nc

nuevo=Dataset('DatosCCMP2','r')

U= np.array(nuevo.variables['Uwind'][:])
V= np.array(nuevo.variables['Vwind'][:])
latn= np.array(nuevo.variables['latitude'][:])
lonn= np.array(nuevo.variables['longitude'][:])
timen= np.array(nuevo.variables['time'][:]).astype(np.float)



file = nc.Dataset('C:/Users/Unalmed/Documents/Codigos/CCMP/'+'2001'+'_')

    print "Se leen datos"
    iso_T_20    = file['so20chgt'][:, 457:627, 229:910] # Depth of 20C isotherm


    y           = file['y'][457:627]
    x           = file['x'][229:910]

    time        = file['time_counter'][:] # Time


print "Creando nc"
A= nc.Dataset('DatosCCMP2', mode='w')


print "Creando dimensiones"
T= A.createDimension('time', len(time_nc))
X= A.createDimension('lon'   , len(lon1))
Y= A.createDimension('lat'   , len(lat1))


print "Creando variables"
T           = A.createVariable('time'       , 'f8', ('time'))
LAT         = A.createVariable('latitude'   , 'f8', ('lat' ))
LON         = A.createVariable('longitude'  , 'f8', ('lon'))
U   = A.createVariable('Uwind'   , 'f8', ('time', 'lat', 'lon'))
V   = A.createVariable('Vwind'   , 'f8', ('time', 'lat', 'lon'))



print "Introduciendo datos en variables"
T[:]           = time_nc
LAT[:]         = lat[lat1]
LON[:]         = lon[lon1]
U[:]    = Uwind_nc
V[:]    = Vwind_nc


print "Asignando atributos de varible a los datos"
T.units           = Wind['time'].units
LAT.units         = Wind['latitude'].units
LON.units         = Wind['longitude'].units
U.units    = Wind['uwnd'].units
V.units    = Wind['vwnd'].units

T.long_name           = Wind['time'].long_name
LAT.long_name         = Wind['latitude'].long_name
LON.long_name         = Wind['longitude'].long_name
U.long_name    = Wind['uwnd'].long_name
V.long_name    = Wind['vwnd'].long_name

T.calendar    = Wind['time'].calendar


A.close()
file.close()




####leer nc creado



     
Fecha = np.array([datetime.datetime(1987,01,01)+\
datetime.timedelta(hours = time_nc[i]) for i in range(len(time_nc))])


Uwind_nc[Uwind_nc==-9999.0]=np.nan
Vwind_nc[Vwind_nc==-9999.0]=np.nan

########magnitud y dirección###########

Vel=(Uwind_nc**2 + Vwind_nc**2)**0.5
Tang=Uwind_nc/Vwind_nc


Dir=np.zeros([len(Tang),len(lat1),len(lon1)])*np.nan
for i in range (len(Tang)):
	for j in range (len(lat1)):
		for k in range (len(lon1)):
			if ((Uwind_nc[i,j,k]>0) & (Vwind_nc[i,j,k]>0)):
				Dir[i,j,k]=math.degrees(math.atan(Tang[i,j,k]))
			elif ((Uwind_nc[i,j,k]>0) & (Vwind_nc[i,j,k]<0)):
				Dir[i,j,k]=180-(math.degrees(math.atan(Tang[i,j,k])))
			elif ((Uwind_nc[i,j,k]<0) & (Vwind_nc[i,j,k]<0)):
				Dir[i,j,k]=180+(math.degrees(math.atan(Tang[i,j,k])))
			else:
				Dir[i,j,k]=360-(math.degrees(math.atan(Tang[i,j,k])))
#################################Rosa de vientos######################

def new_axes(fig, rect):
    ax = WindroseAxes(fig, rect, axisbg='w')
    fig.add_axes(ax)
    return ax


def set_legend(ax):
    l = ax.legend(borderaxespad=-6.8)
    plt.setp(l.get_texts(), fontsize=8)

fig = plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='w')

ax1 = new_axes(fig, [0.1, 0.1, 0.4, 0.4])
ax1.bar(Dir[:,0,0],Vel[:,0,0], normed=True, opening=0.8, edgecolor='white')
set_legend(ax1)

##############################Ciclo Diurno en Mayo ##################################
viento  = {'Magnitud': Vel[:,6,9], 'Direccion': Dir[:,6,9]}

datos=pd.DataFrame(viento, index=Fecha)

ciclidiurno_Vel=np.zeros([4])*np.nan
ciclidiurno_Dir=np.zeros([4])*np.nan

for i,j in enumerate(range(0,23,6)):
	ciclidiurno_Vel[i]=np.nanmean(datos['Magnitud'][(np.where(datos.index.hour==j)[0])])
	ciclidiurno_Dir[i]=np.nanmean(datos['Direccion'][(np.where(datos.index.hour==j)[0])])
	
hora=['00:00','06:00','12:00','18:00']
plt.plot(hora,ciclidiurno_Vel)
plt.show()
	
	
 	



























VientoU=VientoU*-1

#ciclo anual punto 1 SAI

CicloAnual_Viento= np.zeros([12]) * np.NaN

Meses = np.array([fecha[i].month for i in range(len(fecha))])
for k in range(1,13):
    tmpp = np.where(Meses == k)[0]
   
    altura1_tmp= VientoU[tmpp]
    CicloAnual_Viento[k-1]= np.mean(altura1_tmp)

Fig= plt.figure()
plt.rcParams.update({'font.size':14})
plt.plot(CicloAnual_Viento,'-', color='skyblue',lw=3,label='Hs')
x_label = ['Año']
plt.title('Ciclo Anual Altura de Ola Significante', fontsize=24)
plt.xlabel('Mes',fontsize=18)
plt.ylabel('Hs(metros)',fontsize=18)
plt.legend(loc=0)
plt.show()

axes = plt.gca()
axes.set_xlim([0,11])
axes.set_ylim([0.9,2.0])
axes.set_xticks([0,1,2, 3, 4, 5, 6, 7,8, 9, 10 ,11]) #choose which x locations to have ticks
axes.set_xticklabels(['Ene','Feb','Mar','Abr','May','Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic' ]) 
plt.savefig('CicloAnualAltura1.png')

#ciclo anual con pandas

Waves=pd.Series(index=fecha, data=altura1)

WavesM=Waves.resample('M').mean()
WavesD=Waves.resample('D').mean()

WM=np.array(WavesM)
WM=WM[:-6]
WM=np.reshape(WM,(-1,12))
WMM=np.mean(WM,axis=0)
WMS=np.std(WM, axis=0)

plt.plot(WMM)


