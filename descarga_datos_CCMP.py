# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:37:59 2017

@author: yordan
"""


#==============================================================================
"Creando rutas"
#==============================================================================

import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd


ano = '1999'
fechas = pd.date_range(ano+'-01-01', ano+'-12-31', freq='D')
RutasTxt=[]
for date in fechas:
	day   = '%02d' % (date.day,)
	month = '%02d' % (date.month,)
	year  = str(date.year)
	ruta= 'ftp://ftp2.remss.com/ccmp/v02.0/Y'+year+'/M'+month+'/'+'CCMP_Wind_Analysis_'+year+month+day+'_V02.0_L3.0_RSS.nc'
	RutasTxt.append(ruta)    
np.savetxt('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/CCMP/'+ano+'/rutas_temp_'+ano, np.array(RutasTxt), delimiter=" ", fmt="%s")
