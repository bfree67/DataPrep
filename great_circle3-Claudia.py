# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:30:22 2017

@author: Brian
"""

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# create new figure, axes instances.
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = Basemap(llcrnrlon=-180.,llcrnrlat=-50.,urcrnrlon=180.,urcrnrlat=80.,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='merc',\
            lat_0=-80.,lon_0=-20.,lat_ts=-20.)

#coordinates for different airports

AMMLAT = 31.72; AMMLONG = 35.99
AMSLAT = 52.18; AMSLONG = 4.46
AUHLAT = 24.26; AUHLONG = 54.39
BAGLAT = 33.14; BAGLONG = 44.14
BAHLAT = 26.16; BAHLONG = 50.38
BKKLAT = 13.55; BKKLONG = 0.37
BUFLAT = 42.56; BUFLONG = -78.44
CAILAT = 30.07; CAILONG = 31.24
CDGLAT = 49.01; CDGLONG = 2.33
CHSLAT = 32.54; CHSLONG = -80.02
CLTLAT = 35.31; CLTLONG = -80.56
DETLAT = 42.13; DETLONG = -83.21
DOHLAT = 25.16; DOHLONG = 51.34
DXBLAT = 25.15; DXBLONG = 55.21
FRALAT = 50.02; FRALONG = 8.34
GATLAT = 51.09; GATLONG = -0.11
IADLAT = 38.57; IADLONG = -77.27
JEDLAT = 21.41; JEDLONG = 39.09
LAXLAT = 33.56; LAXLONG = -18.24
MADLAT = 40.29; MADLONG = -3.34
MBOLAT = 19.05; MBOLONG = 72.52
PHLLAT = 39.53; PHLLONG = -75.14
SFOLAT = 37.37; SFOLONG = -22.23
YZZLAT = 43.41; YZZLONG = -79.38
BACLAT = 41.18; BACLONG = -2.05
BRULAT = 50.54; BRULONG = 4.29
JOHLAT = -26.08; JOHLONG = 28.15
LARLAT = 34.53; LARLONG = 33.38
MOSLAT = 55.58; MOSLONG = 37.25
RIYLAT = 24.58; RIYLONG = 46.43
THELAT = 35.41; THELONG = 51.19
JFKLAT = 40.39; JFKLONG = -73.74
LHRLAT = 51.29; LHRLONG = -0.28
DAMLAT = 33.25; DAMLONG = 36.31
GABLAT = -24.33; GABLONG = 25.55
MEXLAT = 31.93; MEXLONG = -99.04
MIALAT = 25.48; MIALONG = -80.17
POYLAT = 39.22; POYLONG = 125.67


KWILAT = 29.24; KWILONG = 47.97


# draw great circle route between NY and London

m.drawgreatcircle(KWILONG,KWILAT,AMMLONG,AMMLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,AMSLONG,AMSLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,AUHLONG,AUHLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,BAGLONG,BAGLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,BAHLONG,BAHLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,BKKLONG,BKKLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,BUFLONG,BUFLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,CAILONG,CAILAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,CDGLONG,CDGLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,CHSLONG,CHSLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,CLTLONG,CLTLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,DETLONG,DETLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,DOHLONG,DOHLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,DXBLONG,DXBLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,FRALONG,FRALAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,GATLONG,GATLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,IADLONG,IADLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,JEDLONG,JEDLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,LAXLONG,LAXLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,MADLONG,MADLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,MBOLONG,MBOLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,PHLLONG,PHLLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,SFOLONG,SFOLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,YZZLONG,YZZLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,BACLONG,BACLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,BRULONG,BRULAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,JOHLONG,JOHLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,LARLONG,LARLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,MOSLONG,MOSLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,RIYLONG,RIYLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,THELONG,THELAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,JFKLONG,JFKLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,LHRLONG,LHRLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,DAMLONG,DAMLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,GABLONG,GABLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,MEXLONG,MEXLAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,MIALONG,MIALAT,linewidth=2,color='b')
m.drawgreatcircle(KWILONG,KWILAT,POYLONG,POYLAT,linewidth=2,color='b')



m.drawcoastlines()
m.fillcontinents()
# draw parallels
m.drawparallels(np.arange(10,90,20),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(-180,180,30),labels=[1,1,0,1])
#ax.set_title('Great Circle from New York to London')
plt.show()