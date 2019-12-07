# define a class for processing modis 1km LST

from pyhdf.SD import SD, SDC
from utils import parsemodis
from utils.maskmodis import ModisQuality, Masker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import earthpy.plot as ep
import numpy as np


class getModisLST_1km():

    def __init__(self, MODIS_file, prefix, lats, longs):
        '''
        getModisLST_1km object digests MODIS .hdf files and return parsed data
        based on input spatial coordinates.

        :param MODIS_file (str): the path where modis .hdf saves
        :param prefix (str): specifies the temperature to be extracted, can be either "night" or "day"
        :param lats, longs (float/double list): specify the locations in GCS, and should be in pairs
        '''

        self.MODIS_file = MODIS_file
        self.lats = np.array(lats)
        self.longs = np.array(longs)
        self.modisParser = parsemodis.parseModis(MODIS_file)
        self.prefix = prefix

        quality = ModisQuality.high # set quality to high
        file = SD(MODIS_file, SDC.READ) # open hdf4 file
        # read data according prefix
        if prefix == 'Day':
            # get day time temperature
            self.data = file.select('LST_Day_1km').get()
            masker = Masker(band=MODIS_file, band_name='QC_Day')
        elif prefix == 'Night':
            # get night time temperature
            self.data = file.select('LST_Night_1km').get()
            masker = Masker(band=MODIS_file, band_name='QC_Night')
        else:
            print('Prefix can only be Day or Night')
            return(0)
        self.mask = masker.get_mask(0, 2, quality).astype(int) # QA mask


    def getBound(self):
        '''
        a function to get boundary of MODIS
        '''
        return self.modisParser.retBoundary()


    def printBands(self):
        '''
        a function to print MODIS .hdf information
        '''
        file = SD(self.MODIS_file, SDC.READ)
        datasets_dic = file.datasets()
        for idx,sds in enumerate(datasets_dic.keys()):
            print(idx,sds)


    def runQC(self):
        '''
        a function to run QC on modis data
        '''
        return self.data * self.mask


    def plotQC(self):
        '''
        a function to plot modis QC
        '''
        plt.figure(figsize=(20,10))
        im = plt.imshow(self.mask)
        ep.colorbar(im)
        plt.show()


    def plotData(self):
        '''
        a function to plot LST data
        '''
        plt.figure(figsize=(20,10))
        im = plt.imshow(self.toCelsius(self.data))
        im_bar = plt.colorbar(im, shrink = 0.75)
        im_bar.set_label('LST (Celsius)')
        plt.show()


    def plotOverlap(self, zval=None, z_label=None):
        '''
        a function to plot LST data with spatial points on top
        :param zval (numeric array): specifies the z-value to be plot
                                     long with the spatial points
        '''
        plt.figure(figsize=(20,10))

        # set up image plot
        im = plt.imshow(self.toCelsius(self.data))
        im_bar = plt.colorbar(im, shrink = 0.75)
        # im_bar = ep.colorbar(im)
        # set up labels
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        im_bar.set_label('LST (Celsius)')

        # set up points plot
        _, _, rows, cols = self.toMatrixCoor()
        if zval != None:
            zval = np.array(zval)
            points = plt.scatter(x=cols, y=rows, c=zval,
                                 norm=colors.Normalize(vmin=zval.min(), vmax=zval.max()),
                                 cmap='coolwarm')
            pts_bar = plt.colorbar(points, shrink = 0.75)
            if z_label != None:
                pts_bar.set_label(z_label)
        else:
            plt.scatter(x=cols, y=rows, c='magenta')
        # show
        plt.show()


    def plotPoints(self, zval=None, z_label=None):
        '''
        a function to plot LST data with spatial points on top
        :param zval (numeric array): specifies the z-value to be plot
                                     long with the spatial points
        '''
        plt.figure(figsize=(20,10))
        plt.gca().invert_yaxis()

        # set up points plot
        _, _, rows, cols = self.toMatrixCoor()
        if zval != None:
            zval = np.array(zval)
            points = plt.scatter(x=cols, y=rows, c=zval,
                                 norm=colors.Normalize(vmin=zval.min(), vmax=zval.max()),
                                 cmap='coolwarm')
            pts_bar = plt.colorbar(points, shrink = 0.25)
            if z_label != None:
                pts_bar.set_label(z_label)
        else:
            plt.scatter(x=cols, y=rows)
        # show
        plt.show()


    def toMatrixCoor(self):
        '''
        a function to convert spatial coordinates to matrix coordinates
        '''

        # get geoinformation
        bound = self.modisParser.retBoundary()
        long_res = (bound['max_lon'] - bound['min_lon']) / self.data.shape[1] # col - x
        lat_res = (bound['max_lat'] - bound['min_lat']) / self.data.shape[0] # row - y

        # retreive value based on lat/long
        rows = ((bound['max_lat'] - self.lats) / lat_res).astype(int)
        cols = ((self.longs - bound['min_lon']) / long_res).astype(int)

        # trim those locations out of bounds
        row_outlier = (rows<0) | (rows >= self.data.shape[0])
        col_outlier = (cols<0) | (cols >= self.data.shape[1])
        outliers = row_outlier | col_outlier
        print(f'There are {sum(outliers)} has been excluded due to laying out boundary')

        # updates location parameters
        rows = rows[(outliers == False)]
        cols = cols[(outliers == False)]
        lats = self.lats[(outliers == False)]
        longs = self.longs[(outliers == False)]

        return lats, longs, rows, cols


    def toCelsius(self, modis_data):
        '''
        a function to convert MODIS temperature to Celsius
        '''
        modis_data = (modis_data * 1.0) # convert into float
        modis_data[np.where(modis_data == 0)] = np.nan # set invalided value to nan
        modis_data = modis_data * 0.02 - 273.15 # convert to celsius
        return modis_data


    def getVal(self, dataType = 'qc_temperature'):
        '''
        a function to extract MODIS value

        :param dataType (str): specify the data type to be extracted
        :return np.array(longtidue, latitude, MODIS values)
        '''
        if dataType == 'raw_temperature':
            modis_data = self.toCelsius(modis_data=self.data)
        elif dataType == 'raw':
            modis_data = self.data
        elif dataType == 'qc_temperature':
            modis_data = self.runQC()
            modis_data = self.toCelsius(modis_data=modis_data)
        elif dataType == 'qc_mask':
            modis_data = self.mask

        lats, longs, rows, cols = self.toMatrixCoor()

        # return value from modis_data
        return np.array(list(zip(longs, lats, [modis_data[(row, col)] for row, col in zip(rows, cols)])))


    def createVarName(self):
        '''
        a function to create a variable name for storing extracted modis LST data
        '''
        # define name of the column
        time_range = self.modisParser.retRangeTime()
        return self.prefix+'_'+time_range['RangeBeginningDate']+'_'+time_range['RangeEndingDate']
