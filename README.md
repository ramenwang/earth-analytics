# Earth Observatory Analytics
-- homebuild toolkits for processing remote sensing imagery

## [MODIS LST - python scripts for downloading, preprocessing, and value extraction](https://github.com/ramenwang/earth-analytics/tree/master/MODIS_LST)

Some of the scripts are modifications from pyModis and pyMasker, the copyright should belong to original authors. 
The modified version solves the issue where brew installed GDAL does not support HDF4 driver. However, one still need to build following dependencies:

### Building hdf4 driver on mac

1. get [hdf-4.2.10.tar.gz](https://support.hdfgroup.org/ftp/HDF/releases/HDF4.2.10/src/hdf-4.2.10.tar.gz)
2. unzip hdf-4.2.10.tar.gz and go into the directory
3. $ cd hdf-4.2.10 && ./configure --disable-fortran --enable-production --enable-shared --disable-netcdf --with-zlib=/usr --with-jpeg=/usr/local --prefix=/usr/local/hdf4
4. $ make >& make.out
5. $ make check >& check.out
6. $ make install

### Building pyhdf4 wrapper on mac

1. $ export INCLUDE_DIRS=/usr/local/hdf4/include (make sure consist with the hdf configuration)
2. $ export LIBRARY_DIRS=/usr/local/hdf4/lib (make sure consist with the hdf configuration)
3. $ pip3 install pyhdf

### Other python library
glob, requests, gdal (brew install gdal (on mac), which does not include hdf4 driver)


## [General tools for image processing](https://github.com/ramenwang/earth-analytics/tree/master/general_tools)

This folder are hand crafted tools for the convinience of processing RS imagery data in python environment. It requires some python packages to be installed: rasterio, gdal, earthpy and matplotlib
