import os

import error
import numpy as np
import PIL.Image as Image

if __name__ == '__main__':
    dem_file_list = os.listdir(r'G:test_4')
    print(type(dem_file_list[0]))
    mars_dem_file_list = [x for x in dem_file_list if x[0]=='m']
    moon_dem_file_list = [x for x in dem_file_list if x[0]!='m']
    print(len(mars_dem_file_list))
    print(len(moon_dem_file_list))
    mars_vae_file_list = [x for x in mars_dem_file_list if x.split('_')[4][0]=='v']
    mars_cd_file_list = [x for x in mars_dem_file_list if x.split('_')[4][0]=='c']
    moon_vae_file_list = [x for x in moon_dem_file_list if x.split('_')[3][0]=='v']
    moon_cd_file_list = [x for x in moon_dem_file_list if x.split('_')[3][0]=='c']
    gt_dem_mars_file_list = os.listdir(r'G:test\test')
    gt_dem_moon_file_list = os.listdir(r'G:test\testB')
    print(len(mars_vae_file_list))
    print(len(mars_cd_file_list))
    print(len(moon_vae_file_list))
    print(len(moon_cd_file_list))

    for i in range(len(gt_dem_mars_file_list)):
        dem = Image.open(r'G:test\test' + '\\' + gt_dem_mars_file_list[i])
        dem = np.array(dem)
        dem = Image.fromarray(dem)
        dem.save(r'G:test_4' + '\\' + gt_dem_mars_file_list[i])
