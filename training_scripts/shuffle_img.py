import shutil
import os
import random



if __name__ == '__main__':

    # 指定图片路径
    image_dir_path = r"/home/vipuser/data/testA_crop"
    dem_dir_path = r"/home/vipuser/data/min_test_B_crop"

    # 指定目标路径
    test_image_save_dir = r"/home/vipuser/data/mix_test_A_64"
    test_dem_save_dir = r"/home/vipuser/data/mix_test_B_64"
    train_image_save_dir = r"/home/vipuser/data/mix_train_A_64"
    train_dem_save_dir = r"/home/vipuser/data/mix_train_B_64"

    file_list = os.listdir(image_dir_path)
    print(len(file_list))
    # 打乱文件

    random.shuffle(file_list)
    # 复制文件
    # for i,file in enumerate(file_list):
    #     if(i/len(file_list) < 0.8):
    #         shutil.copy(os.path.join(image_dir_path,file),train_image_save_dir)
    #         shutil.copy(os.path.join(dem_dir_path,file[:-4]+'.tif'),train_dem_save_dir)
    #     else:
    #         shutil.copy(os.path.join(image_dir_path,file),test_image_save_dir)
    #         shutil.copy(os.path.join(dem_dir_path,file[:-4]+'.tif'),test_dem_save_dir)
    #
    #     print('moved:',file)


