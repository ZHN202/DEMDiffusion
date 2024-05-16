import cv2
import os


def crop_image(file, image_path, crop_size, save_dir):
    # 读取原始图片
    if image_path[-4:]=='.tif':
        image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return

    # 获取原始图片的尺寸
    height, width = image.shape[:2]

    # 计算裁剪后的图片数量
    num_rows = height // crop_size[1]
    num_cols = width // crop_size[0]

    # 裁剪图片并保存
    for y in range(num_rows):
        for x in range(num_cols):
            # 计算当前裁剪区域的坐标
            x_start = x * crop_size[0]
            y_start = y * crop_size[1]
            x_end = x_start + crop_size[0]
            y_end = y_start + crop_size[1]

            # 裁剪图片
            cropped_image = image[y_start:y_end, x_start:x_end]
            print('max',max(cropped_image.flatten()))
            print('min',min(cropped_image.flatten()))

            # # 保存裁剪后的图片
            # if image_path[-4:]=='.tif':
            #     cv2.imwrite(save_dir + f"\\"+file+f"_{y}_{x}.tif", cropped_image)
            # else:
            #     cv2.imwrite(save_dir + f"\\"+file+f"_{y}_{x}.png", cropped_image)
            # print(f"Saved "+file+f"_{y}_{x}.jpg")

if __name__ == '__main__':

    # 指定图片路径和裁剪尺寸
    image_dir_path = r"K:\Mars1013_-5~10\Mars1013\min_test_B"
    crop_size = (64, 64)  # 裁剪尺寸为 (100, 100)
    file_list = os.listdir(image_dir_path)
    save_dir = r"K:\Mars1013_-5~10\Mars1013\min_test_B_crop"
    for file in file_list:
        image_path = os.path.join(image_dir_path, file)
        file_name = file.split('.')[0]
        crop_image(file_name, image_path, crop_size, save_dir)
