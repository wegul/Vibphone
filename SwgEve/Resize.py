import os
import cv2

# 所有需要处理的图片的路径 自动遍历文件夹内所有文件（包括子文件） 填写路径全称
traversal_file="F:/2020AccelEve/database/sound/_wav/rgb"
# 修改完成后输出的文件夹
output_file="F:/2020AccelEve/database/sound/_wav/rgb/resize224"
# 变成多少分辨率的方形图
img_width_height=224

def resize_img(img_path,save_path):
    # 读取原图片
    img = cv2.imread(img_path)
    h, w = img.shape[0], img.shape[1]
    print("Original h : " + str(h) + "px Original w : " + str(w) + "px")
    rateh=img_width_height/h
    ratew=img_width_height/w
    # img_processing=img[0:65,:]
    # img_processing = cv2.resize(img, (0, 0), fx=ratew, fy=rateh, interpolation=cv2.INTER_NEAREST)
    # 这里改一下名称，最后三位强制改为jpg
    # if save_path[-3:] == "png":
    #     save_path=save_path.replace("png", "jpg")
    cv2.imwrite(save_path, img_processing)
    print("Save as : " + save_path)

def show_files(path, all_files):
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            # 拼接文件路径
            all_files.append(path+"/"+file)
    return all_files

# 首先检测两个路径是否都可访问，如果水印路径没有则创建
if os.path.isdir(traversal_file):
    print("Check traversal file ok")
else:
    print("Traversal file error")

if os.path.isdir(output_file):
    print("Check water mask file ok")
else:
    print("Water mask file warning,auto create it")
    os.mkdir(output_file)

#首先遍历文件夹，然后对每个文件进行处理
# 传入空的list接收文件名
contents = show_files(traversal_file, [])
# 循环打印show_files函数返回的文件名列表
for content in contents:
    # print(content)
    # 判断是否为图片
    if content.endswith('jpg') or content.endswith('png'):
        print("processing : "+content)
        resize_img(content,output_file + "/" +os.path.basename(content))
