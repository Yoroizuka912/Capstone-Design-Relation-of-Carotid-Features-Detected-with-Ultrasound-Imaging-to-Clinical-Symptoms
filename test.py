from concurrent.futures import ThreadPoolExecutor
import requests
import time
import os
import glob
import numpy as np
import random

def post_request_with_files(url, data, file_paths):
    files = {
        'image': open(file_paths[0], 'rb'),
        'mask': open(file_paths[1], 'rb')
    }
    response = requests.post(url, data=data, files=files)
    print(response.text)
    response.close()
    return None

concurrent_num = 10
urls = ["http://127.0.0.1:5000/upload"] * concurrent_num  # Replace with the actual URLs
data = [
    {"age": 58, "gender": "female", "history": "high_bl"},
    {"age": 61, "gender": "male", "history": "high_bp", "history": "smoke", "history": "alcohol"},
    {"age": 67, "gender": "male", "history": "smoke", "history": "alcohol"},
    {"age": 52, "gender": "male", "history": "high_bp", "history": "high_bl", "history": "alcohol"},
    {"age": 36, "gender": "female"},
    {"age": 68, "gender": "male", "history": "high_bp"},
    {"age": 72, "gender": "male", "history": "high_bp", "history": "diabetes"},
    {"age": 73, "gender": "male", "history": "high_bp", "history": "smoke"},
    {"age": 79, "gender": "male", "history": "high_bp", "history": "diabetes", "history": "smoke"},
    {"age": 60, "gender": "male", "history": "high_bp", "history": "smoke", "history": "alcohol"},
    {"age": 84, "gender": "female"},
    {"age": 42, "gender": "male", "history": "diabetes", "history": "high_bp", "history": "smoke"},
    {"age": 77, "gender": "female", "history": "high_bp", "history": "high_bl"},
    {"age": 60, "gender": "female", "history": "high_bp", "history": "high_bl"},
    {"age": 67, "gender": "male", "history": "high_bp", "history": "high_bl", "history": "alcohol"},
    {"age": 59, "gender": "male", "history": "high_bp", "history": "high_bl", "history": "smoke"},
    {"age": 75, "gender": "male", "history": "diabetes", "history": "smoke"},
    {"age": 80, "gender": "female"},
    {"age": 66, "gender": "male", "history": "diabetes"},
    {"age": 55, "gender": "male", "history": "diabetes", "history": "smoke", "history": "alcohol", "history": "heart"},
    {"age": 69, "gender": "male", "history": "high_bp", "history": "diabetes", "history": "high_bl", "history": "alcohol", "history": "brain", "history": "heart"},
    {"age": 59, "gender": "male", "history": "high_bl", "history": "smoke"},
    {"age": 68, "gender": "female"},
    {"age": 65, "gender": "female", "history": "high_bp", "history": "diabetes", "history": "high_bl"},
    {"age": 62, "gender": "male", "history": "high_bp", "history": "diabetes", "history": "alcohol"},
    {"age": 75, "gender": "male", "history": "diabetes", "history": "smoke"},
    {"age": 52, "gender": "male", "history": "diabetes"},
    {"age": 74, "gender": "male", "history": "high_bp", "history": "smoke", "history": "alcohol"},
    {"age": 73, "gender": "female", "history": "high_bp"},
    {"age": 56, "gender": "female", "history": "diabetes", "history": "high_bl"},
    {"age": 70, "gender": "male", "history": "high_bp", "history": "high_bl"},
    {"age": 64, "gender": "female", "history": "high_bl"},
    {"age": 62, "gender": "male", "history": "diabetes"},
    {"age": 79, "gender": "male", "history": "high_bp", "history": "smoke"},
    {"age": 65, "gender": "female"},
    {"age": 63, "gender": "male", "history": "high_bp", "history": "smoke"} 
]
image_sym_folder, mask_sym_folder = "../image/standard/symptomatic/", "../image/mask/symptomatic/"
image_asym_folder, mask_asym_folder = "../image/standard/asymptomatic/", "../image/mask/asymptomatic/"
image_sym_paths = ["ZOUMINMEI-431.jpg", "QINHUAIPING-451.jpg", "ZHUHUIXIN-498.jpg", "SHENCHANGDONG-503.jpg", "WUJINGYI-537.jpg", "QIUZHIYUAN-540.jpg", "WANGMAOJUN-580.jpg", "QIUSHUHAI-582.jpg", "CAIJIANKUAN-584.jpg", "MALIHUA-599.jpg", "ZHANGFENGLUAN-621.jpg", "WANGKAI-626.jpg", "GUOJIEMEI-668.jpg", "QIUMAONV-680.jpg", "DINGDAOHUA-738.jpg", "MAAXIAOHUA-776.jpg", "XUJIENIAN-778.jpg", "YAOLAIDI-795.jpg", "TANGGUOFU-947.jpg", "JIANGBAOZHONG-954.jpg", "LIUYIGEN-963.jpg", "CGENXIANZHI-1007.jpg", "XIAYINGMEI-1060.jpg", "LIGUILAN-1065.jpg", "LUWEIZHONG-1070.jpg"] 
image_asym_paths = ["SHENYONGCHANG-505.jpg", "WANGJINBING-536.jpg", "FURENYI-538.jpg", "LUFENGDI-573.jpg", "ZENGJIE-622.jpg", "XUHUIMING-679.jpg", "LIUJIE-702.jpg", "XUETINGGANG-706.jpg", "ZHANGWENZHONG-874.jpg", "JIHANMING-921.jpg", "QIANGUANGWEI-933.jpg"]
mask_sym_paths = ["ZOUMINMEI-431.tif", "QINHUAIPING-451.tif", "ZHUHUIXIN-498.tif", "SHENCHANGDONG-503.tif", "WUJINGYI-537.tif", "QIUZHIYUAN-540.tif", "WANGMAOJUN-580.tif", "QIUSHUHAI-582.tif", "CAIJIANKUAN-584.tif", "MALIHUA-599.tif", "ZHANGFENGLUAN-621.tif", "WANGKAI-626.tif", "GUOJIEMEI-668.tif", "QIUMAONV-680.tif", "DINGDAOHUA-738.tif", "MAAXIAOHUA-776.tif", "XUJIENIAN-778.tif", "YAOLAIDI-795.tif", "TANGGUOFU-947.tif", "JIANGBAOZHONG-954.tif", "LIUYIGEN-963.tif", "CGENXIANZHI-1007.tif", "XIAYINGMEI-1060.tif", "LIGUILAN-1065.tif", "LUWEIZHONG-1070.tif"] 
mask_asym_paths = ["SHENYONGCHANG-505.tif", "WANGJINBING-536.tif", "FURENYI-538.tif", "LUFENGDI-573.tif", "ZENGJIE-622.tif", "XUHUIMING-679.tif", "LIUJIE-702.tif", "XUETINGGANG-706.tif", "ZHANGWENZHONG-874.tif", "JIHANMING-921.tif", "QIANGUANGWEI-933.tif"]

image_paths = [(image_sym_folder + image_path) for image_path in image_sym_paths] + [(image_asym_folder + image_path) for image_path in image_asym_paths]
mask_paths = [(mask_sym_folder + mask_path) for mask_path in mask_sym_paths] + [(mask_asym_folder + mask_path) for mask_path in mask_asym_paths]
file_paths = list(zip(image_paths, mask_paths))

random_idx = random.sample(list(range(36)), concurrent_num)

start = time.perf_counter()
with ThreadPoolExecutor() as executor:
    results = list(executor.map(post_request_with_files, urls, np.array(data)[random_idx], np.array(file_paths)[random_idx]))
end = time.perf_counter()
for filename in glob.glob("./static/temp/*"):
    os.remove(filename)
with open("log.txt", "a") as log:
    log.write(str(end - start) + "\n")
print(f"{end - start}s")

