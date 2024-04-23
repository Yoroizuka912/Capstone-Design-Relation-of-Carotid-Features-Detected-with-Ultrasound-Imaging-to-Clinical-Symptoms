import cv2
import numpy as np
from sklearn.cluster import KMeans
from fusion import Fusion
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode
import xgboost as xgb
import pickle
import uuid
import os
import glob


cnn_model_path = "./model/cnn_model.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn_model = Fusion().to(device)
cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

xgb_model_path = "./model/xgb.json"
xgb_model = xgb.Booster()
xgb_model.load_model(xgb_model_path)

lr_model_path = "./model/lr.pkl"
lr_model = pickle.load(open(lr_model_path, 'rb'))


def process_data(history, gender):
    fields = ['high_bp', 'diabetes', 'high_bl', 'smoke', 'alcohol', 'brain', 'heart']
    processed = [1 if gender == 'male' else 0]
    for field in fields:
        if field in history:
            processed.append(1)
        else:
            processed.append(0)
    return processed


def load_image(path, as_int=True, rgb=False):
    if as_int:
        if rgb:
            image = np.array(cv2.imread(path))
        else:
            image = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    else:
        if rgb:
            image = np.array(cv2.imread(path), dtype=np.float64)
        else:
            image = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE), dtype=np.float64)
    return image


# float image
def save_center(image, image_uuid):
    new_size = 320
    new_max = 255
    new_min = 0
    
    h, w = image.shape
    image_centered = image[int(h / 2) - new_size: int(h / 2) + new_size, int(w / 2) - new_size: int(w / 2) + new_size]

    # enhance: linear-normalization
    min, max = np.min(image_centered), np.max(image_centered)
    image_centered = (image_centered - min) * (new_max - new_min) / (max - min) + new_min
    
    centered_image_path = f"./static/temp/center-{image_uuid}.jpg"
    cv2.imwrite(centered_image_path, image_centered)


# int image, int mask
def cluster(image, mask):
    min_area = 10
    n_clusters = 4
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    x, y, w, h = cv2.boundingRect(contours[0])
    
    roi_image = image[y:y+h, x:x+w]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(roi_image.reshape(-1, 1))

    ordered_labels = np.argsort(kmeans.cluster_centers_.sum(axis=1))

    # Create an empty array for new labels
    new_labels = np.zeros_like(kmeans.labels_)

    # Assign new labels
    for i, cluster in enumerate(ordered_labels):
        new_labels[kmeans.labels_ == cluster] = i

    # proportion of pixels in each cluster
    counts = np.bincount(new_labels)
    prop = counts / np.sum(counts)
    return prop


# rgb image
def get_cnn_score(image, prop):
    cnn_model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        prop = torch.tensor(prop, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = cnn_model(image, prop)
        cnn_score = (torch.exp(outputs) / torch.sum(torch.exp(outputs)))[0][1].item()
    return cnn_score


def get_xgb_score(data):
    return xgb_model.predict(xgb.DMatrix([data]))[:,1][0]


def get_stacking_score(cnn_score, xgb_score):
    score = lr_model.predict_proba([[xgb_score, cnn_score]])
    return score[:,1][0]


def final_predict(image_int, image_float, image_uuid, mask_int, data):
    image_center_path = f"./static/temp/center-{image_uuid}.jpg"
    
    save_center(image_float, image_uuid)
    prop = cluster(image_int, mask_int)
    
    image_rgb = transform(read_image(image_center_path, mode=ImageReadMode.RGB).float())
    
    cnn_score = get_cnn_score(image_rgb, prop)
    xgb_score = get_xgb_score(data)
    final_score = get_stacking_score(cnn_score, xgb_score)
    
    return cnn_score, xgb_score, final_score

# int rgb image, int grayscale mask
def generate_highlight(image, mask, image_uuid):
    mask = mask.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    image[..., 2] += mask * 100 
    image = np.clip(image, 0, 255).astype(np.uint8)
    img_highlight_path = f"./static/temp/result-{image_uuid}.jpg"
    cv2.imwrite(img_highlight_path, image)
    return img_highlight_path


def delete_files_with_uuid(target_folder, target_uuid):
    search_pattern = os.path.join(target_folder, f'*{target_uuid}*')
    for filename in glob.glob(search_pattern):
        if 'result' not in filename:
            os.remove(filename)