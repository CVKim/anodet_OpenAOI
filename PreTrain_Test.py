import os
import anodet
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


DATASET_PATH = os.path.realpath("C:\\dataset\\")
MODEL_DATA_PATH = os.path.realpath("C:\\distributions")


dataset = anodet.AnodetDataset(os.path.join(DATASET_PATH, "capsule/train/good"))

# Dataset = sample, label save
# DataLoader = Dataset을 Sample에 쉽게 접근할 수 있도록 순회 가능한 객체로 감쌈
dataloader = DataLoader(dataset, batch_size=32)
print("Number of images in dataset:", len(dataloader.dataset)) # train image 2

patch_core = anodet.PatchCore()
patch_core.fit(dataloader)

torch.save(patch_core.embedding_coreset, os.path.join(MODEL_DATA_PATH, "capsule_embedding_coreset.pt"))

paths = [
    os.path.join(DATASET_PATH, "capsule/test/crack/017.png"),
    os.path.join(DATASET_PATH, "capsule/test/crack/001.png"),
    os.path.join(DATASET_PATH, "capsule/test/poke/000.png"),
    os.path.join(DATASET_PATH, "capsule/test/good/000.png"),
    os.path.join(DATASET_PATH, "capsule/test/good/001.png"),
]

images = []
for path in paths:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    
batch = anodet.to_batch(images, anodet.standard_image_transform, torch.device('cpu'))

embedding_coreset = torch.load(os.path.join(MODEL_DATA_PATH, 'capsule_embedding_coreset.pt'))
patch_core = anodet.PatchCore('resnet18', embedding_coreset=embedding_coreset)
image_scores, score_maps = patch_core.predict(batch)

THRESH = 2.4
score_map_classifications = anodet.classification(score_maps, THRESH)
image_classifications = anodet.classification(image_scores, THRESH)
print("Image scores:", image_scores)
print("Image classifications:", image_classifications)

test_images = np.array(images).copy()

boundary_images = anodet.visualization.framed_boundary_images(test_images, score_map_classifications, image_classifications, padding=40)
heatmap_images = anodet.visualization.heatmap_images(test_images, score_maps, alpha=0.5)
highlighted_images = anodet.visualization.highlighted_images(images, score_map_classifications, color=(128, 0, 128))

for idx in range(len(images)):
    fig, axs = plt.subplots(1, 4, figsize=(12, 6))
    fig.suptitle('Image: ' + str(idx), y=0.75, fontsize=14)
    axs[0].imshow(images[idx])
    axs[1].imshow(boundary_images[idx])
    axs[2].imshow(heatmap_images[idx])
    axs[3].imshow(highlighted_images[idx])
    plt.show()


heatmap_images = anodet.visualization.heatmap_images(test_images, score_maps, alpha=0.5)
tot_img = anodet.visualization.merge_images(heatmap_images, margin=40)
fig, axs = plt.subplots(1, 1, figsize=(10, 6))
plt.imshow(tot_img)
plt.show()