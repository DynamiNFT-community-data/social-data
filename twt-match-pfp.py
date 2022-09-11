#%%
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from itertools import chain
import pandas as pd

print("import done")
#%%
base_model = VGG16(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)


def extract(img):
    img = img.resize((224, 224))  # Resize the image
    img = img.convert("RGB")  # Convert the image color space
    x = image.img_to_array(img)  # Reformat the image
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)[0]  # Extract Features
    return feature / np.linalg.norm(feature)


#%%

folder = "nft-images"
nft_images = []
for path in os.listdir(folder):
    nft_images.append(os.path.join(folder, path))

folder = "pfp-images"
pfp_images = []
for path in os.listdir(folder):
    pfp_images.append(os.path.join(folder, path))


#%% Iterate through nft images and extract Features
all_features = np.zeros(shape=(len(nft_images), 4096))

for i in range(len(nft_images)):
    feature = extract(img=Image.open(nft_images[i]))
    all_features[i] = np.array(feature)

#%%
# Match image
best_matches = pd.DataFrame()
for img in pfp_images:
    image_to_match = img
    query = extract(img=Image.open(image_to_match))  # Extract its features
    dists = np.linalg.norm(all_features - query, axis=1)  # Calculate the similarity (distance) between images
    best_matches = pd.concat(
        [
            best_matches,
            pd.DataFrame({"pfp_img": image_to_match, "nft_img": nft_images, "dist": dists})
            .sort_values("dist")
            .iloc[0:1],
        ]
    )

best_matches
best_matches.plot.hist()

# Exact match threshold < 0.5
best_matches_filtered = best_matches.query("dist < 0.5")
pfp_matches = len(best_matches_filtered)


# %% Plot and check results

# def plot_similar_images2(row):
#     img_list = [row["pfp_img"], row["nft_img"]]
#     # Visualize the result
#     axes = []
#     fig = plt.figure(figsize=(20, 15))
#     for a in range(len(img_list)):
#         axes.append(fig.add_subplot(1, 2, a + 1))
#         plt.axis("off")
#         plt.imshow(Image.open(img_list[a]))
#     fig.tight_layout()
#     fig.suptitle(row["dist"], fontsize=22)
#     plt.show(fig)

# for i, row in best_matches.query('dist < 1').iterrows():
#     plot_similar_images2(row)
