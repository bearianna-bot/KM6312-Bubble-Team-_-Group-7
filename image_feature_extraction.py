import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

IMG_DIR = "/Users/yuanlue/Desktop/python/pythonProject/cnn/processed_images"
CSV_PATH = "/Users/yuanlue/Desktop/python/pythonProject/cnn/shopee_dataset_final.csv"
SAVE_PATH = "eff_embeddings.csv"

df = pd.read_csv(CSV_PATH)
df["image_path"] = df["image"].apply(lambda x: os.path.join(IMG_DIR, x))

# EfficientNet model
eff_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img

# --- Breakpoint Recovery Mechanism ---
if os.path.exists(SAVE_PATH):
    done_df = pd.read_csv(SAVE_PATH)
    done_ids = set(done_df["posting_id"])
else:
    done_df = pd.DataFrame()
    done_ids = set()

# Interim storage of final results
results = []

batch_imgs = []
batch_ids = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    pid = row["posting_id"]
    path = row["image_path"]

    # Already processed ? Skip
    if pid in done_ids:
        continue

    batch_imgs.append(load_image(path))
    batch_ids.append(pid)

    # Run the model every 32 frames
    if len(batch_imgs) == 32:
        arr = np.array(batch_imgs)
        emb = eff_model.predict(arr)

        for i, e in enumerate(emb):
            results.append([batch_ids[i]] + e.tolist())

        batch_imgs, batch_ids = [], []

        # Save every 500 entries
        if len(results) >= 500:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv("eff_embeddings_temp.csv", index=False)

# If there are any remaining batches
if len(batch_imgs) > 0:
    arr = np.array(batch_imgs)
    emb = eff_model.predict(arr)
    for i, e in enumerate(emb):
        results.append([batch_ids[i]] + e.tolist())

# Save the final file
colnames = ["posting_id"] + [f"eff_{i}" for i in range(emb.shape[1])]
final_df = pd.DataFrame(results, columns=colnames)
final_df.to_csv(SAVE_PATH, index=False)

print("EfficientNet embeddings saved!")

from PIL import Image
import imagehash

PHASH_SAVE = "dataset_with_phash.csv"

# Breakpoint recovery
if os.path.exists(PHASH_SAVE):
    df_phash = pd.read_csv(PHASH_SAVE)
    done_hash_ids = set(df_phash["posting_id"])
else:
    df_phash = pd.DataFrame(columns=["posting_id", "phash"])
    done_hash_ids = set()

rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    pid = row["posting_id"]
    path = row["image_path"]

    if pid in done_hash_ids:
        continue

    try:
        img = Image.open(path)
        h = str(imagehash.phash(img))
    except:
        h = None

    rows.append([pid, h])

    # Save every 500 lines
    if len(rows) >= 500:
        temp = pd.DataFrame(rows, columns=["posting_id", "phash"])
        df_phash = pd.concat([df_phash, temp], ignore_index=True)
        df_phash.to_csv(PHASH_SAVE, index=False)
        rows = []

# Save the remainder
if len(rows) > 0:
    temp = pd.DataFrame(rows, columns=["posting_id", "phash"])
    df_phash = pd.concat([df_phash, temp], ignore_index=True)
    df_phash.to_csv(PHASH_SAVE, index=False)

print("pHash saved!")
