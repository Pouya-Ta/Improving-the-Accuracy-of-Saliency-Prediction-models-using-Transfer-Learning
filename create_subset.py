import os, json, random, shutil
import time

# ─── 1️⃣ CONFIGURE THESE PATHS & SIZE ────────────────────────────────
SRC_IMG_DIR    = r"C:\Users\Notebook\Downloads\SALICON\train\train"
SRC_ANN_FILE   = r"C:\Users\Notebook\Downloads\SALICON\fixations_train2014.json"
OUT_SUBSET_DIR = r"C:\Users\Notebook\Downloads\SALICON\salicon_subset"
NUM_SAMPLES    = 1000  
# ─────────────────────────────────────────────────────────────────────

print("➤ Starting subset creation")
print(f"  • Loading annotations from:\n    {SRC_ANN_FILE}")
t0 = time.time()

# ─── 2️⃣ LOAD THE ANNOTATIONS ────────────────────────────────────────
with open(SRC_ANN_FILE, 'r') as f:
    data = json.load(f)

print(f"  • Loaded annotations in {time.time()-t0:.1f}s")
all_images      = data['images']
all_annotations = data['annotations']
categories      = data.get('categories', [])

# ─── 3️⃣ SAMPLE & FILTER ─────────────────────────────────────────────
print(f"  • Sampling {NUM_SAMPLES} images…")
selected = random.sample(all_images, NUM_SAMPLES)
sel_ids  = {img['id'] for img in selected}
subset_anns = [ann for ann in all_annotations if ann['image_id'] in sel_ids]

# ─── 4️⃣ PREP OUTPUT FOLDERS ─────────────────────────────────────────
os.makedirs(os.path.join(OUT_SUBSET_DIR, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(OUT_SUBSET_DIR, 'annotations'),    exist_ok=True)

# ─── 5️⃣ COPY IMAGES & WRITE JSON ────────────────────────────────────
print("  • Copying images and writing subset JSON…")
for img in selected:
    src = os.path.join(SRC_IMG_DIR, img['file_name'])
    dst = os.path.join(OUT_SUBSET_DIR, 'images', 'train', img['file_name'])
    shutil.copy(src, dst)

subset_json = {
    'images':      selected,
    'annotations': subset_anns,
    'categories':  categories
}
with open(os.path.join(OUT_SUBSET_DIR, 'annotations', 'train_subset.json'), 'w') as f:
    json.dump(subset_json, f, indent=2)

print(f"✅ Done! Created {NUM_SAMPLES} samples in {OUT_SUBSET_DIR}")
