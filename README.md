# 🌊 Flood Rapid Mapping using Satellite Data

This project focuses on **rapid segmentation of water bodies** using harmonized Sentinel-2 and Landsat satellite imagery. The solution is essential for applications such as **flood monitoring, water resource management, and environmental conservation**. By leveraging deep learning techniques, the model accurately segments water areas in near real-time and is deployed via a Flask web interface.

---

## 🛰 Dataset Description

The dataset includes **multispectral and contextual satellite data** layers:

### Spectral Bands:

* Coastal Aerosol
* Blue
* Green
* Red
* Near Infrared (NIR)
* Shortwave Infrared 1 (SWIR1)
* Shortwave Infrared 2 (SWIR2)

### Contextual Data:

* QA Band
* MERIT DEM
* Copernicus DEM
* ESA World Cover Map
* Water Occurrence Probability

**Label:**

* Binary Water Mask (stored separately)

**Image Shape:** 128 x 128,
**Channels:** 12,
**Ground Sampling Distance:** 30 meters,
**Patch Size:** 512 x 512 pixels


![bands](https://github.com/user-attachments/assets/ca559762-4bc5-4284-ae2a-ae3eb0de8d9a)

---

## ⚙️ Preprocessing

Preprocessing involved:

* **Maintaining original spatial resolution** to preserve geographic accuracy.
* **Normalization** across all 12 input channels to reduce sensor variability and improve model convergence.
* **Band visualization** was used for inspecting data quality and understanding spectral characteristics.
* Binary masks were matched with input image patches for supervised training.

---

### Dataset Splitting

The dataset consists of a total of **306 images**, which are split into the following subsets:

* **Training Set:** 80% (244 images)
* **Validation Set:** 10% (31 images)
* **Testing Set:** 10% (31 images)

---

## 🧠 Model Architecture & Training

### Stage 1: Custom U-Net from Scratch

* Architecture: Standard U-Net with encoder-decoder design
* Loss: Binary Cross-Entropy (BCE)
* Input Shape: (128, 128, 12)
* Evaluation on Test Set:

  * **IoU**: 0.72
  * **Dice Coefficient**: 0.84

---

### Stage 2: U-Net with Pretrained ResNet-18 Encoder

* Enhanced the architecture by integrating a **pretrained ResNet-18** encoder.
* To match the expected 3-channel input shape of ResNet-18 (**128×128×3**), the **NIR, SWIR1, and Green** bands were selected.
* **Band Selection Justification**: These bands are commonly used in computing the **Normalized Difference Water Index (NDWI)**, which is effective for detecting water bodies, enhancing model sensitivity to flooded regions.
* Loss Function: **Dice Loss**
* Benefit: Transfer learning improved feature extraction and generalization from satellite imagery.
* Final Evaluation on test set:

  * **IoU**: 0.78
  * **Dice Coefficient**: **0.85**

---

## 🌐 Deployment

The final model was deployed using **Flask**, offering a user-friendly web interface for real-time flood segmentation.

### Key Features:

* Upload a satellite patch
* Model performs segmentation in real time
* Returns binary water mask and overlay for quick visual interpretation

![Screenshot 2025-05-28 014031](https://github.com/user-attachments/assets/44c09583-8d42-400d-b018-1ec441cfcea7)

![Uploadin![Screenshot 2025-05-28 014427](https://github.com/user-attachments/assets/7e334c2b-ee1f-4b4f-9d2b-0a95dd23b7f6)

![Screenshot 2025-05-28 014142](https://github.com/user-attachments/assets/ef9871aa-c7d5-4d7e-98f2-d319a337c9ba)
