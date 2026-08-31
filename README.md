# GUI-Specific Information
- The GUI is at TANGERINE_GUI_v2.py, and can be simply run with python TANGERINE_GUI_v2.py.
- A video tutorial on GUI use can be found here https://drive.google.com/file/d/1crL96D4tzsuvU4zcfjJtpoXDIB4c68Xq/view?usp=sharing

## Program details
- For nii.gz files, it will run TANGERINE on the provided preprocessed CT volumes and save the resulting embedding vectors as .npy files. Please ensure that they contain CT volumes that have been clipped to the range [-1200, 800] HU, min-max scaled to [0, 1], and resized to 256 × 256 × 256.
- For DICOM directories, it will recursively search through the selected directory for DICOM series. The program will skip directories containing fewer than 10 DICOM files. It will then sort DICOM slices by their physical position and preprocess each series (min-max scaling between -1200 and 800 HU) and run TANGERINE on these preprocessed CT volumes. It will recreate the entire original directory structure in the selected output folder, but saving each resulting TANGERINE embedding as a .npy file in place of the .dcm files.

## To use the GUI, 
- Clone this repository
- Install packages in requirements.txt
- Furthermore, install SimpleITK (pip install SimpleITK), sv-ttk (pip install sv-ttk), and pydicom (pip install pydicom)
- Download model weights from https://zenodo.org/records/18835750.
- To test that it works, feel free to use the NON_PHI_EXAMPLE_256x256x256.nii.gz file as input, which can be found at https://drive.google.com/file/d/1H0ghash-lTVrEKfuu05wZ44qPJB1qZE9/view?usp=sharing 
    
Here is some sample console output: 
```bash
    (aug25_env) (base) shaun@mac 3D-MAE-MedImaging % python3 TANGERINE_GUI.py
    2026-08-25 11:58:26.049 python3[66068:8547167] +[IMKClient subclass]: chose IMKClient_Legacy
    2026-08-25 11:58:26.049 python3[66068:8547167] +[IMKInputSession subclass]: chose IMKInputSession_Legacy

    TANGERINE model successfully loaded

    Output directory /Users/shaun/Documents/aug_25_test selected

    Loaded /Users/shaun/Documents/dicom_loading/100331_0.nii.gz successfully, has shape torch.Size([256, 256, 256]), min 0.08799999952316284, max 1.0
    TANGERINE successfully run on /Users/shaun/Documents/dicom_loading/100331_0.nii.gz, embedding vector saved to /Users/shaun/Documents/aug_25_test/100331_0.npy
    
    Loaded /Users/shaun/Documents/dicom_loading/100331_1.nii.gz successfully, has shape torch.Size([256, 256, 256]), min 0.08799999952316284, max 1.0
    TANGERINE successfully run on /Users/shaun/Documents/dicom_loading/100331_1.nii.gz, embedding vector saved to /Users/shaun/Documents/aug_25_test/100331_1.npy
```

# 3D Masked Autoencoders for Volumetric Medical Imaging Data

This repository provides a **3D extension of the Masked Autoencoder (MAE) framework**, designed for self-supervised pretraining on **volumetric medical imaging data** (e.g., CT scans). Our method extends MAE to 3D by incorporating **custom volumetric patch embedding** and **Transformer-based feature learning**, enabling efficient representation learning for medical imaging applications.

As part of this framework, we introduce **TANGERINE** (*Thoracic Autoencoder Network Generating Embeddings for Radiological Interpretation for Numerous End-tasks*), a **ViT-Large model pretrained on 98,000 chest CT volumes** for lung screening. TANGERINE demonstrates the effectiveness of this framework and is described in detail in our paper (citation below). We provide the **pretrained encoder weights**, which can be used to initialize fine-tuning for a variety of downstream tasks.


## Key Features

- **3D Extension of MAE**  
  - Adapts the MAE framework for 3D volumetric data.  
  - Employs a specialized **3D patch embedding module** for improved spatial feature extraction.  

- **Computationally Efficient Pretraining**  
  - Utilizes **high masking ratios** to reduce training memory consumption.  
  - Enables **scalable training on large-scale 3D datasets**.  

- **Pretrained ViT Large Model**  
  - TANGERINE, our pretrained ViT-Large model, was trained on 98,000 chest CT volumes for thoracic screening, as detailed in our paper.  
  - This pretrained model can be **readily finetuned** for a wide range of **downstream tasks**.

- **Flexible Finetuning and Inference**  
  - Includes scripts for **supervised finetuning** on downstream classification and segmentation tasks.  
  - Supports **efficient inference** using learned volumetric representations.  

---

## Installation

To use this repository, install the necessary dependencies and set up the environment.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/niccolo246/3D-MAE-MedImaging
   cd 3D-MAE-MedImaging
   ```

2. **Install dependencies:**  
   Ensure you have **Python 3.7+** and a compatible **PyTorch version** installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

For additional dependency requirements, refer to `requirements.txt`.

---

## Pretraining

Pretraining is performed using **distributed training** across multiple GPUs for efficiency. The following script launches pretraining with `torchrun` (4 GPUs):

```bash
# Check GPU availability
nvidia-smi

# Optional: Disable NCCL P2P if needed
export NCCL_P2P_DISABLE=1

# Assign a free port for distributed training
find_free_port() {
    while true; do
        PORT=$(shuf -i 20000-65000 -n 1)
        ss -lpn | grep ":$PORT " > /dev/null
        if [ $? -ne 0 ]; then
            echo $PORT
            return 0
        fi
    done
}
MASTER_PORT=$(find_free_port)
echo "Using port: $MASTER_PORT"

# Run pretraining
torchrun --standalone --nproc_per_node=4 --nnodes=1 --master_port=$MASTER_PORT path/to/main_pretrain.py
```

Modify `path/to/main_pretrain.py` based on your dataset and training parameters.

---

## Finetuning

To adapt the pre-trained model for downstream tasks, finetuning is performed as follows:

```bash
torchrun --standalone --nproc_per_node=4 --nnodes=1 --master_port=$MASTER_PORT path/to/main_finetune.py \
    --finetune /path/to/pretrained_checkpoint.pth \
    --additional_finetune_args
```

Modify `--additional_finetune_args` based on task-specific requirements.

---

## Inference and Prediction

For model inference on new volumetric datasets:

```bash
python3 main_predict.py \
    --input_csv /path/to/input.csv \
    --output_csv /path/to/output_predictions.csv \
    --finetune /path/to/pretrained_checkpoint.pth
```

This script loads the finetuned model and generates predictions.

---

## Custom Dataset Handling

**Important:** Users must create a **custom dataset class** (`Custom3DDataset`) depending on their **data structure**.  
- The dataset class for **pretraining** should be defined in:  
  **`datasets_three_d.py`**  
- The dataset class for **finetuning** should be defined in:  
  **`datasets_three_d_fine.py`**

Each user should modify `Custom3DDataset` to correctly **load, preprocess, and format their data** based on their dataset structure.

---

## Technical Details

### **3D Data Handling**  
- The dataset loader utilizes **SimpleITK** for reading **NIfTI** medical images.  
- Ensures correct axis ordering **([Depth, Height, Width])** for volumetric representation.  
- Includes **optional resampling functions** to standardize input dimensions.

#### **Resampling to 256x256x256**  
An **example resampling function** is provided in `datasets_three_d_fine.py` to **resize input volumes to 256×256×256 resolution**.  
Modify this function as needed to fit specific dataset characteristics.

### **Training and Sampling Strategy**  
- For **single-GPU training**, `DataLoader` is configured with `shuffle=True`.  
- In **distributed training**, `DistributedSampler` is recommended to partition data across GPUs.  
- *(Note: The `DistributedSampler` is included but commented out for single-GPU training.)*

### **Model Architecture**
- Utilizes **Transformer-based MAE architecture** for volumetric feature extraction.  
- Implements **custom 3D patch embedding** to handle medical imaging modalities.  
- Incorporates **high masking ratios** to enhance self-supervised learning efficiency.

---

## Pretrained Model Weights

We provide **TANGERINE pretrained ViT-Large weights** for both the **encoder**, available at the following link:

[Zenodo](https://zenodo.org/records/18835750)


These weights can be directly used for **finetuning** across a wide range of downstream tasks, including **classification**, **segmentation**, and **detection**.

### Example usage

```bash
torchrun --standalone --nproc_per_node=4 --nnodes=1 --master_port=$MASTER_PORT path/to/main_finetune.py \
    --finetune path/to/pretrained_checkpoint.pth \
    --additional_finetune_args
```

---

## Citation & License

This project is licensed under the **CC-BY-NC 4.0** license.  

If you use this repository in academic work, please cite:

McConnell, N., Vasudev, P., Yamada, D. et al. A computationally frugal, open-source chest CT foundation model for thoracic disease detection in lung cancer screening programmes. *Commun Med* **6**, 83 (2026). https://doi.org/10.1038/s43856-025-01328-1

---

## Contact

For questions or contributions, please contact **niccolo.mcconnell.17@ucl.ac.uk** or open an issue on GitHub.


