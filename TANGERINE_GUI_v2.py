import os
os.environ["OS_ACTIVITY_MODE"] = "disable"
os.environ["TK_SILENCE_DEPRECATION"] = "1" # it screams otherwise sorry :(

from tkinter import * 
from tkinter import filedialog 
from tkinter import ttk, messagebox
import sv_ttk
import subprocess 

import torch
import models_vit
import timm
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import pydicom
from skimage.transform import resize

######

# # Widgets: GUI Elements, like buttons, textboxes, images, or labels
# # Windows: Serve as containers to hold or contain these widgets 

def load_TANGERINE(inspect_ = False):
    file_path = filedialog.askopenfilename()
    CHECKPOINT = file_path
    # CHECKPOINT = "tangerine.pth"
    global model 
    model = models_vit.vit_large_patch16_yo(
        num_classes=1,
        drop_path_rate=0,
        global_pool=True,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu"
    )

    # Some checkpoints are wrapped in {"model": ...}
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    msg = model.load_state_dict(checkpoint, strict=False)

    # INSPECTION 

    if inspect_: 
        print("Checkpoint loaded:")
        print(msg)
        print("\nMODEL:")
        print(model)
        print("\nPARAMETERS:")
        for name, param in model.named_parameters():
            print(
                name,
                param.shape,
                "mean =", param.mean().item(),
                "std =", param.std().item()
            )

        print("\nTOTAL PARAMETERS:")
        print(sum(p.numel() for p in model.parameters()))
    model.eval()
    if inspect_: 
        print("Model set to evaluation mode")
    print("TANGERINE model successfully loaded")

def open_nii_gz_files():
    file_paths = filedialog.askopenfilenames()
    NUM_FILES = len(file_paths)
    for i in range(NUM_FILES):
        file_path = file_paths[i]
        img = sitk.ReadImage(file_path)
        arr = sitk.GetArrayFromImage(img)
        
        ct_scan = torch.tensor(arr, dtype=torch.float32)

        print()
        print(f"Loaded {file_path} successfully, has shape {ct_scan.shape}, min {ct_scan.min()}, max {ct_scan.max()}")

        ct_scan = ct_scan.unsqueeze(0).unsqueeze(0)

        with torch.inference_mode():
            emb = model.forward_features(ct_scan)
        
        input_path = Path(file_path)
        output_path = Path(output_dir) / (input_path.name.removesuffix(".nii.gz") + ".npy")
        np.save(output_path, emb.detach().cpu().numpy())

        print(f"TANGERINE successfully run on {file_path}, embedding vector saved to {output_path}")
        print(f"({i + 1}/{NUM_FILES} complete...)")

    print()
    print(f"Finished running TANGERINE on batch of {NUM_FILES} files")
        
def select_output_directory():
    global output_dir 
    output_dir = filedialog.askdirectory(
        title="Select a folder"
    )
    print(f"Output directory {output_dir} selected")

def preprocess_dicom_folder(base_path):
    # Find all DICOM slices in this series.
    slice_paths = list(Path(base_path).glob("*.dcm"))

    # Sort by physical position instead of filename.
    slice_paths.sort(
        key=lambda path: float(
            pydicom.dcmread(
                path, stop_before_pixels=True
            ).ImagePositionPatient[2]
        )
    )

    num_slices = len(slice_paths)
    arr = np.zeros((num_slices, 512, 512), dtype=np.float32)

    for slice_idx, slice_path in enumerate(slice_paths):
        ds = pydicom.dcmread(slice_path)

        arr[slice_idx] = (
            ds.pixel_array.astype(np.float32)
            * float(ds.RescaleSlope)
            + float(ds.RescaleIntercept)
        )

    # Resize to the model's expected input size.
    arr = resize(arr, (256, 256, 256), anti_aliasing=True)

    # Clip to the Hounsfield-unit window and normalize to [0, 1].
    arr[arr > 800] = 800
    arr[arr < -1200] = -1200
    arr = (arr - (-1200)) / (800 - (-1200))

    ct_scan = torch.tensor(arr, dtype=torch.float32)
    ct_scan = ct_scan.unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        emb = model.forward_features(ct_scan)

    return emb

def open_dcm_directory():
    dcm_dir = filedialog.askdirectory(
        title="Select a folder"
    )
    dcm_dir = Path(dcm_dir)

    # Find all directories containing DICOM files.
    directories = [
        directory for directory in dcm_dir.rglob("*")
        if directory.is_dir()
        and len(list(directory.glob("*.dcm"))) >= 10
    ]

    num_directories = len(directories)

    for i, directory in enumerate(directories):
        out_dir = output_dir / directory.relative_to(dcm_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        emb = preprocess_dicom_folder(directory)

        output_path = out_dir / "embedding.npy"
        np.save(output_path, emb.detach().cpu().numpy())

        print(f"TANGERINE successfully run on {directory}")
        print(f"Embedding vector saved to {output_path}")
        print(f"({i + 1}/{num_directories} complete...)")
        print()

    print()
    print(f"Finished running TANGERINE on {num_directories} DICOM series")

window = Tk() # Creates a window 

sv_ttk.set_theme("light")

container = Frame(window)
container.pack(fill="both", expand=True)

canvas = Canvas(container)
canvas.pack(side="left", fill="both", expand=True)

frame = Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

# info frame
info_frame = Frame(frame)
info_frame.pack(fill="x", padx=10, pady=10)

Label(
    info_frame,
    text="TANGERINE Inference Tool for 08/25/26 – Shaun Ng",
    font=("Arial", 16, "bold"),
    anchor="w"
).pack(anchor="w")

Label(
    info_frame,
    text="Download TANGERINE onto your local device.\n"
         "Model parameters are hosted at:\n"
         "https://zenodo.org/records/18835750",
    justify="left",
    anchor="w"
).pack(anchor="w", pady=(5, 0))

# main grid

grid_frame = Frame(frame)
grid_frame.pack(fill="x", padx=10)

Label(grid_frame, text="Step 0:").grid(
    row=0, column=0, sticky="w"
)

Label(grid_frame, text="Step 1:").grid(
    row=1, column=0, sticky="w"
)

Label(grid_frame, text="Step 2:").grid(
    row=2, column=0, sticky="w"
)

Label(grid_frame, text="OR").grid(
    row=2, column=2, sticky="w"
)

model_upload_button = ttk.Button(
    grid_frame,
    text="Load TANGERINE model from .pth file",
    command=load_TANGERINE
)
model_upload_button.grid(row=0, column=1, sticky="w")


folder_select_button = ttk.Button(
    grid_frame,
    text="Select output directory",
    command=select_output_directory
)
folder_select_button.grid(row=1, column=1, sticky="w")


file_upload_button = ttk.Button(
    grid_frame,
    text="Load .nii.gz Files and run TANGERINE",
    command=open_nii_gz_files
)
file_upload_button.grid(row=2, column=1, sticky="w")
    
alt_button = ttk.Button(
    grid_frame,
    text="Load .dcm series and run TANGERINE",
    command=open_dcm_directory
)
alt_button.grid(row=2, column=3, sticky="w")



# doc frame
doc_frame = Frame(frame)
doc_frame.pack(fill="x", padx=10, pady=10)

Label(
    doc_frame,
    text="Additional Information",
    font=("Arial", 16, "bold"),
    anchor="w"
).pack(anchor="w")

Label(
    doc_frame,
    text='''What will the program do?
    • For nii.gz files, it will run TANGERINE on the provided preprocessed CT volumes and save the resulting embedding vectors as .npy files.
    • For DICOM directories, it will recursively search through the selected directory for DICOM series. The program will skip directories containing fewer than 10 DICOM files. 
      It will then sort DICOM slices by their physical position and preprocess each series (min-max scaling between -1200 and 800 HU) and run TANGERINE on these preprocessed CT volumes. 
      It will recreate the entire original directory structure in the selected output folder, but saving each resulting TANGERINE embedding as a .npy file in place of the .dcm files. 
    
Please affirm that
    • You have cloned the GitHub repository https://github.com/niccolo246/3D-MAE-MedImaging 
    • You have placed this file (TANGERINE_GUI.py) in that repository
    • If you are using .nii.gz files, that they contain CT volumes that have been clipped to the range [-1200, 800] HU, min-max scaled to [0, 1], and resized to 256 × 256 × 256. A sample file can be found at 
    
Here is some sample console output: 
    (aug25_env) (base) shaun@mac 3D-MAE-MedImaging % python3 TANGERINE_GUI.py
    2026-08-25 11:58:26.049 python3[66068:8547167] +[IMKClient subclass]: chose IMKClient_Legacy
    2026-08-25 11:58:26.049 python3[66068:8547167] +[IMKInputSession subclass]: chose IMKInputSession_Legacy

    TANGERINE model successfully loaded

    Output directory /Users/shaun/Documents/aug_25_test selected

    Loaded /Users/shaun/Documents/dicom_loading/100331_0.nii.gz successfully, has shape torch.Size([256, 256, 256]), min 0.08799999952316284, max 1.0
    TANGERINE successfully run on /Users/shaun/Documents/dicom_loading/100331_0.nii.gz, embedding vector saved to /Users/shaun/Documents/aug_25_test/100331_0.npy
    
    Loaded /Users/shaun/Documents/dicom_loading/100331_1.nii.gz successfully, has shape torch.Size([256, 256, 256]), min 0.08799999952316284, max 1.0
    TANGERINE successfully run on /Users/shaun/Documents/dicom_loading/100331_1.nii.gz, embedding vector saved to /Users/shaun/Documents/aug_25_test/100331_1.npy
    ''',
    justify="left",
    anchor="w"
).pack(anchor="w", pady=(5, 0))

window.title("TANGERINE Inference GUI v2")
window.mainloop()
