import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd

def nii_loader(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    return data

def read_table(path):
    """
    Read table from CSV or Excel file.
    Returns DataFrame to preserve column names for MTL task.
    """
    if path.endswith('.csv'):
        return pd.read_csv(path)
    else:
        return pd.read_excel(path)

def white0(image, threshold=0):
    "standardize voxels with value > threshold"
    image = image.astype(np.float32)
    mask = (image > threshold).astype(int)

    image_h = image * mask
    image_l = image * (1 - mask)

    mean = np.sum(image_h) / np.sum(mask)
    std = np.sqrt(np.sum(np.abs(image_h - mean)**2 * mask) / np.sum(mask))

    if std > 0:
        ret = (image_h - mean) / std + image_l
    else:
        ret = image * 0.
    return ret

class Integer_Multiple_Batch_Size(torch.utils.data.Dataset):
    
    def __init__(self, folder_dataset, batch_size=8):
        self.folder_dataset = folder_dataset
        self.batch_size = batch_size

        source_dataset_len = len(self.folder_dataset)
        # Fix: avoid doubling data when perfectly divisible
        num_need_to_complement = (self.batch_size - (source_dataset_len % self.batch_size)) % self.batch_size
        
        idx_list = np.arange(0, source_dataset_len)
        if num_need_to_complement > 0:
            complement_idx = idx_list[-num_need_to_complement:]
            self.complemented_idx = np.concatenate([idx_list, complement_idx], axis=0)
        else:
            self.complemented_idx = idx_list
        self.complemented_size = self.complemented_idx.shape[0]
        print(f"Dataset size: {source_dataset_len} -> {self.complemented_size} (padded: {num_need_to_complement})")
        
    def __len__(self):
        return self.complemented_size

    def __getitem__(self, index):
        return self.folder_dataset[self.complemented_idx[index]]
    
class IMG_Folder(torch.utils.data.Dataset):
    def __init__(self, excel_path, data_path, loader=nii_loader, transforms=None, use_diagnosis=False):
        """
        Dataset for brain MRI images with age and optional diagnosis labels.
        
        Args:
            excel_path: Path to CSV/Excel file with metadata
            data_path: Path to directory containing .nii.gz files
            loader: Function to load images
            transforms: Optional image transformations
            use_diagnosis: If True, return diagnosis label for MTL (default: False for backward compatibility)
        """
        self.root = data_path
        self.sub_fns = sorted(os.listdir(self.root))
        self.table_refer = read_table(excel_path)  # Now returns DataFrame
        self.loader = loader
        self.transform = transforms
        self.use_diagnosis = use_diagnosis
        
        # Diagnosis label mapping for ADNI dataset
        self.dx_map = {
            'ADNI_Norm': 0,
            'ADNI_MCI': 1,
            'ADNI_AD': 2
        }
        
        # Build hash map for O(1) lookup (avoid O(N²) iteration)
        print("Building metadata index...")
        self.row_map = {}
        for _, row in self.table_refer.iterrows():
            sid = str(row.iloc[0])  # ID column (first column)
            # Normalize: remove extensions
            sid_base = sid.replace('.nii.gz', '').replace('.nii', '')
            self.row_map[sid_base] = row
        print(f"Indexed {len(self.row_map)} metadata entries")

    def __len__(self):
        return len(self.sub_fns)

    def __getitem__(self, index):
        sub_fn = self.sub_fns[index]
        
        # Normalize filename for matching
        # Handle: ADNI_MCI_sub-0784_nonlin_brain.nii.gz -> ADNI_MCI_sub-0784
        sub_fn_base = sub_fn.replace('.nii.gz', '').replace('.nii', '')
        if sub_fn_base.endswith('_nonlin_brain'):
            sub_fn_base = sub_fn_base.replace('_nonlin_brain', '')
        
        # O(1) lookup instead of O(N) iteration
        row = self.row_map.get(sub_fn_base, None)
        if row is None:
            raise ValueError(f"No metadata found for file: {sub_fn} (normalized: {sub_fn_base})")
        
        # Extract labels from DataFrame
        sid = str(row.iloc[0])
        age = int(row['age']) if 'age' in row else int(row.iloc[1])
        
        # Extract and normalize gender to 0/1
        sx = row['sex'] if 'sex' in row else row.iloc[2]
        if isinstance(sx, str):
            gender = 1 if sx.upper().startswith('M') or sx.upper() == 'MALE' else 0
        else:
            gender = int(sx)
        
        # Extract diagnosis label if MTL mode
        dx_label = None
        if self.use_diagnosis:
            # Support both OASIS format (dx column) and ADNI format (source_dataset column)
            if 'dx' in row:
                # OASIS format: dx column contains 0, 1, or 2 directly
                dx_label = int(row['dx'])
                if dx_label not in [0, 1, 2]:
                    raise ValueError(f"Invalid dx value: {dx_label}. Expected 0, 1, or 2.")
            elif 'source_dataset' in row:
                # ADNI format: source_dataset column contains string like 'ADNI_Norm', 'ADNI_MCI', 'ADNI_AD'
                dx_str = row['source_dataset']
                dx_label = self.dx_map.get(dx_str, -1)
                if dx_label == -1:
                    raise ValueError(f"Unknown diagnosis type: {dx_str}")
            else:
                # Default to 0 (Norm) if no diagnosis info
                print(f"Warning: No diagnosis column found for {sid}, defaulting to 0 (Norm)")
                dx_label = 0
        
        # Load and preprocess image
        sub_path = os.path.join(self.root, sub_fn)
        img = self.loader(sub_path)
        img = white0(img)
        if self.transform is not None:
            img = self.transform(img)
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        
        # Return with or without diagnosis based on mode
        if self.use_diagnosis:
            return (img, sid, age, gender, dx_label)
        else:
            return (img, sid, age, gender)
