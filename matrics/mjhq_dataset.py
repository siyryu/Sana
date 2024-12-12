import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


# Read the JSON file with prompts and categories


class MJHQ30KDataset(Dataset):
    def __init__(self, data_path, meta_path, transform=None):
        self.data_path = data_path
        with open(meta_path) as f:
            self.prompt_data = json.load(f)
        self.image_ids = list(self.prompt_data.keys())

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_data = self.prompt_data[img_id]

        # Get category and construct image path
        category = img_data["category"]
        img_path = os.path.join(self.data_path, category, f"{img_id}.jpg")

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {"image": image, "prompt": img_data["prompt"], "category": category, "image_id": img_id}


class MJHQ30KDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        prompt_data: dict,
        batch_size: int = 32,
        num_workers: int = 0,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.data_path = data_path
        self.prompt_data = prompt_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split

        # Define transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def setup(self, stage=None):
        # Create dataset
        self.full_dataset = MJHQ30KDataset(self.data_path, self.prompt_data, self.transform)

        # Calculate lengths for splits
        total_size = len(self.full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

    def full_dataloader(self):
        return DataLoader(
            self.full_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

if __name__ == "__main__":
    dataset_meta_path = "/home/siyuanyu/HF-cache/hub/datasets--playgroundai--MJHQ-30K/snapshots/15b0a659e066e763d0e9a6cd8f00e25f8af5e084/meta_data.json"
    dataset_path = (
        "/home/siyuanyu/HF-cache/hub/datasets--playgroundai--MJHQ-30K/snapshots/15b0a659e066e763d0e9a6cd8f00e25f8af5e084/"
    )
    # Create and setup the data module
    data_module = MJHQ30KDataModule(dataset_path, dataset_meta_path)
    data_module.setup()

    # Access the dataloaders
    full_loader = data_module.full_dataloader()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print(f"Full batches: {len(full_loader)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
