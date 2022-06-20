import json
import os
import h5py
import PIL.Image as Image
import torch
import torchvision
from torch.utils.data import Dataset
from overrides import overrides
from tqdm import tqdm

FRAME_PATH = "../data/frames/utterances_final"
json_path = "../data/sarcasm_data.json"

class Frames(Dataset):
    def __init__(self, transform_data) -> None:
        self.transform = transform_data

        with open(json_path) as file:
            videos_data_dict = json.load(file)

        for video_id in list(videos_data_dict): 
            video_folder_path = os.path.join(FRAME_PATH, video_id)
            if not os.path.exists(video_folder_path):
                raise FileNotFoundError(f"Directory {video_folder_path} not found, which was referenced in"
                                            f" {FRAME_PATH}")

        self.video_ids = list(videos_data_dict)
        self.frame_count_by_video_id = {video_id: len(os.listdir(os.path.join(FRAME_PATH, video_id)))
                                        for video_id in self.video_ids}

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames = None
        video_folder_path = os.path.join(FRAME_PATH, video_id)

        for i, frame_file_name in enumerate(os.listdir(video_folder_path)):
            frame = Image.open(os.path.join(video_folder_path, frame_file_name))
            if self.transform:
                frame = self.transform(frame)
            if frames is None:
                frames = torch.empty((self.frame_count_by_video_id[video_id], *frame.size()))
            frames[i] = frame

        return {"id": video_id, "frames": frames}
    
    def __len__(self):
        return len(self.video_ids)

    def features_file_path(model, layer) -> str:
        return f"../data/features/{model}_{layer}.hdf5"

def resnet_model():
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet.eval()
    for param in resnet.parameters():
        param.requires_grad = False
    return resnet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_visual_features():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = Frames(transform_data=transforms)
    resnet = resnet_model().to(DEVICE)

    class Identity(torch.nn.Module):
        @overrides
        def forward(self, input):
            return input

    resnet.fc = Identity()

    with h5py.File(Frames.features_file_path("resnet", "res5c"), "w") as res5c_features_file, \
            h5py.File(Frames.features_file_path("resnet", "pool5"), "w") as pool5_features_file:

        for video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            res5c_features_file.create_dataset(video_id, shape=(video_frame_count, 2048, 7, 7))
            pool5_features_file.create_dataset(video_id, shape=(video_frame_count, 2048))

        res5c_output = None

        def avg_pool_hook(module, input, output):
            nonlocal res5c_output
            res5c_output = input[0]

        resnet.avgpool.register_forward_hook(avg_pool_hook)

        total_frame_count = sum(dataset.frame_count_by_video_id[video_id] for video_id in dataset.video_ids)
        with tqdm(total=total_frame_count, desc="Extracting ResNet features") as progress_bar:
            for instance in torch.utils.data.DataLoader(dataset):
                video_id = instance["id"][0]
                frames = instance["frames"][0].to(DEVICE)

                batch_size = 32
                for start_index in range(0, len(frames), batch_size):
                    end_index = min(start_index + batch_size, len(frames))
                    frame_ids_range = range(start_index, end_index)
                    frame_batch = frames[frame_ids_range]

                    avg_pool_value = resnet(frame_batch)

                    res5c_features_file[video_id][frame_ids_range] = res5c_output.cpu()
                    pool5_features_file[video_id][frame_ids_range] = avg_pool_value.cpu()

                    progress_bar.update(len(frame_ids_range))