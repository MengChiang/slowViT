import json
import torch
from torch.utils.data import Dataset
# TODO: Import or define load_video, resize_frames, normalize_frames, to_tensor

class NUYLSushiDataset(Dataset):
    def __init__(self, annotation_file, class_file, video_dir, tokenizer, fps, transform=None):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        with open(class_file, 'r') as f:
            self.main_classes = json.load(f)
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.fps = fps  # Frames per second
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        video_id = annotation['video_id']
        start_time = annotation['start_time']
        end_time = annotation['end_time']
        action_class = annotation['action_class']
        view = annotation['view']  # Load the view information

        # Load and preprocess the video
        video_path = f'{self.video_dir}/{video_id}_{view}.mp4'  # Use the view information in the video path
        frames = load_video(video_path)
        frames = frames[int(start_time*self.fps):int(end_time*self.fps)]  # Use fps to slice the frames
        frames = resize_frames(frames)
        frames = normalize_frames(frames)
        video = to_tensor(frames)

        # Preprocess the action class
        inputs = self.tokenizer(action_class, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Get the main class
        main_class = self.main_classes[action_class]

        return video, input_ids, attention_mask, main_class, view  # Return the view information