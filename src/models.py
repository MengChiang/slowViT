import torch
import torch.nn as nn

class SlowViT(nn.Module):
    def __init__(self, slowfast_model, bert_model, num_classes):
        super(SlowViT, self).__init__()
        self.slowfast_model = slowfast_model
        self.bert_model = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, video, text):
        video_features = self.slowfast_model(video)
        text_features = self.bert_model(text)[0]
        combined_features = torch.cat((video_features, text_features), dim=-1)
        output = self.classifier(combined_features)
        return output