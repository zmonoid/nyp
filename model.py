import torch.nn as nn
import torch
import torch.nn.functional as F


# Network input should be 224x224
class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()
        # Everything except the last linear layer

        if arch.startswith('vgg'):
            self.features = original_model.features
            self.classifier = nn.Sequential(*list(
                original_model.classifier.children())[:-1])
            self.regress = nn.Linear(4096, num_classes)

        elif arch.startswith('resnet'):
            self.features = nn.Sequential(*list(original_model.children())[:
                                                                           -1])
            self.classifier = nn.Sequential(nn.Dropout(),
                                            nn.Linear(2048, num_classes))

        # Freeze those weights
        # for p in self.features.parameters():
        # p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        # y = self.regress(y)
        y = torch.atan(y) * 8
        return y


class IntentionBlock(nn.Module):
    def __init__(self, num_intention=2):
        super(IntentionBlock, self).__init__()
        self.num_intention = num_intention
        self.fc1 = nn.Linear(2048, 2048 * num_intention)
        self.fc2 = nn.Linear(num_intention, 64)
        self.fc3 = nn.Linear(2048 + 64, 1)

    def forward(self, x, intention):
        y2 = self.fc2(intention)

        y1 = F.dropout(x)
        y1 = self.fc1(y1)
        mask = intention.repeat(1, 2048)
        y1 = y1.masked_select(mask.byte())
        y1 = y1.view(y2.size(0), -1)

        y3 = torch.cat((y1, y2), 1)
        y3 = F.relu(y3, True)
        y3 = F.dropout(y3)
        y = self.fc3(y3)
        return y


class IntentionModel(nn.Module):
    def __init__(self, original_model, arch='vgg16', num_intention=2):
        super(IntentionModel, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = IntentionBlock(num_intention=num_intention)

    def forward(self, x, intention):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        f = self.classifier(f, intention)
        return f


class HeatmapModel(nn.Module):
    def __init__(self, original_model, arch='vgg16'):
        super(HeatmapModel, self).__init__()
        self.features = original_model.module.features

    def forward(self, x):
        y = self.features(x)
        y = torch.atan(y) * 8
        return y


# Network input should be 200x66
class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.PReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.PReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.PReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.PReLU(), )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1152, 1164),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(1164, 100),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(100, 50),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(10, 1),
            nn.PReLU(), )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = torch.atan(x) * 8
        return x
