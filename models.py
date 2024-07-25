import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchaudio
from config import LATENT_DIM

class CustomCNNModel(nn.Module):
    """
    A customized CNN model that accepts a single input channel and outputs a specified number of classes.

    Attributes:
        base_model (nn.Module): The base CNN model with modifications.
    """
    
    def __init__(self, num_classes, weights=None, modelstr='resnet18'):
        """
        Initializes the custom model with a single input channel and a custom number of output classes.

        Parameters:
            num_classes (int): The number of output classes for the final classification layer.
            weights (str, optional): The type of pre-trained weights to use (e.g., 'IMAGENET1K_V1').
        """
        super(CustomCNNModel, self).__init__()
        
        # Load the ResNet-18 model, optionally with pre-trained weights
        if modelstr == 'resnet18':
            self.base_model = models.resnet18(weights=weights)
            # Modify the first convolutional layer to accept 1 channel instead of 3
            self.base_model.conv1 = nn.Conv2d(1, self.base_model.conv1.out_channels,
                                          kernel_size=self.base_model.conv1.kernel_size,
                                          stride=self.base_model.conv1.stride,
                                          padding=self.base_model.conv1.padding,
                                          bias=False)
        
            # Modify the final fully connected layer to output the specified number of classes
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        elif modelstr == 'dense121':
            self.base_model = models.densenet121(weights=weights)
            # Modify the first convolutional layer to accept 1 channel instead of 3
            self.base_model.features.conv0 = nn.Conv2d(1, self.base_model.features.conv0.out_channels,
                                          kernel_size=self.base_model.features.conv0.kernel_size,
                                          stride=self.base_model.features.conv0.stride,
                                          padding=self.base_model.features.conv0.padding,
                                          bias=False)
            # Modify the final fully connected layer to output the specified number of classes
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)
        else:
            self.base_model = models.efficientnet_b0(weights=weights)
            self.base_model.features[0][0]  = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.base_model.classifier = torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the model.
        """
        return self.base_model(x)

class ContrastiveCNN(nn.Module):
    """
    A customized CNN model for contrastive learning that accepts a single input channel
    and outputs latent representations of a specified dimension.
    """
    def __init__(self, latent_dim, weights=None, modelstr='resnet18'):
        """
        Initializes the contrastive learning model with a single input channel and a custom latent dimension.
        
        Parameters:
        latent_dim (int): The dimension of the latent representation.
        weights (str, optional): The type of pre-trained weights to use (e.g., 'IMAGENET1K_V1').
        modelstr (str): The type of model to use ('resnet18', 'dense121', or 'efficientnet_b0').
        """
        super(ContrastiveCNN, self).__init__()
        
        if modelstr == 'resnet18':
            self.base_model = models.resnet18(weights=weights)
            self.base_model.conv1 = nn.Conv2d(1, self.base_model.conv1.out_channels,
                                              kernel_size=self.base_model.conv1.kernel_size,
                                              stride=self.base_model.conv1.stride,
                                              padding=self.base_model.conv1.padding,
                                              bias=False)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
            
        elif modelstr == 'dense121':
            self.base_model = models.densenet121(weights=weights)
            self.base_model.features.conv0 = nn.Conv2d(1, self.base_model.features.conv0.out_channels,
                                                       kernel_size=self.base_model.features.conv0.kernel_size,
                                                       stride=self.base_model.features.conv0.stride,
                                                       padding=self.base_model.features.conv0.padding,
                                                       bias=False)
            self.feature_dim = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
            
        else:  # efficientnet_b0
            self.base_model = models.efficientnet_b0(weights=weights)
            self.base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.feature_dim = 1280
            self.base_model.classifier = nn.Identity()
        
        # Add a new projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, latent_dim)
        )
        
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Parameters:
        x (torch.Tensor): The input tensor.
        
        Returns:
        torch.Tensor: The latent representation.
        """
        features = self.base_model(x)
        latent = self.projection_head(features)
        return latent

    def get_features(self, x):
        """
        A method to get the features before the projection head.
        
        Parameters:
        x (torch.Tensor): The input tensor.
        
        Returns:
        torch.Tensor: The features before the projection head.
        """
        return self.base_model(x)


class FinetuningClassifier(nn.Module):
    """
    A classifier that uses a pre-trained ContrastiveCNN model as a feature extractor
    and adds a single trainable linear layer for classification.
    """
    def __init__(self, contrastive_model, num_classes,requires_grad=False):
        """
        Initializes the finetuning classifier.
        
        Parameters:
        contrastive_model (ContrastiveCNN): A pre-trained ContrastiveCNN model.
        num_classes (int): The number of classes for the classification task.
        """
        super(FinetuningClassifier, self).__init__()
        
        self.feature_extractor = contrastive_model
        self.requires_grad =  requires_grad
        
        # Freeze all parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = self.requires_grad
        
        # Add a single trainable linear layer for classification
        self.classifier = nn.Linear(LATENT_DIM, num_classes)
    
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Parameters:
        x (torch.Tensor): The input tensor.
        
        Returns:
        torch.Tensor: The classification logits.
        """
        with torch.no_grad():
            features = self.feature_extractor(x)
        return self.classifier(features)

    def unfreeze_last_n_layers(self, n):
        """
        Unfreezes the last n layers of the feature extractor for fine-tuning.
        
        Parameters:
        n (int): The number of layers to unfreeze, counting from the end.
        """
        trainable_layers = list(self.feature_extractor.modules())[-n:]
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True

# Example usage
if __name__ == "__main__":
    pass
