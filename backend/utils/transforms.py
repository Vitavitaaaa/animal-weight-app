import torchvision.transforms as T


classifier_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])


segmentation_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])


regression_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])