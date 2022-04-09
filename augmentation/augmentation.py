from torchvision import transforms

transforms_list = [(transforms.ColorJitter(brightness=0.1, hue=.1), 'ColorJitter-Small'),
                   (transforms.ColorJitter(brightness=.5, hue=.2), 'ColorJitter-Medium'),
                   (transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), 'GaussianBlur-Small'),
                   (transforms.GaussianBlur(kernel_size=(5, 9), sigma=(5, 10)), 'GaussianBlur-Medium'),
                   (transforms.RandomRotation(degrees=(0, 10)), 'Rotation-Small'),
                   (transforms.RandomRotation(degrees=(0, 180)), 'Rotation-Large'),
                   (transforms.RandomResizedCrop(size=(299, 299), scale=(0.8, 0.8)), 'Crop-Small'),
                   (transforms.RandomResizedCrop(size=(299, 299), scale=(0.45, 0.45)), 'Crop-Large'),
                   (transforms.RandomAdjustSharpness(sharpness_factor=1.2), 'Sharpness-Small'),
                   (transforms.RandomAdjustSharpness(sharpness_factor=2.0), 'Sharpness-Large'),
                   (transforms.RandomAutocontrast(p=0.2), 'Contrast-Small'),
                   (transforms.RandomAutocontrast(p=0.8), 'Contrast-Large'),
                   (transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET), 'AutoAugment-Imagenet')]

resnext_best = [
    transforms.RandomAutocontrast(p=0.8),
    transforms.RandomAdjustSharpness(sharpness_factor=2.0),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomResizedCrop(size=(299, 299), scale=(0.8, 0.8)),
    transforms.RandomRotation(degrees=(0, 180))
]

convnext_best = [transforms.ColorJitter(brightness=0.1, hue=.1),
                 transforms.RandomAutocontrast(p=0.8),
                 transforms.RandomAdjustSharpness(sharpness_factor=2.0),
                 transforms.RandomResizedCrop(size=(299, 299), scale=(0.45, 0.45)),
                 transforms.GaussianBlur(kernel_size=(5, 9), sigma=(5, 10)),
                 transforms.RandomRotation(degrees=(0, 10))]
