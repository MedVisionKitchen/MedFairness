from torchvision import transforms
ISIC2019_LABEL = ['MEL',
                'NV',
                'BCC',
                'AK',
                'BKL',
                'DF',
                'VASC',
                'SCC',
                'UNK']

ISIC2019_SUBGROUP = [['Male','Female'], 
                     ['0~59', '60~85']]

def ISIC2019_Age():
    return (ISIC2019_LABEL, ISIC2019_SUBGROUP[1])


def Transforms(name):
    if name in ["ISIC2019_Age"]:
        data_transforms = {
            'valid': transforms.Compose([
                transforms.CenterCrop(768),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5])
            ]),
            'train': transforms.Compose([
                transforms.CenterCrop(768),
                transforms.Resize(224),
                transforms.RandomRotation(degrees=90), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5])
            ])
        }
        return data_transforms
        
