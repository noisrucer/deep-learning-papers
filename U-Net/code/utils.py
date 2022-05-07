import torch
import torchvision
import torchvision.transforms.functional as TF

def crop_img(original, target):
    '''
    :param original: (B, C, H, W)
    :param target: (B, C, H, W)
    '''
    target_size = target.size()[2]
    original_size = original.size()[2]

    delta = original_size - target_size

    delta_1 = delta // 2
    delta_2 = delta - delta_1

    return original[:, :, delta_1:original_size-delta_2, delta_1:original_size-delta_2]


def concat_imgs_crop(down_tensor, up_tensor):
    '''
    Concatenate 'down_tensor' and 'up_tensor' along dim-1.
    'down_tensor' must be cropped to match the resolution of 'up_tensor'

    :param down_tensor (tensor): from contracting pathway. Shape: (B, C, H, W)
    :param up_tensor (tensor): from expansive pathway. Shape: (B, C, H, W)

    :return cropped tensor
    '''
    cropped_down_tensor = crop_img(down_tensor, up_tensor)
    return torch.cat([cropped_down_tensor, up_tensor], dim=1) # channel-wise concat


def concat_imgs_resize(down_tensor, up_tensor):
    resized_up_tensor = TF.resize(up_tensor, size=down_tensor.shape[2:])
    return torch.cat([down_tensor, resized_up_tensor], dim=1)


def check_accuracy(loader, model, device='cpu'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    print(f"Dice score:{dice_score/len(loader)}")

    model.train()


def save_predictions(loader, model, folder='saved_images', device='cpu'):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )

        torchvision.utils.save_image(
            y.unsqueeze(1), f"{folder}/{idx}.png"
        )
        print("[Predictions Saved]")

    model.train()
