import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import matplotlib.pyplot as plt


def reconstruction(maskgit, x=None, code=None, masked_code=None, unmasked_code=None, mask=None):
    """ For visualization, show the model ability to reconstruct masked img
       :param
        x             -> torch.FloatTensor: bsize x 3 x 256 x 256, the real image
        code          -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
        masked_code   -> torch.LongTensor: bsize x 16 x 16, the masked image tokens
        unmasked_code -> torch.LongTensor: bsize x 16 x 16, the prediction of the transformer
        mask          -> torch.LongTensor: bsize x 16 x 16, the binary mask of the encoded image
       :return
        l_visual      -> torch.LongTensor: bsize x 3 x (256 x ?) x 256, the visualization of the images
    """
    vq = maskgit.ae
    l_visual = []
    with torch.no_grad():
        if x is not None:
            l_visual.append(x.cpu())
        if code is not None:
            code = code.view(code.size(0), maskgit.input_size, maskgit.input_size)
            # Decoding reel code
            _x = vq.decode_code(torch.clamp(code, 0, maskgit.args.codebook_size - 1))
            _x = torch.clamp(_x, -1, 1).cpu()
            l_visual.append(_x)
            if mask is not None:
                # Decoding reel code with mask to hide
                mask = mask.view(code.size(0), 1, maskgit.input_size, maskgit.input_size).float().cpu()
                __x2 = _x * (1 - F.interpolate(mask, (maskgit.args.img_size, maskgit.args.img_size)))
                l_visual.append(__x2.cpu())
        if masked_code is not None:
            # Decoding masked code
            masked_code = masked_code.view(code.size(0), maskgit.input_size, maskgit.input_size)
            __x = vq.decode_code(torch.clamp(masked_code, 0, maskgit.args.codebook_size - 1))
            __x = torch.clamp(__x, -1, 1)
            l_visual.append(__x.cpu())

        if unmasked_code is not None:
            # Decoding predicted code
            unmasked_code = unmasked_code.view(code.size(0), maskgit.input_size, maskgit.input_size)
            ___x = vq.decode_code(torch.clamp(unmasked_code, 0, maskgit.args.codebook_size - 1))
            ___x = torch.clamp(___x, -1, 1)
            l_visual.append(___x.cpu())

    return torch.cat(l_visual, dim=0)


def show_images_grid(batch, nrow=4, padding=2):
    """
    Displays a batch of images concatenated into a single grid using PyTorch's make_grid.

    Args:
        batch (torch.Tensor): Batch of images, shape (B, C, H, W), with values in range [-1, 1].
        nrow (int): Number of images in each row of the grid.
        padding (int): Padding between images in the grid.
    """
    # Unnormalize the tensor from [-1, 1] to [0, 1]
    batch = (batch + 1) / 2.0

    # Clamp to ensure all values are in [0, 1]
    batch = batch.clamp(0, 1)

    # Create the grid
    grid = make_grid(batch, nrow=nrow, padding=padding)

    # Move the grid to CPU and convert to numpy
    grid = grid.permute(1, 2, 0).cpu().numpy()

    # Display the grid
    plt.figure(figsize=(nrow * 2, (len(batch) // nrow + 1) * 2))
    plt.imshow(grid)
    plt.axis("off")
    plt.savefig("./gen_img/generated_images_3.png")  # Save the figure
    plt.close()  # Close the figure to free memory

