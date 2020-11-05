def dice_loss(input, target):
    smooth = 1.
    loss = 0.

    iflat = input[:,0 ].view(-1)
    tflat = target[:, 0].view(-1)
    intersection = (iflat * tflat).sum()
    loss = (1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth)))

    return loss
