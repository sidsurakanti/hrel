"""
FLOPs / params of a pruned net, from layer widths alone

- FLOPs = multiply-accumulates of conv + fc layers (the papers' convention)
- params = conv + fc weights (no BN, no biases)
- width depends only on how MANY filters survive, so no model or mask is needed

two ways to count a conv's input width:
    chain  (HRel paper, true structural savings): input width = previous conv's surviving width
    dense  (hrank's own vgg count, understates savings): input width stays full

for resnets, "dense" means the residual stream stays full width and only conv1 -> conv2 shrinks
"""

VGG_SPATIAL = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
VGG_FULL = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
VGG_HEAD = [512, 512, 10]  # chenyaofo's classifier: 3 fc layers
RESNET_STAGE = [(16, 32), (32, 16), (64, 8)]  # (stream width, spatial) per stage
BLOCKS_PER_STAGE = 9


def vgg_cost(widths, dense_input=False):
    """widths: surviving filters of all 13 convs"""
    macs = params = 0
    cin = 3

    for i, (w, s) in enumerate(zip(widths, VGG_SPATIAL)):
        macs += cin * w * 9 * s * s
        params += cin * w * 9
        cin = VGG_FULL[i] if dense_input else w

    for out in VGG_HEAD:
        macs += cin * out
        params += cin * out
        cin = out

    return macs, params


def resnet_cost(stem, widths, dense_stream=False):
    """stem: surviving stem filters, widths: surviving filters of the 54 block convs"""
    macs = stem * 3 * 9 * 32 * 32
    params = stem * 3 * 9
    prev, stream = stem, stem

    for i, w in enumerate(widths):
        stage, conv = divmod(i, 2 * BLOCKS_PER_STAGE)
        stream_w, s = RESNET_STAGE[stage]
        if conv % 2 == 0:  # conv1 of a block
            cin = stream if dense_stream else prev
            if conv == 0:
                stream = stream_w  # first block of a stage moves the stream to its width
        else:
            cin = prev
        macs += cin * w * 9 * s * s
        params += cin * w * 9
        prev = w

    cin = 64 if dense_stream else prev
    macs += cin * 10
    params += cin * 10
    return macs, params
