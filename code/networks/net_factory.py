from networks.VNet import FBAnet, FBAPnet

def net_factory(net_type="fbanet", in_chns=1, class_num=4, mode = "train", dim=128, dataset='LA'):
    if net_type == "fbapnet" and mode == "train":
        net = FBAPnet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, dim=dim, dataset=dataset).cuda()
    elif net_type == "fbapnet" and mode == "test":
        net = FBAPnet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False, dim=dim, dataset=dataset).cuda()
    elif net_type == "fbanet" and mode == "train":
        net = FBAnet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, dim=dim, dataset=dataset).cuda()
    elif net_type == "fbanet" and mode == "test":
        net = FBAnet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False, dim=dim, dataset=dataset).cuda()
    return net
