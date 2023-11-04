from models.cifar import (alexnet, densenet, resnet,
                          vgg16_bn, vgg19_bn, wrn)
# from models.imagenet import (resnext50, resnext101,
#                               resnext152)


def get_network(network, **kwargs):
    """ 
    定义了一个名为 get_network 的函数，该函数接受一个名为 network 的参数，以及一个可变数量的关键字参数（**kwargs）。
    
    这个函数的主要目的是根据传入的 network 参数，从预定义的网络字典中选择并返回相应的网络函数。
    
    networks中保存了将网络实例化的函数。
    
    在函数内部，我们首先定义了一个名为 networks 的字典。

    这个字典的键是网络的名称（如 'alexnet'，'densenet'，'resnet' 等），值是对应的网络函数。这些网络函数应该已经在其他地方定义，并且可以接受关键字参数。
    
    然后，函数通过 networks[network] 从字典中获取对应的网络函数。这里的 network 应该是一个字符串，对应于 networks 字典中的一个键。

    最后，函数通过 (**kwargs) 调用选中的网络实例化函数，并将所有传入的关键字参数传递给这个函数。然后返回这个函数的结果。
    """
    networks = {
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'wrn': wrn
    }

    return networks[network](**kwargs)

