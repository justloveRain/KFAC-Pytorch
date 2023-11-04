import torch
import torchvision
import torchvision.transforms as transforms


def get_transforms(dataset):
    transform_train = None
    transform_test = None
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=2, root='../data'):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    """
    这段代码是在Python中使用assert语句进行断言。
    断言是一种编程概念，用于在代码中设置检查点。如果断言条件为真，那么程序将正常运行。但是，如果条件为假，那么程序将抛出AssertionError异常。
    在这段代码中，assert后面的条件是trainset is not None and testset is not None。
    这意味着，如果trainset和testset都不是None，那么程序将继续运行。
    否则，程序将抛出AssertionError异常，并显示后面的错误消息'Error, no dataset %s' % dataset。
    """
    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)

    return trainloader, testloader