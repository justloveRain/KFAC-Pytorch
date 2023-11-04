'''Train CIFAR10/CIFAR100 with PyTorch.'''
import argparse
import os
from optimizers import (KFACOptimizer, EKFACOptimizer)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.network_utils import get_network
from utils.data_utils import get_dataloader


# fetch args
parser = argparse.ArgumentParser()


parser.add_argument('--network', default='vgg16_bn', type=str)
parser.add_argument('--depth', default=19, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)

# densenet
""" 
growthRate：
这是一个超参数，用于控制每个密集块（dense block）中特征图（feature map）的数量。
在每个密集块中，每个卷积层都会生成growthRate数量的特征图，这些特征图会被连接到输入特征图上，形成更丰富的特征组合。
growthRate的值越大，生成的特征图就越多，模型的复杂度就越高。

compressionRate：
这是在过渡层（transition layer）中使用的一个超参数。
过渡层的作用是减少特征图的数量和维度，以减少计算量和控制模型复杂度。
compressionRate的值越小，过渡层减少的特征图数量就越多，模型的复杂度就越低。 
"""
parser.add_argument('--growthRate', default=12, type=int)       
parser.add_argument('--compressionRate', default=2, type=int)       

# wrn, densenet
parser.add_argument('--widen_factor', default=1, type=int)      
parser.add_argument('--dropRate', default=0.0, type=float)


parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--resume', '-r', action='store_true')         # resume from checkpoint
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--log_dir', default='runs/pretrain', type=str)


parser.add_argument('--optimizer', default='kfac', type=str)
parser.add_argument('--batch_size', default=64, type=float)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--milestone', default=None, type=str)              # for MultiStepLR
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--stat_decay', default=0.95, type=float)           # for KFAC
parser.add_argument('--damping', default=1e-3, type=float)
parser.add_argument('--kl_clip', default=1e-2, type=float)          # for KFAC
parser.add_argument('--weight_decay', default=3e-3, type=float)
parser.add_argument('--TCov', default=10, type=int)         # for EKFAC
parser.add_argument('--TScal', default=10, type=int)        # for EKFAC
parser.add_argument('--TInv', default=100, type=int)        # for EKFAC


parser.add_argument('--prefix', default=None, type=str)     # prefix for checkpoint
args = parser.parse_args()

# init model
nc = {      # number of classes
    'cifar10': 10,
    'cifar100': 100
}
num_classes = nc[args.dataset]
net = get_network(args.network,
                  depth=args.depth,
                  num_classes=num_classes,
                  growthRate=args.growthRate,
                  compressionRate=args.compressionRate,
                  widen_factor=args.widen_factor,
                  dropRate=args.dropRate)

# get network and put it to device
net = net.to(args.device)

# init dataloader
trainloader, testloader = get_dataloader(dataset=args.dataset,
                                         train_batch_size=args.batch_size,
                                         test_batch_size=256)

#===========================================================================================================#
# init optimizer and lr scheduler
""" 
.lower() 是一个 Python 字符串方法，它将字符串中的所有大写字符转换为小写。
这样做的目的通常是为了消除大小写的差异，使得无论用户如何输入（大写、小写或混合），程序都能正确理解。
"""
optim_name = args.optimizer.lower()
tag = optim_name
if optim_name == 'sgd':
    #default SGD optimizer
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
elif optim_name == 'kfac':
    # K-FAC optimizer
    optimizer = KFACOptimizer(net,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              stat_decay=args.stat_decay,
                              damping=args.damping,
                              kl_clip=args.kl_clip,
                              weight_decay=args.weight_decay,
                              TCov=args.TCov,
                              TInv=args.TInv)
elif optim_name == 'ekfac':
    # EKFAC optimizer
    optimizer = EKFACOptimizer(net,
                               lr=args.learning_rate,
                               momentum=args.momentum,
                               stat_decay=args.stat_decay,
                               damping=args.damping,
                               kl_clip=args.kl_clip,
                               weight_decay=args.weight_decay,
                               TCov=args.TCov,
                               TScal=args.TScal,
                               TInv=args.TInv)
else:
    raise NotImplementedError
#===========================================================================================================#

if args.milestone is None:
    """ 
    args.milestone为None，则使用默认的里程碑，即在训练的50%和75%的时候调整学习率，调整的比例（gamma）为0.1。
    这是通过MultiStepLR类实现的，它接收一个优化器（optimizer）对象，一个里程碑列表和一个gamma值作为参数。 
    """
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1)
else:
    """ 
    如果args.milestone不为None，则将其按照逗号分割，然后转换为整数列表，作为里程碑传递给MultiStepLR类。
    这样，用户就可以通过命令行参数自定义学习率调度器的里程碑。
    split(',') 方法用于将 args.milestone 字符串按照逗号分割，返回一个字符串列表。
    然后，列表推导式 int(_) for _ in ... 遍历这个列表，将每个字符串转换为整数。
    这里的 int(_) 是 Python 的内置函数，用于将一个数字或字符串转换为整数。
    """
    milestone = [int(_) for _ in args.milestone.split(',')]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.1)

# init criterion
criterion = nn.CrossEntropyLoss()

""" 
这段代码是用于在训练神经网络时从检查点（checkpoint）恢复的。检查点是在训练过程中保存的模型状态，可以用于在训练中断后恢复训练，或者用于在训练完成后加载最佳模型。
首先，代码定义了两个变量 start_epoch 和 best_acc，这两个变量用于存储从检查点加载的训练轮次（epoch）和最佳准确率。
然后，如果 args.resume 为真，代码将尝试从检查点恢复。首先，它会检查 args.load_path 指定的文件是否存在。如果不存在，它会抛出一个断言错误。
如果检查点文件存在，代码将使用 torch.load 函数加载检查点。这个函数会返回一个字典，其中包含了保存在检查点中的信息。
接着，代码使用 net.load_state_dict 方法加载模型的状态。这个方法会将模型的参数设置为检查点中保存的参数。
然后，代码从检查点字典中获取并更新 best_acc 和 start_epoch 的值。
最后，代码打印出已经加载的检查点的训练轮次和最佳准确率。
"""
start_epoch = 0
best_acc = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.load_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.load_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))

# init summary writter
"""
这段代码主要用于初始化一个日志写入器，用于记录训练过程中的信息。
首先，它定义了一个日志目录log_dir，这个目录的路径是由多个部分组成的，包括args.log_dir，args.dataset，args.network，args.optimizer，以及一些参数（学习率，权重衰减，阻尼）。
然后，代码检查这个目录是否已经存在。如果不存在，就使用os.makedirs(log_dir)创建它。这是为了确保在写入日志之前，目录已经存在。
最后，使用SummaryWriter(log_dir)创建了一个SummaryWriter对象，并将其赋值给writer。
SummaryWriter是PyTorch的一个功能，它可以将训练过程中的数据（例如损失，准确率等）写入TensorBoard，这样就可以在TensorBoard中可视化这些数据。
总的来说，这段代码的目的是创建一个日志目录，并初始化一个SummaryWriter对象，以便在训练过程中记录和可视化数据。
"""
log_dir = os.path.join(args.log_dir, args.dataset, args.network, args.optimizer,
                       'lr%.3f_wd%.4f_damping%.4f' %            #这里的%操作符将args.learning_rate、args.weight_decay和args.damping的值插入到'lr%.3f_wd%.4f_damping%.4f'字符串中对应的位置，生成一个新的字符串。
                                                                #例如，如果args.learning_rate为0.001，args.weight_decay为0.0001，args.damping为0.0001，那么生成的字符串将会是'lr0.001_wd0.0001_damping0.0001'。
                       (args.learning_rate, args.weight_decay, args.damping))

if not os.path.isdir(log_dir):
    """
    如果log_dir不是一个已经存在的目录（即os.path.isdir(log_dir)返回False），那么就执行后面的代码块。
    最后，os.makedirs(log_dir)这一行代码会创建log_dir指定的目录。如果在创建过程中需要创建多级目录，os.makedirs会自动创建所有需要的中间目录。
    """
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    lr_scheduler.step()
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (tag, lr_scheduler.get_lr()[0], 0, 0, correct, total))

    writer.add_scalar('train/lr', lr_scheduler.get_lr()[0], epoch)

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if optim_name in ['kfac', 'ekfac'] and optimizer.steps % optimizer.TCov == 0:
            # compute true fisher
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                              1).squeeze().cuda()
            loss_sample = criterion(outputs, sampled_y)
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()  # clear the gradient for computing true-fisher.
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (tag, lr_scheduler.get_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (tag,lr_scheduler.get_lr()[0], test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (tag, lr_scheduler.get_lr()[0], test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100.*correct/total

    writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('test/acc', 100. * correct / total, epoch)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'loss': test_loss,
            'args': args
        }

        torch.save(state, '%s/%s_%s_%s%s_best.t7' % (log_dir,
                                                     args.optimizer,
                                                     args.dataset,
                                                     args.network,
                                                     args.depth))
        best_acc = acc


def main():
    for epoch in range(start_epoch, args.epoch):
        train(epoch)
        test(epoch)
    return best_acc


if __name__ == '__main__':
    main()


