import math

import torch
import torch.optim as optim

from utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.kfac_utils import update_running_stat


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True):
        # legitimation check
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.CovAHandler = ComputeCovA()
        """ compute the covariance of the activation """
        self.CovGHandler = ComputeCovG()
        """ compute the covariance of the gradient """
        self.batch_averaged = batch_averaged
        """ bool markers for whether the gradient is batch averaged """

        
        self.known_modules = {'Linear', 'Conv2d'}
        """ dictionary for modules: {Linear,Conv2d} """

        self.modules = []
        """ list for saving modules temporarily """

        self.grad_outputs = {}
        """ buffer for saving the gradient output """
        self.model = model
        self._prepare_model()           #初始化准备模型，注册hook

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        """ buffer for saving the running estimates of the covariance of the activation and gradient """
        self.Q_a, self.Q_g = {}, {}
        """ buffer for saving the eigenvectors of the covariance of the activation and gradient """
        self.d_a, self.d_g = {}, {}
        """ buffer for saving the eigenvalues of the covariance of the activation and gradient """
        self.stat_decay = stat_decay
        """ parameter determines the time scale for the moving average """

        self.kl_clip = kl_clip
        self.TCov = TCov
        """ the period for computing the covariance of the activation and gradient """
        self.TInv = TInv
        """ the period for updating the inverse of the covariance """

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            # 计算module对应层激活的协方差矩阵，(classmethod方法)调用了ComputeCovA类中compute_cov_a函数，参数ainput[0].data，参数layer是module对应层。
            aa = self.CovAHandler(input[0].data, module)        
            # Initialize buffers
            if self.steps == 0:     
                """
                首先，aa.new(aa.size(0)) 创建了一个与 aa 具有相同数据类型和设备的新张量，但大小为 aa.size(0)，即 aa 的第一维度的大小。
                这意味着，如果 aa 是一个形状为 (n, m) 的二维张量，那么 aa.new(aa.size(0)) 将会创建一个形状为 (n,) 的一维张量。
                然后，.fill_(1) 是一个就地（in-place）操作，它将新创建的张量的所有元素填充为 1。
                最后，torch.diag() 函数接收一个一维张量作为输入，并返回一个二维张量，其中对角线上的元素是输入张量的元素，其余元素都为 0。
                因此，torch.diag(aa.new(aa.size(0)).fill_(1)) 将返回一个形状为 (n, n) 的对角矩阵，对角线上的元素都为 1，其余元素都为 0。
                总的来说，这段代码的目的是创建一个与 aa 的第一维度大小相同的单位矩阵。
                """
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))         # 为module对应层激活的协方差矩阵准备的缓存
            update_running_stat(aa, self.m_aa[module], self.stat_decay)             # exponential moving average

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            # 计算module对应层梯度的协方差矩阵，(classmethod方法)调用了ComputeCovG类中compute_cov_g函数，参数g是grad_output[0].data，参数layer是module对应层，参数self.batch_averaged对应batch_averaged。
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)     #计算梯度的协方差矩阵
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))         # 为module对应层梯度的协方差矩阵准备的缓存
            update_running_stat(gg, self.m_gg[module], self.stat_decay)             # exponential moving average

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():         #遍历模型的所有模块(层)
            """
            在这段代码中，module 是一个对象，module.__class__ 获取了创建 module 对象的类。
            然后 .__name__ 获取了这个类的名称。这种方式常常用于动态地获取对象的类型信息。
            例如，如果 module 是一个由 numpy.ndarray 类创建的对象，那么 module.__class__.__name__ 将返回 'ndarray'。
            """
            classname = module.__class__.__name__       #获取模块(层)的名称
            if classname in self.known_modules:     #如果模块(层)的名称在字典中
                self.modules.append(module)         #把模块(层)放入列表module中
                """
                注册一个钩子函数时，PyTorch会自动传递两个参数给这个钩子函数：一个是模块自身（即module），另一个是输入的元组（即input）。
                在每次前向传播之前，调用_save_input函数，并自动传递模块自身和输入的元组作为参数
                """
                module.register_forward_pre_hook(self._save_input)  
                """
                这里是 register_backward_hook 的一般用法：
                hook_handle = module.register_backward_hook(hook_function)
                注册的 hook_function 必须有三个参数：module、grad_input 和 grad_output。
                当反向传播到达该模块时，PyTorch 自动调用这个钩子函数，并传入以下三个参数：
                    module：当前钩子所在的模块。
                    grad_input：该模块的输入梯度的元组。
                    grad_output：该模块的输出梯度的元组。
                这意味着，当你在类中定义一个方法并将其注册为钩子时，PyTorch 会在适当的时候自动传递这些参数给该方法。
                因此，你在注册钩子时不需要手动传递这些参数，PyTorch 会在内部处理这些细节。
                此外，由于 _save_grad_output 是类的一个方法，第一个参数 self 是隐式传递的，代表类实例本身，这是 Python 方法调用的一个标准行为。

                在你的代码示例中，self._save_grad_output 作为一个方法被注册为钩子。
                当反向传播发生时，PyTorch 会自动调用它并传入 module、grad_input 和 grad_output 作为参数。
                这就是为什么在调用 self._save_grad_output 时不需要手动传入这四个参数的原因。
                """      
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))     #打印模块(层)和序号
                count += 1

    def _update_inv(self, m):
        """
        Do eigen decomposition of kronecker faction for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        :function: 

            m_aa=Q_a*(d_a)*Q_a^T

            m_gg=Q_g*(d_g)*Q_g^T
        """
        eps = 1e-10  # for numerical stability
        self.d_a[m], self.Q_a[m] = torch.symeig(
            self.m_aa[m], eigenvectors=True)
        self.d_g[m], self.Q_g[m] = torch.symeig(
            self.m_gg[m], eigenvectors=True)

        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the mth layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            param_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh) 卷积核梯度转矩阵
        else:
            param_grad_mat = m.weight.grad.data             #全连接层权重梯度不做处理
        if m.bias is not None:
            param_grad_mat = torch.cat([param_grad_mat, m.bias.grad.data.view(-1, 1)], 1)       #卷积和全连接处理一样，偏置梯度转为列向量拼接到权重梯度矩阵的最后一列
        return param_grad_mat

    def _get_natural_grad(self, m, param_grad_mat, damping):
        """
        :param m:  the mth layer
        :param param_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m` th layer
        """
        # param_grad_mat is of output_dim * input_dim
        # inv((ss')) param_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ param_grad_mat @ [Q_a (1/R_a) Q_a^T]，where R_g = diag(d_g) and R_a = diag(d_a).
        # (F^-1 + lambda*I) * nebla(h) = Q_g * (Q_g^T * nebla(h) * Q_a * (1/d_g * d_a^T + lambda)) * Q_a^T
        # V_1 = Q_g^T * nebla(h) * Q_a 
        # V_2 = V_1 * (1/d_g * d_a^T + lambda)
        # V_3 = Q_g * V_2 * Q_a^T
        v1 = self.Q_g[m].t() @ param_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        #卷积kfac论文附录A.2第26页 Q_g是关于梯度的kronecker因子 Q_a是关于激活的kronecker因子
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()                                      
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            #v[:, :-1] 和 v[:, -1:] 分别表示取 v 的所有行，但列分别取除了最后一列的所有列和最后一列。对应的是权重和偏置各自的fvp。再把形状对应变回去
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):        #卷积kfac论文附录A.1
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.TCov:          #正则化
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:                                       #动量
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf

                p.data.add_(-group['lr'], d_p)

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:      #逐层遍历，更新F逆的kronecker因子和计算fvp
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m)
            param_grad_mat = self._get_matrix_form_grad(m, classname)       #获取第m层的(偏置拼接到权重矩阵最后一列得到的)梯度矩阵
            fvp = self._get_natural_grad(m, param_grad_mat, damping)          #获取第m层的权重和偏置的fvp
            updates[m] = fvp          #第m层的权重和偏置的fvp放入updates中
        self._kl_clip_and_update_grad(updates, lr)              #kl剪裁和更新梯度

        self._step(closure)         #update the parameters
        self.steps += 1

        return fvp
