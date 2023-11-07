import torch
import torch.nn as nn
import torch.nn.functional as F


def try_contiguous(x):
    """
    test x is contiguousor not, 
    for acquiring the covariance of gradient execlusively in conv2d layer.
    """
    if not x.is_contiguous():
        """
        contiguous()函数的作用是使得张量（Tensor）在内存中连续。
        当对张量进行某些操作（如transpose、permute等）后，张量的数据在内存中可能不再是连续的。
        这时就需要调用contiguous()函数来使得张量在内存中变得连续，以便于进行后续的操作。
        """
        x = x.contiguous()
    return x

#??????????????????????????????????????????????????????????????????????
def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def update_running_stat(aa, m_aa, stat_decay):
    """ 
    update the running estimates for the covariance of activation or gradient.
        ie. exponential moving average in the end of 3.1 in the paper.
    
    :param aa: the covariance of activation or gradient.
    :param m_aa: the updated running estimates of covariance of activation or gradient.
    :param stat_decay: the parameter determines the time scale for the moving average.
    :return: None
    Function: m_aa = stat_decay*m_aa + (1-stat_decay)*aa
    """
    # using inplace operation to save memory!
    m_aa *= stat_decay / (1 - stat_decay)
    m_aa += aa
    m_aa *= (1 - stat_decay)


class ComputeCovA:
    """
    以下是代码执行的顺序：

    1.self.CovAHandler(input[0].data, module) 调用 ComputeCovA 实例。
    
    2.ComputeCovA 类的 compute_cov_a 方法被触发，传入 a（即 input[0].data）和 layer（即 module）。调用__call__ 方法检查 layer 的类型并计算CovA。
    
    3.如果 layer 是 nn.Linear 或 nn.Conv2d 类型，相应的方法会计算并返回协方差矩阵。
        
        在 conv2d 静态方法中，它首先提取卷积层的输入特征图的补丁（通过 _extract_patches 函数，该函数在代码中未提供），然后将这些补丁重塑为二维张量，以便计算协方差矩阵。如果卷积层包含偏置，它会在特征图的每个补丁上添加一个常数特征。最后，它计算并返回特征图补丁的转置与其自身的乘积除以批量大小的结果，这是协方差矩阵的估计。 
        
        在 linear 静态方法中，它处理线性层的激活。如果线性层包含偏置，它会在激活上添加一个常数特征。然后，它计算并返回激活的转置与其自身的乘积除以批量大小的结果，这同样是协方差矩阵的估计。

    ================================================
      
    在Python中，__call__方法是一个特殊的方法，它使得一个类的实例可以像函数一样被调用。当你对一个类的实例进行调用操作时，Python会自动执行这个__call__方法。

        例如，假设我们有一个类Foo，它定义了一个__call__方法：
            
            class Foo:
                
                def __call__(self, x, y):
                
                    return x + y

        然后我们创建一个Foo的实例：

            foo_instance = Foo()

        现在我们可以像调用函数一样调用foo_instance：

            result = foo_instance(1, 2)  # 这将调用Foo类的__call__方法
            
            print(result)  # 输出：3
    """
    @classmethod
    def compute_cov_a(cls, a, layer):
        """
        :param a: intermediates' activation
        :param layer: the corresponding layer
        :return:__call__ returns the covariance of activation
        """
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Conv2d):
            cov_a = cls.cova_conv2d(a, layer)
        elif isinstance(layer, nn.Linear):
            cov_a = cls.cova_linear(a, layer)
        else:
            # FIXME(CW): for extension to other layers. eg.BN layers
            # raise NotImplementedError
            cov_a = None
        return cov_a

    @staticmethod
    def cova_conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        # FIXME(CW): do we need to divide the output feature map's size?
        return a.t() @ (a / batch_size)

    @staticmethod
    def cova_linear(a, layer):
        # a size: batch_size * in_dim
        # cov_a: in_dim * in_dim
        batch_size = a.size(0)
        if layer.bias is not None:              #如果有偏置参数，就在a的最后一列加上全1的一列向量，之前我们把偏置参数拼接到了权重矩阵的最后一列
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)     
        return a.t() @ (a / batch_size)         #(1/batch_size)*中间变量激活的内积


class ComputeCovG:

    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:__call__ returns the covariance of gradient
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.covg_conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g = cls.covg_linear(g, layer, batch_averaged)
        else:
            # FIXME(CW): for extension to other layers. eg.BN layers
            # raise NotImplementedError
            cov_g = None

        return cov_g

    @staticmethod
    def covg_conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))

        return cov_g

    @staticmethod
    def covg_linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        # cov_g: out_dim*out_dim
        batch_size = g.size(0)
        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)        #cancel batch_averaged
        else:
            cov_g = g.t() @ (g / batch_size)        #(1/batch_size)*中间变量梯度的内积
        return cov_g


class ComputeMatGrad:

    @classmethod
    def __call__(cls, input, grad_output, layer):
        if isinstance(layer, nn.Linear):
            grad = cls.linear(input, grad_output, layer)
        elif isinstance(layer, nn.Conv2d):
            grad = cls.conv2d(input, grad_output, layer)
        else:
            raise NotImplementedError
        return grad

    @staticmethod
    def linear(input, grad_output, layer):
        """
        :param input: batch_size * input_dim
        :param grad_output: batch_size * output_dim
        :param layer: [nn.module] output_dim * input_dim
        :return: batch_size * output_dim * (input_dim + [1 if with bias])
        """
        with torch.no_grad():
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.unsqueeze(1)
            grad_output = grad_output.unsqueeze(2)
            grad = torch.bmm(grad_output, input)
        return grad

    @staticmethod
    def conv2d(input, grad_output, layer):
        """
        :param input: batch_size * in_c * in_h * in_w
        :param grad_output: batch_size * out_c * h * w
        :param layer: nn.module batch_size * out_c * (in_c*k_h*k_w + [1 if with bias])
        :return:
        """
        with torch.no_grad():
            input = _extract_patches(input, layer.kernel_size, layer.stride, layer.padding)
            input = input.view(-1, input.size(-1))  # b * hw * in_c*kh*kw
            grad_output = grad_output.transpose(1, 2).transpose(2, 3)
            grad_output = try_contiguous(grad_output).view(grad_output.size(0), -1, grad_output.size(-1))
            # b * hw * out_c
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.view(grad_output.size(0), -1, input.size(-1))  # b * hw * in_c*kh*kw
            grad = torch.einsum('abm,abn->amn', (grad_output, input))
        return grad
    

if __name__ == '__main__':
    def test_ComputeCovA():
        pass

    def test_ComputeCovG():
        pass






