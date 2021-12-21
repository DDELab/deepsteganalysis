import torch
import torch.nn.functional


class ConvolutionalFilterManifold(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, transposed=False, bias=True, manifold_channels=32, activation=torch.nn.LeakyReLU, manifold_bias=True):
        super(ConvolutionalFilterManifold, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.transposed = transposed
        self.dilation = dilation
        self.padding = padding

        self.manifold = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=manifold_channels, kernel_size=8, bias=manifold_bias),
            activation(),
            torch.nn.Conv2d(in_channels=manifold_channels, out_channels=manifold_channels, kernel_size=1, bias=manifold_bias),
            activation()
        )

        self.weight_transform = torch.nn.ConvTranspose2d(in_channels=manifold_channels, out_channels=out_channels * in_channels, kernel_size=self.kernel_size, bias=manifold_bias)

        if bias:
            self.bias_transfrom = torch.nn.Conv2d(in_channels=manifold_channels, out_channels=out_channels, kernel_size=1, bias=manifold_bias)
        else:
            self.bias_transfrom = None

    def forward(self, q, x):
        m = self.manifold(q)

        batch_size = x.shape[0]
        x = x.view(1, batch_size * self.in_channels, x.shape[2], x.shape[3])

        w = self.weight_transform(m)

        if self.bias_transfrom is not None:
            b = self.bias_transfrom(m).view(self.out_channels * batch_size)
        else:
            b = None

        if not self.transposed:
            w = w.view(self.out_channels * batch_size, self.in_channels, self.kernel_size, self.kernel_size)
            x = torch.nn.functional.conv2d(x, w, bias=b, stride=self.stride, groups=batch_size, dilation=self.dilation, padding=self.padding)
        else:
            w = w.view(self.in_channels * batch_size, self.out_channels, self.kernel_size, self.kernel_size)
            x = torch.nn.functional.conv_transpose2d(x, w, bias=b, stride=self.stride, groups=batch_size, dilation=self.dilation, padding=self.padding)

        x = x.reshape(batch_size, self.out_channels, x.shape[2], x.shape[3])

        return x
