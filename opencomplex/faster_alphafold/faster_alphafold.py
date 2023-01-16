import os
import torch
from torch import nn, autograd

path='opencomplex/faster_alphafold/libths_faster_alphafold.so'
#torch.ops.load_library(path)

class LayerNormFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, gamma, beta, residual=None):
        input_tensor = input_tensor.contiguous()
        if residual is not None:
            residual = residual.contiguous()

        # result[] = output, mean, var_rsqrt, input_tensor if residual is None else input_add_residual
        result = torch.ops.FasterAlphaFold.LayerNorm_forward(input_tensor, gamma, beta, residual)
        ctx.save_for_backward(gamma, result[1], result[2], result[3])
        ctx.add_residual = residual is not None
        return result[0]

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        gamma, mean, var_rsqrt, input_add_residual = ctx.saved_tensors

        #grad[] = grad_in, grad_gamma, grad_beta, [grad_residual]
        grad = torch.ops.FasterAlphaFold.LayerNorm_backward(grad_out, gamma, mean, var_rsqrt, input_add_residual, ctx.add_residual)
        return grad[0], grad[1], grad[2], None if not ctx.add_residual else grad[3]

def FasterLayerNorm(input_tensor, weight, bias, residual=None):
    return LayerNormFunction.apply(input_tensor, weight, bias, residual)


class MatMulFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_A, input_B, transpose_a = False, transpose_b = False, scale = 1.0):
        input_A = input_A.contiguous()
        input_B = input_B.contiguous()
        matmul_out = torch.ops.FasterAlphaFold.MatMul_forward(
            input_A, input_B, transpose_a, transpose_b, scale)
        ctx.transpose_a = transpose_a
        ctx.transpose_b = transpose_b
        ctx.scale = scale
        ctx.save_for_backward(input_A, input_B)
        return matmul_out

    @staticmethod
    def backward(ctx, grad_out):
        input_A, input_B = ctx.saved_tensors
        grad_A, grad_B = torch.ops.FasterAlphaFold.MatMul_backward(
            grad_out, input_A, input_B, ctx.transpose_a, ctx.transpose_b, ctx.scale)
        return grad_A, grad_B, None, None, None

def faster_matmul(input_A, input_B, transpose_a = False, transpose_b = False, scale = 1.0):
    return MatMulFunction.apply(input_A, input_B, transpose_a, transpose_b, scale)


class SoftmaxFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, mask_tensor=None, head_num=1):
        input_tensor = input_tensor.contiguous()
        softmax_out = torch.ops.FasterAlphaFold.Softmax_forward(input_tensor, mask_tensor, head_num)
        ctx.save_for_backward(softmax_out)
        return softmax_out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        (softmax_out,) = ctx.saved_tensors
        grad_in = torch.ops.FasterAlphaFold.Softmax_backward(grad_out, softmax_out)
        return grad_in, None, None

def faster_softmax(input_tensor, mask_tensor=None, head_num=1):
    return SoftmaxFunction.apply(input_tensor, mask_tensor, head_num)


class TriangleUpdateABGFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, mask_tensor, c_z, c_hidden):
        a, b, g = torch.ops.FasterAlphaFold.TriangleUpdateABG_forward(input_tensor, mask_tensor, c_z, c_hidden)
        ctx.save_for_backward(input_tensor, mask_tensor)
        ctx.c_z = c_z
        ctx.c_hidden = c_hidden
        return a, b, g

    @staticmethod
    def backward(ctx, grad_a, grad_b, grad_g):
        input_tensor, mask_tensor = ctx.saved_tensors
        input_grad, mask_grad = torch.ops.FasterAlphaFold.TriangleUpdateABG_backward(
            grad_a.contiguous(), grad_b.contiguous(), grad_g.contiguous(),
            input_tensor, mask_tensor)
        return input_grad, mask_grad, None, None

class TriangleUpdateABG(nn.Module):
    def __init__(self, c_z, c_hidden):
        super(TriangleUpdateABG, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden

    def forward(self, input_tensor, mask_tensor):
        return TriangleUpdateABGFunction.apply(input_tensor, mask_tensor, self.c_z, self.c_hidden)

    def extra_repr(self):
        return 'TriangleUpdateABG c_z={}, c_hidden={}'.format(self.c_z, self.c_hidden)


class AttentionSplitQKVFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, head_num=1):
        input_tensor = input_tensor.contiguous()
        q_out, k_out, v_out = torch.ops.FasterAlphaFold.AttentionSplitQKV_forward(input_tensor, head_num)
        return q_out, k_out, v_out

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        grad_qkv = torch.ops.FasterAlphaFold.AttentionSplitQKV_backward(grad_q.contiguous(), grad_k.contiguous(), grad_v.contiguous())
        return grad_qkv, None

def attention_split_qkv(input_tensor, head_num=1):
    return AttentionSplitQKVFunction.apply(input_tensor, head_num)

# class LinearFunction(autograd.Function):
#     @staticmethod
#     def forward(ctx, input_tensor, weight, bias, act_gelu = False, dropout_rate = 0.0):
#         bias_out = torch.Tensor(0)
#         dropout_mask = torch.Tensor(0)
#         if act_gelu == True or dropout_rate > 0.0:
#             output, bias_out, dropout_mask = Linear.forward_gelu_dropout(input_tensor, weight, bias, act_gelu, dropout_rate)
#         else:
#             output = Linear.forward(input_tensor, weight, bias)
#         ctx.save_for_backward(input_tensor, weight, bias_out, dropout_mask)
#         ctx.act_gelu = act_gelu
#         ctx.dropout_rate = dropout_rate
#         return output

#     @staticmethod
#     def backward(ctx, grad_out):
#         input_tensor, weight, bias_out, dropout_mask = ctx.saved_tensors
#         if ctx.act_gelu == True or ctx.dropout_rate > 0.0:
#             grad_in, grad_weight, grad_bias = Linear.backward_gelu_dropout(
#                 grad_out, input_tensor, weight, ctx.act_gelu, ctx.dropout_rate, bias_out, dropout_mask)
#         else:
#             grad_in, grad_weight, grad_bias = Linear.backward(
#                 grad_out, input_tensor, weight)
#         return grad_in, grad_weight, grad_bias, None, None

# class FasterLinear(nn.Module):
#     def __init__(self, in_features, out_features, act_gelu = False, dropout_rate = 0.0, assign=None):
#         super(FasterLinear, self).__init__()

#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.bias = nn.Parameter(torch.Tensor(out_features))
#         self.act_gelu = act_gelu
#         self.dropout_rate = dropout_rate

#         if assign:
#             self.weight.data.fill_(assign[0])
#             self.bias.data.fill_(assign[1])
#         else:
#             torch.nn.init.normal_(self.weight, mean=0, std=1)
#             torch.nn.init.normal_(self.bias, mean=0, std=1)

#     def forward(self, input_tensor):
#         return LinearFunction.apply(input_tensor, self.weight, self.bias, self.act_gelu, self.dropout_rate)

#     def extra_repr(self):
#         return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


# class TransposeFunction(autograd.Function):
#     @staticmethod
#     def forward(ctx, input_tensor, transpose_type):
#         transpose_out = Transpose4d.forward(input_tensor, transpose_type)
#         ctx.transpose_type = transpose_type
#         return transpose_out

#     @staticmethod
#     def backward(ctx, grad_out):
#         grad_in = Transpose4d.backward(grad_out, ctx.transpose_type)
#         return grad_in, None


# def faster_transpose(input_tensor, transpose_type = "0213"):
#     return TransposeFunction.apply(input_tensor, transpose_type)
