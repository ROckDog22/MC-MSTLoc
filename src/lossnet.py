# """gnt xent loss"""
#
# import mindspore.nn as nn
# import mindspore
# from mindspore.ops import operations as P
# import torch
#
# class LossNet(nn.Cell):
#     """modified loss function"""
#
#     def __init__(self, temp=0.1):
#         super(LossNet, self).__init__()
#         self.exp = P.Exp()
#         self.t = P.Transpose()
#         self.matmul = P.MatMul()
#         self.sum = P.ReduceSum()
#         self.sum_keep_dim = P.ReduceSum(keep_dims=True)
#         self.log = P.Log()
#         self.mean = P.ReduceMean()
#         self.shape = P.Shape()
#         self.eye = P.Eye()
#         self.temp = temp
#
#     def diag_part_new(self, inputs, batch_size):
#         eye_matrix = self.eye(batch_size, batch_size, mindspore.float32)
#         inputs = inputs * eye_matrix
#         inputs = self.sum_keep_dim(inputs, 1)
#         return inputs
#
#     def construct(self, x, y, z_aux):
#         """forward function"""
#         batch_size = self.shape(x)[0]
#
#         perm = (1, 0)
#         mat_x_x = self.exp(self.matmul(x, self.t(x, perm) / self.temp))
#         mat_y_y = self.exp(self.matmul(y, self.t(y, perm) / self.temp))
#         mat_x_y = self.exp(self.matmul(x, self.t(y, perm) / self.temp))
#
#         mat_aux_x = self.exp(self.matmul(x, self.t(z_aux, perm) / self.temp))
#         mat_aux_y = self.exp(self.matmul(y, self.t(z_aux, perm) / self.temp))
#         mat_aux_z_x = self.exp(self.matmul(z_aux, self.t(x, perm) / self.temp))
#         mat_aux_z_y = self.exp(self.matmul(z_aux, self.t(y, perm) / self.temp))
#         tpppp = self.diag_part_new(mat_x_y, batch_size)
#         tppp = self.sum_keep_dim(mat_x_y, 1) - self.diag_part_new(mat_x_y, batch_size) + self.sum_keep_dim(mat_x_x, 1) - self.diag_part_new(mat_x_x, batch_size) +self.sum_keep_dim(mat_y_y, 1) - self.diag_part_new(mat_y_y, batch_size)
#         z = self.mean(-2 * self.log(tpppp / tppp))
#         z = self.log(tpppp / tppp)
#         loss_mutual = self.mean(-2 * self.log(self.diag_part_new(mat_x_y, batch_size) / (
#             self.sum_keep_dim(mat_x_y, 1) - self.diag_part_new(mat_x_y, batch_size) +
#             self.sum_keep_dim(mat_x_x, 1) - self.diag_part_new(mat_x_x, batch_size) +
#             self.sum_keep_dim(mat_y_y, 1) - self.diag_part_new(mat_y_y, batch_size))))
#
#         loss_aux_x = self.mean(-self.log((self.diag_part_new(mat_aux_x, batch_size) / (
#             self.sum_keep_dim(mat_aux_x, 1) - self.diag_part_new(mat_aux_x, batch_size)))))
#
#         loss_aux_y = self.mean(-self.log((self.diag_part_new(mat_aux_y, batch_size) / (
#             self.sum_keep_dim(mat_aux_y, 1) - self.diag_part_new(mat_aux_y, batch_size)))))
#
#         loss_aux_z_x = self.mean(- self.log((self.diag_part_new(mat_aux_z_x, batch_size) / (
#             self.sum_keep_dim(mat_aux_z_x, 1) - self.diag_part_new(mat_aux_z_x, batch_size)))))
#
#         loss_aux_z_y = self.mean(-self.log((self.diag_part_new(mat_aux_z_y, batch_size) / (
#             self.sum_keep_dim(mat_aux_z_y, 1) - self.diag_part_new(mat_aux_z_y, batch_size)))))
#
#         loss = loss_mutual + loss_aux_x + loss_aux_y + loss_aux_z_x + loss_aux_z_y
#
#         return loss
#
# # -1599.4438
# if __name__ == '__main__':
#     from mindspore.common.initializer import One, Normal
#     a =  mindspore.ops.ones((32, 128), mindspore.float32)/10
#     b = a/4
#     c = a/2
#     model = LossNet()
#     d = model(a,b,c)
#     print(d)


"""gnt xent loss"""

import torch.nn as nn
import torch


class LossNet(nn.Module):
    """modified loss function"""

    def __init__(self, temp=0.1):
        super(LossNet, self).__init__()
        self.exp = torch.exp
        self.t = torch.permute
        self.matmul = torch.matmul
        self.sum = torch.sum
        self.sum_keep_dim = torch.sum
        self.log = torch.log
        self.mean = torch.mean
        self.eye = torch.eye
        self.temp = temp

    def diag_part_new(self, inputs, batch_size):
        # .cuda()
        eye_matrix = self.eye(batch_size, dtype=torch.float32)
        inputs = inputs * eye_matrix
        inputs = self.sum_keep_dim(inputs, 1)
        return inputs


    def forward(self, x):
        """forward function"""
        batch_size = int(x.size(0)/3)
        x = torch.nn.functional.normalize(x, dim=1)
        x,y,z_aux = torch.split(x, batch_size, 0)
        perm = (1, 0)
        mat_x_x = self.exp(self.matmul(x, self.t(x, perm) / self.temp))
        mat_y_y = self.exp(self.matmul(y, self.t(y, perm) / self.temp))
        mat_x_y = self.exp(self.matmul(x, self.t(y, perm) / self.temp))


        mat_aux_x = self.exp(self.matmul(x, self.t(z_aux, perm) / self.temp))
        mat_aux_y = self.exp(self.matmul(y, self.t(z_aux, perm) / self.temp))
        mat_aux_z_x = self.exp(self.matmul(z_aux, self.t(x, perm) / self.temp))
        mat_aux_z_y = self.exp(self.matmul(z_aux, self.t(y, perm) / self.temp))


        loss_mutual = self.mean(-2 * self.log(self.diag_part_new(mat_x_y, batch_size) / (
                self.sum_keep_dim(mat_x_y, 1, keepdim=True) - self.diag_part_new(mat_x_y, batch_size) +
                self.sum_keep_dim(mat_x_x, 1, keepdim=True) - self.diag_part_new(mat_x_x, batch_size) +
                self.sum_keep_dim(mat_y_y, 1, keepdim=True) - self.diag_part_new(mat_y_y, batch_size))))

        loss_aux_x = self.mean(-self.log((self.diag_part_new(mat_aux_x, batch_size) / (
                self.sum_keep_dim(mat_aux_x, 1, keepdim=True) - self.diag_part_new(mat_aux_x, batch_size)))))

        loss_aux_y = self.mean(-self.log((self.diag_part_new(mat_aux_y, batch_size) / (
                self.sum_keep_dim(mat_aux_y, 1, keepdim=True) - self.diag_part_new(mat_aux_y, batch_size)))))

        loss_aux_z_x = self.mean(- self.log((self.diag_part_new(mat_aux_z_x, batch_size) / (
                self.sum_keep_dim(mat_aux_z_x, 1, keepdim=True) - self.diag_part_new(mat_aux_z_x, batch_size)))))

        loss_aux_z_y = self.mean(-self.log((self.diag_part_new(mat_aux_z_y, batch_size) / (
                self.sum_keep_dim(mat_aux_z_y, 1, keepdim=True) - self.diag_part_new(mat_aux_z_y, batch_size)))))

        loss = loss_mutual + loss_aux_x + loss_aux_y + loss_aux_z_x + loss_aux_z_y

        return loss

class MylossNet(nn.Module):
    """modified loss function"""

    def __init__(self, temp=0.1):
        super(MylossNet, self).__init__()
        self.exp = torch.exp
        self.t = torch.permute
        self.matmul = torch.matmul
        self.sum = torch.sum
        self.sum_keep_dim = torch.sum
        self.log = torch.log
        self.mean = torch.mean
        self.eye = torch.eye
        self.temp = temp

    def diag_part_new(self, inputs, batch_size):
        # .cuda()
        eye_matrix = self.eye(batch_size, dtype=torch.float32).cuda()
        inputs = inputs * eye_matrix
        inputs = self.sum_keep_dim(inputs, 1)
        return inputs


    def forward(self, x,y,z):
        """forward function"""
        batch_size = int(x.size(0)/2)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        z = torch.nn.functional.normalize(z, dim=1)

        x1,x2 = torch.split(x, batch_size, 0)
        y1,y2 = torch.split(y, batch_size, 0)
        z1,z2 = torch.split(z, batch_size, 0)

        perm = (1, 0)
        mat_x1_x2 = self.exp(self.matmul(x1, self.t(x2, perm) / self.temp))
        mat_x1_y1 = self.exp(self.matmul(x1, self.t(y1, perm) / self.temp))
        mat_x1_z1 = self.exp(self.matmul(x1, self.t(z1, perm) / self.temp))

        a = self.diag_part_new(mat_x1_x2, batch_size) / (
                self.sum_keep_dim(mat_x1_y1, 1, keepdim=True)+
                self.sum_keep_dim(mat_x1_z1, 1, keepdim=True))
        loss_low = self.mean(-self.log(a))

        mat_y1_y2 = self.exp(self.matmul(y1, self.t(y2, perm) / self.temp))
        mat_y1_x1 = self.exp(self.matmul(y1, self.t(x1, perm) / self.temp))
        mat_y1_z1 = self.exp(self.matmul(y1, self.t(z1, perm) / self.temp))

        loss_mid = self.mean(-self.log(self.diag_part_new(mat_y1_y2, batch_size) / (
                self.sum_keep_dim(mat_y1_x1, 1, keepdim=True)+
                self.sum_keep_dim(mat_y1_z1, 1, keepdim=True))))

        mat_z1_z2 = self.exp(self.matmul(z1, self.t(z2, perm) / self.temp))
        mat_z1_x1 = self.exp(self.matmul(z1, self.t(x1, perm) / self.temp))
        mat_z1_y1 = self.exp(self.matmul(z1, self.t(y1, perm) / self.temp))



        loss_high = self.mean(-self.log(self.diag_part_new(mat_z1_z2, batch_size) / (
                self.sum_keep_dim(mat_z1_x1, 1, keepdim=True)+
                self.sum_keep_dim(mat_z1_y1, 1, keepdim=True))))

        loss_scale = loss_low + loss_mid + loss_high

        xyz1 = torch.cat((x1, y1, z1), dim=1)
        xyz2 = torch.cat((x2, y2, z2), dim=1)

        mat12 = self.exp(self.matmul(xyz1, self.t(xyz2, perm) / self.temp))
        loss_view = self.mean(-self.log(self.diag_part_new(mat12, batch_size) / (
                self.sum_keep_dim(mat12, 1, keepdim=True))))

        return loss_view + loss_scale



class SIMCLRNet(nn.Module):
    """modified loss function"""

    def __init__(self, temp=0.1):
        super(SIMCLRNet, self).__init__()
        self.exp = torch.exp
        self.t = torch.permute
        self.matmul = torch.matmul
        self.sum = torch.sum
        self.sum_keep_dim = torch.sum
        self.log = torch.log
        self.mean = torch.mean
        self.eye = torch.eye
        self.temp = temp

    def diag_part_new(self, inputs, batch_size):
        eye_matrix = self.eye(batch_size, dtype=torch.float32).cuda()
        inputs = inputs * eye_matrix
        inputs = self.sum_keep_dim(inputs, 1)
        return inputs

    def forward(self, x):
        """forward function"""
        batch_size = int(x.size(0)/2)
        x = torch.nn.functional.normalize(x, dim=1)
        x,y = torch.split(x, batch_size, 0)
        perm = (1, 0)
        mat_x_x = self.exp(self.matmul(x, self.t(x, perm) / self.temp))
        mat_x_y = self.exp(self.matmul(x, self.t(y, perm) / self.temp))

        loss_mutual = self.mean(-1 * self.log(self.diag_part_new(mat_x_y, batch_size) / (
                self.sum_keep_dim(mat_x_y, 1, keepdim=True) - self.diag_part_new(mat_x_y, batch_size) +
                self.sum_keep_dim(mat_x_x, 1, keepdim=True)
                )))

        loss = loss_mutual

        return loss


# -1599.4438
if __name__ == '__main__':
    # a = torch.ones((96, 128), dtype=torch.float32)/10
    # b = a/4
    # c = b/2
    # model = LossNet()
    # b = model(a)
    # print(b)

    # simclr
    a = torch.ones((96, 128), dtype=torch.float32)/10
    model = SIMCLRNet()
    b = model(a)
    print(b)
