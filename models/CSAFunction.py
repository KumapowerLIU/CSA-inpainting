import torch
from util.NonparametricShift import NonparametricShift
from util.MaxCoord import MaxCoord
import util.util as util
import torch.nn as nn
import torch


from torch.autograd import Variable
class CSAFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask, shift_sz, stride, triple_w, flag, nonmask_point_idx,mask_point_idx ,flatten_offsets, sp_x, sp_y):
        assert input.dim() == 4, "Input Dim has to be 4"
        ctx.triple_w = triple_w
        ctx.flag = flag
        ctx.flatten_offsets = flatten_offsets


        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        c = c_real
        ctx.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor



        assert mask.dim() == 2, "Mask dimension must be 2"


        # bz is the batchsize of this GPU
        output_lst = ctx.Tensor(ctx.bz, c, ctx.h, ctx.w)
        ind_lst = torch.LongTensor(ctx.bz, ctx.h*ctx.w, ctx.h, ctx.w)

        if torch.cuda.is_available:
            ind_lst = ind_lst.cuda()
            nonmask_point_idx = nonmask_point_idx.cuda()
            mask_point_idx = mask_point_idx.cuda()
            sp_x = sp_x.cuda()
            sp_y = sp_y.cuda()

        for idx in range(ctx.bz):


            inpatch = input.narrow(0, idx, 1)
            output = input.narrow(0, idx, 1)

            Nonparm = NonparametricShift()

            _, conv_enc, conv_new_dec,_,known_patch, unknown_patch = Nonparm.buildAutoencoder(inpatch.squeeze(), False, False, nonmask_point_idx,mask_point_idx,  shift_sz, stride)

            output_var = Variable(output)
            tmp1 = conv_enc(output_var)


            maxcoor = MaxCoord()

            kbar, ind, vmax = maxcoor.update_output(tmp1.data, sp_x, sp_y)
            real_patches = kbar.size(1) + torch.sum(ctx.flag)
            vamx_mask=vmax.index_select(0,mask_point_idx)
            _, _, kbar_h, kbar_w = kbar.size()
            out_new = unknown_patch.clone()
            out_new=out_new.zero_()
            mask_num=torch.sum(ctx.flag)

            in_attention=ctx.Tensor (mask_num,real_patches).zero_()

            kbar = ctx.Tensor(1, real_patches, kbar_h, kbar_w).zero_()
            ind_laten=0
            for i in range(kbar_h):
                for j in range(kbar_w):
                    indx = i*kbar_w + j
                    check=torch.eq(mask_point_idx, indx )
                    non_r_ch = ind[indx]
                    offset = ctx.flatten_offsets[non_r_ch]

                    correct_ch = int(non_r_ch + offset)
                    if(check.sum()>=1):
                        known_region=known_patch[non_r_ch]
                        unknown_region=unknown_patch[ind_laten]

                        if ind_laten==0:
                            out_new[ind_laten]=known_region
                            in_attention[ind_laten,correct_ch]=1
                            kbar[:, :, i, j] = torch.unsqueeze(in_attention[ind_laten], 0)
                        elif ind_laten!=0:
                            little_value = unknown_region.clone()
                            ininconv = out_new[ind_laten - 1].clone()
                            ininconv = torch.unsqueeze(ininconv, 0)

                            value_2 = little_value * (1 / (little_value.norm(2) + 1e-8))
                            conv_enc_2 = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False)
                            value_2 = torch.unsqueeze(value_2, 0)
                            conv_enc_2.weight.data = value_2

                            ininconv_var = Variable(ininconv)

                            at_value = conv_enc_2(ininconv_var)
                            at_value_m = at_value.data
                            at_value_m=at_value_m.squeeze()

                            at_final_new = at_value_m / (at_value_m + vamx_mask[ind_laten])
                            at_final_ori = vamx_mask[ind_laten] / (at_value_m + vamx_mask[ind_laten])
                            out_new[ind_laten] = (at_final_new) * out_new[ind_laten - 1] + (at_final_ori) * known_region
                            in_attention[ind_laten]=in_attention[ind_laten-1]*at_final_new.item()
                            in_attention[ind_laten,correct_ch]=in_attention[ind_laten,correct_ch]+at_final_ori.item()
                            kbar[:, :, i, j] = torch.unsqueeze(in_attention[ind_laten], 0)
                        ind_laten+=1
                    else:

                        kbar[:,  correct_ch , i, j] = 1
            kbar_var = Variable(kbar)
            result_tmp_var = conv_new_dec(kbar_var)
            result_tmp = result_tmp_var.data
            output_lst[idx] = result_tmp
            ind_lst[idx] = kbar.squeeze()


        output = output_lst

        ctx.ind_lst = ind_lst
        return output



    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst


        c = grad_output.size(1)



        grad_swapped_all = grad_output.clone()

        spatial_size = ctx.h * ctx.w

        W_mat_all = Variable(ctx.Tensor(ctx.bz, spatial_size, spatial_size).zero_())
        for idx in range(ctx.bz):
            W_mat = W_mat_all.select(0, idx).clone()
            back_attention=ind_lst[idx ].clone()
            for i in range(ctx.h):
                for j in range(ctx.w):
                    indx = i * ctx.h + j
                    W_mat[indx] = back_attention[:,i,j]


            W_mat_t = W_mat.t()

            # view(c/3,-1):t() makes each line be a gradient of certain position which is c/3 channels.
            grad_swapped_weighted = torch.mm(W_mat_t, grad_swapped_all[idx].view(c , -1).t())

            # Then transpose it back
            grad_swapped_weighted = grad_swapped_weighted.t().contiguous().view(1, c , ctx.h, ctx.w)
            grad_swapped_all[idx] = torch.add(grad_swapped_all[idx], grad_swapped_weighted.mul(ctx.triple_w))

        # note the input channel and the output channel are all c, as no mask input for now.
        grad_input =grad_swapped_all

        return grad_input, None, None, None, None, None, None, None, None, None, None
