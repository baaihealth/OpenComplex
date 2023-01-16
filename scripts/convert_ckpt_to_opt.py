#%%
import torch

# %%
ckpt_path = ''
new_ckpt_path = ''
sd = torch.load(ckpt_path)

# %%
def convert_module(module):
    keys = list(module.keys())

    for key in keys:
        if key.endswith('mha.linear_q.weight'):
            if 'template_pointwise_att.mha' in key:
                # c_q != c_k, faster attention not enabled for this layer
                continue
            prefix = key[:-16]

            linear_q = module[f'{prefix}.linear_q.weight']
            linear_k = module[f'{prefix}.linear_k.weight']
            linear_v = module[f'{prefix}.linear_v.weight']

            del module[f'{prefix}.linear_q.weight']
            del module[f'{prefix}.linear_k.weight']
            del module[f'{prefix}.linear_v.weight']

            linear_qkv = torch.cat((linear_q, linear_k, linear_v), 0)
            module[f'{prefix}.linear_qkv.weight'] = linear_qkv

        elif key.endswith('linear_a_p.weight'):
            prefix = key[:-18]

            linear_a_p_weight = module[f'{prefix}.linear_a_p.weight']
            linear_a_p_bias = module[f'{prefix}.linear_a_p.bias']
            linear_a_g_weight = module[f'{prefix}.linear_a_g.weight']
            linear_a_g_bias = module[f'{prefix}.linear_a_g.bias']
            linear_b_p_weight = module[f'{prefix}.linear_b_p.weight']
            linear_b_p_bias = module[f'{prefix}.linear_b_p.bias']
            linear_b_g_weight = module[f'{prefix}.linear_b_g.weight']
            linear_b_g_bias = module[f'{prefix}.linear_b_g.bias']
            linear_g_weight = module[f'{prefix}.linear_g.weight']
            linear_g_bias = module[f'{prefix}.linear_g.bias']

            del module[f'{prefix}.linear_a_p.weight']
            del module[f'{prefix}.linear_a_p.bias']
            del module[f'{prefix}.linear_a_g.weight']
            del module[f'{prefix}.linear_a_g.bias']
            del module[f'{prefix}.linear_b_p.weight']
            del module[f'{prefix}.linear_b_p.bias']
            del module[f'{prefix}.linear_b_g.weight']
            del module[f'{prefix}.linear_b_g.bias']
            del module[f'{prefix}.linear_g.weight']
            del module[f'{prefix}.linear_g.bias']

            linear_a_pg_b_pg_g_weight = torch.cat((linear_a_p_weight, linear_a_g_weight, linear_b_p_weight, linear_b_g_weight,
                    linear_g_weight), 0)
            linear_a_pg_b_pg_g_bias = torch.cat((linear_a_p_bias, linear_a_g_bias, linear_b_p_bias, linear_b_g_bias,
                    linear_g_bias), 0)
            module[f'{prefix}.linear_a_pg_b_pg_g.weight'] = linear_a_pg_b_pg_g_weight
            module[f'{prefix}.linear_a_pg_b_pg_g.bias'] = linear_a_pg_b_pg_g_bias

        elif key.endswith('outer_product_mean.linear_1.weight'):
            prefix = key[:-16]

            linear_1_weight = module[f'{prefix}.linear_1.weight']
            linear_1_bias = module[f'{prefix}.linear_1.bias']
            linear_2_weight = module[f'{prefix}.linear_2.weight']
            linear_2_bias = module[f'{prefix}.linear_2.bias']

            del module[f'{prefix}.linear_1.weight']
            del module[f'{prefix}.linear_1.bias']
            del module[f'{prefix}.linear_2.weight']
            del module[f'{prefix}.linear_2.bias']

            linear_12_weight = torch.cat((linear_1_weight, linear_2_weight), 0)
            linear_12_bias = torch.cat((linear_1_bias, linear_2_bias), 0)
            module[f'{prefix}.linear_12.weight'] = linear_12_weight
            module[f'{prefix}.linear_12.bias'] = linear_12_bias

convert_module(sd['module'])
convert_module(sd['ema']['params'])

# %%
torch.save(sd, new_ckpt_path)

# %%
