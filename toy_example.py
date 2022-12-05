import modeling_crossattn
model=modeling_crossattn.CrossTransformer()
def set_unfreeze_block(model,block_list):
    for name, param in model.named_parameters():
        for block in block_list:#if block in block_list
            if block in name:
                param.requires_grad = True
                break
            else:
                param.requires_grad = False
    return model
block_list=["head","fc_norm","norm1","attn","cross_norm","norm2"]
set_unfreeze_block(model,block_list)
print("1")