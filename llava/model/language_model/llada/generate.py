import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_num_transfer_tokens_sch(mask_index, steps,schedule=None,schedule_kwargs=None):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    if schedule is None:
        return get_num_transfer_tokens(mask_index,steps)
    if schedule_kwargs is None:
        schedule_kwargs = {}
   
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = int(min(steps,mask_num[0]))
    t = torch.linspace(0, 1, steps+1)
    # at least one sample per step
    if schedule =='logit_normal':
      sigmas = sigmoid_normal_cdf(t)
    elif schedule =='shift':
      sigmas = logit_normal_schedule(schedule_kwargs.get('shift',3),t)
    elif schedule == 'cosine':
        sigmas = cosine_schedule(t)
    else:
      sigmas = t
    sigmas = sigmas.to(mask_num.device)
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    
    for i in range(mask_num.size(0)):
      # print(sigmas.shape)
      sigmas_sample = (sigmas*mask_num[i]).to(torch.int64)
      # print(sigmas_sample)
      sigmas_sample = sigmas_sample[1:]-sigmas_sample[:-1]
      # print(sigmas_sample)
      # fix detal
      sigmas_sample = torch.clamp(sigmas_sample,1,None) # should only increase
      delta = sigmas_sample.sum() - mask_num[i]
    #   breakpoint()
      assert delta>=0
      j = 0
      
      while delta > 0:
        j = j % len(sigmas_sample) 
        if sigmas_sample[j] == 1:
          j += 1
          continue
        
        delta -= 1
        sigmas_sample[j] -= 1
        j += 1
    #   breakpoint()
      assert sigmas_sample.sum()==mask_num[i]
      num_transfer_tokens[i] = sigmas_sample#.to(torch.int64)
    return num_transfer_tokens.flip(-1)

def linear(y):
    return y

def cosine_schedule(x):
    """
    Cosine schedule mapping [0, 1] -> [1, 0]
    """
    x = np.clip(x, 0, 1)
    return 1-0.5 * (1 + np.cos(np.pi * x))

def sigmoid_normal_cdf(y):
    # y must be in (0, 1)
    logit_y = torch.log(y / (1 - y))
    return 0.5 * (1 + torch.erf(logit_y / torch.sqrt(torch.tensor(2.0))))
def logit_normal_schedule(shift,sigmas):
    # shift = 1 / shift
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas
import os
DEBUG_PRINT_OUTPUT = os.environ.get('DEBUG_PRINT_OUTPUT',False)
@ torch.no_grad()
def generate(model, prompt=None, steps=None, max_new_tokens=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,inputs_embeds=None, position_ids=None,attention_mask=None,
              tokenizer=None,
                verbose=False,
                step_per_block=None,
                prefix_lm=False,
                schedule=None,
                schedule_kwargs=None,
                draft_tokens=None,
                step_ratio=None,
                image_position=None, #changed!
                num_visual_tokens=None, #changed!
             **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    #change for ablation study !!!
    # max_new_tokens = 2 # ai2d:2, realworldqa:2, infovqa_val:32, docvqa_val:32, mme:2, mmbench_en_dev:2
    block_length = max_new_tokens
    visual_pruned = False
    # steps = max_new_tokens # min(steps,max_new_tokens)
    # steps = steps/2 # for docvqa_val and infovqa_val
    # if step_ratio:
    #     steps = int(max_new_tokens*step_ratio)
    gen_length = max_new_tokens
    assert position_ids is None
    if prompt is None:
        assert inputs_embeds is not None
        bsz, seq_len = inputs_embeds.shape[:2]
        prompt = torch.full((bsz, seq_len), 0, dtype=torch.long).to(model.device)
    past_key_values = None
    if prefix_lm:
        past_key_values = model(None,input_embeddings=inputs_embeds,use_cache=True).attn_key_values
        # breakpoint()
        x = torch.full((bsz, gen_length), mask_id, dtype=torch.long).to(model.device)
        prompt = torch.full((bsz, 0), 0, dtype=torch.long).to(model.device)
        # x[:, :prompt.shape[1]] = prompt.clone()
    else:
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    prompt_len = prompt.shape[1] # changes
    # assert prompt.shape[0] == 1
    if draft_tokens is not None:
        assert draft_tokens.shape[1] <= gen_length
        x[:, prompt.shape[1]:prompt.shape[1]+draft_tokens.shape[1]] = draft_tokens.clone()

    # if block_length < gen_length:
    #    block_length = gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert ( steps % num_blocks == 0) or step_per_block is not None
    steps = steps // num_blocks
  
    if step_per_block:
        steps = min(step_per_block,block_length)
        assert step_ratio is None, 'Please do not pass both step_ratio and step_per_block'
    # step_ratio = 0.5
    # schedule = 'shift'
    # schedule_kwargs = dict(shift=3)
    # breakpoint()
    if step_ratio:
        steps = int(steps*step_ratio)

    # print(steps,step_per_block,block_length,draft_tokens.shape[-1])
    # NFE = 0
    if verbose:
        history = []
    # print(f"[DEBUG] Original x shape: {x.shape}")
    # print("steps:",steps)
    # print("num_blocks:",num_blocks)
    # print("gen_length:",gen_length)
    # print("block_length:",block_length)
    # exit()
    print()
    for num_block in range(num_blocks):

        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_sch(block_mask_index, steps,schedule=schedule,schedule_kwargs=schedule_kwargs)
        if DEBUG_PRINT_OUTPUT:
            print(f"Block: {num_block + 1}/{num_blocks}, Steps per Block: {steps}, Block Length: {block_length}")
            print(f"Tokens generated per step {num_transfer_tokens[0]}")
        
        for i in range(steps):
            # print(i)
            mask_index = (x == mask_id)
            block_mask_index = mask_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:]
            # print(mask_index.sum())
            if block_mask_index.sum() == 0:
                continue
            # NFE += 2
            if cfg_scale > 0.:
                assert NotImplementedError('cfg_scale > 0. is not supported.')
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                #
                logits = model(x_,input_embeds_inference=[inputs_embeds,None]).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                inputs_embeds_curr = model.transformer.wte(x)
                #print(tokenizer.batch_decode(x)[0].replace('<|endoftext|>',''))
                # print((x==mask_id).sum())
                # breakpoint()
                if prefix_lm:
                    # breakpoint()
                    logits = model(None,input_embeddings=inputs_embeds_curr,past_key_values=past_key_values).logits
                else:
                    #model will go this way
                    # if inputs_embeds is not None:
                    #     inputs_embeds_curr[:,:inputs_embeds.shape[1]] = inputs_embeds
                    #change!
                    if (num_block == 0 and i == 0 and (not visual_pruned)and image_position is not None and num_visual_tokens is not None and num_visual_tokens > 0):
                        output = model(None,input_embeddings=inputs_embeds_curr, output_attentions = True)
                        attn_mean_map = torch.stack(output.attentions, dim=0).mean(dim=(0, 2)).squeeze(0)
                    else:
                        output = model(None,input_embeddings=inputs_embeds_curr, output_attentions = False)
                    # output = model(None,input_embeddings=inputs_embeds_curr, output_attentions = True)
                    # print(output.logits.shape)
                    # print(len(output.attentions),output.attentions[0].shape)
                    # exit()
                    logits = model(None,input_embeddings=inputs_embeds_curr).logits
            # logits = logits.cpu()
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'entrophy':
                epsilon = 1e-10
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                log_probs = torch.log(probs + epsilon)
                x0_p = torch.sum(probs * log_probs, dim=-1)
            elif remasking == 'margin':
                ## similar to margin algo in Dream
                p = F.softmax(logits.to(torch.float64), dim=-1)
                sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                top1_probs = sorted_probs[:, :, 0] 
                top2_probs = sorted_probs[:, :, 1] 
                x0_p = top1_probs - top2_probs 
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                try:
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                except:
                    breakpoint()
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            if verbose:
                history.append(x.clone().cpu())
            #change!
            if (num_block == 0 and i == 0 and (not visual_pruned)and image_position is not None and num_visual_tokens is not None and num_visual_tokens > 0):
                remask_indices = (x == mask_id)[0]  # [T]
                attn_avg = attn_mean_map
                vis_start, vis_end = image_position, image_position + num_visual_tokens
                if remask_indices.numel() > 0:
                    if vis_end > vis_start:
                        response_attn = attn_avg[remask_indices][:, vis_start:vis_end]  # [N_remask, N_vis]
                        vis_importance = response_attn.mean(dim=0)                       # [N_vis]

                        # D) prune 50% lowest-importance visual tokens
                        keep_ratio = 0.50
                        keep_n = max(1, int((vis_end - vis_start) * keep_ratio))
                        _, keep_local_idx = torch.topk(vis_importance, k=keep_n, largest=True)
                        keep_local_idx = torch.sort(keep_local_idx).values  # ascending
                        # Build a keep-mask over TOTAL sequence positions
                        T = x.shape[1]
                        global_keep = torch.ones(T, dtype=torch.bool, device=x.device)
                        # first drop all visual tokens
                        global_keep[vis_start:vis_end] = False
                        # then re-enable the kept ones
                        global_keep[vis_start + keep_local_idx] = True
                        # E) apply slicing to all sequence-aligned tensors
                        x = x[:, global_keep]
                        prompt_keep = global_keep[:prompt.shape[1]]
                        prompt = prompt[:, prompt_keep]
                        inputs_embeds = inputs_embeds[:, prompt_keep, :]
                        prompt_index = (x != mask_id)
                        prompt_len = prompt.shape[1]
                        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
                        # F) update accounting variables after pruning
                        pruned_count = int((vis_end - vis_start) - keep_n)
                        num_visual_tokens = keep_n
                        visual_pruned = True
                        print(f"[Visual Pruning] Block {num_block} Step {i}: pruned {pruned_count} visual tokens "
                                f"(kept {keep_n}/{vis_end - vis_start}).")
                    
    # breakpoint()
    # print(f"NFE: {NFE} Num Blocks: {num_blocks}")
    if verbose:
        return x,history
    print(f"[DEBUG] Final x shape: {x.shape}")
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
   

if __name__ == '__main__':
    main()