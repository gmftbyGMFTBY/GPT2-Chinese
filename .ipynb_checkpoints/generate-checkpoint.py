import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from pytorch_transformers import GPT2LMHeadModel
import ipdb
import numpy as np
from tqdm import tqdm
import random
import pickle


def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True



def normalization(data):
    _range = np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True)
    return (data - np.min(data, axis=1, keepdims=True)) / _range


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, context, length, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0,
                    device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]


def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu'):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return generate


# 通过命令行参数--fast_pattern，指定模式
def generate(n_ctx, model, context, length, tokenizer, temperature=1, top_k=0, top_p=0.0, repitition_penalty=1.0, device='cpu',
             is_fast_pattern=False):
    if is_fast_pattern:
        return fast_sample_sequence(model, context, length, temperature=temperature, top_k=top_k, top_p=top_p,
                                    device=device)
    else:
        return sample_sequence(model, context, length, n_ctx, tokenizer=tokenizer, temperature=temperature, top_k=top_k, top_p=top_p,
                               repitition_penalty=repitition_penalty, device=device)
    
    
# ========== my own class and function ========== #
class tool:
    
    '''
    The tool for computing the Mutual Information and the SRF function
    '''
    
    def __init__(self, model, tokenizer, maxlen=15):
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        print('Init the tool class over')
    
    def process_raw_text(self, raw_text):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(raw_text))
    
    def pad_seq(self, response):
        # response: [batch, l]
        seq, seql = [], []
        pmaxlen = min(self.maxlen, max([len(i) for i in response]))
        for i in response:
            if len(i) >= pmaxlen:
                seq.append(i[:pmaxlen])
                seql.append(pmaxlen)
            else:
                seq.append(i + ['unk'] * (pmaxlen - len(i)))
                seql.append(len(i))
        return seq, seql
        
    def possibility_sentence(self, context, response, temperature=1, batch_size=1):
        '''
        :param response: list of [response is the list of the token string]
        :param context: context maybe the empty
        '''
        # pad
        response, seql = self.pad_seq(response)    # [batch, l]
        # serlized the string to tensor
        if context:
            # append the split token between the context and response
            # context: [batch, l_context]
            context += '。'
            context = self.process_raw_text(context)[-self.maxlen:]    # [l_context]
            context = [context for i in range(batch_size)]    # [batch, l_context]
        else:
            context = []    # [batch, 1]
            for i in response:
                kk = self.process_raw_text(i[0])
                if len(kk) > 1: kk = [kk[0]]
                context.append(kk)
            response = [i[1:] for i in response]    # [batch, l]
            seql = [i-1 for i in seql]    # [l]
        inputs = torch.LongTensor(context).cuda()
        length = len(response[0])    # [length]
        
        if len(context[0]) > 1:
            _, past = self.model(inputs[:, :-1], None)[:2]
            prev = inputs[:, -1].view(-1, 1)
        else:
            past = None
            prev = inputs
        # ipdb.set_trace()
        possibility = []
        with torch.no_grad():
            for i in range(length):
                output = self.model(prev, past=past)
                output, past = output[:2]     # output [128, 1, 13317]
                output = output.squeeze(1) / temperature    # [128, 13317]
                next_tokens = []
                for res in response:
                    kk = self.tokenizer.convert_tokens_to_ids(res[i])
                    if isinstance(kk, list): kk = kk[0]
                    next_tokens.append(kk)
                x = list(range(batch_size))
                output = torch.softmax(output, dim=-1)
                p = output[x, next_tokens]    # [batch]
                for k in range(p.shape[0]):
                    if p[k] < 1e-7:
                        p[k] = 1e-7
                possibility.append(p)
                next_tokens = torch.tensor(next_tokens).cuda()
                next_tokens = next_tokens.view(-1, 1)    # [128, 1]
                prev = next_tokens    # [128, 1]

        # [batch, length]
        possibility = torch.stack(possibility).transpose(0, 1)
        return possibility, seql
    
    def p_t_s(self, response, context=None, batch_size=128):
        # response and context are the raw text
        # be careful of the case that only contains one character.
        res = []
        for i in response:
            i = list(i)
            if len(i) == 1: i = ['unk', i[0]]
            res.append(i)
        response = res
        
        p, l = self.possibility_sentence(context, response, batch_size=batch_size)    # [batch, length], l: [batch]
        
        for i in range(len(l)):
            p[i, l[i]:] = 1.0
            
        pause = torch.prod(p, 1)    # [batch]
        for i in range(pause.shape[0]):
            if pause[i] == 0:
                pause[i] = 1e-45
        # ipdb.set_trace()
        return pause
    
    
def load_txt(file):
    dataset = []
    with open(file) as f:
        for line in f.readlines():
            if not line:
                line = '哈哈'
            dataset.append(''.join(line.strip().split()))
    print(f'[!] load file from {file} over')
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=-1, type=int, required=False, help='生成长度')
    parser.add_argument('--batch_size', default=128, type=int, required=False, help='生成的batch size')
    parser.add_argument('--nsamples', default=10, type=int, required=False, help='生成几个样本')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='model/final_model/config.json', type=str, required=False,
                        help='模型参数')
    # use the vocab.txt
    parser.add_argument('--tokenizer_path', default='cache/vocab.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
    parser.add_argument('--prefix', default='<s>', type=str, required=False, help='生成文章的开头')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--fast_pattern', action='store_true', help='采用更加快的方式生成文本')
    parser.add_argument('--save_samples', action='store_true', help='保存产生的样本')
    parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.no_wordpiece:
        from tokenizations import tokenization_bert_without_wordpiece as tokenization_bert
    elif args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    length = args.length
    batch_size = args.batch_size
    nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.cuda()
    model.eval()
            
    t = tool(model, tokenizer, maxlen=15)
    folder, sss = 'xiaohuangji', 256
    
    # load the dialog data
    src, tgt = load_txt(f'./data/{folder}/src-train.txt'), load_txt(f'./data/{folder}/tgt-train.txt')
    
    # sample 256 sentence from target dataset that is very short
    tgtlength = [-len(i) if len(i) <= 8 else -np.inf for i in tgt]
    pp = torch.softmax(torch.tensor(tgtlength, dtype=torch.float), dim=0).numpy()
    sidx = np.random.choice(list(range(len(tgt))), sss, p=pp)
    
    ptgt = []
    for i in sidx:
        ptgt.append(tgt[i])
    tgt = ptgt
    sl, tl = [len(i) for i in src], [len(i) for i in tgt]
    print(f'[!] sl avg length: {round(np.mean(sl), 4)}, tl avg length: {round(np.mean(tl), 4)}')
    
    # compute the weight matrix: [45000, 45000]
    pt_matrix = np.zeros([len(tgt)])    # [m]
    pts_matrix = np.zeros([len(src), len(tgt)])    # [n, m]
    pst_matrix = np.zeros([len(tgt), len(src)])    # [m, n]
    print(f'[!] pt matrix shape: {pt_matrix.shape}')
    print(f'[!] pts matrix shape: {pts_matrix.shape}')
    print(f'[!] pst matrix shape: {pst_matrix.shape}')
    
    # pt, the possibility of each sentence (128)
    for i in tqdm(range(0, len(tgt), batch_size)):
        batch = tgt[i:i+batch_size]
        pt_matrix[i:i+batch_size] = t.p_t_s(batch, context=None,
                                            batch_size=len(batch)).cpu().numpy()
    
    # pts, p(t|s) the samples in the batch have the same context
    for i in tqdm(range(len(src))):
        for j in range(0, len(tgt), batch_size):
            batch, context = tgt[j:j+batch_size], src[i]
            pts_matrix[i, j:j+batch_size] = t.p_t_s(batch, context=context, batch_size=len(batch)).cpu().numpy()
    
    
    # pst, p(s|t) the reverse possibility computing
    for i in tqdm(range(len(tgt))):
        for j in range(0, len(src), batch_size):
            batch, context = src[j:j+batch_size], tgt[i]
            pst_matrix[i, j:j+batch_size] = t.p_t_s(batch, context=context, batch_size=len(batch)).cpu().numpy()
    
    with open(f'./data/{folder}/PT.pkl', 'wb') as f:
        pickle.dump(pt_matrix, f)
        
    with open(f'./data/{folder}/PTS.pkl', 'wb') as f:
        pickle.dump(pts_matrix, f)
        
    with open(f'./data/{folder}/PST.pkl', 'wb') as f:
        pickle.dump(pst_matrix, f)
        
    pt_matrix = normalization(np.log(pt_matrix).reshape(1, -1)).reshape(-1)
    pts_matrix = normalization(np.log(pts_matrix))
    pst_matrix = normalization(np.log(pst_matrix).T)
    
    # ignore the zero
    pt_matrix += 1e-20
    pts_matrix += 1e-20
    pst_matrix += 1e-20
    
    SRF = 2 * pst_matrix / (pt_matrix + pts_matrix)
    # fix the 128 wrong case in the SRF matrix
    for i in range(len(sidx)):
        SRF[sidx[i], i] = -np.inf     # ban it
        
    with open(f'./data/{folder}/SRF.pkl', 'wb') as f:
        pickle.dump([sidx, SRF], f)
    print(f'[!] save file into ./data/{folder}/SRF_matrix, shape: {SRF.shape}')


if __name__ == '__main__':
    main()
        
        