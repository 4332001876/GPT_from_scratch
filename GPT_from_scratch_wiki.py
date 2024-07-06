# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List
from transformers import AutoTokenizer
from tqdm import tqdm
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_kaggle_env = True

class Config:
    pretrained_tokenizer_name = "./qwen_tokenizer" # "google-bert/bert-base-uncased"
    input_file = "input.txt"
    save_model_path = "./"

if is_kaggle_env:
    Config.pretrained_tokenizer_name = "./qwen_tokenizer" # "openai-community/gpt2"
    Config.input_file = "./tmp/input.txt"
    Config.save_model_path = "./tmp/"


# %%
class Tokenizer:
    def __init__(
        self,
        dataPath:str
        ):
        with open(dataPath,"r",encoding="utf-8") as f:
            self.dataset = f.read()
        self.generate_vocabulary()

    def generate_vocabulary(
        self,
        ):
        self.char2index = {}
        self.index2char = {}
        """
        TODO:
        """
        # token 0 is reserved for padding
        self.char2index["[pad]"] = 0
        self.index2char[0] = "[pad]"

        for c in self.dataset:
            if c not in self.char2index:
                self.char2index[c] = len(self.char2index)
                self.index2char[len(self.index2char)] = c

    def encode(
        self,
        sentence : str,
        ) -> torch.Tensor:
        """
        TODO:
        例子, 假设A-Z 对应的token是1-26, 句子开始，结束符号的token是0。
        input  : "ABCD"
        output : Tensor([0,1,2,3]) 

        注意: 为了后续实验方便，输出Tensor的数据类型dtype 为torch.long。
        """
        return torch.tensor([self.char2index[c] for c in sentence], dtype=torch.long)

    def decode(
        self,
        tokens : torch.Tensor,
        ) -> str:
        """
        TODO:
        例子, 假设A-Z 对应的token是1-26, 句子开始，结束符号的token是0。
        input : Tensor([0,1,2,3]) 
        output : "ABCD"
        """
        return "".join([self.index2char[i] for i in tokens if i != 0])
    
    def __len__(self):
        return len(self.char2index)
    
class PretrainedTokenizer:
    def __init__(
        self,
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.pretrained_tokenizer_name)

    def encode(
        self,
        sentence : str,
        ) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(sentence, add_special_tokens=True), dtype=torch.long)
    
    def decode(
        self,
        tokens : torch.Tensor,
        ) -> str:
        return self.tokenizer.decode(tokens)
    
    def __len__(self):
        return len(self.tokenizer)

def test_tokenizer():
    sentence = "hello world"
    tokenizer = Tokenizer(Config.input_file)
    tokens = tokenizer.encode(sentence)
    print("Test Tokenizer:")
    print("Input Sentence:", sentence)
    print("Tokenized Sentence:", tokens)
    print("Decoded Sentence:", tokenizer.decode(tokens))
    assert tokenizer.decode(tokens) == sentence
    print("Test Tokenizer Passed")

    pretrained_tokenizer = PretrainedTokenizer()
    tokens = pretrained_tokenizer.encode(sentence)
    print("Test Pretrained Tokenizer:")
    print("Input Sentence:", sentence)
    print("Tokenized Sentence:", tokens)
    print("Decoded Sentence:", pretrained_tokenizer.decode(tokens))
    assert pretrained_tokenizer.decode(tokens) == sentence
    print("Test Pretrained Tokenizer Passed")

# test_tokenizer()

# %%
class ShakespeareDataset(Dataset):
    def __init__(self, filepath, tokenizer, chunk_size):
        self.tokenizer = tokenizer
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        self.encoded = self.tokenizer.encode(text)
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.encoded) - self.chunk_size # 这里还多减去了一个1哈，因为最后一个字符是没有下一个字符作为label的

    def __getitem__(self, idx):
        # TODO: 提取一段文本(长度为 chunk_size）作为输入，以及这段文本的每一个字符的下一个字符作为标签
        # example(not correspond to real text): chunk = tensor([ 0, 20, 49, 58, 59])
        #         label = tensor([20, 49, 58, 59, 19])
        # decoded chunk: "The "
        # decoded label: "he T"

        chunk = self.encoded[idx:idx+self.chunk_size]
        label = self.encoded[idx+1:idx+self.chunk_size+1]
        return chunk, label

class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, chunk_size):
        from modelscope.msdatasets import MsDataset
        ds =  MsDataset.load('AI-ModelScope/wikipedia-cn-20230720-filtered', subset_name='default', split='train')
        self.text_list = []
        for item in ds:
            self.text_list.append(item['completion'])
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.encoded_list = []

        for i, text in enumerate(self.text_list):
            log_step = 1000
            if i % log_step == log_step-1:
                print(f"Processed {i+1} documents")
            self.encoded_list.append(self.tokenizer.encode(text))

        self.item_slice = [] # (doc_idx, start_idx, end_idx)
        for i, encoded in enumerate(self.encoded_list):
            if len(encoded) > self.chunk_size:
                for j in range(0, len(encoded)-self.chunk_size):
                    self.item_slice.append((i, j, j+self.chunk_size))
            # else:
                # self.item_slice.append((encoded, 0, len(encoded)-1))
    
    def __len__(self):
        return len(self.item_slice)
    
    def __getitem__(self, idx):
        doc_idx, start_idx, end_idx = self.item_slice[idx]
        encoded = self.encoded_list[doc_idx]
        chunk = encoded[start_idx:end_idx]
        label = encoded[start_idx+1:end_idx+1]
        return chunk, label

# tokenizer = Tokenizer(dataPath=Config.input_file)
tokenizer = PretrainedTokenizer()
# enc = tiktoken.get_encoding(Config.input_file)
# enc = tiktoken.encoding_for_model("gpt-4o")

def create_dataloader(filepath, tokenizer, chunk_size, batch_size, shuffle=True):
    # dataset = ShakespeareDataset(filepath, tokenizer, chunk_size)
    dataset = WikiTextDataset(tokenizer, chunk_size)
    train_dataset,val_dataset = torch.utils.data.random_split(dataset,[int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, val_dataloader


# train_dataloader,val_dataloader = create_dataloader(Config.input_file, tokenizer, chunk_size=200, batch_size=2)


# %%
class HeadAttention(nn.Module):
    def __init__(self, seq_len:int, embed_size:int, hidden_size:int):
        super().__init__()
        # embed_size: dimension for input embedding vector
        # hidden_size: dimension for hidden vector. eg. x:(..., embed_size) --to_q--> query_vector:(..., hidden_size)

        # a triangular bool matrix for mask
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))
        
        # TODO: init three matrix, to_q, to_k, to_v.
        self.to_q = nn.Linear(embed_size, hidden_size)
        self.to_k = nn.Linear(embed_size, hidden_size)
        self.to_v = nn.Linear(embed_size, hidden_size)

    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size)
        # return (batch_size, seq_len, hidden_size)
        # TODO: implement the attention mechanism
        
        q = self.to_q(inputs) # (batch_size, seq_len, hidden_size)
        k = self.to_k(inputs) # (batch_size, seq_len, hidden_size)
        v = self.to_v(inputs) # (batch_size, seq_len, hidden_size)
        attention_score = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5) # 这里除了个根号d_k，忠于原文, (batch_size, seq_len, seq_len)
        attention_score = attention_score * self.tril[:attention_score.size(1), :attention_score.size(2)]
        attention_score = F.softmax(attention_score, dim=-1)
        attention_output = torch.matmul(attention_score, v)
        
        return attention_output

# %%
class MultiHeadAttention(nn.Module):
    # MultiHeadAttention is consist of many HeadAttention output.
    # concat all this head attention output o_i, then merge them with a projection matrix W_o, as [o_1, o_2, ...] x W_o
    # The reason for using multi-head attention is that we want each head to be able to extract different features
    def __init__(self, n_heads:int, head_size:int, seq_len:int, embed_size:int):
        # n_heads is the number of head attention
        # head_size is the hidden_size in each HeadAttention
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size

        # head_size = embed_size // n_heads
        #TODO: implement heads and projection
        # a triangular bool matrix for mask
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

        # 为了计算效率，不采用ModuleList方案
        self.to_q = nn.Linear(embed_size, n_heads * head_size)
        self.to_k = nn.Linear(embed_size, n_heads * head_size)
        self.to_v = nn.Linear(embed_size, n_heads * head_size)

        self.to_o = nn.Linear(n_heads * head_size, embed_size)

        self.attn_dropout = nn.Dropout(0.1)
        self.to_o_dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size), make sure embed_size=n_heads x head_size
        # return: (batch_size, seq_len, embed_size)
        # TODO:
        q = self.to_q(inputs).view(inputs.size(0), inputs.size(1), self.n_heads, -1).transpose(1, 2) # (B, S, H, E) -> (B, H, S, E)
        k = self.to_k(inputs).view(inputs.size(0), inputs.size(1), self.n_heads, -1).transpose(1, 2) # (B, S, H, E) -> (B, H, S, E)
        v = self.to_v(inputs).view(inputs.size(0), inputs.size(1), self.n_heads, -1).transpose(1, 2) # (B, S, H, E) -> (B, H, S, E)
        attention_score = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attention_score = attention_score.masked_fill(self.tril[:,:,:attention_score.size(2), :attention_score.size(3)] == 0, float('-inf'))
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.attn_dropout(attention_score)
        attention_output = torch.matmul(attention_score, v) # (B, H, S, E)
        attention_output = attention_output.transpose(1, 2).contiguous().view(inputs.size(0), inputs.size(1), -1) # (B, H, S, E) -> (B, S, H, E) -> (B, S, E)
        
        return self.to_o_dropout(self.to_o(attention_output))


# %%
class Expert(nn.Module):
    def __init__(self, embed_size:int):
        super().__init__()
        #TODO: init two linear layer
        self.fc1 = nn.Linear(embed_size, 4*embed_size)
        self.fc2 = nn.Linear(4*embed_size, embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, embed_size)
        # -> mid: (batch_size, seq_len, 4 x embed_size)
        # -> outputs: (batch_size, seq_len, embed_size)
        mid = F.gelu(self.fc1(inputs))
        outputs = self.fc2(mid)
        outputs = self.dropout(outputs)
        return outputs

# %%
# First define the top k router module
class TopkRouter(nn.Module):
    def __init__(self, embed_size, num_experts, active_experts):
        ## TODO
        ## embed_size : dimension of embedding 
        ## num_experts : how many Experts per layer
        ## active_experts: only active_experts out of num_experts are selected to process Embeddings per token.

        ## 初始化参数
        super().__init__()
        self.embed_size = embed_size
        self.num_experts = num_experts
        self.active_experts = active_experts

        self.fc = nn.Linear(embed_size, num_experts)

    def forward(self, inputs):
        ## TODO
        ## 完成这部分时，注意使用Softmax()对router_output做标准化。同时注意这部分所用操作的可导性。
        ## 输入值
        ## inputs is the output tensor from multihead self attention block, shape (B:batch size, T: seq_len, C: embed_size)
        ## 返回值
        ## router_output: normalized weight of Experts, 即教程中的 \alpha
        ## indices:   index of selected Experts, 即教程中的 index
        router_output = self.fc(inputs)
        _, indices = torch.topk(router_output, self.active_experts, dim=-1)
        # print("top_values:", top_values)
        # print("indices:", indices)
        # mask = an tensor of -inf shape like router_output
        # mask the top_values
        mask = torch.zeros_like(router_output)
        mask.scatter_(-1, indices, 1)
        router_output = router_output.masked_fill(mask==0, -float('inf'))
        # mask.scatter(-1, indices, top_values)
        # print("router_output:", router_output)
        router_output = F.softmax(router_output, dim=-1)

        return router_output, indices
    
def test_topk_router():
    embed_size = 8
    num_experts = 4
    active_experts = 2
    router = TopkRouter(embed_size, num_experts, active_experts)
    inputs = torch.randn(1, 3, embed_size)
    router_output, indices = router(inputs)
    print("Test TopkRouter:")
    print("Input shape:", inputs.shape)
    print("Router Output shape:", router_output.shape)
    print("Indices shape:", indices.shape)
    print("Router Output:", router_output)
    print("Indices:", indices)
    print("Test TopkRouter Passed")

# test_topk_router()

# %%
class SparseMoE(nn.Module):
    def __init__(self, embed_size:int, num_experts:int, active_experts:int):
        ## TODO
        super().__init__()
        self.router = TopkRouter(embed_size, num_experts, active_experts)
        self.experts = nn.ModuleList([Expert(embed_size) for _ in range(num_experts)])

    def forward(self, inputs):
        ## TODO
        router_output, indices = self.router(inputs) # router_output: (B, seq_len, num_experts)
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(router_output[:, :, i:i+1] * expert(inputs))
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        final_output = expert_outputs.sum(dim=-1)
        return final_output

# %%
class Block(nn.Module):
    # Transformer basic block, consist of MultiHeadAttention, FeedForward and layer normalization
    def __init__(self, embed_size:int, n_heads:int, seq_len:int, num_experts:int, active_experts:int):
        super().__init__()
        # TODO: implement block structure
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attn = MultiHeadAttention(n_heads, embed_size//n_heads, seq_len, embed_size)
        self.ff = SparseMoE(embed_size, num_experts, active_experts)

    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size)
        #TODO: forward with residual connection
        attn_output = self.attn(self.norm1(inputs)) + inputs
        ff_output = self.ff(self.norm2(attn_output)) + attn_output
        return ff_output

# %%
class SparseMoETransformer(nn.Module):
    # Transformer decoder, consist of 
    # token embedding layer and position_embedding(position_embedding 可以理解为对位置编码，感兴趣的同学可以查阅原文，这里可以看为vocab_len = seq_len的Embedding)
    # a stack of Transformer basic block
    # a layernorm and output linear layer
    def __init__(self, vocab_size:int, seq_len:int, embed_size:int, n_layers:int, n_heads:int, num_experts:int, active_experts:int):
        # vocab_size is the number of word in vocabulary dict
        # seq_len is the sequence length/sentence length
        # embed_size is the embedding vector dimension
        super().__init__()
        # TODO: 
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(seq_len, embed_size)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(embed_size, n_heads, seq_len, num_experts, active_experts) for _ in range(n_layers)])
        self.seq_len = seq_len

        self.next_token_ff = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, inputs, labels=None):
        # labels: the (ground) true output 
        # TODO: implement the forward function of the transformer

        # inputs:(batch_size, seq_len, )
        batch_size, seq_len, = inputs.shape
        # embedding:(batch_size, seq_len, embed_size)
        embedding = self.token_embedding(inputs) + self.position_embedding(torch.arange(seq_len, device=device))
        embedding = self.dropout(embedding)

        # embedding:(batch_size, seq_len, embed_size)
        for block in self.blocks:
            embedding = block(embedding)

        # logits:(batch_size, seq_len, vocab_size)
        logits = self.next_token_ff(embedding)
        
        # compute the loss
        
        if labels is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(batch_size * seq_len, vocab_size)
            labels = labels.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, labels)
        return logits, loss
    def generate(self, inputs, max_new_tokens):
        inputs = torch.tensor(tokenizer.encode(inputs)).unsqueeze(0)
        device = next(self.parameters()).device  
        inputs = inputs.to(device)
        if inputs.size(1) > self.seq_len:
            inputs = inputs[:, :self.seq_len]
        generated = inputs
        for _ in range(max_new_tokens):
            if generated.size(1) > self.seq_len:
                generated_input = generated[:, -self.seq_len:]
            else:
                generated_input = generated
            logits, _ = self.forward(generated_input)
            last_logits = logits[:, -1, :]  
            next_token_ids = torch.argmax(last_logits, dim=-1)  
            next_token_ids = next_token_ids.unsqueeze(-1)  
            generated = torch.cat([generated, next_token_ids], dim=1)  
        return generated


# %%
def train(model, optimizer, dataloader, epoch, device, gradient_accumulation_steps=2):
    def test_generate(prompt, max_new_tokens=256):
        print(tokenizer.decode(model.generate(prompt,max_new_tokens=max_new_tokens)[0].tolist()))
    model.train()
    total_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch} Loss: 0.0000')
    optimizer.zero_grad()
    for i, (inputs, targets) in pbar:
        # TODO: implement the training process, and compute the training loss and validation loss
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits, loss = model(inputs, targets)
        total_loss += loss.item()
        pbar.set_description('Epoch %d Loss: %.4f'%(epoch, loss.item()))

        loss = loss / gradient_accumulation_steps
        loss.backward()
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        log_step = 2500
        if i % log_step == log_step - 1:
            torch.save(model.state_dict(), Config.save_model_path + 'model_ckpt.pth')
            test_generate("I could pick my lance")
            test_generate("To be or not to be")
            test_generate("法国是")
            test_generate("据称，")
            # print(f'Epoch {epoch} Batch {i} Loss: {loss.item()}')

    print(f'Epoch {epoch} Loss: {total_loss / len(dataloader)}')

    return total_loss / len(dataloader)

def validate(model, dataloader, epoch, device):
    model.eval()
    # TODO: 实现验证函数。与训练函数类似，但不需要计算梯度。
    with torch.no_grad():
        total_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch} Validation Loss: 0.0000')
        for i, (inputs, targets) in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, loss = model(inputs, targets)
            total_loss += loss.item()
            pbar.set_description('Epoch %d Validation Loss: %.4f'%(epoch, loss.item()))
        print(f'Epoch {epoch} Validation Loss: {total_loss / len(dataloader)}')
    return total_loss / len(dataloader)
    

# %%
gc.collect()
torch.cuda.empty_cache()

context_len = 64
train_dataloader, valid_dataloader = create_dataloader(Config.input_file, tokenizer, chunk_size=context_len, batch_size=64)
# model = SparseMoETransformer(vocab_size=len(tokenizer.char2index), seq_len=context_len, embed_size=64, n_layers=3, n_heads=8, num_experts=8, active_experts=2).to(device)
model = SparseMoETransformer(vocab_size=len(tokenizer), seq_len=context_len, embed_size=256, n_layers=4, n_heads=8, num_experts=2, active_experts=1).to(device)
# tiny-story-33M: dim=768, heads=16, layers=4, 
# lr = 5e-4, lr_schedule = constant, wd=0.1, adam_beta1=0.9, adam_beta2 = 0.95, context_length=512, batch_size=80, gradient_accumulation_steps=16

# Optimizer 会根据模型的输出和真实标签计算梯度，然后利用反向传播算法更新模型的参数。
# 在本实验中你可以将 Optimizer 视作黑盒，只需要知道如何使用即可。
# 找一个合适的 Optimizer。对不同的任务，模型，最适合的优化器是不一样的，你可以先尝试最常用的 Adam，如果有兴趣可以看看其他的优化器。
# docs see: https://pytorch.org/docs/stable/optim.html 
# AdamW万岁！
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

gc.collect()
torch.cuda.empty_cache()

# model.load_state_dict(torch.load(Config.save_model_path + 'model_19_v2.pth'))

def test_generate(prompt, max_new_tokens=256):
    print(tokenizer.decode(model.generate(prompt,max_new_tokens=max_new_tokens)[0].tolist()))

# 训练模型
def run(model, train_dataloader, valid_dataloader, device, epochs=10):
    train_loss_result = []
    valid_loss_result = []
    for epoch in range(epochs):
        train_loss = train(model, optimizer, train_dataloader, epoch, device, gradient_accumulation_steps=16)
        valid_loss = validate(model, valid_dataloader, epoch, device)
        train_loss_result.append(train_loss)
        valid_loss_result.append(valid_loss)
        print(f'Epoch {epoch} Train Loss: {train_loss}, Valid Loss: {valid_loss}')
        torch.save(model.state_dict(), Config.save_model_path + 'model_%d_v3.pth'%epoch)
        test_generate("I could pick my lance")
        test_generate("To be or not to be")
        test_generate("法国是")
        test_generate("据称，")

    #TODO: 用 matplotlib plot 训练过程中的 loss 变化
    import matplotlib.pyplot as plt
    x_axis = list(range(epochs))
    plt.plot(x_axis, train_loss_result, label='train_loss')
    plt.plot(x_axis, valid_loss_result, label='valid_loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.savefig(Config.save_model_path + 'loss_v3.png')
    print("train loss:", train_loss_result)
    print("valid loss:", valid_loss_result)


is_train = True
if is_train:
    run(model, train_dataloader, valid_dataloader, device, epochs=2)

    # 保存模型
    torch.save(model.state_dict(), Config.save_model_path + 'model_v3.pth')

    model.load_state_dict(torch.load(Config.save_model_path + 'model_v3.pth'))
else:
    test_generate("To be or not to")
    test_generate("法国是")
    test_generate("据称，")



