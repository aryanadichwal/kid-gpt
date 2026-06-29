import torch
import torch.nn as nn
from torch.nn import functional as F


#hyperparameters 
batch_size = 64  #How many "practice sentences" the model studies at the same time 
block_size = 256  #How many characters the model can see at once when predicting the next character
max_iters = 5000  #How many times the model will practice and update itself
eval_interval = 500 #Every 500 iterations, check how well the model is doing
learning_rate = 3e-4  #How big of a step the model takes when correcting its mistakes
device="cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
eval_iters = 200 # iterations to estimate the loss
n_embd =384  #Each character gets converted into a list of 384 numbers that capture its meaning
n_head = 6  #The attention mechanism is split into 6 parallel "readers", each focusing on different patterns in the same text.
n_layer = 6  #How many transformer blocks are stacked on top of each other
dropout = 0.2 #while training, randomly switch off 20% of neurons to prevent the model from memorizing instead of actually learning

print(f"Using device: {device}")



torch.manual_seed(1337)

with open ( "toddler_dataset.txt", "r", encoding="utf-8") as f:
    text=f.read()

#here are all the unique characters that occur in this text
chars=sorted(list(set(text)))
vocab_size=len(chars) #size of vocabulary

#creating a mapping from characters to integers
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s: [stoi[c] for c in s]
decode=lambda l: "".join([itos[i]for i in l])


# creating train data and validation data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) #first 90% will be train data

train_data= data[:n]
val_data=data[n:]


#creating data loader
def get_batch(split):
    #generating small batch of data of inputs X and targets Y
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # torch.randint( high , size ) 
    x = torch.stack([data[i:i+block_size] for i in ix]) # gives the input sequence of length block_size
    y = torch.stack([data[i+1:i+1+block_size]for i in ix]) # gives the target sequence of length block_size
    x , y = x.to(device), y.to(device) #moves tensores to GPU if available
    return x,y




class Head(nn.Module):
   # one head of self attention ( makes token aware of other tokens in the context) 

    def __init__(self, head_size):
       super().__init__()
       self.key = nn.Linear(n_embd, head_size, bias=False)
       self.query= nn.Linear(n_embd, head_size, bias=False)
       self.value= nn.Linear(n_embd, head_size, bias=False)
       self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) #lower triangular matrix of ones

       self.dropout=nn.Dropout(dropout)

    def forward(self, x):
       B,T,C = x.shape
       k=self.key(x) #(B,T,head_size)
       q=self.query(x)  

       wei = q @ k.transpose(-2,-1) * C**-0.5 #(B,T,head_size) @ (B,head_size,T) => (B,T,T)  
       wei = wei.masked_fill(self.tril[:T,:T]==0, float("-inf")) #masking the upper triangular part of the matrix 
       wei = F.softmax(wei, dim=-1)
       wei = self.dropout(wei)
       v=self.value(x) #(B,T,head_size) 
       out = wei @ v  #Take the value vectors and mix them together according to the attention weights.
       return out # out is attention weight 

class MultiHeadAttention(nn.Module):
    # multiple heads of self attention in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()  
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)] )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    

    

class FeedForward(nn.Module): # To make tokens independently think on the data they gathered 
    # a simple linear layer

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd ),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)
    

class block(nn.Module):
    # transformer block: communication followed by computation 

    def __init__(self,n_embd,n_head):
        # n_head: the number of heads we'd like 
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    
        x = x + self.ffwd(self.ln2(x))  
        return x
    





# Simple Bigram Language Model
class Bigramlanguagemodel(nn.Module):
    def __init__(self):
        super().__init__()
        #creating a lookup table for each token
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) #embedding layer that maps each token to a vector of size n_embd
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.block = nn.Sequential(*[block(n_embd, n_head=n_head) for _ in range (n_layer)])
        # self.sa_heads = MultiHeadAttention(4 , n_embd//4) #4 heads of 8-dimensional self attention // sa( self attention)
        # self.ffwd = FeedForward(n_embd) 
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size) #linear layer that maps the embedding vector to a vector of size vocab_size // lm( )



    def forward(self,idx,targets=None):
        B,T = idx.shape 

        #Creating a tensor of logits(prediction scores) for every possible next token
        tok_embd = self.token_embedding_table(idx) #(Batch, Time, Channels(n_embd))
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) #(Time, Channels(n_embd))
        x = tok_embd + pos_embd #(Batch, Time, Channels(n_embd))
        x = self.block(x) # apply one head of self attention 
        x = self.ln_f(x)
        logits=self.lm_head(x) #(Batch, Time, Channels(vocab_size))

        if targets == None:
            loss=None
        else:
            B ,T ,C = logits.shape
            logits = logits.view(B*T,C) #Stretching (B,T,C) 3d to 2d
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits, loss 
  
    def generate(self, idx, max_new_tokens): #to take (B,T) and make it (B,T+1)=>(B,T+2) as many as max tokens
        #idx (B,T) is array of current contex
        for _ in range(max_new_tokens):
        
            idx_cond=idx[:, -block_size:] #getting the last block_size tokens from the context
            logits, loss = self(idx_cond) #getting the predictions for the last block_size tokens
            logits = logits[:, -1, :] #focuses only on the most recent token for generating the next token // becomees (B,C)
            
            # applying softmax to get probabilities
            probs=F.softmax(logits, dim=1)

            #sample from distrtibution
            idx_next = torch.multinomial(probs, num_samples=1)#(B,1)

            #appending sampled index to running sequence
            idx = torch.cat((idx,idx_next),dim=1)#(B,T+1)
        return idx
    
model=Bigramlanguagemodel()
m=model.to(device)



@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ["train", "val"]:
        losses=torch.zeros(eval_iters)  #starts as tensor([0., 0., 0., 0., 0.])
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k]=loss.item() #converts it into a regular Python number
        out[split]=losses.mean() #mean of all the losses
    model.train()
    return out



#creating optimizer basically a function that updates the weights of the model based on the gradients computed during backpropagation
optimizer=torch.optim.AdamW(m.parameters(),lr=learning_rate)

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
      losses=estimate_loss()
      print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses['val']:.4f}")


    #sample a batch of data
    xb, yb = get_batch("train")
    

    #evaluate the loss
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # figures out what mistakes were made.
    optimizer.step() #adjusts The understanding


#generate from the model
context= torch.zeros((1,1), dtype=torch.long, device=device) #starting with a zero tensor
# print(decode(m.generate(context,max_new_tokens=500)[0].tolist())) #generating
open('output.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

