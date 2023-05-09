import torch
import torch.nn as nn
from utils import quantize_image, dequantize

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(*[nn.Linear(embed_dim, embed_dim*4), nn.GELU(), nn.Linear(embed_dim*4, embed_dim)])

    def forward(self, x):
        '''
        x - [seq_len, B, embed_dim]
        '''
        # define mask to perform masked attention
        # <todo>
        attn_mask = torch.full((x.shape[0], x.shape[0]), -float("Inf"), dtype=x.dtype).to(x) # [seq_len, seq_len]
        attn_mask = torch.triu(attn_mask, diagonal=1) # [seq_len, seq_len]

        mha_out, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False) # [seq_len, B, embed_dim]
        x = self.ln_1(x + mha_out) # [seq_len, B, embed_dim]
        ff_out = self.ffn(x) # [seq_len, B, embed_dim]
        x = x + ff_out # [seq_len, B, embed_dim]

        return self.ln_2(x) # [seq_len, B, embed_dim]

class GPT2(nn.Module):
    def __init__(self, embed_dim, num_vocab, img_h, img_w, num_blocks, num_heads):
        super().__init__()
        
        self.embed_dim = embed_dim

        self.tokenize = nn.Embedding(num_vocab, embed_dim) # define lookup table with size [num_clustersxembed_dim]
        self.pos_enc = nn.Embedding(img_h*img_w, embed_dim)
        self.sos = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.layers.append(DecoderBlock(embed_dim, num_heads))

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.out = nn.Linear(embed_dim, num_vocab)

    def forward(self, x):
        '''
        x -> [seq_len, B]
        '''

        assert torch.is_tensor(x)
        assert len(x.shape) == 2

        length, batch = x.shape

        # get embeddings for tokens.
        tok_emb = self.tokenize(x) # [seq_len, B, embed_dim]
        assert tok_emb.shape == (length, batch, self.embed_dim)

        # define and append start-of-sequence token to the previous embedding matrix.
        sos_tok_emb = torch.ones(1, batch, self.embed_dim).to(x)*self.sos # [1, B, embed_dim]
        tok_emb = torch.cat([sos_tok_emb, tok_emb[0:-1,:,:]], dim=0) # [seq_len, B, embed_dim]
        assert tok_emb.shape == (length, batch, self.embed_dim)

        # define and add positonal encodings i.e. a learnable embedding for every position.
        position_indices = torch.arange(length).to(x).unsqueeze(-1) # [seq_len, 1]
        pos_enc_emb = self.pos_enc(position_indices).expand_as(tok_emb) # [seq_len, B, embed_dim]
        tok_emb = tok_emb + pos_enc_emb # [seq_len, B, embed_dim]
        assert tok_emb.shape == (length, batch, self.embed_dim)

        # pass it through the decoder block.
        for layer in self.layers:
            tok_emb = layer(tok_emb) # [seq_len, B, embed_dim]

        assert tok_emb.shape == (length, batch, self.embed_dim)

        # pass final output embeddings through linear layer to get logits.
        tok_emb = self.layer_norm(tok_emb) # [seq_len, B, embed_dim]
        logits = self.out(tok_emb) # [seq_len, B, embed_dim]

        return logits
        