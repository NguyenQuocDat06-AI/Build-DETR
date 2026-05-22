import torch
import torch.nn as nn
import copy

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()

        # Multihead Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed Forward Network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation Function
        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """
        src: Tensor feature from CNN Backbone shape [HW, Batch, d_model]
        pos: Tensor positional encoding shape [HW, Batch, d_model]
        """
        
        # Self-Attention
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed Forward Network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """
        src: Tensor feature from CNN Backbone shape [HW, Batch, d_model]
        pos: Tensor positional encoding shape [HW, Batch, d_model]
        """
        for layer in self.layers:
            src = layer(src, pos)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()

        # Multihead Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Multihead Cross-Attention
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed Forward Network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation Function
        self.activation = nn.ReLU()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                pos: torch.Tensor = None, query_pos: torch.Tensor = None,
                ) -> torch.Tensor:
        """
        tgt: Tensor query shape [Query_num, Batch, d_model]
        memory: Tensor encoded feature from Encoder shape [HW, Batch, d_model]
        pos: Tensor positional encoding shape [HW, Batch, d_model]
        query_pos: Tensor positional encoding shape [Query_num, Batch, d_model]
        """
        # Self-Attention
        q = k  = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-Attention
        q = tgt if query_pos is None else tgt + query_pos
        k = memory if pos is None else memory + pos
        tgt2 = self.multihead_attn(q,k,value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed Forward Network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                pos: torch.Tensor = None, query_pos: torch.Tensor = None,
                ) -> torch.Tensor:
        """
        tgt: Tensor query shape [Query_num, Batch, d_model]
        memory: Tensor encoded feature from Encoder shape [HW, Batch, d_model]
        pos: Tensor positional encoding shape [HW, Batch, d_model]
        query_pos: Tensor positional encoding shape [Query_num, Batch, d_model]
        """
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, pos, query_pos)
            intermediate.append(output)
        return torch.stack(intermediate)

class Transformer(nn.Module):
    """
    Transformer Architecture
    - Encode feature map for 2D backbone
    - Decode with Object Queries
    """
    def __init__(self, d_model: int =  256, nhead: int=8,
                num_encoder_layers: int=6,
                num_decoder_layers: int=6,
                dim_feedforward: int=2048,
                dropout: float=0.1):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: torch.Tensor, query_embed: torch.Tensor,
                pos_embed: torch.Tensor):
        b,c,h,w = src.shape

        src = src.flatten(2).permute(2,0,1) # [HW,Batch,C]
        pos_embed = pos_embed.flatten(2).permute(2,0,1) # [HW,Batch,C]

        num_queries = query_embed.shape[0]
        query_pos = query_embed.unsqueeze(1).repeat(1, b, 1)

        tgt = torch.zeros(num_queries, b, c, device = src.device)

        memory = self.encoder(src, pos=pos_embed)

        hs = self.decoder(tgt, memory, pos = pos_embed, query_pos = query_pos)

        return hs.permute(0,2,1,3)
