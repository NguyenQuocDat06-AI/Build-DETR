import torch
from transformer import Transformer

if __name__ == "__main__":
    batch_size = 2
    c,h,w = 256,19,25
    num_queries = 100

    dummy_src = torch.randn(batch_size, c, h, w)
    dummy_pos = torch.randn(batch_size, c, h, w)
    dummy_query_embed = torch.randn(num_queries, c)
    
    transformer = Transformer()
    output = transformer(dummy_src, dummy_query_embed, dummy_pos)
    print(f"Shape of feature maps: {output.shape}")