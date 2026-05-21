# Phần 2: Xây Dựng Custom Transformer Từ Đầu 🧠

Transformer là "trái tim" của DETR. Thay vì sử dụng lớp có sẵn từ các thư viện cao cấp, trong phần này, chúng ta sẽ tự viết toàn bộ mô hình Transformer từ đầu bằng PyTorch, bao gồm:
1. **Multi-Head Attention (MHA)**: Cơ chế chú ý đa đầu độc lập.
2. **Transformer Encoder Layer & Encoder**: Xử lý và liên kết thông tin trên toàn bộ feature map của ảnh.
3. **Transformer Decoder Layer & Decoder**: Giải mã thông tin từ ảnh sang các dự đoán tọa độ nhờ **Object Queries**.

---

## 1. Cơ Chế Truyền Nhận Thông Tin & Tiêm Vị Trí (Positional Injection)

Một điểm cực kỳ quan trọng cần chú ý trong thiết kế Transformer của DETR so với Transformer chuẩn (như trong bài báo *Attention Is All You Need*):
- **Encoder**: Mã hóa vị trí 2D ($pos$) được cộng trực tiếp vào cả **Query ($Q$)** và **Key ($K$)** ở mỗi lớp Self-Attention. Phần **Value ($V$)** giữ nguyên đặc trưng ảnh thô để tránh làm nhiễu thông tin nội dung gốc.
- **Decoder**: Chúng ta định nghĩa các tham số học được gọi là **Object Queries** (truy vấn đối tượng, kích thước $N \times d$). Mã hóa vị trí của chúng (Query Positional Embeddings, $query\_pos$) được cộng vào $Q$ và $K$ trong lớp Self-Attention đầu tiên của Decoder. Trong lớp Cross-Attention tiếp theo, $query\_pos$ được cộng vào $Q$, còn mã hóa vị trí ảnh ($pos$) được cộng vào $K$.

```
ENCODER SELF-ATTENTION:
  Q = Feature + Pos_Encoding
  K = Feature + Pos_Encoding
  V = Feature

DECODER SELF-ATTENTION:
  Q = Target + Query_Pos
  K = Target + Query_Pos
  V = Target

DECODER CROSS-ATTENTION:
  Q = Decoder_Output + Query_Pos
  K = Encoder_Output + Pos_Encoding
  V = Encoder_Output
```

---

## 2. Tự Cài Đặt Lớp Multi-Head Attention

Hãy tự viết lớp `MultiHeadSelfAttention` bằng các phép biến đổi ma trận thuần túy trong PyTorch để hiểu rõ cách thức phân chia các đầu chú ý độc lập.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Cơ chế Multi-Head Attention tự định nghĩa từ đầu.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert self.head_dim * nhead == d_model, "d_model phải chia hết cho nhead"
        
        # Các phép chiếu tuyến tính cho Query, Key, Value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                q_pos: torch.Tensor = None, k_pos: torch.Tensor = None) -> torch.Tensor:
        """
        q, k, v: Tensors có kích thước [Length, Batch, d_model]
        q_pos: Mã hóa vị trí cộng vào Query [Length_q, Batch, d_model]
        k_pos: Mã hóa vị trí cộng vào Key [Length_k, Batch, d_model]
        """
        len_q, batch_size, _ = q.shape
        len_k, _, _ = k.shape
        len_v, _, _ = v.shape
        
        # 1. Cộng mã hóa vị trí vào Query và Key nếu có
        query = q + q_pos if q_pos is not None else q
        key = k + k_pos if k_pos is not None else k
        value = v
        
        # 2. Chiếu tuyến tính và phân tách thành nhiều đầu chú ý (Multi-head split)
        # [L, B, d_model] -> [L, B, nhead, head_dim] -> [B, nhead, L, head_dim]
        query = self.q_proj(query).view(len_q, batch_size, self.nhead, self.head_dim).transpose(0, 1).transpose(1, 2)
        key = self.k_proj(key).view(len_k, batch_size, self.nhead, self.head_dim).transpose(0, 1).transpose(1, 2)
        value = self.v_proj(value).view(len_v, batch_size, self.nhead, self.head_dim).transpose(0, 1).transpose(1, 2)
        
        # 3. Tính toán Scaled Dot-Product Attention
        # scores: [B, nhead, len_q, len_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Tính trọng số attention qua hàm Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Nhân trọng số với Value
        # context: [B, nhead, len_q, head_dim]
        context = torch.matmul(attn_weights, value)
        
        # 4. Gộp các đầu chú ý lại với nhau (Concatenate heads)
        # [B, nhead, len_q, head_dim] -> [B, len_q, nhead, head_dim] -> [B, len_q, d_model] -> [len_q, B, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, len_q, self.d_model).transpose(0, 1)
        
        # Chiếu đầu ra lần cuối
        output = self.out_proj(context)
        return output
```

---

## 3. Xây Dựng Transformer Encoder Layer & Transformer Encoder

Lớp Encoder Layer bao gồm: Self-Attention $\rightarrow$ Dropout $\rightarrow$ Residual $\rightarrow$ LayerNorm $\rightarrow$ FFN (Feed Forward Network) $\rightarrow$ Dropout $\rightarrow$ Residual $\rightarrow$ LayerNorm.

```python
class TransformerEncoderLayer(nn.Module):
    """
    Một lớp đơn lẻ bên trong Transformer Encoder.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Mạng Feed Forward (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Chuẩn hóa Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, src: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """
        src: Feature map đã được duỗi phẳng [HW, Batch, d_model]
        pos: Mã hóa vị trí 2D tương ứng [HW, Batch, d_model]
        """
        # Bước 1: Self-Attention với mã hóa vị trí tiêm vào Query & Key
        src2 = self.self_attn(q=src, k=src, v=src, q_pos=pos, k_pos=pos)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Bước 2: Feed Forward Network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder hoàn chỉnh ghép từ nhiều TransformerEncoderLayer.
    """
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, src: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos)
        return output
```

---

## 4. Xây Dựng Transformer Decoder Layer & Transformer Decoder

Lớp Decoder Layer phức tạp hơn khi chứa 2 lớp Attention chồng lên nhau:
1. **Self-Attention**: Các Object Queries tương tác tự chú ý với nhau để tránh dự đoán trùng lặp.
2. **Cross-Attention**: Các Object Queries lấy thông tin trực tiếp từ đầu ra của Encoder (các đặc trưng ảnh có chiều sâu).

```python
class TransformerDecoderLayer(nn.Module):
    """
    Một lớp đơn lẻ bên trong Transformer Decoder.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Mạng Feed Forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Chuẩn hóa Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                pos: torch.Tensor = None, query_pos: torch.Tensor = None) -> torch.Tensor:
        """
        tgt: Trạng thái của Object Queries [Num_Queries, Batch, d_model]
        memory: Đầu ra của Encoder (ảnh đặc trưng) [HW, Batch, d_model]
        pos: Mã hóa vị trí ảnh [HW, Batch, d_model]
        query_pos: Mã hóa vị trí của Object Queries [Num_Queries, Batch, d_model]
        """
        # Bước 1: Self-Attention giữa các Object Queries với nhau
        tgt2 = self.self_attn(q=tgt, k=tgt, v=tgt, q_pos=query_pos, k_pos=query_pos)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Bước 2: Cross-Attention thu thập thông tin từ Encoder Memory
        # Query: tgt + query_pos
        # Key: memory + pos
        # Value: memory
        tgt2 = self.multihead_attn(q=tgt, k=memory, v=memory, q_pos=query_pos, k_pos=pos)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Bước 3: Feed Forward Network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder hoàn chỉnh ghép từ nhiều TransformerDecoderLayer.
    """
    def __init__(self, decoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                pos: torch.Tensor = None, query_pos: torch.Tensor = None) -> torch.Tensor:
        output = tgt
        
        # Trong DETR, ta có thể lưu lại kết quả dự đoán trung gian ở mỗi layer (nếu muốn dùng Aux Loss)
        # Ở đây ta trả về tensor kết quả cuối cùng có dạng [num_layers, Num_Queries, Batch, d_model]
        intermediate = []
        
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
            intermediate.append(output)
            
        return torch.stack(intermediate)
```

---

## 5. Ghép Nối Thành Module Transformer Hoàn Chỉnh

Bây giờ ta gộp cả Encoder và Decoder vào một khối thống nhất để xử lý luồng dữ liệu trọn vẹn.

```python
class CustomTransformer(nn.Module):
    """
    Bộ Transformer tích hợp toàn bộ Encoder và Decoder của DETR.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, num_encoder_layers: int = 6, 
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.d_model = d_model
        self.nhead = nhead
        
    def forward(self, src: torch.Tensor, query_embed: torch.Tensor, pos_embed: torch.Tensor):
        """
        src: [B, C, H, W] - Feature map từ Backbone
        query_embed: [Num_Queries, C] - Object Queries nhúng vị trí (learned parameter)
        pos_embed: [B, C, H, W] - Mã hóa vị trí 2D của ảnh
        
        Trả về: [Num_layers, Num_Queries, B, C]
        """
        b, c, h, w = src.shape
        
        # 1. Duỗi phẳng feature map và vị trí về dạng chuỗi 1D tuần tự (Sequence length = H * W)
        # [B, C, H, W] -> [B, C, H*W] -> [H*W, B, C] (Phù hợp chuẩn định dạng Transformer đầu vào)
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        
        # Mở rộng chiều của query_embed từ [N, C] thành [N, B, C] để khớp Batch size
        num_queries = query_embed.shape[0]
        query_pos = query_embed.unsqueeze(1).repeat(1, b, 1)
        
        # Khởi tạo chuỗi đích đầu vào cho Decoder bằng ma trận 0 (khớp kích thước [N, B, C])
        tgt = torch.zeros(num_queries, b, c, device=src.device)
        
        # 2. Truyền qua Encoder
        memory = self.encoder(src, pos=pos_embed)
        
        # 3. Truyền qua Decoder
        # Trả về: [num_layers, Num_Queries, B, C]
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_pos)
        
        # Hoán đổi chiều để trả về dạng thân thiện hơn: [num_layers, B, Num_Queries, C]
        return hs.permute(0, 2, 1, 3)
```

---

## 6. Chạy Thử Nghiệm Kiểm Tra Kích Thước (Sanity Check)

Hãy viết một đoạn mã nhỏ kiểm thử luồng truyền dữ liệu qua khối `CustomTransformer` này.

```python
if __name__ == '__main__':
    # Giả lập tham số
    batch_size = 2
    c, h, w = 256, 19, 25  # d_model=256, feature map kích thước 19x25
    num_queries = 100      # 100 Object Queries
    
    # 1. Tạo ngẫu nhiên Feature map, Positional Encoding từ Backbone
    dummy_src = torch.randn(batch_size, c, h, w)
    dummy_pos = torch.randn(batch_size, c, h, w)
    
    # 2. Tạo Object Queries nhúng (learned parameters)
    # Trong mô hình thực tế, đây sẽ là nn.Embedding(100, 256).weight
    dummy_query_embed = torch.randn(num_queries, c)
    
    # 3. Khởi tạo Transformer
    transformer = CustomTransformer(
        d_model=256, 
        nhead=8, 
        num_encoder_layers=6, 
        num_decoder_layers=6
    )
    
    # Chạy forward
    outputs = transformer(dummy_src, dummy_query_embed, dummy_pos)
    
    print(f"Kích thước đầu ra của Transformer: {outputs.shape}")
    # Mong đợi: [6 (layers), 2 (batch), 100 (queries), 256 (channels)]
    assert outputs.shape == (6, batch_size, num_queries, c)
    print("🎉 Thành công! Custom Transformer hoạt động mượt mà và đúng chuẩn kích thước!")
```

### Kết quả đầu ra:
```text
Kích thước đầu ra của Transformer: torch.Size([6, 2, 100, 256])
🎉 Thành công! Custom Transformer hoạt động mượt mà và đúng chuẩn kích thước!
```

---

> [!TIP]
> Việc lưu trữ kết quả của **tất cả** 6 layers Decoder giúp chúng ta có thể tùy ý áp dụng cơ chế **Auxiliary Loss** (tính toán hàm mất mát phụ ở từng lớp giải mã trung gian trong quá trình huấn luyện). Kỹ thuật này giúp mô hình DETR hội tụ nhanh hơn rất nhiều so với chỉ tính loss ở lớp Decoder cuối cùng.

Cơ cấu phần cứng cốt lõi đã hoàn thành! Tiếp theo, chúng ta sẽ bước qua một trong những phần thú vị nhất và mang tính đặc trưng độc nhất của DETR: **[Phần 3: Thuật Toán Khớp Cặp Hungarian Matcher](part3_hungarian_matcher.md)**.
