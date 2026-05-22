# Phần 2: Xây Dựng Custom Transformer (Sử Dụng PyTorch Built-in Modules) 🧠

Transformer là "trái tim" của DETR. Thay vì tự viết lại các phép toán nhân ma trận từ con số không, trong phần này, chúng ta sẽ sử dụng module `nn.MultiheadAttention` cực kỳ tối ưu có sẵn của PyTorch. Tuy nhiên, chúng ta vẫn sẽ tự thiết kế lớp Encoder và Decoder để đảm bảo tính năng đặc thù của DETR: **Tiêm vị trí (Positional Injection)** ở mỗi lớp.

---

## 1. Cơ Chế Truyền Nhận Thông Tin & Tiêm Vị Trí

Một điểm cực kỳ quan trọng cần chú ý trong thiết kế Transformer của DETR so với Transformer chuẩn:
- **Encoder**: Mã hóa vị trí 2D ($pos$) được cộng trực tiếp vào cả **Query ($Q$)** và **Key ($K$)** ở mỗi lớp Self-Attention. Phần **Value ($V$)** giữ nguyên đặc trưng ảnh thô để tránh làm nhiễu thông tin.
- **Decoder**: Chúng ta định nghĩa các tham số học được gọi là **Object Queries**. Mã hóa vị trí của chúng ($query\_pos$) được cộng vào $Q$ và $K$ trong lớp Self-Attention đầu tiên của Decoder. Trong lớp Cross-Attention tiếp theo, $query\_pos$ được cộng vào $Q$, còn mã hóa vị trí ảnh ($pos$) được cộng vào $K$.

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

## 2. Xây Dựng Transformer Encoder Layer & Transformer Encoder

Chúng ta sẽ sử dụng `nn.MultiheadAttention` của PyTorch. Để tiêm vị trí, ta chỉ cần cộng `pos` vào `src` trước khi truyền vào vị trí của Query và Key.

```python
import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    """
    Một lớp đơn lẻ bên trong Transformer Encoder.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        # Sử dụng module có sẵn của PyTorch
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
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
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src)[0]
        
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

## 3. Xây Dựng Transformer Decoder Layer & Transformer Decoder

Lớp Decoder Layer chứa 2 lớp Attention chồng lên nhau. Việc dùng `nn.MultiheadAttention` làm cho đoạn code trở nên vô cùng gọn gàng so với tự code ma trận:

```python
class TransformerDecoderLayer(nn.Module):
    """
    Một lớp đơn lẻ bên trong Transformer Decoder.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
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
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Bước 2: Cross-Attention thu thập thông tin từ Encoder Memory
        q = tgt if query_pos is None else tgt + query_pos
        k = memory if pos is None else memory + pos
        tgt2 = self.multihead_attn(q, k, value=memory)[0]
        
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
        intermediate = []
        
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
            intermediate.append(output)
            
        # Trả về tất cả các bước trung gian phục vụ Auxiliary Loss
        return torch.stack(intermediate)
```

---

## 4. Ghép Nối Thành Module Transformer Hoàn Chỉnh

Phần khối tổng này không có gì thay đổi so với bản trước, vì nó chỉ gọi tới Encoder và Decoder:

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
        b, c, h, w = src.shape
        
        # Duỗi phẳng feature map và vị trí
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        
        num_queries = query_embed.shape[0]
        query_pos = query_embed.unsqueeze(1).repeat(1, b, 1)
        
        tgt = torch.zeros(num_queries, b, c, device=src.device)
        
        memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_pos)
        
        return hs.permute(0, 2, 1, 3)
```

---

## 5. Chạy Thử Nghiệm Kiểm Tra Kích Thước (Sanity Check)

```python
if __name__ == '__main__':
    batch_size = 2
    c, h, w = 256, 19, 25
    num_queries = 100
    
    dummy_src = torch.randn(batch_size, c, h, w)
    dummy_pos = torch.randn(batch_size, c, h, w)
    dummy_query_embed = torch.randn(num_queries, c)
    
    transformer = CustomTransformer()
    outputs = transformer(dummy_src, dummy_query_embed, dummy_pos)
    
    print(f"Kích thước đầu ra của Transformer: {outputs.shape}")
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
> Việc sử dụng trực tiếp `nn.MultiheadAttention` của PyTorch không chỉ giúp code gọn gàng, mà nó còn được tối ưu hóa cực kỳ tốt bằng C++/CUDA dưới nền, giúp mô hình huấn luyện nhanh hơn và tốn ít bộ nhớ VRAM hơn so với việc tự nhân ma trận thủ công.

Cơ cấu phần cứng cốt lõi đã hoàn thành! Tiếp theo, chúng ta sẽ bước qua một trong những phần thú vị nhất và mang tính đặc trưng độc nhất của DETR: **[Phần 3: Thuật Toán Khớp Cặp Hungarian Matcher](part3_hungarian_matcher.md)**.
