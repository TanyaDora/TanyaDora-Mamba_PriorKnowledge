import torch
import torch.nn as nn
from attention_layer import MultiHeadAttention
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
from mamba_ssm import Mamba

# Base class defining S-Mamba architecture elements
class BaseExperiment(nn.Module):
    def __init__(self, configs):
        super(BaseExperiment, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        
        # Common components
        # Embedding layer for encoder input
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, 
                                                  configs.embed, configs.freq, configs.dropout)
        # Encoder made up of multiple EncoderLayers with Mamba
        self.encoder = Encoder(
            [EncoderLayer(
                Mamba(d_model=configs.d_model, d_state=configs.d_state, 
                      d_conv=2, expand=1),
                Mamba(d_model=configs.d_model, d_state=configs.d_state, 
                      d_conv=2, expand=1),
                configs.d_model, configs.d_ff, 
                dropout=configs.dropout, activation=configs.activation
            ) for _ in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Final linear projection to prediction length
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
    # Optional input normalization
    def normalize(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            return x_enc, means, stdev
        return x_enc, None, None
    
    # Denormalize output predictions
    def denormalize(self, dec_out, means, stdev):
        if self.use_norm and means is not None and stdev is not None:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
    
# Basic S-Mamba model with embeddings
class Experiment1(BaseExperiment):
    """Basic: x_enc -> embedding -> encoder -> projection -> output"""
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
       
        _, _, N = x_enc.shape
        # Apply embedding -> encoder -> projection
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :2] 
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]
    
# Linear projection before encoder (no embedding)
class Experiment2(BaseExperiment):
    """Basic: x_enc -> encoder -> projection -> output"""
    def __init__(self, configs):
        super(Experiment2, self).__init__(configs)
        self.input_dim = configs.enc_in 
        self.output_dim = configs.target_dim
        self.input_projection = nn.Linear(self.input_dim, configs.d_model)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        _, _, N = x_enc.shape

        x_enc = self.input_projection(x_enc) # added Linear Layer
        enc_out, _ = self.encoder(x_enc, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :2]  
        
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]
    
# Adds attention after embedding
class Experiment3(BaseExperiment):
    """With attention: x_enc -> embedding -> attention -> encoder -> projection -> output"""
    def __init__(self, configs):
        super(Experiment3, self).__init__(configs)
        self.attention = MultiHeadAttention(
            d_model=configs.d_model,
            num_heads=4,
            dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        attn_out, _ = self.attention(enc_out) # Multihead Attetion
        enc_out = enc_out + attn_out
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :2]  
        
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]

# Attention applied before embedding       
class Experiment4(BaseExperiment):
    """Attention first: x_enc -> attention -> embedding -> encoder -> projection -> output"""
    def __init__(self, configs):
        super(Experiment4, self).__init__(configs)
        self.input_dim = configs.enc_in 
        self.output_dim = configs.target_dim
        
        self.feature_projection = nn.Linear(self.input_dim, configs.d_model)
        self.attention = MultiHeadAttention(
            d_model=self.input_dim,
            num_heads=4,
            dropout=configs.dropout
        )
        
        # Additional layer to match dimensions after attention
        self.post_attention_proj = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Input shape: [batch_size, seq_len, num_features]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        # Apply attention across features
        attn_out, _ = self.attention(x_enc)  
        x_enc = self.post_attention_proj(attn_out)
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [batch_size, seq_len, d_model]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :2]  # Only predict GPP and ET
        
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]


# Applies attention only on last 2 features (prior knowledge)
class Experiment5(BaseExperiment):
    """Prior knowledge attention: x_enc -> attention(prior) -> embedding -> encoder -> projection -> output"""
    def __init__(self, configs):
        super(Experiment5, self).__init__(configs)
        
        self.prior_projection = nn.Linear(2, configs.d_model)
        
        # Attention on projected prior features
        self.attention = MultiHeadAttention(
            d_model=configs.d_model,  
            num_heads=4,  
            dropout=configs.dropout
        )
        
        self.final_projection = nn.Linear(configs.d_model, 2)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        # Split features
        prior_features = x_enc[:, :, -2:]  # Extract GPP_pred and ET_pred
        other_features = x_enc[:, :, :-2]  # Other 8 features
        
        # Process prior features through attention
        # [batch_size, seq_len, 2] -> [batch_size, seq_len, d_model]
        prior_projected = self.prior_projection(prior_features)
        
        # Apply attention on projected features
        prior_attn, _ = self.attention(prior_projected)
        
        # Project back to original dimension
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, 2]
        prior_processed = self.final_projection(prior_attn)
        
        # Combine processed prior features with other features
        # [batch_size, seq_len, num_features]
        x_enc = torch.cat([other_features, prior_processed], dim=2)
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :2]  # Select GPP and ET predictions
        
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]

# Same as Experiment5 but skips embedding layer
class Experiment6(BaseExperiment):
    """Prior knowledge attention: x_enc -> attention(prior) -> encoder -> projection -> output"""
    def __init__(self, configs):
        super(Experiment6, self).__init__(configs)
        
        self.prior_projection = nn.Linear(2, configs.d_model)
        self.input_projection = nn.Linear(configs.enc_in - 2, configs.d_model)
        
        # Attention for prior features
        self.prior_attention = MultiHeadAttention(
            d_model=configs.d_model,
            num_heads=4,
            dropout=configs.dropout
        )
        
        self.final_projection = nn.Linear(configs.d_model, 2)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        # Split features
        prior_features = x_enc[:, :, -2:]  # Extract GPP_pred and ET_pred
        other_features = x_enc[:, :, :-2]  # Other features
        
        # Project features
        prior_projected = self.prior_projection(prior_features)
        other_projected = self.input_projection(other_features)
        
        # Process prior features through attention
        prior_attn, _ = self.prior_attention(prior_projected)
        
        # Combine processed features
        combined_features = other_projected + prior_attn
        
        # Encoder pipeline without embedding
        enc_out, _ = self.encoder(combined_features, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :2]
        
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]