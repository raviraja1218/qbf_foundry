# src/models/classical_nn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# constants: 20 amino acids + PAD
AA_VOCAB = 20
DEFAULT_SEQ_LEN = 300
VOCAB_SIZE = 20
PAD_IDX = 20

class ProteinVAE(nn.Module):
    def __init__(
        self,
        vocab_size=AA_VOCAB,
        seq_len=DEFAULT_SEQ_LEN,
        embed_dim=64,
        phys_dim=5,
        phys_proj_dim=32,
        enc_hidden=128,
        latent_dim=64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.phys_dim = phys_dim
        self.phys_proj_dim = phys_proj_dim
        self.enc_hidden = enc_hidden
        self.latent_dim = latent_dim

        # embedding for sequence tokens (includes PAD index)
        self.embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embed_dim, padding_idx=PAD_IDX)

        # small projector for phys_props
        self.phys_proj = nn.Sequential(
            nn.Linear(phys_dim, phys_proj_dim),
            nn.ReLU(),
            nn.LayerNorm(phys_proj_dim),
        )

        # encoder: GRU over concatenated [embed || phys_proj]
        enc_input_dim = embed_dim + phys_proj_dim
        self.encoder = nn.GRU(input_size=enc_input_dim, hidden_size=enc_hidden, batch_first=True, bidirectional=False)

        # bottleneck
        self.fc_mu = nn.Linear(enc_hidden, latent_dim)
        self.fc_logvar = nn.Linear(enc_hidden, latent_dim)

        # decoder: map latent -> seq_len * vocab (simple, stable)
        # you can replace with an RNN decoder later
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, seq_len * (vocab_size + 1)),  # include PAD slot in logits (but training ignores PAD)
        )

    def encode(self, x_seq, phys):
        """
        x_seq: LongTensor (B, seq_len)
        phys: FloatTensor (B, seq_len, phys_dim)
        """
        emb = self.embedding(x_seq)  # (B, L, embed_dim)
        phys_proj = self.phys_proj(phys)  # (B, L, phys_proj_dim)
        enc_in = torch.cat([emb, phys_proj], dim=-1)  # (B, L, embed+physproj)
        out, h = self.encoder(enc_in)  # out (B, L, enc_hidden), h (1, B, enc_hidden)
        h = h.squeeze(0)  # (B, enc_hidden)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        z: (B, latent_dim)
        returns: logits (B, seq_len, vocab_size+1)
        """
        out = self.fc_dec(z)  # (B, seq_len*(vocab+1))
        out = out.view(-1, self.seq_len, self.vocab_size + 1)
        return out

    def forward(self, x_seq, phys):
        mu, logvar = self.encode(x_seq, phys)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

def vae_loss(recon_logits, target_seq, mu, logvar, pad_idx=PAD_IDX, kl_beta=1.0):
    """
    recon_logits: (B, L, V)
    target_seq: (B, L) long
    """
    B, L, V = recon_logits.shape
    # reshape for cross-entropy: (B*L, V) and (B*L,)
    logits = recon_logits.view(-1, V)
    tgt = target_seq.view(-1)
    # ignore pad index
    ce = F.cross_entropy(logits, tgt, ignore_index=pad_idx, reduction='sum')
    # KLD
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = ce + kl_beta * KLD
    return loss, ce.item(), KLD.item()
