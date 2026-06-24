"""
MT-MNER model (re-implementation based on the paper's described architecture).

Pipeline (Fig. 2 of the paper):
  1. Feature extraction:
       - RoBERTa   -> token-level text features      f^T  (B, L_T, d)
       - ViT       -> local/patch visual features    f^L  (B, L_P, d)
       - ResNet-50 -> global visual feature          f^G  (B, d)   (after linear compression)
  2. Multi-view contrastive learning:
       - coarse-grained: mean-pooled text  <-> global image (symmetric InfoNCE)
       - fine-grained:   token <-> patch, Top-K rank-based contrast
  3. Two-stage feature fusion:
       - hierarchical: 3x CAM + 2x GFM  -> f_hi
       - global:       attention over (text, global image) conditioned on f_hi -> f_glo
  4. CRF decoding over f_glo

All models are loaded from local paths (not from HuggingFace).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, ViTModel
import torchvision
import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #
class ProjectionHead(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


class CrossAttentionModule(nn.Module):
    """CAM: standard multi-head cross-attention + LayerNorm (Fig.2 inset)."""

    def __init__(self, d, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, q, k, v, key_padding_mask=None):
        out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        return self.norm(out + q)


class GatedFusionModule(nn.Module):
    """
    GFM (Fig.2 inset): two inputs go through Tanh + MLP, an alpha gate balances
    them with weights (alpha) and (1 - alpha), combined by element-wise product.
    """

    def __init__(self, d):
        super().__init__()
        self.mlp_a = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.mlp_b = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.gate = nn.Linear(2 * d, d)

    def forward(self, a, b):
        ha = self.mlp_a(a)
        hb = self.mlp_b(b)
        alpha = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return alpha * ha + (1.0 - alpha) * hb


class CRF(nn.Module):
    """Minimal linear-chain CRF (Viterbi decode + NLL loss)."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start = nn.Parameter(torch.randn(num_tags))
        self.end = nn.Parameter(torch.randn(num_tags))
        self.trans = nn.Parameter(torch.randn(num_tags, num_tags))

    def _score(self, emissions, tags, mask):
        B, L, _ = emissions.shape
        score = self.start[tags[:, 0]] + emissions[torch.arange(B), 0, tags[:, 0]]
        for t in range(1, L):
            m = mask[:, t]
            score = score + (self.trans[tags[:, t - 1], tags[:, t]]
                             + emissions[torch.arange(B), t, tags[:, t]]) * m
        last = mask.sum(1).long() - 1
        score = score + self.end[tags[torch.arange(B), last]]
        return score

    def _norm(self, emissions, mask):
        B, L, T = emissions.shape
        alpha = self.start.unsqueeze(0) + emissions[:, 0]
        for t in range(1, L):
            broadcast = alpha.unsqueeze(2) + self.trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            new_alpha = torch.logsumexp(broadcast, dim=1)
            m = mask[:, t].unsqueeze(1)
            alpha = torch.where(m.bool(), new_alpha, alpha)
        alpha = alpha + self.end.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def forward(self, emissions, tags, mask):
        """Negative log-likelihood loss."""
        return (self._norm(emissions, mask) - self._score(emissions, tags, mask)).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        B, L, T = emissions.shape
        history = []
        score = self.start.unsqueeze(0) + emissions[:, 0]
        for t in range(1, L):
            broadcast = score.unsqueeze(2) + self.trans.unsqueeze(0)
            best, idx = broadcast.max(dim=1)
            score_t = best + emissions[:, t]
            m = mask[:, t].unsqueeze(1).bool()
            score = torch.where(m, score_t, score)
            history.append(idx)
        score = score + self.end.unsqueeze(0)
        best_paths = []
        last = mask.sum(1).long() - 1
        for b in range(B):
            best_tag = score[b].argmax().item()
            path = [best_tag]
            for idx in reversed(history[: last[b].item()]):
                best_tag = idx[b][best_tag].item()
                path.append(best_tag)
            best_paths.append(path[::-1])
        return best_paths


# --------------------------------------------------------------------------- #
#  Main model
# --------------------------------------------------------------------------- #
class MTMNER(nn.Module):
    def __init__(self, num_tags, d=768, n_heads=8, proj_dim=256,
                 roberta_name="FacebookAI/roberta-base",
                 vit_name="google/vit-base-patch16-224",
                 freeze_text_layers=5, temperature=0.1, top_k=5):
        super().__init__()
        self.d = d
        self.tau = temperature
        self.top_k = top_k

        # --- encoders ---
        # RoBERTa & ViT: download from HuggingFace (auto-cached)
        # ResNet-50: torchvision handles download automatically
        self.text_encoder = RobertaModel.from_pretrained(
            roberta_name, ignore_mismatched_sizes=True
        )
        self.vit = ViTModel.from_pretrained(
            vit_name, ignore_mismatched_sizes=True
        )
        resnet_weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.resnet = nn.Sequential(*list(torchvision.models.resnet50(weights=resnet_weights).children())[:-1])  # drop fc -> (B,2048,1,1)
        self.res_proj = nn.Linear(2048, d)                          # Eq.(5): 2048 -> d

        self._freeze_text(freeze_text_layers)

        # --- projection heads for contrastive learning ---
        self.proj_t = ProjectionHead(d, proj_dim)
        self.proj_g = ProjectionHead(d, proj_dim)
        self.proj_tok = ProjectionHead(d, proj_dim)
        self.proj_pat = ProjectionHead(d, proj_dim)

        # --- two-stage fusion ---
        self.cam1 = CrossAttentionModule(d, n_heads)   # global img (Q) x text (K,V)  -> f1
        self.cam2 = CrossAttentionModule(d, n_heads)   # f1 (Q) x local img (K,V)     -> f2
        self.cam3 = CrossAttentionModule(d, n_heads)   # f2 (Q) x global img (K,V)    -> f3
        self.gfm1 = GatedFusionModule(d)
        self.gfm2 = GatedFusionModule(d)

        self.mlp_t = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.mlp_g = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.attn_t = CrossAttentionModule(d, n_heads)
        self.attn_g = CrossAttentionModule(d, n_heads)
        self.mlp_out = nn.Linear(2 * d, d)

        # --- decoder ---
        self.classifier = nn.Linear(d, num_tags)
        self.crf = CRF(num_tags)

    def _freeze_text(self, k):
        if k <= 0:
            return
        for p in self.text_encoder.embeddings.parameters():
            p.requires_grad = False
        for layer in self.text_encoder.encoder.layer[:k]:
            for p in layer.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------ #
    #  feature extraction
    # ------------------------------------------------------------------ #
    def extract_features(self, input_ids, attention_mask, pixel_values_vit, pixel_values_res):
        f_t = self.text_encoder(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state   # (B,L_T,d)
        f_l = self.vit(pixel_values=pixel_values_vit).last_hidden_state            # (B,L_P+1,d)
        res = self.resnet(pixel_values_res).flatten(1)                             # (B,2048)
        f_g = self.res_proj(res)                                                   # (B,d)
        return f_t, f_l, f_g

    # ------------------------------------------------------------------ #
    #  contrastive losses
    # ------------------------------------------------------------------ #
    def coarse_contrastive(self, f_t, f_g, attention_mask):
        """Coarse-grained symmetric InfoNCE (Eqs.6-9): mean-pooled text (Eq.6)
        vs global image, projected + L2-normalized, in-batch negatives."""
        # mean-pool text over valid tokens -> global text vector f^{T1} (Eq.6)
        m = attention_mask.unsqueeze(-1).float()
        t_mean = (f_t * m).sum(1) / m.sum(1).clamp(min=1e-6)           # (B,d)
        zt = F.normalize(self.proj_t(t_mean), dim=-1)
        zg = F.normalize(self.proj_g(f_g), dim=-1)
        logits = zt @ zg.t() / self.tau                                # (B,B)
        target = torch.arange(logits.size(0), device=logits.device)
        # symmetric InfoNCE (text->image and image->text)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))

    def fine_contrastive(self, f_t, f_l, attention_mask):
        """
        Fine-grained rank-based contrast, following the paper's Eqs.(10)-(21).

        Procedure:
          1. token-patch similarity matrix S^{TL} = h^T (h^L)^T          (Eq.10-11)
          2. row-wise sort; Top-K positions -> important mask M (Eq.12-14),
              the complement -> unimportant mask M_bar
          3. build two masked "views" of each modality (Eq.17-18):
                 Z   = feature gated by important mask        (positive view)
                 Z_bar = feature gated by unimportant mask    (negative view)
          4. self-supervised InfoNCE: pull (H, Z) together, push (H, Z_bar)
              apart, for both text and image sides (Eq.19-21).
        """
        zt = F.normalize(self.proj_tok(f_t), dim=-1)        # (B,L_T,p)
        zp = F.normalize(self.proj_pat(f_l), dim=-1)        # (B,L_P,p)
        sim = torch.bmm(zt, zp.transpose(1, 2))             # (B,L_T,L_P)  Eq.(10-11)

        # --- Top-K important / unimportant masks over patches per token (Eq.12-14)
        k = min(self.top_k, sim.size(-1))
        topk_idx = sim.topk(k, dim=-1).indices              # (B,L_T,k)
        m_imp = torch.zeros_like(sim).scatter_(2, topk_idx, 1.0)   # important (1=keep)
        m_uni = 1.0 - m_imp                                        # unimportant

        # token-conditioned patch summaries (important vs unimportant views)
        # weighted-average patch features under each mask -> per-token vectors
        def masked_patch_view(mask):
            w = mask / mask.sum(-1, keepdim=True).clamp(min=1e-6)  # (B,L_T,L_P)
            return torch.bmm(w, zp)                                # (B,L_T,p)

        z_imp = masked_patch_view(m_imp)        # positive view  Z      (Eq.17)
        z_uni = masked_patch_view(m_uni)        # negative view  Z_bar  (Eq.18)

        # anchor = token feature itself (H); InfoNCE over {positive, negative} (Eq.19)
        tok_mask = attention_mask.float()                          # (B,L_T)
        pos = (zt * z_imp).sum(-1) / self.tau                      # (B,L_T)
        neg = (zt * z_uni).sum(-1) / self.tau
        logits_t = torch.stack([pos, neg], dim=-1)                 # (B,L_T,2)
        labels_t = torch.zeros_like(pos, dtype=torch.long)         # positive = index 0
        loss_t = F.cross_entropy(logits_t.reshape(-1, 2),
                                 labels_t.reshape(-1), reduction="none")
        loss_t = (loss_t * tok_mask.reshape(-1)).sum() / tok_mask.sum().clamp(min=1e-6)

        # --- symmetric image side: patch anchored to token views (Eq.20) ---
        sim_p = sim.transpose(1, 2)                                # (B,L_P,L_T)
        kp = min(self.top_k, sim_p.size(-1))
        topk_idx_p = sim_p.topk(kp, dim=-1).indices
        mp_imp = torch.zeros_like(sim_p).scatter_(2, topk_idx_p, 1.0)
        mp_uni = 1.0 - mp_imp

        def masked_token_view(mask):
            w = mask / mask.sum(-1, keepdim=True).clamp(min=1e-6)  # (B,L_P,L_T)
            return torch.bmm(w, zt)                                # (B,L_P,p)

        zp_imp = masked_token_view(mp_imp)
        zp_uni = masked_token_view(mp_uni)
        pos_p = (zp * zp_imp).sum(-1) / self.tau
        neg_p = (zp * zp_uni).sum(-1) / self.tau
        logits_p = torch.stack([pos_p, neg_p], dim=-1)
        labels_p = torch.zeros_like(pos_p, dtype=torch.long)
        loss_p = F.cross_entropy(logits_p.reshape(-1, 2), labels_p.reshape(-1))

        return 0.5 * (loss_t + loss_p)                            # Eq.(21)

    # ------------------------------------------------------------------ #
    #  two-stage fusion
    # ------------------------------------------------------------------ #
    def fuse(self, f_t, f_l, f_g, attention_mask):
        L_T = f_t.size(1)
        g = f_g.unsqueeze(1).expand(-1, L_T, -1)            # broadcast global -> seq len
        pad = (attention_mask == 0)

        f1 = self.cam1(g, f_t, f_t, key_padding_mask=pad)   # global(Q) x text(K,V)
        f2 = self.cam2(f1, f_l, f_l)                        # f1(Q) x local(K,V)
        f3 = self.cam3(f2, g, g)                            # f2(Q) x global(K,V)
        f_hi = self.gfm1(f1, self.gfm2(f2, f3))             # Eq.(29)

        t2 = self.attn_t(f_t + self.mlp_t(f_t), f_hi, f_hi)
        g2 = self.attn_g(g + self.mlp_g(g), f_hi, f_hi)
        f_glo = self.mlp_out(torch.cat([t2, g2], dim=-1))   # (B,L_T,d)
        return f_glo

    # ------------------------------------------------------------------ #
    #  forward
    # ------------------------------------------------------------------ #
    def forward(self, input_ids, attention_mask, pixel_values_vit,
                pixel_values_res, labels=None, lambda_s=1.0):
        f_t, f_l, f_g = self.extract_features(
            input_ids, attention_mask, pixel_values_vit, pixel_values_res)

        f_glo = self.fuse(f_t, f_l, f_g, attention_mask)
        emissions = self.classifier(f_glo)                  # (B,L_T,num_tags)

        if labels is None:
            return self.crf.decode(emissions, attention_mask)

        # CRF needs real tags; map -100 (ignored sub-tokens / pad) to 'O'=0 but
        # exclude them from the path via the mask.
        crf_mask = (labels != -100) & (attention_mask.bool())
        crf_mask[:, 0] = True                               # CRF requires first step valid
        safe_tags = labels.clone()
        safe_tags[labels == -100] = 0

        loss_ner = self.crf(emissions, safe_tags, crf_mask.long())
        loss_coarse = self.coarse_contrastive(f_t, f_g, attention_mask)
        loss_fine = self.fine_contrastive(f_t, f_l, attention_mask)
        loss = loss_ner + lambda_s * (loss_coarse + loss_fine)   # Eq.(33)

        return {
            "loss": loss,
            "loss_ner": loss_ner.detach(),
            "loss_coarse": loss_coarse.detach(),
            "loss_fine": loss_fine.detach(),
            "emissions": emissions,
            "logits": emissions,  # alias for compatibility
        }

    def predict(self, input_ids, attention_mask, pixel_values_vit, pixel_values_res):
        """Inference: predict entity labels via CRF decode"""
        return self.forward(
            input_ids, attention_mask,
            pixel_values_vit, pixel_values_res, labels=None
        )

    def get_contrastive_features(self, input_ids, attention_mask,
                                  pixel_values_vit, pixel_values_res):
        """Get intermediate features for analysis"""
        f_t, f_l, f_g = self.extract_features(
            input_ids, attention_mask, pixel_values_vit, pixel_values_res)
        f_glo = self.fuse(f_t, f_l, f_g, attention_mask)
        return {
            'f_t': f_t,
            'f_l': f_l,
            'f_g': f_g,
            'f_glo': f_glo,
        }
