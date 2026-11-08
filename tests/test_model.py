import torch
import torch.nn.functional as F
from model import ModelConfig, Transformer

TINY = dict(d_model=64, n_head=4, d_ff=160, vocab_size=128, max_seq_len=32)


def test_baseline_size_and_depth_is_the_only_difference():
    deep, shallow = Transformer(ModelConfig(n_layer=12)), Transformer(ModelConfig(n_layer=4))
    assert 120e6 < deep.num_params() < 128e6  # ~GPT-2-small scale (123.6M with tied embeddings)
    deep_sd, shallow_sd = deep.state_dict(), shallow.state_dict()
    assert set(shallow_sd) <= set(deep_sd)
    assert all(deep_sd[k].shape == v.shape for k, v in shallow_sd.items())
    extra = {k.split(".")[1] for k in set(deep_sd) - set(shallow_sd)}
    assert extra == {str(i) for i in range(4, 12)}  # only blocks 4..11 differ


def test_forward_is_lm_head_of_hidden():
    torch.manual_seed(0)
    model = Transformer(ModelConfig(n_layer=2, **TINY))
    idx = torch.randint(0, 128, (2, 10))
    h = model.hidden(idx)
    assert h.shape == (2, 10, 64)
    assert torch.allclose(model(idx), model.lm_head(h))
    assert model.lm_head.weight is model.tok_emb.weight


def test_causal():
    torch.manual_seed(0)
    model = Transformer(ModelConfig(n_layer=2, **TINY))
    a = torch.randint(0, 128, (1, 16))
    b = a.clone()
    b[0, 8:] = torch.randint(0, 128, (8,))
    assert torch.allclose(model(a)[0, :8], model(b)[0, :8], atol=1e-6)
    assert not torch.allclose(model(a)[0, 8:], model(b)[0, 8:])


def test_loss_matches_cross_entropy_and_ignores_minus_one():
    torch.manual_seed(0)
    model = Transformer(ModelConfig(n_layer=2, **TINY))
    x, y = torch.randint(0, 128, (2, 12)), torch.randint(0, 128, (2, 12))
    y[:, :5] = -1
    expected = F.cross_entropy(model(x)[:, 5:].reshape(-1, 128), y[:, 5:].reshape(-1))
    assert torch.allclose(model(x, y), expected, atol=1e-6)


def test_overfits_a_batch():
    torch.manual_seed(0)
    model = Transformer(ModelConfig(n_layer=2, **TINY))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 128, (4, 24))
    for _ in range(150):
        loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
