
torch.manual_seed(42)
np.random.seed(42)

model = EditFlowsTransformer(
    vocab_size=V+2,  # +2 for PAD + BOS tokens
    hidden_dim=512,
    num_layers=8,
    num_heads=32,
    max_seq_len=2*L,
    pad_token_id=V,
    bos_token_id=V+1,
)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

# Print some model statistics and details
print(f"Model: {model.__class__.__name__}")
print(f"  Vocab size: {model.vocab_size}")
print(f"  Hidden dim: {model.hidden_dim}")
print(f"  Num layers: {model.num_layers}")
print(f"  Max seq len: {model.max_seq_len}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Print some details about the optimizer
print(f"Optimizer: {optim.__class__.__name__}")
print(f"  Learning rate: {optim.defaults['lr']}")