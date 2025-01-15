import torch

optimizer = torch.optim.AdamW([torch.tensor(0.0, requires_grad=True)], lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Simulate epochs
for epoch in range(15):
    print(f"Epoch {epoch}: Learning rate = {optimizer.param_groups[0]['lr']}")
    scheduler.step()