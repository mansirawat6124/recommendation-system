import torch
import torch.nn as nn
import torch.optim as optim

user_count, item_count = 10, 10
ratings = torch.randint(0, 6, (user_count, item_count), dtype=torch.float)

class Recommender(nn.Module):
    def __init__(self, users, items, factors=5):
        super().__init__()
        self.user_factors = nn.Embedding(users, factors)
        self.item_factors = nn.Embedding(items, factors)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

model = Recommender(user_count, item_count)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

users = torch.randint(0, user_count, (50,))
items = torch.randint(0, item_count, (50,))
ratings_sampled = ratings[users, items]

for _ in range(100):
    optimizer.zero_grad()
    predictions = model(users, items)
    loss = loss_fn(predictions, ratings_sampled)
    loss.backward()
    optimizer.step()

user_id, item_id = 3, 7
predicted_rating = model(torch.tensor([user_id]), torch.tensor([item_id]))
print(f"Predicted rating for user {user_id} on item {item_id}: {predicted_rating.item()}")
