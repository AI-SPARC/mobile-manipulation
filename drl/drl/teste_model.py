import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Rede neural ---
class PolicyNet(nn.Module):
    def __init__(self, input_dim=3000, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# --- Cubo de pontos (pré-gerado) ---
def generate_cube_points(size=10, spacing=0.01):
    coords = np.linspace(0, (size - 1) * spacing, size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

    center = np.array([np.mean(coords)] * 3)
    dists = np.linalg.norm(points - center, axis=1)
    sorted_idx = np.argsort(-dists)
    points_sorted = points[sorted_idx]

    # --- ALTERAÇÃO REALIZADA AQUI ---
    # A normalização foi removida. A função agora retorna os pontos brutos.
    # Isso permite que a rede não fique restrita a um intervalo específico.
    return points_sorted.flatten().astype(np.float32)


# --- Recompensa ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Recompensa ---
def compute_reward(action, target=None):
    if target is None:
        target = torch.tensor([90.044, -20.055, 40.088], device=action.device)
    else:
        target = target.to(action.device)
    return -torch.norm(action - target)



def main():
    model = PolicyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.000025e-1)  # taxa mais alta
    print(torch.cuda.is_available())  # True se houver GPU CUDA

    # Pré-gera observação fixa
    obs_vector = generate_cube_points()
    obs = torch.tensor(obs_vector, dtype=torch.float32, device=device)

    rewards = []
    chosen_points = []

    episodes = 100000
    batch_size = 128

    obs_batch = obs.repeat(batch_size, 1).to(device)

    for episode in range(episodes):
        # Forward
        action_mean = model(obs_batch)  # sem tanh

        # Política com ruído mínimo
        dist = torch.distributions.Normal(action_mean, torch.ones_like(action_mean, device=device) * 1e-12)
        actions = dist.rsample()
        logprob = dist.log_prob(actions).sum(dim=1)

        # Recompensa
        rewards_batch = torch.stack([compute_reward(a) for a in actions])

        # Loss REINFORCE
        loss = -(logprob * rewards_batch).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = rewards_batch.mean().item()
        rewards.append(avg_reward)

        # Guardar o ponto mais preciso (média do batch)
        chosen_points.append(action_mean[0].detach().cpu().numpy())

        # Print do ponto escolhido
        if (episode + 1) % 40 == 0:
            print(f"Ep {episode+1} | reward médio={avg_reward:.4f} | ponto escolhido={action_mean[0].detach().cpu().numpy()}")

        if (episode + 1) % 100 == 0:
            print(f"Ep {episode+1} | reward médio últimos 200={np.mean(rewards[-200:]):.4f}")


    # Último ponto
    print("Último ponto escolhido:", chosen_points[-1])

    # Curva de recompensas
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa média")
    plt.title("Curva de recompensa")
    plt.show()

    # Evolução dos pontos escolhidos
    chosen_points = np.array(chosen_points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(chosen_points[:, 0], chosen_points[:, 1], chosen_points[:, 2], marker="o")
    ax.set_title("Evolução dos pontos escolhidos")
    plt.show()


if __name__ == "__main__":
    main()