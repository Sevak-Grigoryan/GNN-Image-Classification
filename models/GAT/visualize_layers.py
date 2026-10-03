import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from PIL import Image
import networkx as nx
from model_GAT import GAT

# ---------------- CONFIG ----------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GAT.pth")
IMAGE_PATH = os.environ.get("GNN_SAMPLE_IMAGE", "path/to/sample_image.bmp")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

DEVICE = "cuda"
H, W = 80, 50
NODE_SIZE = 20
EDGE_COLOR = "#333333"
EDGE_ALPHA = 0.3

# ------------- UTILS -------------------
def make_grid_8(H, W):
    edges = []
    for r in range(H):
        for c in range(W):
            idx = r * W + c
            if c < W-1: edges.append([idx, idx+1])
            if r < H-1: edges.append([idx, idx+W])
            if r < H-1 and c < W-1: edges.append([idx, idx+W+1])
            if r < H-1 and c > 0: edges.append([idx, idx+W-1])
    return torch.tensor(edges, dtype=torch.long).t()

def pixel_positions(H, W):
    return {r*W+c:(c,-r) for r in range(H) for c in range(W)}

def load_model(path, device):
    model = GAT().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_image(image_path):
    transform = T.Compose([T.Resize((H, W)), T.ToTensor()])
    pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(pil)
    x = img_tensor.reshape(img_tensor.shape[0], -1).t().to(DEVICE)
    return pil, img_tensor, x

@torch.no_grad()
def forward_layers(model, x, edge_index):
    h1 = F.relu(model.norm1(model.gat1(x, edge_index)))
    h2 = F.relu(model.norm2(model.gat2(h1, edge_index)))
    h3 = F.relu(model.norm3(model.gat3(h2, edge_index)))
    return [h1, h2, h3]

def draw_graph(ax, h, edge_index, pos):
    node_colors = h[:, 0].cpu().numpy()  # use first channel
    G = nx.Graph()
    num_nodes = h.shape[0]
    for n in range(num_nodes):
        G.add_node(n)
    e0, e1 = edge_index.cpu().numpy()
    for s, d in zip(e0, e1):
        G.add_edge(s, d)
    nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(),
        width=0.5, alpha=EDGE_ALPHA,
        edge_color=EDGE_COLOR, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=range(num_nodes),
        node_color=node_colors, cmap='hot',
        node_size=NODE_SIZE, ax=ax
    )
    ax.set_aspect('equal')
    ax.axis('off')
    sm = ScalarMappable(cmap='hot')
    sm.set_array(node_colors)
    plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.02, label="Embedding (channel 0)")

# ---------------- MAIN -----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = load_model(MODEL_PATH, DEVICE)
    pil, img_tensor, x = load_image(IMAGE_PATH)
    edge_index = make_grid_8(H, W).to(DEVICE)
    pos = pixel_positions(H, W)

    layers = forward_layers(model, x, edge_index)

    for i, h in enumerate(layers, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.1]})
        ax_img, ax_graph = axes

        # LEFT: Original Image
        ax_img.imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
        ax_img.set_title("Original Image", fontsize=14)
        ax_img.axis("off")

        # RIGHT: Graph Visualization
        draw_graph(ax_graph, h, edge_index, pos)
        ax_graph.set_title(f"GAT Image Graph — Layer h{i}", fontsize=14)

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"image_vs_graph_layer_{i}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
