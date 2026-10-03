import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from PIL import Image
import networkx as nx
from model import ChebNet

# CONFIG
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ChebNet_46.pth")
IMAGE_PATH =  os.environ.get("GNN_SAMPLE_IMAGE", "path/to/sample_image.bmp")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

DEVICE = "cuda"
H, W = 80, 50
NODE_SIZE = 3
EDGE_COLOR = "#333333"
EDGE_ALPHA = 0.3

# UTILS
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
    model = ChebNet().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_image(image_path):
    transform = T.Compose([T.Resize((H, W)), T.ToTensor()])
    pil = Image.open(image_path).convert("RGB")
    img = transform(pil)
    x = img.reshape(img.shape[0], -1).t().to(DEVICE)
    return pil, x

@torch.no_grad()
def forward_layers(model, x, edge_index):
    h1 = F.relu(model.norm1(model.conv1(x, edge_index)))
    h2 = F.relu(model.norm2(model.conv2(h1, edge_index)))
    h3 = F.relu(model.norm3(model.conv3(h2, edge_index)))
    # h4 = F.relu(model.norm4(model.conv4(h3, edge_index)))

    return [h1, h2, h3]

def draw_graph(ax, h, edge_index, pos):
    node_colors = h[:,0].cpu().numpy()  # օգտագործում ենք առաջին channel
    G = nx.Graph()
    num_nodes = h.shape[0]
    for n in range(num_nodes): G.add_node(n)
    e0, e1 = edge_index.cpu().numpy()
    for s,d in zip(e0,e1): G.add_edge(s,d)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=0.5, alpha=EDGE_ALPHA, edge_color=EDGE_COLOR, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=range(num_nodes), node_color=node_colors, cmap='hot', node_size=NODE_SIZE, ax=ax)
    ax.set_aspect('equal')
    ax.axis('off')
    sm = ScalarMappable(cmap='hot')
    sm.set_array(node_colors)
    plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.02, label="Embedding (channel 0)")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model(MODEL_PATH, DEVICE)
    pil, x = load_image(IMAGE_PATH)
    edge_index = make_grid_8(H, W).to(DEVICE)
    pos = pixel_positions(H, W)
    layers = forward_layers(model, x, edge_index)

    for i,h in enumerate(layers,start=1):
        fig, axes = plt.subplots(1,2, figsize=(12,6))
        axes[0].imshow(pil)
        axes[0].axis('off')
        axes[0].set_title("Original Image")
        draw_graph(axes[1], h, edge_index, pos)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR,f"layer_{i}_graph.png"), dpi=300)
        plt.close(fig)
        print(f"[SAVED] layer {i} visualization")

if __name__=="__main__":
    main()
