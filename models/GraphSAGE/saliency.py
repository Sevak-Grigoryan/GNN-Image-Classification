import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image
import networkx as nx

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, global_mean_pool

from model import GraphSAGE  
# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GraphSAGE.pth")
IMAGE_PATH = os.environ.get("GNN_SAMPLE_IMAGE", "path/to/sample_image.bmp")
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

H, W = 80, 50
NODE_SIZE = 3
EDGE_ALPHA = 0.25
EDGE_WIDTH = 0.35
NODE_CMAP = "hot"

# ----------------------------
# Grid graph (8-neighbors) with edge attributes
# ----------------------------
def make_grid_8_with_edges(H, W, device):
    edges = []
    edge_attrs = []
    
    for r in range(H):
        for c in range(W):
            i = r * W + c
            neighbors = []
            
            # Right
            if c < W - 1: 
                neighbors.append((i + 1, [1, 0, 0, 0, 0]))  # direction: right
            # Down
            if r < H - 1: 
                neighbors.append((i + W, [0, 1, 0, 0, 0]))  # direction: down
            # Down-right diagonal
            if r < H - 1 and c < W - 1: 
                neighbors.append((i + W + 1, [0, 0, 1, 0, 0]))  # direction: down-right
            # Down-left diagonal
            if r < H - 1 and c > 0: 
                neighbors.append((i + W - 1, [0, 0, 0, 1, 0]))  # direction: down-left
            
            for j, attr in neighbors:
                edges.append([i, j])
                edge_attrs.append(attr)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # Make undirected by flipping edges
    edge_index_flip = edge_index.flip(0)
    edge_attr_flip = edge_attr.clone()
    
    edge_index = torch.cat([edge_index, edge_index_flip], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr_flip], dim=0)
    
    return edge_index.to(device), edge_attr.to(device)

def pixel_positions(H, W):
    return {r * W + c: (c, -r) for r in range(H) for c in range(W)}

# ----------------------------
# Load model + image
# ----------------------------
def load_model(path, device):
    model = GraphSAGE().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def load_image_as_x(path, H, W, device):
    transform = T.Compose([T.Resize((H, W)), T.ToTensor()])
    pil = Image.open(path).convert("RGB")
    img = transform(pil)              # [3,H,W]
    x = img.reshape(3, -1).T.to(device)  # [N,3]
    return pil, img, x

# ----------------------------
# Forward with layer outputs (GINE version)
# ----------------------------
def forward_with_layers(model, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch

    # ----------------------------
    # Block 1
    # ----------------------------
    h1 = model.conv1(x, edge_index)
    h1 = F.relu(model.norm1(h1))
    h1.retain_grad()
    x1 = F.dropout(h1, p=0.2, training=False)  # no training, just inference

    # ----------------------------
    # Block 2 (residual)
    # ----------------------------
    h2 = model.conv2(x1, edge_index)
    h2 = x1 + F.elu(model.norm2(h2))  # residual connection
    h2.retain_grad()
    x2 = F.dropout(h2, p=0.2, training=False)

    # ----------------------------
    # Block 3 (residual)
    # ----------------------------
    h3 = model.conv3(x2, edge_index)
    h3 = x2 + F.elu(model.norm3(h3))
    h3.retain_grad()
    x3 = h3  # no dropout applied here

    # ----------------------------
    # Block 4 (residual)
    # ----------------------------
    # You don't have norm4, so using norm3 again (or you can add norm4 to the model)
    h4 = model.conv4(x3, edge_index)
    h4 = x3 + F.elu(model.norm3(h4))
    h4.retain_grad()

    # ----------------------------
    # Global pooling + MLP head
    # ----------------------------
    g = torch.cat([
        global_mean_pool(h4, batch),
        global_max_pool(h4, batch)
    ], dim=1)

    g1 = F.relu(model.fc1(g))
    logits = model.fc2(g1)

    return logits, [h1, h2, h3, h4]


def compute_grad_times_activation(model, x, edge_index, edge_attr, target_class=None):
    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    logits, hs = forward_with_layers(model, data)
    logits_1d = logits[0]

    pred = int(torch.argmax(logits_1d))
    target = target = target_class if target_class is not None else pred
    
    score = logits_1d[target]
    model.zero_grad()
    score.backward(retain_graph=True)

    saliencies = []
    for h in hs:
    
        weights = torch.mean(h.grad, dim=0, keepdim=True)
        cam = (weights * h).sum(dim=1) 
        
        cam = F.relu(cam)
        
        cam_np = cam.detach().cpu().numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
            
        saliencies.append(cam_np)

    return saliencies, pred, logits_1d.detach().cpu().numpy()
# ----------------------------
# Draw graph saliency (correct geometry)
# ----------------------------
def draw_graph_saliency(ax, node_imp, edge_index, pos):
    G = nx.Graph()
    G.add_nodes_from(range(len(node_imp)))

    e0, e1 = edge_index.cpu().numpy()
    for s, d in zip(e0, e1):
        G.add_edge(int(s), int(d))

    nx.draw_networkx_edges(
        G, pos,
        width=EDGE_WIDTH,
        alpha=EDGE_ALPHA,
        ax=ax
    )

    vmin, vmax = np.percentile(node_imp, [5, 95])
    norm = Normalize(vmin=vmin, vmax=vmax)

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_imp,
        node_size=NODE_SIZE,
        cmap=NODE_CMAP,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-H + 0.5, 0.5)
    ax.axis("off")

    sm = ScalarMappable(norm=norm, cmap=NODE_CMAP)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)

# ----------------------------
# MAIN
# ----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = load_model(MODEL_PATH, DEVICE)
    _, img_tensor, x = load_image_as_x(IMAGE_PATH, H, W, DEVICE)
    edge_index, edge_attr = make_grid_8_with_edges(H, W, DEVICE)
    pos = pixel_positions(H, W)

    saliencies, pred, logits = compute_grad_times_activation(
        model, x, edge_index, edge_attr
    )

    print("Predicted class:", pred)
    print("Logits:", logits)

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    for i, node_imp in enumerate(saliencies, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(img_np)
        axes[0].axis("off")
        axes[0].set_title(f"Input ({H}×{W})")

        draw_graph_saliency(
            axes[1], node_imp, edge_index, pos
        )
        axes[1].set_title(f"Layer {i}")

        out = os.path.join(OUT_DIR, f"GraphSage_FALSE_CAR-HUMAN_{i}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("[SAVED]", out)

if __name__ == "__main__":
    main()