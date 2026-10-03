import os
import torch
import numpy as np
import networkx as nx
from PIL import Image
import torch_geometric
from model import GINE  
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from torch_geometric.data import Data
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from torch_geometric.nn import global_max_pool


# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GINE_48.pth")
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
    model = GINE().to(device)
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
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

    # 1. Encode
    x = F.relu(model.node_encoder(x))
    edge_attr_encoded = F.relu(model.edge_encoder(edge_attr))

    # 2. GINE layer 1 + Residual
    h1_conv = model.conv1(x, edge_index, edge_attr_encoded)
    h1_conv = model.norm1(h1_conv, batch)
    h1_conv = F.elu(h1_conv)
    x = x + h1_conv  # Սա ձեր մոդելի իրական տրամաբանությունն է
    x.retain_grad()
    layer1_out = x

    # 3. GINE layer 2 + Residual
    h2_conv = model.conv2(layer1_out, edge_index, edge_attr_encoded)
    h2_conv = model.norm2(h2_conv, batch)
    h2_conv = F.elu(h2_conv)
    x = layer1_out + h2_conv
    x.retain_grad()
    layer2_out = x

    # 4. GINE layer 3 + Residual
    h3_conv = model.conv3(layer2_out, edge_index, edge_attr_encoded)
    h3_conv = model.norm3(h3_conv, batch)
    h3_conv = F.elu(h3_conv)
    x = layer2_out + h3_conv
    x.retain_grad()
    layer3_out = x

    # 5. Global pooling & Classifier
    g = global_max_pool(layer3_out, batch)
    logits = F.relu(model.lin1(g))
    logits = model.lin2(logits)

    return logits, [layer1_out, layer2_out, layer3_out]

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

        out = os.path.join(OUT_DIR, f"GINE_HUMAN_{i}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("[SAVED]", out)

if __name__ == "__main__":
    main()