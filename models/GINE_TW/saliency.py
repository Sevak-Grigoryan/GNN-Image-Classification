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
from torch_geometric.nn import global_mean_pool, global_max_pool

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GINE_TW.pth")
IMAGE_PATH = os.environ.get("GNN_SAMPLE_IMAGE", "path/to/sample_image.bmp")
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

H, W = 80, 50
NODE_SIZE = 3
EDGE_ALPHA = 0.25
EDGE_WIDTH = 0.35
NODE_CMAP = "hot"

# ----------------------------
# Grid graph (8-neighbors)
# ----------------------------
def make_grid_8(H, W):
    edges = []
    for r in range(H):
        for c in range(W):
            i = r * W + c
            if c < W - 1: edges.append([i, i + 1])
            if r < H - 1: edges.append([i, i + W])
            if r < H - 1 and c < W - 1: edges.append([i, i + W + 1])
            if r < H - 1 and c > 0: edges.append([i, i + W - 1])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index

def pixel_positions(H, W):
    return {r * W + c: (c, -r) for r in range(H) for c in range(W)}

# ---------------- IMAGE LOADING ----------------
def load_image_as_x(path, H, W, device):
    transform = T.Compose([T.Resize((H, W)), T.ToTensor()])
    pil = Image.open(path).convert("RGB")
    img = transform(pil)  # [3,H,W]
    x = img.reshape(3, -1).T.to(device)  # [N,3]
    return pil, img, x

# ---------------- MODEL LOADING ----------------
def load_model(path, device):
    from model import GINE
    model = GINE().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# ---------------- FORWARD + LAYERS ----------------
def forward_with_layers(model, data):
    x, edge_index, edge_attr, batch = (
        data.x, data.edge_index, data.edge_attr, data.batch
    )

    edge_attr = edge_attr * model.edge_weight.unsqueeze(0)

    x = model.node_encoder(x)
    edge_attr = model.edge_encoder(edge_attr)

    h1 = F.relu(model.norm1(model.conv1(x, edge_index, edge_attr), batch))
    x = x + h1
    h1.retain_grad()

    h2 = F.relu(model.norm2(model.conv2(x, edge_index, edge_attr), batch))
    x = x + h2
    h2.retain_grad()

    h3 = F.relu(model.norm3(model.conv3(x, edge_index, edge_attr), batch))
    x = x + h3
    h3.retain_grad()

    h4 = F.relu(model.norm4(model.conv4(x, edge_index, edge_attr), batch))
    x = x + h4
    h4.retain_grad()

    xg = global_max_pool(x, batch)
    logits = model.lin2(F.relu(model.lin1(xg)))

    return logits, [h1, h2, h3]
# ---------------- GRAD × ACTIVATION ----------------
def compute_grad_times_activation(model, x, edge_index, target_class=None):
    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    edge_attr = torch.ones(edge_index.size(1), 5, device=x.device)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    logits, hs = forward_with_layers(model, data)
    logits_1d = logits[0]
    pred = int(torch.argmax(logits_1d))
    if target_class is None:
        target_class = pred

    score = logits_1d[target_class]
    model.zero_grad(set_to_none=True)
    score.backward()

    saliencies = [(h.detach() * h.grad).abs().sum(dim=1).cpu().numpy() for h in hs]
    return saliencies, pred, logits_1d.detach().cpu().numpy()

# ---------------- DRAW GRAPH ----------------
def draw_graph_saliency(ax, node_imp, edge_index, pos):
    G = nx.Graph()
    G.add_nodes_from(range(len(node_imp)))
    e0, e1 = edge_index.cpu().numpy()
    for s, d in zip(e0, e1):
        G.add_edge(int(s), int(d))

    nx.draw_networkx_edges(G, pos, width=EDGE_WIDTH, alpha=EDGE_ALPHA, ax=ax)

    vmin, vmax = np.percentile(node_imp, [5, 95])
    norm = Normalize(vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(
        G, pos, node_color=node_imp, node_size=3,
        cmap=NODE_CMAP, vmin=vmin, vmax=vmax, ax=ax
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-H + 0.5, 0.5)
    ax.axis("off")
    sm = ScalarMappable(norm=norm, cmap=NODE_CMAP)
    sm.set_array([])
    plt.colorbar(sm, ax=ax)

# ---------------- MAIN ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = load_model(MODEL_PATH, DEVICE)
    _, img_tensor, x = load_image_as_x(IMAGE_PATH, H, W, DEVICE)
    edge_index = make_grid_8(H, W).to(DEVICE)
    pos = pixel_positions(H, W)

    saliencies, pred, logits = compute_grad_times_activation(model, x, edge_index)

    print("Predicted class:", pred)
    print("Logits:", logits)

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    for i, node_imp in enumerate(saliencies, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_np)
        axes[0].axis("off")
        axes[0].set_title(f"Input ({H}x{W})")

        draw_graph_saliency(axes[1], node_imp, edge_index, pos)
        axes[1].set_title(f"Layer {i}")

        out_file = os.path.join(OUT_DIR, f"GINE_TW_layer_HUMAN_{i}.png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[SAVED]", out_file)

if __name__ == "__main__":
    main()


# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T
# import matplotlib.pyplot as plt
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize
# from PIL import Image
# import networkx as nx

# import torch_geometric
# from torch_geometric.data import Data
# from torch_geometric.nn import global_mean_pool, global_max_pool

# # ----------------------------
# # CONFIG
# # ----------------------------
# MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GINE_TW.pth")
# IMAGE_PATH = os.environ.get("GNN_SAMPLE_IMAGE", "path/to/sample_image.bmp")
# OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# H, W = 80, 50
# NODE_SIZE = 8
# EDGE_ALPHA = 0.25
# EDGE_WIDTH = 0.35
# NODE_CMAP = "hot"

# # ----------------------------
# # Grid graph (8-neighbors)
# # ----------------------------
# def make_grid_8(H, W):
#     edges = []
#     for r in range(H):
#         for c in range(W):
#             i = r * W + c
#             if c < W - 1: edges.append([i, i + 1])
#             if r < H - 1: edges.append([i, i + W])
#             if r < H - 1 and c < W - 1: edges.append([i, i + W + 1])
#             if r < H - 1 and c > 0: edges.append([i, i + W - 1])
#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#     edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
#     return edge_index

# def pixel_positions(H, W):
#     return {r * W + c: (c, -r) for r in range(H) for c in range(W)}

# # ---------------- IMAGE LOADING ----------------
# def load_image_as_x(path, H, W, device):
#     transform = T.Compose([T.Resize((H, W)), T.ToTensor()])
#     pil = Image.open(path).convert("RGB")
#     img = transform(pil)  # [3,H,W]
#     x = img.reshape(3, -1).T.to(device)  # [N,3]
#     return pil, img, x

# # ---------------- MODEL LOADING ----------------
# def load_model(path, device):
#     from model import GINE
#     model = GINE().to(device)
#     state = torch.load(path, map_location=device)
#     model.load_state_dict(state, strict=True)
#     model.eval()
#     return model

# # ---------------- FORWARD + LAYERS ----------------
# def forward_with_layers(model, data):
#     x, edge_index, edge_attr, batch = (
#         data.x, data.edge_index, data.edge_attr, data.batch
#     )

#     edge_attr = edge_attr * model.edge_weight.unsqueeze(0)

#     x = model.node_encoder(x)
#     edge_attr = model.edge_encoder(edge_attr)

#     h1 = F.relu(model.norm1(model.conv1(x, edge_index, edge_attr), batch))
#     x = x + h1
#     h1.retain_grad()

#     h2 = F.relu(model.norm2(model.conv2(x, edge_index, edge_attr), batch))
#     x = x + h2
#     h2.retain_grad()

#     h3 = F.relu(model.norm3(model.conv3(x, edge_index, edge_attr), batch))
#     x = x + h3
#     h3.retain_grad()

#     h4 = F.relu(model.norm4(model.conv4(x, edge_index, edge_attr), batch))
#     x = x + h4
#     h4.retain_grad()

#     xg = global_max_pool(x, batch)
#     logits = model.lin2(F.relu(model.lin1(xg)))

#     return logits, [h1, h2, h3]

# # ---------------- GRAD × ACTIVATION ----------------
# def compute_grad_times_activation(model, x, edge_index, target_class=None):
#     batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
#     edge_attr = torch.ones(edge_index.size(1), 5, device=x.device)
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

#     logits, hs = forward_with_layers(model, data)
#     logits_1d = logits[0]
#     pred = int(torch.argmax(logits_1d))
#     if target_class is None:
#         target_class = pred

#     score = logits_1d[target_class]
#     model.zero_grad(set_to_none=True)
#     score.backward()

#     saliencies = [(h.detach() * h.grad).abs().sum(dim=1).cpu().numpy() for h in hs]
#     return saliencies, pred, logits_1d.detach().cpu().numpy()

# # ---------------- DRAW GRAPH ----------------
# def draw_graph_saliency(fig, ax, node_imp, edge_index, pos, layer_num):
#     G = nx.Graph()
#     G.add_nodes_from(range(len(node_imp)))
#     e0, e1 = edge_index.cpu().numpy()
#     for s, d in zip(e0, e1):
#         G.add_edge(int(s), int(d))

#     nx.draw_networkx_edges(G, pos, width=EDGE_WIDTH, alpha=EDGE_ALPHA, ax=ax)

#     vmin, vmax = np.percentile(node_imp, [5, 95])
#     norm = Normalize(vmin=vmin, vmax=vmax)
#     nx.draw_networkx_nodes(
#         G, pos, node_color=node_imp, node_size=NODE_SIZE,
#         cmap=NODE_CMAP, vmin=vmin, vmax=vmax, ax=ax
#     )

#     ax.set_aspect("equal", adjustable="box")
#     ax.set_xlim(-0.5, W - 0.5)
#     ax.set_ylim(-H + 0.5, 0.5)
#     ax.axis("off")
#     ax.set_title(f"GINE Layer {layer_num}", fontsize=14, fontweight='bold')
    
#     sm = ScalarMappable(norm=norm, cmap=NODE_CMAP)
#     sm.set_array([])
#     plt.colorbar(sm, ax=ax)

# # ---------------- MAIN ----------------
# def main():
#     os.makedirs(OUT_DIR, exist_ok=True)

#     model = load_model(MODEL_PATH, DEVICE)
#     _, img_tensor, x = load_image_as_x(IMAGE_PATH, H, W, DEVICE)
#     edge_index = make_grid_8(H, W).to(DEVICE)
#     pos = pixel_positions(H, W)

#     saliencies, pred, logits = compute_grad_times_activation(model, x, edge_index)

#     print("Predicted class:", pred)
#     print("Logits:", logits)

#     for i, node_imp in enumerate(saliencies, start=1):
#         fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
#         draw_graph_saliency(fig, ax, node_imp, edge_index, pos, i)

#         out_file = os.path.join(OUT_DIR, f"GINE_TW_layer_{i}.png")
#         plt.tight_layout()
#         plt.savefig(out_file, dpi=300, bbox_inches="tight")
#         plt.close(fig)
#         print("[SAVED]", out_file)

# if __name__ == "__main__":
#     main()