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

from model import GCN   # your model

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GCN.pth")
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

# ----------------------------
# Load model + image
# ----------------------------
def load_model(path, device):
    model = GCN().to(device)
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
# Forward with layer outputs
# ----------------------------
def forward_with_layers(model, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch

    # Forward through GCN layers + BatchNorm + ReLU
    h1 = F.relu(model.bn1(model.conv1(x, edge_index)))
    h1.retain_grad()

    h2 = F.relu(model.bn2(model.conv2(h1, edge_index)))
    h2.retain_grad()

    h3 = F.relu(model.bn3(model.conv3(h2, edge_index)))
    h3.retain_grad()

    h4 = F.relu(model.bn4(model.conv4(h3, edge_index)))
    h4.retain_grad()

    # Global pooling (same as original model)
    g = torch.cat([global_mean_pool(h4, batch),
                   global_max_pool(h4, batch)], dim=1)

    # Dropout + Fully connected layers
    g = F.dropout(g, p=model.dropout, training=model.training)
    g1 = F.relu(model.lin1(g))
    logits = model.lin2(g1)

    # Return logits and list of intermediate embeddings
    return logits, [h1, h2, h3, h4]


# ----------------------------
# Gradient × Activation saliency
# ----------------------------
def compute_grad_times_activation(model, x, edge_index, target_class=None):
    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    data = Data(x=x, edge_index=edge_index, batch=batch)

    logits, hs = forward_with_layers(model, data)
    logits_1d = logits[0]

    pred = int(torch.argmax(logits_1d))
    if target_class is None:
        target_class = pred

    score = logits_1d[target_class]

    model.zero_grad(set_to_none=True)
    score.backward()

    saliencies = []
    for h in hs:
        imp = (h.detach() * h.grad).abs().sum(dim=1)
        saliencies.append(imp.cpu().numpy())

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
    edge_index = make_grid_8(H, W).to(DEVICE)
    pos = pixel_positions(H, W)

    saliencies, pred, logits = compute_grad_times_activation(
        model, x, edge_index
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

        out = os.path.join(OUT_DIR, f"GCN_layer_NOISE_{i}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("[SAVED]", out)

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

# from torch_geometric.data import Data
# from torch_geometric.nn import global_mean_pool, global_max_pool

# from model import GCN  # ձեր GCN model-ը

# # ----------------------------
# # CONFIG
# # ----------------------------
# MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GCN.pth")
# IMAGE_PATH = os.environ.get("GNN_SAMPLE_IMAGE", "path/to/sample_image.bmp")
# OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# H, W = 80, 50
# NODE_SIZE = 3
# EDGE_ALPHA = 0.25
# EDGE_WIDTH = 0.35
# NODE_CMAP = "hot"

# # ----------------------------
# # Helper functions
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

# def load_model(path, device):
#     model = GCN().to(device)
#     state = torch.load(path, map_location=device)
#     model.load_state_dict(state, strict=True)
#     model.eval()
#     return model

# def load_image_as_x(path, H, W, device):
#     transform = T.Compose([T.Resize((H, W)), T.ToTensor()])
#     pil = Image.open(path).convert("RGB")
#     img = transform(pil)              # [3,H,W]
#     x = img.reshape(3, -1).T.to(device)  # [N,3]
#     return pil, img, x

# # ----------------------------
# # Forward with intermediate layers
# # ----------------------------
# def forward_with_layers(model, data):
#     x, edge_index, batch = data.x, data.edge_index, data.batch

#     h1 = F.relu(model.norm1(model.conv1(x, edge_index)))
#     h1.retain_grad()

#     h2_raw = model.conv2(h1, edge_index)
#     h2 = F.relu(model.norm2(h2_raw + model.res2(h1)))
#     h2.retain_grad()

#     h3_raw = model.conv3(h2, edge_index)
#     h3 = F.relu(model.norm3(h3_raw + model.res3(h2)))
#     h3.retain_grad()

#     h4_raw = model.conv4(h3, edge_index)
#     h4 = F.relu(model.norm4(h4_raw + model.res4(h3)))
#     h4.retain_grad()

#     h5_raw = model.conv5(h4, edge_index)
#     h5 = F.relu(model.norm5(h5_raw + model.res5(h4)))
#     h5.retain_grad()

#     # Jumping Knowledge (final layer)
#     h_jk = h1 + h5
#     h_jk.retain_grad()

#     g = torch.cat([global_mean_pool(h_jk, batch),
#                    global_max_pool(h_jk, batch)], dim=1)
#     g.retain_grad()

#     g = F.dropout(g, p=model.dropout, training=False)
#     g1 = F.relu(model.lin1(g))
#     g1.retain_grad()

#     logits = model.lin2(g1)

#     # Return effective saliency tensors
#     return logits, [h1, h2, h3, h4]  # h_jk replaces last layer

# # ----------------------------
# # Gradient × Activation saliency
# # ----------------------------
# def compute_grad_times_activation(model, x, edge_index, target_class=None):
#     batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
#     data = Data(x=x, edge_index=edge_index, batch=batch)

#     logits, hs = forward_with_layers(model, data)
#     logits_1d = logits[0]

#     pred = int(torch.argmax(logits_1d))
#     if target_class is None:
#         target_class = pred

#     score = logits_1d[target_class]

#     model.zero_grad(set_to_none=True)
#     score.backward()

#     saliencies = []
#     for h in hs:
#         imp = (h.detach() * h.grad).abs().sum(dim=1)
#         saliencies.append(imp.cpu().numpy())

#     return saliencies, pred, logits_1d.detach().cpu().numpy()

# # ----------------------------
# # Draw graph saliency
# # ----------------------------
# def draw_graph_saliency(ax, node_imp, edge_index, pos, vmin=None, vmax=None):
#     G = nx.Graph()
#     G.add_nodes_from(range(len(node_imp)))

#     e0, e1 = edge_index.cpu().numpy()
#     for s, d in zip(e0, e1):
#         G.add_edge(int(s), int(d))

#     nx.draw_networkx_edges(
#         G, pos,
#         width=EDGE_WIDTH,
#         alpha=EDGE_ALPHA,
#         ax=ax
#     )

#     if vmin is None or vmax is None:
#         vmin, vmax = np.percentile(node_imp, [5, 95])
#     norm = Normalize(vmin=vmin, vmax=vmax)

#     nx.draw_networkx_nodes(
#         G, pos,
#         node_color=node_imp,
#         node_size=NODE_SIZE,
#         cmap=NODE_CMAP,
#         vmin=vmin,
#         vmax=vmax,
#         ax=ax
#     )

#     ax.set_aspect("equal", adjustable="box")
#     ax.set_xlim(-0.5, W - 0.5)
#     ax.set_ylim(-H + 0.5, 0.5)
#     ax.axis("off")

#     # sm = ScalarMappable(norm=norm, cmap=NODE_CMAP)
#     # sm.set_array([])
#     # plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)

# # ----------------------------
# # MAIN
# # ----------------------------
# def main():
#     os.makedirs(OUT_DIR, exist_ok=True)

#     model = load_model(MODEL_PATH, DEVICE)
#     _, img_tensor, x = load_image_as_x(IMAGE_PATH, H, W, DEVICE)
#     edge_index = make_grid_8(H, W).to(DEVICE)
#     pos = pixel_positions(H, W)

#     saliencies, pred, logits = compute_grad_times_activation(model, x, edge_index)

#     print("Predicted class:", pred)
#     print("Logits:", logits)

#     img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

#     # ---- CONSISTENT COLOR SCALE ----
#     all_vals = np.concatenate(saliencies)
#     vmin, vmax = np.percentile(all_vals, [5, 95])

#     # ---- SINGLE FIGURE: Original + all layers ----
#     num_layers = len(saliencies)
#     fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 5))

#     # Original image
#     axes[0].imshow(img_np)
#     axes[0].axis("off")
#     axes[0].set_title("Original Image")

#     # Saliency per layer
#     for i, node_imp in enumerate(saliencies):
#         draw_graph_saliency(axes[i + 1], node_imp, edge_index, pos, vmin=vmin, vmax=vmax)
#         axes[i + 1].set_title(f"GCN Layer {i + 1}")

#     out_path = os.path.join(OUT_DIR, "GCN_all_layers.png")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     print("[SAVED]", out_path)

# if __name__ == "__main__":
#     main()

# import os
# import torch
# import numpy as np
# import torch.nn.functional as F
# import torchvision.transforms as T
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from PIL import Image
# import networkx as nx
# from torch_geometric.data import Data
# from torch_geometric.nn import global_mean_pool, global_max_pool
# from model import GCN  # your GCN model

# # CONFIG
# MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GCN.pth")
# IMAGE_PATH = os.environ.get("GNN_SAMPLE_IMAGE", "path/to/sample_image.bmp")
# OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CLASS_NAMES = ["Car", "Human", "Noise"]
# H, W = 80, 50
# NODE_SIZE = 4
# EDGE_ALPHA = 0.1
# EDGE_WIDTH = 0.2
# NODE_CMAP = "hot"

# # -------------------
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

# def load_model(path, device):
#     model = GCN().to(device)
#     state = torch.load(path, map_location=device)
#     model.load_state_dict(state)
#     model.eval()
#     return model

# def load_image_as_x(path, H, W, device):
#     transform = T.Compose([T.Resize((H, W)), T.ToTensor()])
#     pil = Image.open(path).convert("RGB")
#     img = transform(pil)
#     x = img.reshape(3, -1).T.to(device)  # [N,3]
#     return pil, img, x

# # -------------------
# def forward_with_layers(model, data):
#     x, edge_index, batch = data.x, data.edge_index, data.batch

#     # Forward through GCN layers + BatchNorm + ReLU
#     h1 = F.relu(model.bn1(model.conv1(x, edge_index)))
#     h1.retain_grad()

#     h2 = F.relu(model.bn2(model.conv2(h1, edge_index)))
#     h2.retain_grad()

#     h3 = F.relu(model.bn3(model.conv3(h2, edge_index)))
#     h3.retain_grad()

#     h4 = F.relu(model.bn4(model.conv4(h3, edge_index)))
#     h4.retain_grad()

#     # Global pooling (same as original model)
#     g = torch.cat([global_mean_pool(h4, batch),
#                    global_max_pool(h4, batch)], dim=1)

#     # Dropout + Fully connected layers
#     g = F.dropout(g, p=model.dropout, training=model.training)
#     g1 = F.relu(model.lin1(g))
#     logits = model.lin2(g1)

#     # Return logits and list of intermediate embeddings
#     return logits, [h1, h2, h3, h4]


# # -------------------
# def compute_grad_times_activation(model, x, edge_index, target_class=None):
#     batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
#     data = Data(x=x, edge_index=edge_index, batch=batch)

#     logits, hs = forward_with_layers(model, data)
#     pred = int(torch.argmax(logits))
#     target_class = pred if target_class is None else target_class

#     model.zero_grad()
#     logits[0, target_class].backward()

#     saliencies = []
#     for h in hs:
#         imp = (h.detach() * h.grad).abs().sum(dim=1).cpu().numpy()
#         saliencies.append(imp)
#     return saliencies, pred, F.softmax(logits, dim=1).detach().cpu().numpy()

# # -------------------
# def draw_graph_saliency(ax, node_imp, edge_index, pos, title):
#     G = nx.Graph()
#     G.add_nodes_from(range(len(node_imp)))
#     e0, e1 = edge_index.cpu().numpy()
#     for s, d in zip(e0, e1):
#         G.add_edge(int(s), int(d))
#     vmin, vmax = np.percentile(node_imp, [5, 95])
#     norm = Normalize(vmin=vmin, vmax=vmax)
#     node_colors = plt.get_cmap(NODE_CMAP)(norm(node_imp))
#     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=NODE_SIZE, ax=ax)
#     nx.draw_networkx_edges(G, pos, width=EDGE_WIDTH, alpha=EDGE_ALPHA, ax=ax)
#     ax.set_aspect("equal")
#     ax.axis("off")
#     ax.set_title(title)

# # -------------------
# def main():
#     os.makedirs(OUT_DIR, exist_ok=True)
#     model = load_model(MODEL_PATH, DEVICE)
#     _, img_tensor, x = load_image_as_x(IMAGE_PATH, H, W, DEVICE)
#     edge_index = make_grid_8(H, W).to(DEVICE)
#     pos = pixel_positions(H, W)

#     saliencies, pred, probs = compute_grad_times_activation(model, x, edge_index)
#     print(f"Predicted: {CLASS_NAMES[pred]} ({probs[0,pred]*100:.2f}%)")

#     img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

#     fig, axes = plt.subplots(1, len(saliencies)+1, figsize=(4*(len(saliencies)+1), 5))
#     axes[0].imshow(img_np)
#     axes[0].axis("off")
#     axes[0].set_title("Original Image")

#     layer_titles = ["Layer1", "Layer2", "Layer3", "Layer4", "JK Fusion"]
#     for i, node_imp in enumerate(saliencies):
#         draw_graph_saliency(axes[i+1], node_imp, edge_index, pos, layer_titles[i])

#     out_path = os.path.join(OUT_DIR, "GCN_Saliency_Full.png")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300)
#     plt.show()
#     print("[SAVED]", out_path)

# if __name__ == "__main__":
#     main()
