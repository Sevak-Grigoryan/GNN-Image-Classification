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

from model_GIN import GIN
from data_processing import make_grid  # must return edge_index [2, E]
from directions import IMAGE_H, IMAGE_W


# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GINE_TW.pth")
IMAGE_PATH = os.environ.get("GNN_SAMPLE_IMAGE", "path/to/sample_image.bmp")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Visualization tuning
SAMPLE_NODES = 4000
EDGE_THRESHOLD = 0.25
NODE_SIZE = 3
EDGE_WIDTH_MIN = 0.2
EDGE_WIDTH_MAX = 2.0

NODE_CMAP = "hot"
EDGE_CMAP = "hot"

# RBF sigma for edge_attr computed from RGB differences
SIGMA = 0.5


# -----------------------------
# UTILS
# -----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pixel_positions(H: int, W: int):
    return {r * W + c: (c, -r) for r in range(H) for c in range(W)}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_model(model_path: str, device: str):
    model = GIN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_image_as_graph_inputs(image_path: str, H: int, W: int, device: str):
    transform = T.Compose([T.Resize((H, W)), T.ToTensor()])
    pil = Image.open(image_path).convert("RGB")
    img = transform(pil)  # [C,H,W]
    C, HH, WW = img.shape
    assert HH == H and WW == W, "Resize mismatch"

    x = img.reshape(C, -1).t().to(device)      # [N, 3]
    edge_index = make_grid(H, W).to(device)    # [2, E]

    # default edge_attr (you can override later with RBF-based one)
    edge_attr = torch.ones((edge_index.size(1), 1), device=device)

    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    return img, x, edge_index, edge_attr, batch


@torch.no_grad()
def forward_layers(model: GIN, x, edge_index, edge_attr, batch):
    x = x[:, :3]
    x = model.encoder(x)

    h1 = model.conv1(x, edge_index, edge_attr)
    h1 = model.norm1(h1, batch)
    h1 = F.relu(h1)
    x1 = x + h1

    h2 = model.conv2(x1, edge_index, edge_attr)
    h2 = model.norm2(h2, batch)
    h2 = F.relu(h2)
    x2 = x1 + h2

    h3 = model.conv3(x2, edge_index, edge_attr)
    h3 = model.norm3(h3, batch)
    h3 = F.relu(h3)
    x3 = x2 + h3

    return [x1, x2, x3]


def compute_node_importance(h: torch.Tensor) -> np.ndarray:
    return h.norm(dim=1).detach().cpu().numpy()


def compute_edge_attr_from_x(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    sigma: float = 0.5
) -> torch.Tensor:
    """
    Build edge_attr [E, 1] from RGB similarity using an RBF kernel:
      w = exp(-||rgb_src - rgb_dst||^2 / (2*sigma^2))
    """
    rgb = x[:, :3]  # [N, 3]

    src = edge_index[0]
    dst = edge_index[1]

    diff = rgb[src] - rgb[dst]                # [E, 3]
    dist_sq = (diff * diff).sum(dim=1)        # [E]
    w = torch.exp(-dist_sq / (2 * sigma * sigma + 1e-12))
    return w.unsqueeze(1).float()             # [E, 1]


def compute_edge_importance(h: torch.Tensor,
                            edge_index: torch.Tensor,
                            edge_attr: torch.Tensor) -> np.ndarray:
    """
    GINE-style edge importance: magnitude of the message (h_src + edge_attr)
    Note: edge_attr is [E,1] and will broadcast across hidden dim.
    """
    src, dst = edge_index
    msg = h[src] + edge_attr   # broadcasting [E,1] -> [E,hidden]
    imp = msg.norm(dim=1)      # [E]
    return imp.detach().cpu().numpy()


def draw_image_graph_with_colorbars(
    ax,
    h: torch.Tensor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    H: int,
    W: int,
    pos: dict,
    sigma: float = 0.5,
    sample_nodes: int = 4000,
    edge_threshold: float = 0.25,
    node_size: int = 3,
    edge_width_min: float = 0.2,
    edge_width_max: float = 2.0,
    node_cmap: str = "hot",
    edge_cmap: str = "viridis",
):
    num_nodes = H * W
    sample_nodes = int(min(sample_nodes, num_nodes))
    if sample_nodes <= 0:
        raise ValueError("sample_nodes must be >= 1")

    idx = np.random.choice(num_nodes, sample_nodes, replace=False)

    node_imp = compute_node_importance(h)  # [N]

    # ✅ compute edge_attr from x correctly (NO self, NO undefined vars)
    edge_attr_rbf = compute_edge_attr_from_x(x, edge_index, sigma=sigma)  # [E,1]
    edge_imp = compute_edge_importance(h, edge_index, edge_attr=edge_attr_rbf)  # [E]

    e0 = edge_index[0].detach().cpu().numpy()
    e1 = edge_index[1].detach().cpu().numpy()
    in_sample = np.isin(e0, idx) & np.isin(e1, idx)

    e0 = e0[in_sample]
    e1 = e1[in_sample]
    edge_imp_f = edge_imp[in_sample]

    strong = edge_imp_f > edge_threshold
    e0 = e0[strong]
    e1 = e1[strong]
    edge_imp_f = edge_imp_f[strong]

    G = nx.Graph()
    for n in idx:
        G.add_node(int(n), value=float(node_imp[n]))
    for s, d, w in zip(e0, e1, edge_imp_f):
        G.add_edge(int(s), int(d), weight=float(w))

    node_norm = Normalize(vmin=float(node_imp[idx].min()), vmax=float(node_imp[idx].max()))

    if G.number_of_edges() > 0:
        w_vals = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
        edge_norm = Normalize(vmin=float(w_vals.min()), vmax=float(w_vals.max()))
        w_scaled = (w_vals - w_vals.min()) / (w_vals.max() - w_vals.min() + 1e-12)
        widths = edge_width_min + (edge_width_max - edge_width_min) * w_scaled
        edge_colors = w_vals
    else:
        edge_norm = Normalize(vmin=0.0, vmax=1.0)
        widths = []
        edge_colors = []

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(G.edges()),
        width=widths,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap(edge_cmap),
        edge_vmin=edge_norm.vmin,
        edge_vmax=edge_norm.vmax,
        alpha=0.85,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=idx.tolist(),
        node_size=node_size,
        node_color=node_imp[idx],
        cmap=plt.get_cmap(node_cmap),
        vmin=node_norm.vmin,
        vmax=node_norm.vmax,
        alpha=0.95,
        ax=ax,
    )

    ax.axis("off")

    sm_nodes = ScalarMappable(norm=node_norm, cmap=plt.get_cmap(node_cmap))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.045, pad=0.01)
    cbar_nodes.set_label("Node importance (‖hᵢ‖₂)", rotation=90)

    sm_edges = ScalarMappable(norm=edge_norm, cmap=plt.get_cmap(edge_cmap))
    sm_edges.set_array([])
    cbar_edges = plt.colorbar(sm_edges, ax=ax, fraction=0.045, pad=0.08)
    cbar_edges.set_label("Edge importance (‖h_src + e‖₂)", rotation=90)


def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    model = load_model(MODEL_PATH, DEVICE)
    img, x, edge_index, edge_attr, batch = load_image_as_graph_inputs(
        IMAGE_PATH, IMAGE_H, IMAGE_W, DEVICE
    )

    _, H, W = img.shape
    print(f"Device: {DEVICE}")
    print(f"Image size: {H}x{W}")
    print(f"Number of nodes: {H*W}")
    print(f"Edge count: {edge_index.shape[1]}")

    pos = pixel_positions(H, W)

    # forward layers using the SAME edge_attr you want for the model
    # If you want the model to use RBF edge_attr too, uncomment these 2 lines:
    # edge_attr = compute_edge_attr_from_x(x, edge_index, sigma=SIGMA).to(DEVICE)
    # (then forward_layers uses that edge_attr)

    layers = forward_layers(model, x, edge_index, edge_attr, batch)

    for i, h in enumerate(layers, start=1):
        fig, ax = plt.subplots(figsize=(9, 9))

        draw_image_graph_with_colorbars(
            ax=ax,
            h=h,
            x=x,                       # ✅ needed for edge_attr computation
            edge_index=edge_index,
            H=H,
            W=W,
            pos=pos,
            sigma=SIGMA,               # ✅ no self.sigma
            sample_nodes=SAMPLE_NODES,
            edge_threshold=EDGE_THRESHOLD,
            node_size=NODE_SIZE,
            edge_width_min=EDGE_WIDTH_MIN,
            edge_width_max=EDGE_WIDTH_MAX,
            node_cmap=NODE_CMAP,
            edge_cmap=EDGE_CMAP,
        )

        ax.set_title(f"GINE (GIN) Image Graph — Layer x{i}", fontsize=14)
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"image_graph_edges_x{i}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
