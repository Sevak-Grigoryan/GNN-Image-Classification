import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
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

from model import GraphSAGE
from utils.directions import IMAGE_H, IMAGE_W


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")
IMAGE_PATH = os.environ.get("GNN_SAMPLE_IMAGE", "path/to/sample_image.bmp")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

DEVICE = "cpu"
SEED = 42

SAMPLE_NODES = IMAGE_H * IMAGE_W  
EDGE_THRESHOLD = 0.0             
NODE_SIZE = 3
EDGE_WIDTH_MIN = 0.2
EDGE_WIDTH_MAX = 2.0

NODE_CMAP = "hot"
EDGE_CMAP = "viridis"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def pixel_positions(H: int, W: int):
    return {r * W + c: (c, -r) for r in range(H) for c in range(W)}

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

def load_model(model_path: str, device: str):
    model = GraphSAGE().to(device)
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
    x = img.reshape(C, -1).t().to(device)  # [N, C]
    edge_index = make_grid_8(H, W).to(device)  # 8-neighbor edges
    return img, x, edge_index

@torch.no_grad()
def forward_layers(model, x, edge_index):
    h1 = F.relu(model.b1(model.conv1(x, edge_index)))
    h2 = F.relu(model.b2(model.conv2(h1, edge_index)))
    h3 = F.relu(model.b3(model.conv3(h2, edge_index)))
    h4 = F.relu(model.b4(model.conv4(h3, edge_index)))
    return [h1, h2, h3, h4]

def compute_node_importance(h: torch.Tensor) -> np.ndarray:
    return h.norm(dim=1).detach().cpu().numpy()

def compute_edge_importance(h: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
    src, dst = edge_index
    sim = F.cosine_similarity(h[src], h[dst], dim=1)
    return sim.detach().cpu().numpy()

def draw_image_graph_with_colorbars(
    ax,
    h: torch.Tensor,
    edge_index: torch.Tensor,
    H: int,
    W: int,
    pos: dict,
    sample_nodes: int = SAMPLE_NODES,
    edge_threshold: float = EDGE_THRESHOLD,
    node_size: int = NODE_SIZE,
    edge_width_min: float = EDGE_WIDTH_MIN,
    edge_width_max: float = EDGE_WIDTH_MAX,
    node_cmap: str = NODE_CMAP,
    edge_cmap: str = EDGE_CMAP,
):
    num_nodes = H * W
    sample_nodes = int(min(sample_nodes, num_nodes))
    idx = np.arange(num_nodes)  # բոլոր նոդերը

    node_imp = compute_node_importance(h)
    edge_imp = compute_edge_importance(h, edge_index)

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
        G, pos, edgelist=list(G.edges()),
        width=widths,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap(edge_cmap),
        edge_vmin=edge_norm.vmin,
        edge_vmax=edge_norm.vmax,
        alpha=0.85,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos, nodelist=idx.tolist(),
        node_size=node_size,
        node_color=node_imp[idx],
        cmap=plt.get_cmap(node_cmap),
        vmin=node_norm.vmin,
        vmax=node_norm.vmax,
        alpha=0.95,
        ax=ax
    )

    ax.axis("off")

    sm_nodes = ScalarMappable(norm=node_norm, cmap=plt.get_cmap(node_cmap))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.045, pad=0.01)
    cbar_nodes.set_label("Node importance (‖hᵢ‖₂)", rotation=90)

    sm_edges = ScalarMappable(norm=edge_norm, cmap=plt.get_cmap(edge_cmap))
    sm_edges.set_array([])
    cbar_edges = plt.colorbar(sm_edges, ax=ax, fraction=0.045, pad=0.08)
    cbar_edges.set_label("Edge importance (cosine sim)", rotation=90)


def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    model = load_model(MODEL_PATH, DEVICE)
    img, x, edge_index = load_image_as_graph_inputs(IMAGE_PATH, IMAGE_H, IMAGE_W, DEVICE)

    _, H, W = img.shape
    pos = pixel_positions(H, W)

    print(f"Image size: {H}x{W}, Number of nodes: {H*W}, Number of edges: {edge_index.shape[1]}")

    layers = forward_layers(model, x, edge_index)

    for i, h in enumerate(layers, start=1):
        fig, ax = plt.subplots(figsize=(9, 9))
        draw_image_graph_with_colorbars(
            ax=ax, h=h, edge_index=edge_index,
            H=H, W=W, pos=pos,
            sample_nodes=SAMPLE_NODES,
            edge_threshold=EDGE_THRESHOLD,
            node_size=NODE_SIZE,
            edge_width_min=EDGE_WIDTH_MIN,
            edge_width_max=EDGE_WIDTH_MAX,
            node_cmap=NODE_CMAP,
            edge_cmap=EDGE_CMAP
        )
        ax.set_title(f"GraphSAGE Image Graph — Layer h{i}", fontsize=14)
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"image_graph_edges_h{i}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
