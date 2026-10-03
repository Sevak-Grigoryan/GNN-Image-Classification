import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# -------------------- CONFIG --------------------
IMG_PATH = os.environ.get("GNN_SAMPLE_IMAGE", "path/to/sample_image.bmp")   # <-- փոխիր քո նկարի path-ով
H, W = 224, 224           # resize size (pixels = nodes)
ALPHA = 8.0              # color similarity scaling
DIAG = True              # 8-neighborhood (all 8 surrounding pixels)

# -------------------- 1) CREATE EDGE INDEX --------------------
def make_grid_edge_index(H, W, diag=True):
    edges = []
    for r in range(H):
        for c in range(W):
            i = r * W + c
            nbrs = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
            if diag:
                nbrs += [(r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1)]
            for rr, cc in nbrs:
                if 0 <= rr < H and 0 <= cc < W:
                    j = rr * W + cc
                    edges.append((i, j))
    edges += [(j,i) for i,j in edges]  # make undirected
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

# -------------------- 2) LOAD IMAGE AND MAKE NODES --------------------
def load_image_tensor(img_path, H, W):
    img = Image.open(img_path).convert("RGB")
    return T.Compose([T.Resize((H,W)), T.ToTensor()])(img)  # [3,H,W]

def image_to_nodes(img_t):
    return img_t.view(3,-1).t()  # [H*W, 3]

# -------------------- 3) COMPUTE EDGE WEIGHTS --------------------
def edge_weights_from_rgb(x, edge_index, alpha=8.0):
    src, dst = edge_index
    diff = x[src] - x[dst]
    dist = torch.linalg.norm(diff, dim=1)
    w = torch.exp(-alpha*dist)   # 0..1
    return w

# -------------------- 4) VISUALIZE FULL GRAPH --------------------
def visualize_full_graph(img_t, edge_index, w, H, W):
    img_np = img_t.permute(1,2,0).cpu().numpy()
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    ww  = w.cpu().numpy()

    plt.figure(figsize=(8,8))
    plt.imshow(img_np)
    plt.axis("off")
    plt.title("Pixel-level Graph (weighted edges)")

    # normalize edge width
    wmin, wmax = float(ww.min()), float(ww.max())
    denom = (wmax-wmin+1e-9)

    # draw all edges
    for s,d,val in zip(src,dst,ww):
        sr, sc = divmod(int(s), W)
        dr, dc = divmod(int(d), W)
        lw = 0.1 + 3.0*((val-wmin)/denom)
        plt.plot([sc,dc],[sr,dr],color='yellow',alpha=0.3,linewidth=lw)

    # draw all nodes
    for r in range(H):
        for c in range(W):
            plt.scatter(c,r,color=img_np[r,c],s=15)

    plt.gca().invert_yaxis()  # to match image coordinate system
    plt.show()

# -------------------- MAIN --------------------
img_t = load_image_tensor(IMG_PATH,H,W)
x = image_to_nodes(img_t)
edge_index = make_grid_edge_index(H,W,diag=DIAG)
w = edge_weights_from_rgb(x,edge_index,alpha=ALPHA)

print("Nodes:", x.shape, "Edges:", edge_index.shape, "Edge weights:", w.shape)
visualize_full_graph(img_t,edge_index,w,H,W)



