"""
This module defines a custom dataset class `Image2Graph` for converting images into graph representations.
Each image is treated as a graph with nodes representing pixels, and edges defined by the grid-based
neighborhood (up, down, left, right). This can be useful for tasks like graph neural networks or other
graph-based models for image analysis.
The following components are included:

- `make_grid(grid_h, grid_w)`: Generates a grid's edge index, where nodes are connected by edges 
  defined by 4-connected neighbors (up, down, left, right).
  
- `image2nodes(image)`: Converts an image (3D tensor) into a 2D tensor of node features, where each 
  pixel is treated as a node.

- `Image2Graph`: A custom dataset class that loads images from a directory, resizes them, converts 
  them into graph representations, and returns the nodes, edges, and labels for each image.

Dependencies:
- `torch`: The core deep learning library for tensor operations.
- `torch_geometric`: A library for working with graph-structured data.
- `torchvision`: A library for image transformations and dataset loading.
- `PIL`: Python Imaging Library for image loading.

Usage Example:
    dataset = Image2Graph(root_dir='/path/to/dataset', classes=['class1', 'class2'])
    data = dataset[0]  # Returns a graph representation of the first image
    print(data.x.shape)  # Shape of the node features tensor
    print(data.edge_index.shape)  # Shape of the edge index tensor
"""

import os
import PIL
import torch
import typing
import torchvision
import torch_geometric
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  
from utils.directions import (
    DATASET_ROOT,
    CLASSES,
    IMAGE_H,
    IMAGE_W,
    GRID_H,
    GRID_W,
    DEVICE
)


def make_grid(grid_h: int,
              grid_w: int) -> torch.Tensor:


    edges = []
    for row in range(grid_h):
        for col in range(grid_w):
            node_id = row * grid_w + col
            neighbors = [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1)
            ]

            for n_row, n_col in neighbors:
                if 0 <= n_row < grid_h and 0 <= n_col < grid_w:
                    neighbor_id = n_row * grid_w + n_col
                    edges.append((node_id, neighbor_id))
    edges = edges + [(j, i) for i, j in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index
    
def image2nodes(image: torch.Tensor) -> torch.Tensor:


    C,H,W = image.shape
    x = image.reshape(C, -1).t()
    return x

class Image2Graph(torch.utils.data.Dataset):


    def __init__(
            self,
            cnn,
            root_dir: str = DATASET_ROOT,
            classes: typing.List[str] = CLASSES,
            grid_h: int = GRID_H,
            grid_w: int = GRID_W,
            image_h: int = IMAGE_H,
            image_w: int = IMAGE_W,
            
    ):
        self.root_dir = root_dir
        self.classes = classes
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.image_h = image_h
        self.image_w = image_w
        self.cnn = cnn.to('cuda')
        self.cnn.eval()
        for p in self.cnn.parameters():
            p.requires_grad = False

        self.class2idx = {
            label: index
            for index, label in enumerate(classes)
        }
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_h, self.image_w)),
            torchvision.transforms.ToTensor(),
        ])
        self.samples = []
        for class_name in classes:
            folder = os.path.join(root_dir, class_name)
            if not os.path.exists(folder):
                continue

            for filename in os.listdir(folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(folder, filename)
                    label = self.class2idx[class_name]
                    self.samples.append((file_path, label))

        self.edge_index = make_grid(self.grid_h, self.grid_w)

        print(f"[INFO] Loaded {len(self.samples)} images from {root_dir}")
        print(f"[INFO] Each image → {self.grid_h}x{self.grid_w} = {self.grid_h * self.grid_w} nodes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = PIL.Image.open(img_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

        with torch.no_grad():
            feat = self.cnn(image)  # [1, C, H_feat, W_feat]
        # print(f"CNN output shape: {feat.shape}")  

        # CNN output: [1, C, H_feat, W_feat]
        C, H_feat, W_feat = feat.shape[1], feat.shape[2], feat.shape[3]
        x = feat.squeeze(0).permute(1, 2, 0).reshape(-1, C)  # [H_feat*W_feat, C]

        # 4️⃣ Generate edge_index based on CNN output spatial size
        edge_index = make_grid(H_feat, W_feat)

        # 5️⃣ Optional: store actual grid size for debugging
        self.grid_h = H_feat
        self.grid_w = W_feat

        # 6️⃣ Return PyG Data object
        data = torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(label, dtype=torch.long)
        )

        # 7️⃣ Print shapes for debugging (remove in production)
        # print(f"[DEBUG] Image {idx}: x={x.shape}, edge_index={edge_index.shape}, label={label}")

        return data


# from model_CNN import CNN
# cnn = CNN().to(DEVICE)
# dataset = Image2Graph(
#     root_dir=DATASET_ROOT,
#     cnn=cnn,
#     classes=CLASSES,
#     grid_h=GRID_H,
#     grid_w=GRID_W,
#     image_h=IMAGE_H,
#     image_w=IMAGE_W,
# )

# graph_sample = dataset[0] 
# # Print node features
# print("Node features (x) shape:", graph_sample.x.shape)
# # Expected: [grid_h * grid_w, CNN_output_channels], e.g., [12800, 64]

# # Print edge index
# print("Edge index shape:", graph_sample.edge_index.shape)
# # Expected: [2, num_edges], e.g., [2, ~2*grid_h*grid_w]

# # Print label
# print("Label:", graph_sample.y)

# # Optional: peek at some values
# print("First 5 node features:\n", graph_sample.x[:5])
# print("First 10 edges:\n", graph_sample.edge_index[:, :10])
