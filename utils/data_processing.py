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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from utils.directions import (
    DATASET_ROOT,
    CLASSES,
    IMAGE_H,
    IMAGE_W,
    GRID_H,
    GRID_W
)


def make_grid(grid_h: int,
              grid_w: int) -> torch.Tensor:

    """
    Creates a 2D grid graph connectivity (edge index) for a grid of given height and width.

    Each cell in the grid is treated as a node, and edges are created between
    4-connected neighboring cells (up, down, left, right).
    The resulting graph is undirected, meaning that for every edge (i → j),
    the reverse edge (j → i) is also included.

    Node indexing follows row-major order:
        node_id = row * grid_w + col

    Args:
        grid_h (int): Number of rows (height) in the grid.
        grid_w (int): Number of columns (width) in the grid.

    Returns:
        torch.Tensor: A tensor of shape (2, E) representing the edge indices,
                      where E is the number of edges. The first row contains
                      source node indices, and the second row contains
                      destination node indices.

    Example:
        >>> edge_index = make_grid(2, 3)
        >>> edge_index.shape
        torch.Size([2, E])
    """

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

    """
    Converts a 3D image tensor into a 2D tensor by flattening the spatial dimensions.

    Args:
        image (torch.Tensor): A 3D tensor representing an image with shape (C, H, W),
                              where C is the number of channels (e.g., 3 for RGB),
                              H is the height, and W is the width of the image.

    Returns:
        torch.Tensor: A 2D tensor with shape (H*W, C), where each row corresponds 
                      to a pixel and each column corresponds to a channel of the image.

    Example:
        >>> image = torch.randn(3, 80, 50)  # Example RGB image of size 64x64
        >>> nodes = image2nodes(image)
        >>> nodes.shape
        torch.Size([4000, 3])  # 4000 pixels, 3 channels
    """

    C,H,W = image.shape
    x = image.reshape(C, -1).t()
    return x

class Image2Graph(torch.utils.data.Dataset):

    """
    A custom dataset class that converts images into graph representations, where each image is
    treated as a graph with nodes corresponding to pixels and edges defined by grid neighbors.

    Each image is transformed into a set of nodes and edges, where the image is divided into a grid
    of size (grid_h, grid_w). Each pixel in the grid is represented as a node, and edges are created
    based on the 4-connected neighbors (up, down, left, right).

    The dataset can be used for tasks like graph-based neural networks or vision transformers that
    operate on graph data.

    Args:
        root_dir (str): The root directory containing subdirectories of image classes. Each subdirectory
                         corresponds to a class, and it contains the images belonging to that class.
                         Default is `DATASET_ROOT`.
        classes (List[str]): A list of class names (subdirectory names) to classify the images into.
                             Default is `CLASSES`.
        grid_h (int): The number of rows in the grid representation of the image. Default is `GRID_H`.
        grid_w (int): The number of columns in the grid representation of the image. Default is `GRID_W`.
        image_h (int): The height of the image after resizing. Default is `IMAGE_H`.
        image_w (int): The width of the image after resizing. Default is `IMAGE_W`.

    Attributes:
        class2idx (dict): A dictionary mapping each class name to a unique index.
        transform (torchvision.transforms.Compose): A transformation pipeline to resize the image and
                                                    convert it to a tensor.
        samples (list): A list of tuples, where each tuple contains the path to an image and its class label.
        edge_index (torch.Tensor): The edge index representing the connectivity of nodes in the grid.
    
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a graph representation of the image at the given index `idx`, including:
                          - `x`: A tensor of node features (pixel values).
                          - `edge_index`: A tensor representing edges between nodes (pixels).
                          - `y`: The class label of the image.

    Example:
        dataset = Image2Graph(root_dir='/path/to/dataset', classes=['class1', 'class2'])
        data = dataset[0]  # Returns a graph representation of the first image
        print(data.x.shape)  # Shape of the node features tensor
        print(data.edge_index.shape)  # Shape of the edge index tensor
    """

    def __init__(
            self,
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

    def __getitem__(self, idx: int) -> torch_geometric.data.Data:
        img_path, label = self.samples[idx]

        image = PIL.Image.open(img_path).convert("RGB")
        image = self.transform(image)

        x = image2nodes(image)
        assert x.size(0) == self.grid_h * self.grid_w, \
            f"x has {x.size(0)} nodes, but grid is {self.grid_h}x{self.grid_w}"

        data = torch_geometric.data.Data(
            x=x,
            edge_index=self.edge_index,
            y=torch.tensor(label, dtype=torch.long)
        )
        return data