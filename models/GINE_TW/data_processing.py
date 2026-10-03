# import os
# import sys
# import typing
# import PIL
# import torch
# import torchvision
# import torch_geometric
# from torch_geometric.data import Data


# from directions import (
#     DATASET_ROOT,
#     CLASSES,
#     IMAGE_H,
#     IMAGE_W,
#     GRID_H,
#     GRID_W
# )


# def make_grid(grid_h: int, grid_w: int):
#     edges = []
#     edge_pairs = []

#     for r in range(grid_h):
#         for c in range(grid_w):
#             src = r * grid_w + c
#             neighbors = [
#                 (r - 1, c),
#                 (r + 1, c),
#                 (r, c - 1),
#                 (r, c + 1)
#             ]

#             for nr, nc in neighbors:
#                 if 0 <= nr < grid_h and 0 <= nc < grid_w:
#                     dst = nr * grid_w + nc
#                     edges.append((src, dst))
#                     edge_pairs.append((src, dst))

#     edges = edges + [(j, i) for i, j in edges]
#     edge_pairs = edge_pairs + [(j, i) for i, j in edge_pairs]

#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#     edge_pairs = torch.tensor(edge_pairs, dtype=torch.long)

#     return edge_index, edge_pairs



# def image2nodes(image: torch.Tensor) -> torch.Tensor:

#     C, H, W = image.shape
#     x = image.reshape(C, -1).t()
#     return x



# def spatial_edge_attr(edge_pairs: torch.Tensor, grid_w: int) -> torch.Tensor:
#     src, dst = edge_pairs[:, 0], edge_pairs[:, 1]

#     r1, c1 = src // grid_w, src % grid_w
#     r2, c2 = dst // grid_w, dst % grid_w

#     dr = (r2 - r1).unsqueeze(1).float()
#     dc = (c2 - c1).unsqueeze(1).float()

#     return torch.cat([dr, dc], dim=1)  # [E, 2]


# def compute_edge_attr(
#     x: torch.Tensor,
#     edge_pairs: torch.Tensor,
#     grid_w: int
# ) -> torch.Tensor:
#     """
#     x: [num_nodes, 3]
#     edge_pairs: [num_edges, 2]
#     return: [num_edges, 5]
#     """

#     src = edge_pairs[:, 0]
#     dst = edge_pairs[:, 1]

#     color_diff = torch.abs(x[src] - x[dst])       # [E, 3]
#     spatial = spatial_edge_attr(edge_pairs, grid_w)  # [E, 2]

#     edge_attr = torch.cat([color_diff, spatial], dim=1)
#     return edge_attr

# class Image2Graph(torch.utils.data.Dataset):

#     def __init__(
#         self,
#         root_dir: str = DATASET_ROOT,
#         classes: typing.List[str] = CLASSES,
#         grid_h: int = GRID_H,
#         grid_w: int = GRID_W,
#         image_h: int = IMAGE_H,
#         image_w: int = IMAGE_W,
#     ):
#         self.root_dir = root_dir
#         self.classes = classes
#         self.grid_h = grid_h
#         self.grid_w = grid_w
#         self.image_h = image_h
#         self.image_w = image_w

#         self.class2idx = {c: i for i, c in enumerate(classes)}

#         self.transform = torchvision.transforms.Compose([
#             torchvision.transforms.Resize((self.image_h, self.image_w)),
#             torchvision.transforms.ToTensor(),
#         ])

#         self.samples = []
#         for class_name in classes:
#             folder = os.path.join(root_dir, class_name)
#             if not os.path.exists(folder):
#                 continue

#             for fname in os.listdir(folder):
#                 if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#                     self.samples.append(
#                         (os.path.join(folder, fname),
#                          self.class2idx[class_name])
#                     )

#         self.edge_index, self.edge_pairs = make_grid(
#             self.grid_h, self.grid_w
#         )

#         print(f"[INFO] Loaded {len(self.samples)} images")
#         print(f"[INFO] Nodes per graph: {self.grid_h * self.grid_w}")
#         print(f"[INFO] Edge dim for GINE: 5")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx: int) -> Data:
#         img_path, label = self.samples[idx]

#         image = PIL.Image.open(img_path).convert("RGB")
#         image = self.transform(image)

#         x = image2nodes(image)

#         assert x.size(0) == self.grid_h * self.grid_w

#         edge_attr = compute_edge_attr(
#             x,
#             self.edge_pairs,
#             self.grid_w
#         )

#         data = Data(
#             x=x,
#             edge_index=self.edge_index,
#             edge_attr=edge_attr, 
#             y=torch.tensor(label, dtype=torch.long)
#         )

#         return data


import os
import sys
import typing
import PIL
import torch
import torchvision
import torch_geometric
from torch_geometric.data import Data


from directions import (
    DATASET_ROOT,
    CLASSES,
    IMAGE_H,
    IMAGE_W,
    GRID_H,
    GRID_W
)


def make_grid(grid_h: int, grid_w: int):
    edges = []
    edge_pairs = []

    for r in range(grid_h):
        for c in range(grid_w):
            src = r * grid_w + c
            neighbors = [
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1)
            ]

            for nr, nc in neighbors:
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    dst = nr * grid_w + nc
                    edges.append((src, dst))
                    edge_pairs.append((src, dst))

    edges = edges + [(j, i) for i, j in edges]
    edge_pairs = edge_pairs + [(j, i) for i, j in edge_pairs]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_pairs = torch.tensor(edge_pairs, dtype=torch.long)

    return edge_index, edge_pairs



def image2nodes(image: torch.Tensor) -> torch.Tensor:

    C, H, W = image.shape
    x = image.reshape(C, -1).t()
    return x



def spatial_edge_attr(edge_pairs: torch.Tensor, grid_w: int) -> torch.Tensor:
    src, dst = edge_pairs[:, 0], edge_pairs[:, 1]

    r1, c1 = src // grid_w, src % grid_w
    r2, c2 = dst // grid_w, dst % grid_w

    dr = (r2 - r1).unsqueeze(1).float()
    dc = (c2 - c1).unsqueeze(1).float()

    return torch.cat([dr, dc], dim=1)  # [E, 2]


def compute_edge_attr(
    x: torch.Tensor,
    edge_pairs: torch.Tensor,
    grid_w: int
) -> torch.Tensor:
    """
    x: [num_nodes, 3]
    edge_pairs: [num_edges, 2]
    return: [num_edges, 5]
    """

    src = edge_pairs[:, 0]
    dst = edge_pairs[:, 1]

    color_diff = torch.abs(x[src] - x[dst])       # [E, 3]
    spatial = spatial_edge_attr(edge_pairs, grid_w)  # [E, 2]

    edge_attr = torch.cat([color_diff, spatial], dim=1)
    return edge_attr

class Image2Graph(torch.utils.data.Dataset):

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

        self.class2idx = {c: i for i, c in enumerate(classes)}

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_h, self.image_w)),
            torchvision.transforms.ToTensor(),
        ])

        self.samples = []
        for class_name in classes:
            folder = os.path.join(root_dir, class_name)
            if not os.path.exists(folder):
                continue

            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append(
                        (os.path.join(folder, fname),
                         self.class2idx[class_name])
                    )

        self.edge_index, self.edge_pairs = make_grid(
            self.grid_h, self.grid_w
        )

        print(f"[INFO] Loaded {len(self.samples)} images")
        print(f"[INFO] Nodes per graph: {self.grid_h * self.grid_w}")
        print(f"[INFO] Edge dim for GINE: 5")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        img_path, label = self.samples[idx]

        image = PIL.Image.open(img_path).convert("RGB")
        image = self.transform(image)

        x = image2nodes(image)

        assert x.size(0) == self.grid_h * self.grid_w

        edge_attr = compute_edge_attr(
            x,
            self.edge_pairs,
            self.grid_w
        )

        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=edge_attr, 
            y=torch.tensor(label, dtype=torch.long)
        )

        return data