import torch
from typing import Tuple

def detect_3d_edges(cell, threshold=0.5, device='cuda'):
    """
    Detect edges in a 3D tensor using 3D Sobel filters.

    Args:
        cell: torch.Tensor
            A 3D tensor representing the cell volume. The tensor should have shape (C, D, H, W).
        threshold: float
            A threshold value to convert edge responses to binary edges.

    Returns:
        edges: torch.Tensor
            A 3D tensor representing the edge responses.
        binary_edges: torch.Tensor
            A 3D tensor representing the binary edges.
    """

    # Ensure the tensor is on GPU and has the correct shape
    cell = cell.to(device)
    if cell.dim() == 3:
        cell = cell.unsqueeze(0).unsqueeze(0)
    
    def create_sobel_filters():
        sobel_x = torch.tensor([
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
        ], dtype=torch.float32)
        
        sobel_y = sobel_x.permute(1, 0, 2)
        sobel_z = sobel_x.permute(2, 1, 0)
        
        return sobel_x, sobel_y, sobel_z

    sobel_x, sobel_y, sobel_z = create_sobel_filters()
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).cuda()
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).cuda()
    sobel_z = sobel_z.unsqueeze(0).unsqueeze(0).cuda()

    # Apply the filters
    edges_x = torch.nn.functional.conv3d(cell, sobel_x, padding=1)
    edges_y = torch.nn.functional.conv3d(cell, sobel_y, padding=1)
    edges_z = torch.nn.functional.conv3d(cell, sobel_z, padding=1)

    # Combine edge responses
    edges = torch.sqrt(edges_x**2 + edges_y**2 + edges_z**2)

    # Optional: Apply threshold for binary edges
    binary_edges = (edges > threshold).float()

    return edges, binary_edges


def detect_outgoing_edges(edge_cube, threshold=0.5, device='cuda'):
    """
    Detect edges that are going out of the cube.
    
    Args:
    edge_cube (torch.Tensor): The edge detection cube, shape (1, 1, depth, height, width)
    threshold (float): Threshold for considering a point as an edge
    
    Returns:
    tuple: A tuple containing the location of the outgoing edges in the cube (shape (depth, height, width))
    
    """
    # Ensure the tensor is on GPU
    edge_cube = edge_cube.to(device)
    
    # Get the dimensions of the cube
    _, _, depth, height, width = edge_cube.shape
    
    # Create a mask for the surface of the cube
    surface_mask = torch.zeros_like(edge_cube, device='cuda')
    surface_mask[:,:,0,:,:] = 1  # Front face
    surface_mask[:,:,-1,:,:] = 1 # Back face
    surface_mask[:,:,:,0,:] = 1  # Top face
    surface_mask[:,:,:,-1,:] = 1 # Bottom face
    surface_mask[:,:,:,:,0] = 1  # Left face
    surface_mask[:,:,:,:,-1] = 1 # Right face

    # Apply threshold to edge_cube
    edge_points = (edge_cube > threshold).float()

    # Multiply edge_points with surface_mask to get only the edges on the surface
    outgoing_edges = edge_points * surface_mask

    # find locations of outgoing edges
    outgoing_edges = outgoing_edges.squeeze().nonzero().t()
    return outgoing_edges