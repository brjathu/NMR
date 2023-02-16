import torch
import neural_renderer.cuda.rasterize as rasterize_cuda
import time

DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100
DEFAULT_EPS = 1e-4
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)

def face_visibility(faces, image_size, near=DEFAULT_NEAR, far=DEFAULT_FAR, eps=DEFAULT_EPS):

    batch_size = faces.shape[0]
    device = faces.device
    face_index_map = torch.empty(batch_size, image_size, image_size, device=device, dtype=torch.int).fill_(-1)
    weight_map = torch.empty(batch_size, image_size, image_size, 3, device=device, dtype=torch.float32).fill_(0.0)
    depth_map = torch.empty(batch_size, image_size, image_size, device=device, dtype=torch.float32).fill_(far)
    rgb_map = torch.empty(1, device=device, dtype=torch.float32).fill_(0)
    face_inv_map = torch.empty(1, device=device, dtype=torch.float32).fill_(0)
    num_faces = faces.shape[1]
    block_size = 4
    buffer_size = 512
    face_visibility = torch.empty(batch_size, num_faces, device=device, dtype=torch.int).fill_(0)
    face_list = torch.empty(batch_size, (image_size-1)//block_size+1, (image_size-1)//block_size+1, buffer_size, device=device, dtype=torch.int).fill_(0)
    faces_inv = torch.zeros_like(faces)
    # tstart = time.time()
    _, _, _, _, face_visibility =  rasterize_cuda.forward_face_index_map(faces, face_index_map, weight_map,
                                    depth_map, face_inv_map, faces_inv, face_visibility,
                                    face_list,
                                    image_size, block_size, near, far,
                                    False, False,
                                    False, True)

    return face_visibility
