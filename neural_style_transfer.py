import jittor as jt
import jittor.transform as transform
from jittor import nn
from PIL import Image
import os

def preprocess(image_path, max_edge=512):
    """
    Load and preprocess an image, keeping aspect ratio, and limiting max edge.

    Args:
        image_path (str): Path to image
        max_edge (int): Maximum size for the longer edge

    Returns:
        img_tensor (jt.Var): Tensor of shape [1, 3, H, W]
    """
    # 1. Load image
    image = Image.open(image_path).convert('RGB')
    w, h = image.size

    # 2. Compute scaling factor to resize long edge
    scale = max_edge / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 3. Define preprocessing transform
    tsfm = transform.Compose([
        transform.Resize((new_h, new_w)),
        transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Apply transform and add batch dimension
    img_tensor = tsfm(image)  # [3, H, W]
    img_tensor = jt.unsqueeze(img_tensor, 0)  # [1, 3, H, W]

    return jt.array(img_tensor)

def deprocess(img_tensor, clip=True):
    """
    Convert a preprocessed image tensor back to a PIL image.

    Args:
        img_tensor (jt.Var or jt.array): Tensor of shape [1, 3, H, W] or [3, H, W]
        clip (bool): Whether to clip pixel values to [0, 1]

    Returns:
        PIL.Image: Deprocessed image
    """
    if len(img_tensor.shape) == 4:
        img_tensor = img_tensor[0]  # [3, H, W]

    # Undo normalization
    mean = jt.Var([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = jt.Var([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img_tensor = img_tensor * std + mean

    # Clip to [0,1]
    if clip:
        img_tensor = jt.clamp(img_tensor, 0.0, 1.0)

    # Convert to HWC and uint8
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
    img_pil = Image.fromarray(img_np)

    return img_pil

class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        vgg,
        content_layer=19,
        style_layers=[1, 5, 10, 17, 24]):
        """
        Extract content representation and style representation from am image using the vgg16 net.
            
        Args:
            vgg: pre-trained vgg net
            content_layer: index of the layer to extract content features (default conv4_2)
            style_layers: list of layer indices to extract style features
                      (default conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
        
        Returns:

            
        """
        self.features = vgg.features
        self.content_layer = content_layer
        self.style_layer = style_layers

    def execute(self, x):
        content_feat = None
        style_feats = []

        for i, layer in self.features.layers.items():
            x = layer(x)

            idx = int(i)

            if idx == self.content_layer:
                content_feat = x

            if idx in self.style_layer:
                style_feats.append(x)
        
        return content_feat, style_feats

def compute_content_loss(synthesis_content_rep, target_content_rep):
    """
    Compute the style loss between the synthesis image and the target style image.
    
    It is computed as the mean squared error (MSE) between two feature maps.

    Args:
        synthesis_content_rep (Tensor): Feature representation of the synthesis image 
                                        at the content layer.
        target_content_rep (Tensor): Feature representation of the target content 
                                     image at the same content layer.

    Returns:
        loss: Scalar content loss.
    """
    # Ensure the target features are detached (no gradients)
    target_content_rep = target_content_rep.stop_grad()

    # Compute mean squared error between synthesis and target content features
    loss = jt.mean((synthesis_content_rep - target_content_rep) ** 2) 

    return loss

def gram_matrix(feature):
    """
    Compute the Gram matrix for a feature map.
    
    Args:
        feature (Tensor): Feature map of shape [N, C, H, W]
    
    Returns:
        G: Gram matrix of shape [N, C, C]
    """
    N, C, H, W = feature.shape
    # flatten spatial dimensions
    F = feature.reshape(N, C, H * W)
    # compute Gram matrix
    G = jt.matmul(F, F.transpose(0, 2, 1)) / (C * H * W)
    return G

def compute_style_loss(synthesis_style_reps, target_style_reps, layer_weights=None):
    """
    Compute the style loss between the synthesis image and the target style image.

    Style loss is calculated as the weighted sum of mean squared errors (MSE) 
    between the Gram matrices of the feature maps at multiple layers.

    Args:
        synthesis_style_reps (list of Tensor): Feature maps of the synthesis image 
                                               from the style layers.
        target_style_reps (list of Tensor): Feature maps of the target style image 
                                            from the same layers.
        layer_weights (list of float, optional): Weight for each layer's style loss.
                                                 If None, all layers have equal weight.

    Returns:
        Tensor: Scalar style loss.
    """
    if layer_weights is None:
        # default: equal weight for all layers
        layer_weights = [1.0 / len(synthesis_style_reps)] * len(synthesis_style_reps)
    
    loss = 0.0
    for w, syn_feat, tgt_feat in zip(layer_weights, synthesis_style_reps, target_style_reps):
        tgt_gram = gram_matrix(tgt_feat.stop_grad())
        syn_gram = gram_matrix(syn_feat)
        loss += w * jt.mean((syn_gram - tgt_gram) ** 2) 

    return loss

def compute_tv_loss(img):
    """
    Compute Total Variation (TV) loss for an image.

    TV loss encourages spatial smoothness in the synthesized image.
    Typically used as a regularizer in style transfer tasks.

    Args:
        img (jt.Var): Synthesized image tensor of shape (1, 3, H, W).

    Returns:
        loss: Scalar TV loss.
    """
    # horizontal differences (left-right)
    diff_x = img[:, :, :, :-1] - img[:, :, :, 1:]

    # vertical differences (top-bottom)
    diff_y = img[:, :, :-1, :] - img[:, :, 1:, :]

    # TV loss = sum of absolute differences (L1)  
    # or squared differences (L2); classic style transfer uses L2.
    loss = jt.sum(diff_x * diff_x) + jt.sum(diff_y * diff_y)

    return loss




