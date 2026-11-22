"""Generates textures/skin_diffuse.jpg (2048x2048) for the MakeHuman base-mesh UV layout.

MakeHuman's own skin textures live in a separate assets package that is not part of this repository, so a
neutral procedural skin texture is generated here instead: blotchy tone variation, fine grain and pores.
It is deterministic (fixed seed). Replace the file with any other 2048x2048 skin texture to restyle the face.
Requires numpy and Pillow.
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 2048


def layer(rng, cells, size=SIZE):
    """Smooth value noise: random grid upsampled bicubically, range 0..1."""
    g = rng.random((cells, cells)).astype(np.float32)
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def main(out=os.path.join(HERE, "textures", "skin_diffuse.jpg")):
    rng = np.random.default_rng(1234)
    base = np.array([0.80, 0.60, 0.50], dtype=np.float32)
    blotch = layer(rng, 6) * 0.6 + layer(rng, 18) * 0.4                  # slow tone variation
    red = layer(rng, 14)                                                  # redness patches
    pores = layer(rng, 512) - 0.5                                         # fine grain
    tex = base[None, None, :] * (0.90 + 0.18 * blotch[..., None])
    tex[..., 0] += 0.05 * red
    tex[..., 1] -= 0.03 * red
    tex += 0.03 * pores[..., None]
    img = Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8), "RGB")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out, quality=92)
    return out


if __name__ == "__main__":
    print("wrote", main())
