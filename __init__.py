# Segmind API Python Client
# Version: 1.0.0

from segmind_api import SegmindAPI
from segmind_models import (
    SDXL,
    SDOutpainting,
    QRGenerator,
    Word2Img,
    BackgroundRemoval,
    Codeformer,
    SAM,
    FaceSwap,
    ControlNet,
    Veo3,
    FluxKontextPro,
    LLaVA13B
)

# For backward compatibility with the original segmind-py package
SD2_1 = SDXL  # Alias for SDXL
Kadinsky = FluxKontextPro  # Alias for FluxKontextPro
SD1_5 = Word2Img  # Alias for Word2Img
ERSGAN = BackgroundRemoval  # Alias for BackgroundRemoval

__all__ = [
    'SegmindAPI',
    'SDXL',
    'SDOutpainting',
    'QRGenerator',
    'Word2Img',
    'BackgroundRemoval',
    'Codeformer',
    'SAM',
    'FaceSwap',
    'ControlNet',
    'Veo3',
    'FluxKontextPro',
    'LLaVA13B',
    # Aliases
    'SD2_1',
    'Kadinsky',
    'SD1_5',
    'ERSGAN'
]