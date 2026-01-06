import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import openslide


def rasterize_xml_mask(
  xml_path: Path,
  slide: openslide.OpenSlide,
  layer_ids: list[int],
  BASE_MAG: float = 40.0,
  MASK_BASE_MAG: float = 2.5,
) -> tuple[int, dict[int, np.ndarray], np.ndarray]:
  """
  Parses an XML annotation file and rasterizes specified regions into binary masks.

  This function reads vector coordinates from the XML, scales them down from the
  base magnification (e.g., 40x) to a target mask magnification (e.g., 2.5x),
  and draws filled polygons.

  Args:
      xml_path (Path): Path to the XML annotation file (e.g., from Aperio ImageScope).
      slide (openslide.OpenSlide): The source WSI object (used for dimensions).
      layer_ids (list[int]): List of 'Id' attributes from the XML to process
      BASE_MAG (float): The native magnification of the WSI (usually 20x or 40x).
      MASK_BASE_MAG (float): The target magnification for the output mask.

  Returns:
      int: The factor by which the image was downscaled

      dict[int, np.ndarray]: A dictionary mapping layer IDs to binary masks.
                             Each mask is a numpy array of shape (Hm, Wm).

      np.ndarray: A single union mask (np.ndarray) of shape (Hm, Wm).
                  This represents the logical OR of all requested layers (1 where
                  any annotation exists, 0 otherwise).
  """

  # 1. Calculate Downsample Factor
  downsample = int(round(BASE_MAG / MASK_BASE_MAG))

  # 2. Determine Mask Dimensions
  W0, H0 = slide.level_dimensions[0]

  # Calculate mask dimensions using integer division
  Hm, Wm = H0 // downsample, W0 // downsample

  # 3. Initialize Masks
  masks = {lid: np.zeros((Hm, Wm), dtype=np.uint8) for lid in layer_ids}
  union_mask = np.zeros((Hm, Wm), dtype=np.uint8)

  # 4. Parse XML
  try:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
  except ET.ParseError as e:
    warnings.warn(f"Failed to parse XML: {xml_path}. Error: {e}")
    return downsample, masks, union_mask

  # 5. Extract and Draw Polygons
  for anno in root.iter("Annotation"):
    # Get the annotation ID (default to -1 if missing)
    lid = int(anno.attrib.get("Id", -1))

    # Skip layers we didn't ask for
    if lid not in layer_ids:
      continue

    for region in anno.iter("Region"):
      pts = []
      for v in region.iter("Vertex"):
        # Coordinate Transformation:
        # Map global WSI coordinates (Level 0) to Mask coordinates
        x = int(float(v.attrib["X"]) / downsample)
        y = int(float(v.attrib["Y"]) / downsample)
        pts.append((x, y))

      # Need at least 3 points to make a polygon
      if len(pts) >= 3:
        # Reshape to (-1, 1, 2) required by cv2.fillPoly
        cnt = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

        # Draw the filled polygon (1 = foreground)
        # Note: This modifies the array inside the 'masks' dict in-place
        cv2.fillPoly(masks[lid], [cnt], 1)

  # 6. Get Union Mask
  for lid in layer_ids:
    union_mask |= masks[lid]

  return downsample, masks, union_mask
