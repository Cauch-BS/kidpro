import numpy as np
import openslide


# ============================================================
# Magnification utilities
# ============================================================
def _objective_power(slide: openslide.OpenSlide) -> float | None:
  for k in (
    "openslide.objective-power",
    "aperio.AppMag",
    "hamamatsu.XObjective",
  ):
    if k in slide.properties:
      try:
        return float(slide.properties[k])
      except Exception:
        pass
  return None


def pick_level_for_target_mag(
  slide: openslide.OpenSlide, target_mag: float, default_base: float = 40.0
) -> tuple[int, float, float]:
  """
  Selects the best pyramid level for a specific target magnification.

  This function calculates which available level in the Whole Slide Image (WSI)
  is closest to the requested magnification. It also calculates a scaling factor
  to resize that level's image to match the target exactly.

  Args:
      slide (openslide.OpenSlide): The WSI object.
      target_mag (float): The desired magnification (e.g., 20.0 for 20x).
      default_base (float): Fallback base magnification if metadata is missing
                            (usually 40.0 or 20.0).

  Returns:
      tuple[int, float, float]:
          - lvl (int): The index of the best pyramid level to use.
          - scale_img (float): The resize factor needed to adjust the selected
            level to the exact target magnification (e.g., 0.5 means shrink by half).
          - base_mag (float): The detected (or default) base magnification of the slide.
  """

  base_mag = _objective_power(slide) or default_base
  desired_down = base_mag / float(target_mag)

  downs = list(slide.level_downsamples)
  lvl = int(np.argmin([abs(d - desired_down) for d in downs]))

  used_down = downs[lvl]
  scale_img = used_down / desired_down
  return lvl, scale_img, base_mag
