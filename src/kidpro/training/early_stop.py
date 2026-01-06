from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
  patience: int = 5
  min_delta: float = 0.0
  mode: str = "min"  # "min" or "max"

  best_score: float | None = None
  counter: int = 0
  early_stop: bool = False

  def step(self, score: float) -> bool:
    if self.best_score is None:
      self.best_score = score
      return True

    improved = (
      score < self.best_score - self.min_delta
      if self.mode == "min"
      else score > self.best_score + self.min_delta
    )

    if improved:
      self.best_score = score
      self.counter = 0
      return True

    self.counter += 1
    if self.counter >= self.patience:
      self.early_stop = True
    return False
