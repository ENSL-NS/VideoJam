from fractions import Fraction


DEFAULTE_FRAME_RATE = 30

class FrameSampler:
  def __init__(self, fps=30) -> None:
    self.fps = fps
    self.fraction = Fraction(self.fps, DEFAULTE_FRAME_RATE)
    self.index = 1
  
  def keep_next(self) -> bool:
    if self.fps == DEFAULTE_FRAME_RATE:
      return True
    
    if self.index % self.fraction.denominator != 0: # keep
      self.index += 1
      return True
    else: # drop
      self.index = 1
      return False