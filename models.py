
from pydantic import BaseModel, Field
from typing import List

class SpeakRequest(BaseModel):
    nums: List[int]
    speed: str = Field(default="Medium", pattern="^(Slow|Medium|Fast)$")