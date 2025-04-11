from typing import NamedTuple

class Affine3DTransform(NamedTuple):
    a: float; b: float; c: float
    d: float; e: float; f: float
    g: float; h: float; i: float
    j: float; k: float; l: float

    @classmethod
    def from_list(cls, elements: list[float]) -> 'Affine3DTransform':
        if len(elements) != 12:
            raise ValueError("Affine3DTransform requires 12 elements")
        return cls(*elements)