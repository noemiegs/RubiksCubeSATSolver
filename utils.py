from enum import Enum
from typing import Literal


CornerPos = Literal[0, 1, 2, 3, 4, 5, 6, 7]
EdgePos = int
CenterPos = int
CornerOrientation = Literal[0, 1, 2]
EdgeOrientation = Literal[0, 1]
Size = tuple[int, int, int]


class Color(Enum):
    RED = 0
    ORANGE = 1
    BLUE = 2
    GREEN = 3
    WHITE = 4
    YELLOW = 5

    def to_rgb(self) -> tuple[int, int, int]:
        return {
            Color.RED: (255, 0, 0),
            Color.BLUE: (0, 0, 255),
            Color.GREEN: (0, 255, 0),
            Color.YELLOW: (255, 255, 0),
            Color.ORANGE: (255, 165, 0),
            Color.WHITE: (255, 255, 255),
        }[self]


class Direction(Enum):
    CLOCKWISE = 0
    HALF_TURN = 1
    COUNTERCLOCKWISE = 2

    @staticmethod
    def from_str(s: str) -> "Direction":
        return {
            "": Direction.CLOCKWISE,
            "2": Direction.HALF_TURN,
            "'": Direction.COUNTERCLOCKWISE,
        }[s]

    def opposite(self) -> "Direction":
        return {
            Direction.CLOCKWISE: Direction.COUNTERCLOCKWISE,
            Direction.HALF_TURN: Direction.HALF_TURN,
            Direction.COUNTERCLOCKWISE: Direction.CLOCKWISE,
        }[self]

    def to_str(self) -> str:
        return {
            Direction.CLOCKWISE: "",
            Direction.HALF_TURN: "2",
            Direction.COUNTERCLOCKWISE: "'",
        }[self]

    def __lt__(self, other: "Direction") -> bool:
        return self.value < other.value


class Face(Enum):
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3
    FRONT = 4
    BACK = 5

    def get_vertices_idx(self) -> list[int]:
        return {
            Face.FRONT: [0, 1, 3, 2],
            Face.BACK: [4, 5, 7, 6],
            Face.LEFT: [0, 4, 6, 2],
            Face.RIGHT: [1, 5, 7, 3],
            Face.TOP: [0, 1, 5, 4],
            Face.BOTTOM: [2, 3, 7, 6],
        }[self]

    @staticmethod
    def from_str(s: str) -> "Face":
        return {
            "F": Face.FRONT,
            "B": Face.BACK,
            "L": Face.LEFT,
            "R": Face.RIGHT,
            "U": Face.TOP,
            "D": Face.BOTTOM,
        }[s]

    def opposite(self) -> "Face":
        return {
            Face.FRONT: Face.BACK,
            Face.BACK: Face.FRONT,
            Face.LEFT: Face.RIGHT,
            Face.RIGHT: Face.LEFT,
            Face.TOP: Face.BOTTOM,
            Face.BOTTOM: Face.TOP,
        }[self]

    def to_str(self) -> str:
        return {
            Face.FRONT: "F",
            Face.BACK: "B",
            Face.LEFT: "L",
            Face.RIGHT: "R",
            Face.TOP: "U",
            Face.BOTTOM: "D",
        }[self]

    def __lt__(self, other: "Face") -> bool:
        return self.value < other.value