from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from rubiks_cube_3_3_3 import (
    Direction,
    Face,
    CornerPos,
    EdgePos,
    CenterPos,
    CornerOrientation,
    EdgeOrientation,
)


class Variable(ABC):
    def __init__(self, id: int) -> None:
        self.id = id
        self.is_true = True

    def __get_subclass_attrs(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k in self.__class__.__init__.__code__.co_varnames
        }

    def copy(self) -> "Variable":
        return self.__class__(**self.__get_subclass_attrs())

    @staticmethod
    def n_vars() -> int: ...

    @staticmethod
    def from_int(var: int) -> "Variable": ...

    def id_repr(self) -> str:
        return ("" if self.is_true else "-") + str(self.id)

    def __neg__(self) -> "Variable":
        neg = self.copy()
        neg.is_true = not neg.is_true
        return neg

    def __invert__(self) -> "Variable":
        return -self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.__get_subclass_attrs()}"

    def __mul__(self, other: int) -> "Variable":
        if other == 1:
            return self
        if other == -1:
            return -self
        raise ValueError(f"Invalid literal {other}")

    def __rmul__(self, other: int) -> "Variable":
        return self * other


Clause = list[Variable]
NamedClause = tuple[str, Clause]

TPos = TypeVar("TPos", CornerPos, EdgePos, CenterPos)
TOrientation = TypeVar("TOrientation", CornerOrientation, EdgeOrientation)


class VariableParent(ABC):
    @staticmethod
    def n_vars() -> int: ...

    @staticmethod
    def g(pos: TPos) -> tuple[int, int, int]: ...

    @staticmethod
    def g_inv(c_x: int, c_y: int, c_z: int) -> TPos: ...


class VariableState(Variable, Generic[TPos], ABC):
    def __init__(self, pos: TPos, t: int) -> None:
        self.pos: TPos = pos
        self.t = t

        super().__init__(self.compute_id())

    @abstractmethod
    def compute_id(self) -> int: ...

    @staticmethod
    def parent() -> type[VariableParent]: ...

    def g(self, pos: TPos) -> tuple[int, int, int]:
        return self.parent().g(pos)

    def g_inv(self, c_x: int, c_y: int, c_z: int) -> TPos:
        return self.parent().g_inv(c_x, c_y, c_z)

    def will_rotate(self, face: Face, depth: int) -> bool:
        c_x, c_y, c_z = self.g(self.pos)

        if face == Face.RIGHT:
            return c_x == 2 - 1 - depth  # TODO: Replace 2 with the size of the cube
        if face == Face.BOTTOM:
            return c_y == 2 - 1 - depth
        if face == Face.BACK:
            return c_z == 2 - 1 - depth
        raise ValueError(f"Invalid face: {face}")

    def rotate_cube(self, face: Face, direction: Direction, depth: int) -> TPos:
        assert face in {Face.RIGHT, Face.BOTTOM, Face.BACK}, f"Invalid face: {face}"

        c_x, c_y, c_z = self.g(self.pos)

        if not self.will_rotate(face, depth):
            return self.pos

        def rotate_1(face: Face, c_x: int, c_y: int, c_z: int) -> tuple[int, int, int]:
            if face == Face.RIGHT:
                return c_x, c_z, 1 - c_y
            if face == Face.BOTTOM:
                return 1 - c_z, c_y, c_x
            if face == Face.BACK:
                return c_y, 1 - c_x, c_z
            raise ValueError(f"Invalid face: {face}")

        def rotate_i(
            face: Face, c_x: int, c_y: int, c_z: int, n_rotation: int
        ) -> tuple[int, int, int]:
            assert n_rotation > 0, f"Invalid rotations: {n_rotation}"

            if n_rotation == 1:
                return rotate_1(face, c_x, c_y, c_z)

            return rotate_i(face, *rotate_1(face, c_x, c_y, c_z), n_rotation - 1)

        return self.g_inv(*rotate_i(face, c_x, c_y, c_z, 1 + direction.value))

    @abstractmethod
    def rotate(
        self, face: Face, direction: Direction, depth: int
    ) -> "VariableState": ...


class VariableX(VariableState[TPos], Generic[TPos], ABC):
    def __init__(self, pos: TPos, idx: TPos, t: int) -> None:
        self.idx: TPos = idx
        super().__init__(pos, t)

    def rotate(self, face: Face, direction: Direction, depth: int) -> "VariableX":
        return self.__class__(
            self.rotate_cube(face, direction, depth),
            self.idx,
            self.t + 1,
        )


class VariableTheta(VariableState[TPos], Generic[TPos, TOrientation], ABC):
    def __init__(self, pos: TPos, orientation: TOrientation, t: int) -> None:
        self.orientation: TOrientation = orientation
        super().__init__(pos, t)
