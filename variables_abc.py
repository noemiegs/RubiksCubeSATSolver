from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar, cast

import numpy as np

from utils import (
    Direction,
    Face,
    CornerPos,
    EdgePos,
    CenterPos,
    CornerOrientation,
    EdgeOrientation,
)


class Variable(ABC):
    cube_size: int
    t_max: int

    def __init__(self, t: int, is_true: bool = True) -> None:
        self.t = t
        self.id = self.compute_id()
        self.is_true = is_true

    def __get_subclass_attrs(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k in self.__class__.__init__.__code__.co_varnames
        }

    def copy(self) -> "Variable":
        return self.__class__(**self.__get_subclass_attrs())

    @abstractmethod
    def compute_id(self) -> int: ...

    @classmethod
    @abstractmethod
    def offset(cls) -> int: ...

    @classmethod
    @abstractmethod
    def n_vars(cls) -> int: ...

    @classmethod
    @abstractmethod
    def from_int(cls, var: int) -> "Variable": ...

    def id_repr(self) -> str:
        return ("" if self.is_true else "-") + str(self.id)

    def __neg__(self) -> "Variable":
        neg = self.copy()
        neg.is_true = not neg.is_true
        return neg

    def __invert__(self) -> "Variable":
        return -self

    def __repr__(self) -> str:
        return f"{'' if self.is_true else 'not '}{self.__class__.__qualname__}{self.__get_subclass_attrs()}"

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
TIdx = TypeVar(
    "TIdx", CornerPos, EdgePos, CenterPos, CornerOrientation, EdgeOrientation
)


class VariableParent(ABC, Generic[TPos]):
    @classmethod
    @abstractmethod
    def n_vars(cls) -> int: ...

    @classmethod
    @abstractmethod
    def g(cls, pos: TPos) -> tuple[int, int, int]: ...

    @classmethod
    @abstractmethod
    def g_inv(cls, c_x: int, c_y: int, c_z: int) -> TPos: ...

    @classmethod
    @abstractmethod
    def n_pos(cls) -> int: ...

    @classmethod
    def pos_range(cls) -> Iterable[TPos]:
        return range(cls.n_pos())  # type: ignore


class VariableState(Variable, Generic[TPos, TIdx], ABC):
    def __init__(self, pos: TPos, idx: TIdx, t: int, is_true: bool = True) -> None:
        self.pos: TPos = pos
        self.idx: TIdx = idx

        super().__init__(t, is_true)

    @classmethod
    @abstractmethod
    def parent(cls) -> type[VariableParent[TPos]]: ...

    @classmethod
    @abstractmethod
    def n_idx(cls) -> int: ...

    @classmethod
    def idx_range(cls) -> Iterable[TIdx]:
        return range(cls.n_idx())  # type: ignore

    @classmethod
    def n_pos(cls) -> int:
        return cls.parent().n_pos()

    @classmethod
    def pos_range(cls) -> Iterable[TPos]:
        return cls.parent().pos_range()

    @classmethod
    def from_int(cls, var: int) -> "VariableState[TPos, TIdx]":
        var -= cls.offset()
        n_pos = cls.n_pos()
        n_idx = cls.n_idx()

        return cls(
            var % n_pos,  # type: ignore
            (var // n_pos) % n_idx,  # type: ignore
            (var // (n_idx * n_pos)) % (Variable.t_max + 1),
        )

    @classmethod
    def n_vars(cls) -> int:
        return cls.n_pos() * cls.n_idx() * (Variable.t_max + 1)

    def compute_id(self) -> int:
        return (
            self.offset()
            + self.pos
            + self.idx * self.n_pos()
            + self.t * self.n_idx() * self.n_pos()
        )

    def g(self, pos: TPos) -> tuple[int, int, int]:
        return self.parent().g(pos)

    def g_inv(self, c_x: int, c_y: int, c_z: int) -> TPos:
        return self.parent().g_inv(c_x, c_y, c_z)

    def will_rotate(self, face: Face, depth: int) -> bool:
        c_x, c_y, c_z = self.g(self.pos)

        if face == Face.RIGHT:
            return c_x == Variable.cube_size - 1 - depth
        if face == Face.BOTTOM:
            return c_y == Variable.cube_size - 1 - depth
        if face == Face.BACK:
            return c_z == Variable.cube_size - 1 - depth
        raise ValueError(f"Invalid face: {face}")

    def rotate_cube(self, face: Face, direction: Direction, depth: int) -> TPos:
        assert face in {Face.RIGHT, Face.BOTTOM, Face.BACK}, f"Invalid face: {face}"

        c_x, c_y, c_z = self.g(self.pos)

        if not self.will_rotate(face, depth):
            return self.pos

        def rotate_1(face: Face, c_x: int, c_y: int, c_z: int) -> tuple[int, int, int]:
            if face == Face.RIGHT:
                return c_x, c_z, Variable.cube_size - c_y - 1
            if face == Face.BOTTOM:
                return Variable.cube_size - c_z - 1, c_y, c_x
            if face == Face.BACK:
                return c_y, Variable.cube_size - c_x - 1, c_z
            raise ValueError(f"Invalid face: {face}")

        def rotate_i(
            face: Face, c_x: int, c_y: int, c_z: int, n_rotation: int
        ) -> tuple[int, int, int]:
            assert n_rotation > 0, f"Invalid rotations: {n_rotation}"

            if n_rotation == 1:
                return rotate_1(face, c_x, c_y, c_z)

            return rotate_i(face, *rotate_1(face, c_x, c_y, c_z), n_rotation - 1)

        return self.g_inv(*rotate_i(face, c_x, c_y, c_z, direction.value))

    @abstractmethod
    def rotate(
        self, face: Face, direction: Direction, depth: int
    ) -> "VariableState[TPos, TIdx]": ...


class VariableX(VariableState[TPos, TIdx], Generic[TPos, TIdx]):
    def rotate(
        self, face: Face, direction: Direction, depth: int
    ) -> "VariableX[TPos, TIdx]":
        return self.__class__(
            self.rotate_cube(face, direction, depth),
            self.idx,
            self.t + 1,
        )

    @classmethod
    def encode(cls, decoded_idx: int) -> tuple[int, ...]:
        return tuple(
            1 if s == "1" else -1
            for s in np.binary_repr(decoded_idx, width=cls.n_idx())
        )

    @classmethod
    def from_decoded(
        cls, pos: TPos, idx_decoded: TIdx, t: int
    ) -> tuple["VariableState[TPos, TIdx]", ...]:
        return tuple(
            sign * cls(pos, cast(TIdx, idx), t)
            for idx, sign in enumerate(cls.encode(idx_decoded))
        )  # type: ignore

    @classmethod
    @abstractmethod
    def pos_to_idx(cls, pos: TPos) -> TIdx: ...


class VariableTheta(VariableState[TPos, TIdx], Generic[TPos, TIdx]): ...
