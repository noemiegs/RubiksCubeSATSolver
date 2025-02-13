from typing import Iterable, cast

import numpy as np

from utils import (
    Direction,
    EdgeOrientation,
    Face,
    CornerPos,
    EdgePos,
    CenterPos,
    CornerOrientation,
)
from variables_abc import Variable, VariableParent, VariableX, VariableTheta


class Var:
    faces = [Face.RIGHT, Face.BOTTOM, Face.BACK]
    directions = [Direction.CLOCKWISE, Direction.HALF_TURN, Direction.COUNTERCLOCKWISE]
    depths: list[int] = []

    class Corners(VariableParent[CornerPos]):
        class x(VariableX[CornerPos]):
            def compute_id(self) -> int:
                return self.pos + self.idx * 8 + self.t * 24 + 1

            @staticmethod
            def parent() -> type["Var.Corners"]:
                return Var.Corners

            @staticmethod
            def n_vars() -> int:
                return 24 * (Variable.t_max + 1)

            @staticmethod
            def from_int(var: int) -> "Var.Corners.x":
                var -= 1
                return Var.Corners.x(
                    var % 8,  # type: ignore
                    (var // 8) % 3,  # type: ignore
                    (var // 24) % (Variable.t_max + 1),
                )

            @staticmethod
            def encode(pos: CornerPos) -> tuple[int, int, int]:
                return 1 if pos & 1 else -1, 1 if pos & 2 else -1, 1 if pos & 4 else -1

            @staticmethod
            def from_decoded(
                pos: CornerPos, decoded: CornerPos, t: int
            ) -> tuple["Var.Corners.x", "Var.Corners.x", "Var.Corners.x"]:
                return tuple(
                    sign * Var.Corners.x(pos, idx, t)
                    for idx, sign in enumerate(Var.Corners.x.encode(decoded))
                )  # type: ignore

        class theta(VariableTheta[CornerPos, CornerOrientation]):
            def compute_id(self) -> int:
                return (
                    Var.Corners.x.n_vars()
                    + self.pos
                    + self.orientation * 8
                    + self.t * 24
                    + 1
                )

            @staticmethod
            def parent() -> type["Var.Corners"]:
                return Var.Corners

            @staticmethod
            def n_vars() -> int:
                return 24 * (Variable.t_max + 1)

            @staticmethod
            def from_int(var: int) -> "Var.Corners.theta":
                var -= 1 + Var.Corners.x.n_vars()

                return Var.Corners.theta(
                    var % 8,  # type: ignore
                    (var // 8) % 3,  # type: ignore
                    (var // 24) % (Variable.t_max + 1),
                )

            @staticmethod
            def orientation_range() -> Iterable[CornerOrientation]:
                return range(3)  # type: ignore

            def rotate(
                self,
                face: Face,
                direction: Direction,
                depth: int,
            ) -> "Var.Corners.theta":
                if not self.will_rotate(face, depth):
                    return Var.Corners.theta(self.pos, self.orientation, self.t + 1)

                new_pos = self.rotate_cube(face, direction, depth)

                if direction == Direction.HALF_TURN:
                    return Var.Corners.theta(new_pos, self.orientation, self.t + 1)

                def s(
                    i: CornerOrientation,
                    j: CornerOrientation,
                    orientation: CornerOrientation,
                ) -> CornerOrientation:
                    if orientation == i:
                        return j
                    if orientation == j:
                        return i
                    return orientation

                if face == Face.RIGHT:
                    new_orientation = s(0, 2, self.orientation)
                elif face == Face.BOTTOM:
                    new_orientation = s(0, 1, self.orientation)
                elif face == Face.BACK:
                    new_orientation = s(1, 2, self.orientation)
                else:
                    raise ValueError(f"Invalid face: {face}")

                return Var.Corners.theta(new_pos, new_orientation, self.t + 1)

        @staticmethod
        def n_vars() -> int:
            return Var.Corners.x.n_vars() + Var.Corners.theta.n_vars()

        @staticmethod
        def g(pos: CornerPos) -> tuple[int, int, int]:
            return (
                (pos % 2) * (Variable.cube_size - 1),
                ((pos // 2) % 2) * (Variable.cube_size - 1),
                ((pos // 4) % 2) * (Variable.cube_size - 1),
            )

        @staticmethod
        def g_inv(c_x: int, c_y: int, c_z: int) -> CornerPos:
            return cast(
                CornerPos,
                int(c_x != 0) + 2 * int(c_y != 0) + 4 * int(c_z != 0),
            )

        @staticmethod
        def pos_range() -> Iterable[CornerPos]:
            return range(8)  # type: ignore

    class Edges(VariableParent[EdgePos]):
        class x(VariableX[EdgePos]):
            def compute_id(self) -> int:
                n_idx = 12 * (Variable.cube_size - 2)
                log_n_idx = int(np.log2(n_idx)) + 1

                return (
                    Var.Corners.n_vars()
                    + self.pos
                    + self.idx * n_idx
                    + self.t * n_idx * log_n_idx
                    + 1
                )

            @staticmethod
            def parent() -> type["Var.Edges"]:
                return Var.Edges

            @staticmethod
            def n_vars() -> int:
                n_idx = 12 * (Variable.cube_size - 2)

                if n_idx == 0:
                    return 0

                log_n_idx = int(np.log2(n_idx)) + 1
                return (Variable.t_max + 1) * n_idx * log_n_idx

            @staticmethod
            def from_int(var: int) -> "Var.Edges.x":
                var -= 1 + Var.Corners.n_vars()
                n_idx = 12 * (Variable.cube_size - 2)
                log_n_idx = int(np.log2(n_idx)) + 1

                return Var.Edges.x(
                    var % n_idx,
                    (var // n_idx) % log_n_idx,
                    (var // (n_idx * log_n_idx)) % (Variable.t_max + 1),
                )

            @staticmethod
            def encode(pos: EdgePos) -> tuple[int, ...]:
                n_idx = 12 * (Variable.cube_size - 2)
                log_n_idx = int(np.log2(n_idx)) + 1

                return tuple(
                    1 if s == "1" else -1 for s in np.binary_repr(pos, width=log_n_idx)
                )

            @staticmethod
            def from_decoded(
                pos: EdgePos, decoded: EdgePos, t: int
            ) -> tuple["Var.Corners.x", ...]:
                return tuple(
                    sign * Var.Edges.x(pos, idx, t)
                    for idx, sign in enumerate(Var.Edges.x.encode(decoded))
                )  # type: ignore

        class theta(VariableTheta[EdgePos, EdgeOrientation]):
            def compute_id(self) -> int:
                n_idx = 12 * (Variable.cube_size - 2)

                return (
                    Var.Corners.n_vars()
                    + Var.Edges.x.n_vars()
                    + self.pos
                    + self.orientation * n_idx
                    + self.t * 2 * n_idx
                    + 1
                )

            @staticmethod
            def parent() -> type["Var.Edges"]:
                return Var.Edges

            @staticmethod
            def n_vars() -> int:
                n_idx = 12 * (Variable.cube_size - 2)
                return 2 * n_idx * (Variable.t_max + 1)

            @staticmethod
            def from_int(var: int) -> "Var.Edges.theta":
                var -= 1 + Var.Corners.n_vars() + Var.Edges.x.n_vars()
                n_idx = 12 * (Variable.cube_size - 2)

                return Var.Edges.theta(
                    var % n_idx,  # type: ignore
                    (var // n_idx) % 2,  # type: ignore
                    (var // (2 * n_idx)) % (Variable.t_max + 1),
                )

            @staticmethod
            def orientation_range() -> Iterable[EdgeOrientation]:
                return range(2)  # type: ignore

            def rotate(
                self,
                face: Face,
                direction: Direction,
                depth: int,
            ) -> "Var.Edges.theta":
                if not self.will_rotate(face, depth):
                    return Var.Edges.theta(self.pos, self.orientation, self.t + 1)

                new_pos = self.rotate_cube(face, direction, depth)

                if direction == Direction.HALF_TURN:
                    return Var.Edges.theta(new_pos, self.orientation, self.t + 1)

                if face != Face.RIGHT and depth == 0:
                    return Var.Edges.theta(new_pos, self.orientation, self.t + 1)

                return Var.Edges.theta(new_pos, 1 - self.orientation, self.t + 1)

        @staticmethod
        def n_vars() -> int:
            return Var.Edges.x.n_vars() + Var.Edges.theta.n_vars()

        @staticmethod
        def g(pos: EdgePos) -> tuple[int, int, int]:
            axis_pos = (pos % (Variable.cube_size - 2)) + 1

            pos_ = pos // (Variable.cube_size - 2)
            other_coords = [
                (pos_ % 2) * (Variable.cube_size - 1),
                ((pos_ // 2) % 2) * (Variable.cube_size - 1),
            ]

            axis = pos_ // 4
            other_coords.insert(axis, axis_pos)

            return other_coords[0], other_coords[1], other_coords[2]

        @staticmethod
        def g_inv(c_x: int, c_y: int, c_z: int) -> EdgePos:
            coords = [c_x, c_y, c_z]

            axis = np.argmax([0 < p < Variable.cube_size - 1 for p in coords]).item()
            other_coords = [int(p != 0) for i, p in enumerate(coords) if i != axis]

            return cast(
                EdgePos,
                (other_coords[0] + other_coords[1] * 2 + axis * 4)
                * (Variable.cube_size - 2)
                + coords[axis]
                - 1,
            )

        @staticmethod
        def pos_range() -> Iterable[EdgePos]:
            return range(12 * (Variable.cube_size - 2))

    class Centers(VariableParent[CenterPos]):
        class x(VariableX[CenterPos]):
            def compute_id(self) -> int:
                n_idx = 6 * (Variable.cube_size - 2) ** 2
                log_n_idx = int(np.log2(n_idx)) + 1

                return (
                    Var.Corners.n_vars()
                    + Var.Edges.n_vars()
                    + self.pos
                    + self.idx * n_idx
                    + self.t * n_idx * log_n_idx
                    + 1
                )

            @staticmethod
            def parent() -> type["Var.Centers"]:
                return Var.Centers

            @staticmethod
            def n_vars() -> int:
                n_idx = 6 * (Variable.cube_size - 2) ** 2

                if n_idx == 0:
                    return 0

                log_n_idx = int(np.log2(n_idx)) + 1
                return n_idx * log_n_idx * (Variable.t_max + 1)

            @staticmethod
            def from_int(var: int) -> "Var.Centers.x":
                var -= 1 + Var.Corners.n_vars() + Var.Edges.n_vars()
                n_idx = 6 * (Variable.cube_size - 2) ** 2
                log_n_idx = int(np.log2(n_idx)) + 1

                return Var.Centers.x(
                    var % n_idx,
                    (var // n_idx) % log_n_idx,
                    (var // (n_idx * log_n_idx)) % (Variable.t_max + 1),
                )

            @staticmethod
            def encode(pos: CenterPos) -> tuple[int, ...]:
                n_idx = 6 * (Variable.cube_size - 2) ** 2
                log_n_idx = int(np.log2(n_idx)) + 1

                return tuple(
                    1 if s == "1" else -1 for s in np.binary_repr(pos, width=log_n_idx)
                )

            @staticmethod
            def from_decoded(
                pos: CenterPos, decoded: CenterPos, t: int
            ) -> tuple["Var.Centers.x", ...]:
                return tuple(
                    sign * Var.Centers.x(pos, idx, t)
                    for idx, sign in enumerate(Var.Centers.x.encode(decoded))
                )  # type: ignore

        @staticmethod
        def n_vars() -> int:
            return Var.Centers.x.n_vars()

        @staticmethod
        def g(pos: CenterPos) -> tuple[int, int, int]:
            gamma = (Variable.cube_size - 2) ** 2
            alpha = pos // gamma
            beta = pos % gamma
            axis = alpha // 2
            i = alpha % 2

            a = 1 + beta % (Variable.cube_size - 2)
            b = 1 + beta // (Variable.cube_size - 2)
            pos_axis = i * (Variable.cube_size - 1)

            coordinates = [a, b]
            coordinates.insert(axis, pos_axis)

            c_x, c_y, c_z = coordinates[0], coordinates[1], coordinates[2]

            return c_x, c_y, c_z

        @staticmethod
        def g_inv(c_x: int, c_y: int, c_z: int) -> CenterPos:
            gamma = (Variable.cube_size - 2) ** 2
            x_0 = 0
            x_max = gamma
            y_0 = 2 * gamma
            y_max = 3 * gamma
            z_0 = 4 * gamma
            z_max = 5 * gamma

            center_pos = 0

            if c_x == 0:
                center_pos = x_0 + (c_y - 1) + (Variable.cube_size - 2) * (c_z - 1)

            elif c_x == Variable.cube_size - 1:
                center_pos = x_max + (c_y - 1) + (Variable.cube_size - 2) * (c_z - 1)

            elif c_y == 0:
                center_pos = y_0 + (c_x - 1) + (Variable.cube_size - 2) * (c_z - 1)

            elif c_y == Variable.cube_size - 1:
                center_pos = y_max + (c_x - 1) + (Variable.cube_size - 2) * (c_z - 1)

            elif c_z == 0:
                center_pos = z_0 + (c_x - 1) + (Variable.cube_size - 2) * (c_y - 1)

            elif c_z == Variable.cube_size - 1:
                center_pos = z_max + (c_x - 1) + (Variable.cube_size - 2) * (c_y - 1)

            return cast(CenterPos, center_pos)

        @staticmethod
        def pos_range() -> Iterable[CenterPos]:
            return range(6 * (Variable.cube_size - 2) ** 2)

    class Actions(Variable):
        def __init__(self, face: Face, direction: Direction, depth: int, t: int):
            self.face = face
            self.direction = direction
            self.depth = depth
            self.t = t

            super().__init__()

        def compute_id(self) -> int:
            return (
                Var.Corners.n_vars()
                + Var.Edges.n_vars()
                + Var.Centers.n_vars()
                + self.direction.value
                + Var.faces.index(self.face) * 3
                + self.depth * 9
                + self.t * 9 * (Variable.cube_size - 1)
                + 1
            )

        @staticmethod
        def n_vars() -> int:
            return 9 * (Variable.t_max + 1) * (Variable.cube_size - 1)

        @staticmethod
        def from_int(var: int) -> "Var.Actions":
            var -= 1 + Var.Corners.n_vars() + Var.Edges.n_vars() + Var.Centers.n_vars()

            return Var.Actions(
                Var.faces[(var // 3) % 3],
                Var.directions[var % 3],
                (var // 9) % (Variable.cube_size - 1),
                (var // (9 * (Variable.cube_size - 1))),
            )

    @staticmethod
    def n_vars() -> int:
        return (
            Var.Corners.n_vars()
            + Var.Edges.n_vars()
            + Var.Centers.n_vars()
            + Var.Actions.n_vars()
        )

    @staticmethod
    def from_int(var: int) -> Variable:
        var_class: list[type[Variable]] = [
            Var.Corners.x,
            Var.Corners.theta,
            Var.Edges.x,
            Var.Edges.theta,
            Var.Centers.x,
            Var.Actions,
        ]

        lower_bound = 1
        for cls in var_class:
            n_vars = cls.n_vars()

            if lower_bound <= var < lower_bound + n_vars:
                return cls.from_int(var)
            lower_bound += n_vars

        raise ValueError(f"Invalid variable: {var}")
