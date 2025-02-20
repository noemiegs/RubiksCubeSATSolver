from typing import cast
import numpy as np

from utils import (
    CenterIdx,
    Direction,
    EdgeIdx,
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
        class x(VariableX[CornerPos, CornerPos]):
            @classmethod
            def offset(cls) -> int:
                return 1

            @classmethod
            def parent(cls) -> type["Var.Corners"]:
                return Var.Corners

            @classmethod
            def n_idx(cls) -> int:
                return 3

            @classmethod
            def pos_to_idx(cls, pos: CornerPos) -> CornerPos:
                return pos

        class theta(VariableTheta[CornerPos, CornerOrientation]):
            @classmethod
            def n_idx(cls) -> int:
                return 3

            @classmethod
            def offset(cls) -> int:
                return 1 + Var.Corners.x.n_vars()

            @classmethod
            def parent(cls) -> type["Var.Corners"]:
                return Var.Corners

            def rotate(
                self,
                face: Face,
                direction: Direction,
                depth: int,
            ) -> "Var.Corners.theta":
                if not self.will_rotate(face, depth):
                    return Var.Corners.theta(self.pos, self.idx, self.t + 1)

                new_pos = self.rotate_cube(face, direction, depth)

                if direction == Direction.HALF_TURN:
                    return Var.Corners.theta(new_pos, self.idx, self.t + 1)

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
                    new_orientation = s(0, 2, self.idx)
                elif face == Face.BOTTOM:
                    new_orientation = s(0, 1, self.idx)
                elif face == Face.BACK:
                    new_orientation = s(1, 2, self.idx)
                else:
                    raise ValueError(f"Invalid face: {face}")

                return Var.Corners.theta(new_pos, new_orientation, self.t + 1)

        @classmethod
        def n_vars(cls) -> int:
            return Var.Corners.x.n_vars() + Var.Corners.theta.n_vars()

        @classmethod
        def g(cls, pos: CornerPos) -> tuple[int, int, int]:
            return (
                (pos % 2) * (Variable.cube_size - 1),
                ((pos // 2) % 2) * (Variable.cube_size - 1),
                ((pos // 4) % 2) * (Variable.cube_size - 1),
            )

        @classmethod
        def g_inv(cls, c_x: int, c_y: int, c_z: int) -> CornerPos:
            return cast(
                CornerPos,
                int(c_x != 0) + 2 * int(c_y != 0) + 4 * int(c_z != 0),
            )

        @classmethod
        def n_pos(cls) -> int:
            return 8

    class Edges(VariableParent[EdgePos]):
        class x(VariableX[EdgePos, EdgeIdx]):
            @classmethod
            def offset(cls) -> int:
                return 1 + Var.Corners.n_vars()

            @classmethod
            def parent(cls) -> type["Var.Edges"]:
                return Var.Edges

            @classmethod
            def n_idx(cls) -> int:
                if Variable.cube_size <= 2:
                    return 0
                return 4

            @classmethod
            def pos_to_idx(cls, pos: EdgePos) -> EdgeIdx:
                x, y, z = Var.Edges.g(pos)
                axis = np.argmax(
                    [0 < p < Variable.cube_size - 1 for p in [x, y, z]]
                ).item()
                other_coords = [
                    0 if p == 0 else 1 for i, p in enumerate([x, y, z]) if i != axis
                ]
                return cast(EdgeIdx, 4 * axis + 2 * other_coords[1] + other_coords[0])

        class theta(VariableTheta[EdgePos, EdgeOrientation]):
            @classmethod
            def n_idx(cls) -> int:
                return 1

            @classmethod
            def offset(cls) -> int:
                return 1 + Var.Corners.n_vars() + Var.Edges.x.n_vars()

            @classmethod
            def parent(cls) -> type["Var.Edges"]:
                return Var.Edges

            def rotate(
                self,
                face: Face,
                direction: Direction,
                depth: int,
            ) -> "Var.Edges.theta":
                if not self.will_rotate(face, depth):
                    return Var.Edges.theta(self.pos, 0, self.t + 1, self.is_true)

                new_pos = self.rotate_cube(face, direction, depth)

                if direction == Direction.HALF_TURN:
                    return Var.Edges.theta(new_pos, 0, self.t + 1, self.is_true)

                if face != Face.RIGHT and depth == 0:
                    return Var.Edges.theta(new_pos, 0, self.t + 1, self.is_true)

                return Var.Edges.theta(new_pos, 0, self.t + 1, not self.is_true)

        @classmethod
        def n_vars(cls) -> int:
            return Var.Edges.x.n_vars() + Var.Edges.theta.n_vars()

        @classmethod
        def g(cls, pos: EdgePos) -> tuple[int, int, int]:
            axis_pos = (pos % (Variable.cube_size - 2)) + 1

            pos_ = pos // (Variable.cube_size - 2)
            other_coords = [
                (pos_ % 2) * (Variable.cube_size - 1),
                ((pos_ // 2) % 2) * (Variable.cube_size - 1),
            ]

            axis = pos_ // 4
            other_coords.insert(axis, axis_pos)

            return other_coords[0], other_coords[1], other_coords[2]

        @classmethod
        def g_inv(cls, c_x: int, c_y: int, c_z: int) -> EdgePos:
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

        @classmethod
        def n_pos(cls) -> int:
            return 12 * (Variable.cube_size - 2)

    class Centers(VariableParent[CenterPos]):
        class x(VariableX[CenterPos, CenterIdx]):
            @classmethod
            def offset(cls) -> int:
                return 1 + Var.Corners.n_vars() + Var.Edges.n_vars()

            @classmethod
            def parent(cls) -> type["Var.Centers"]:
                return Var.Centers

            @classmethod
            def n_idx(cls) -> int:
                if Variable.cube_size <= 2:
                    return 0
                return 3

            @classmethod
            def pos_to_idx(cls, pos: CenterPos) -> CenterIdx:
                x, y, z = Var.Centers.g(pos)
                axis = np.argmax(
                    [p == 0 or p == Variable.cube_size - 1 for p in [x, y, z]]
                ).item()
                axis_pos = 0 if [x, y, z][axis] == 0 else 1

                return cast(CenterIdx, 2 * axis + axis_pos)

        @classmethod
        def n_vars(cls) -> int:
            return Var.Centers.x.n_vars()

        @classmethod
        def g(cls, pos: CenterPos) -> tuple[int, int, int]:
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

        @classmethod
        def g_inv(cls, c_x: int, c_y: int, c_z: int) -> CenterPos:
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

        @classmethod
        def n_pos(cls) -> int:
            return 6 * (Variable.cube_size - 2) ** 2

    class Actions(Variable):
        def __init__(
            self,
            face: Face,
            direction: Direction,
            depth: int,
            t: int,
            is_true: bool = True,
        ):
            self.face = face
            self.direction = direction
            self.depth = depth

            super().__init__(t, is_true)

        def compute_id(self) -> int:
            return (
                self.offset()
                + Var.directions.index(self.direction)
                + Var.faces.index(self.face) * 3
                + self.depth * 9
                + self.t * 9 * (Variable.cube_size - 1)
            )

        @classmethod
        def offset(cls) -> int:
            return 1 + Var.Corners.n_vars() + Var.Edges.n_vars() + Var.Centers.n_vars()

        @classmethod
        def n_vars(cls) -> int:
            return 9 * (Variable.t_max + 1) * (Variable.cube_size - 1)

        @classmethod
        def from_int(cls, var: int) -> "Var.Actions":
            var -= cls.offset()

            return Var.Actions(
                Var.faces[(var // 3) % 3],
                Var.directions[var % 3],
                (var // 9) % (Variable.cube_size - 1),
                (var // (9 * (Variable.cube_size - 1))),
            )

        def get_params(self) -> tuple[Face, Direction, int, int]:
            return self.face, self.direction, self.depth, self.t

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
