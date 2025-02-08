from typing import cast

import numpy as np

from rubiks_cube_3_3_3 import (
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
    t_max: int
    faces = [Face.RIGHT, Face.BOTTOM, Face.BACK]
    directions = [Direction.CLOCKWISE, Direction.HALF_TURN, Direction.COUNTERCLOCKWISE]
    size: int = 2
    depths = list(range(size - 1))

    class Corners(VariableParent):
        class x(VariableX[CornerPos]):
            def compute_id(self) -> int:
                return self.pos + self.idx * 8 + self.t * 64 + 1

            @staticmethod
            def parent() -> type["Var.Corners"]:
                return Var.Corners

            @staticmethod
            def n_vars() -> int:
                return 64 * (Var.t_max + 1)

            @staticmethod
            def from_int(var: int) -> Variable:
                var -= 1
                return Var.Corners.x(
                    var % 8,  # type: ignore
                    (var // 8) % 8,  # type: ignore
                    (var // 64) % (Var.t_max + 1),
                )

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
                return 24 * (Var.t_max + 1)

            @staticmethod
            def from_int(var: int) -> Variable:
                var -= 1 + Var.Corners.x.n_vars()

                return Var.Corners.theta(
                    var % 8,  # type: ignore
                    (var // 8) % 3,  # type: ignore
                    (var // 24) % (Var.t_max + 1),
                )

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
                pos % Var.size,
                (pos // Var.size) % Var.size,
                (pos // (Var.size**2)) % Var.size,
            )

        @staticmethod
        def g_inv(c_x: int, c_y: int, c_z: int) -> CornerPos:
            return cast(CornerPos, c_x + Var.size * c_y + Var.size * c_z)

    class Edges(VariableParent):
        class x(VariableX[EdgePos]):
            def compute_id(self) -> int:
                return (
                    Var.Corners.n_vars() + self.pos + self.idx * 12 + self.t * 144 + 1
                )

            @staticmethod
            def parent() -> type["Var.Edges"]:
                return Var.Edges

            @staticmethod
            def n_vars() -> int:
                return 144 * (Var.t_max + 1) * 0

            @staticmethod
            def from_int(var: int) -> Variable:
                var -= 1 + Var.Corners.n_vars()

                return Var.Edges.x(
                    var % 12,  # type: ignore
                    (var // 12) % 12,  # type: ignore
                    (var // 144) % (Var.t_max + 1),
                )

        class theta(VariableTheta[EdgePos, EdgeOrientation]):
            def compute_id(self) -> int:
                return (
                    Var.Corners.n_vars()
                    + Var.Edges.x.n_vars()
                    + self.pos
                    + self.orientation * 12
                    + self.t * 24
                    + 1
                )

            @staticmethod
            def parent() -> type["Var.Edges"]:
                return Var.Edges

            @staticmethod
            def n_vars() -> int:
                return 24 * (Var.t_max + 1) * 0

            @staticmethod
            def from_int(var: int) -> Variable:
                var -= 1 + Var.Corners.n_vars() + Var.Edges.x.n_vars()

                return Var.Edges.theta(
                    var % 12,  # type: ignore
                    (var // 12) % 2,  # type: ignore
                    (var // 24) % (Var.t_max + 1),
                )

            def rotate(
                self,
                face: Face,
                direction: Direction,
                depth: int,
            ) -> "Var.Edges.theta":
                if direction == Direction.HALF_TURN:
                    return Var.Edges.theta(self.pos, self.orientation, self.t + 1)

                if face != Face.RIGHT:
                    return Var.Edges.theta(self.pos, self.orientation, self.t + 1)

                if not self.will_rotate(face, depth):
                    return Var.Edges.theta(self.pos, self.orientation, self.t + 1)

                return Var.Edges.theta(self.pos, 1 - self.orientation, self.t + 1)

        @staticmethod
        def n_vars() -> int:
            return Var.Edges.x.n_vars() + Var.Edges.theta.n_vars()

        # @staticmethod
        # def g(pos: EdgePos) -> tuple[int, int, int]:

        # @staticmethod
        # def g_inv(c_x: int, c_y: int, c_z: int) -> EdgePos:
        #     coords = (c_x, c_y, c_z)
        #     axis = np.argmax([0 < p < Var.size - 1 for p in coords])
            
        #     other_coords = [p for i, p in enumerate(coords) if i != axis]

    class Centers(VariableParent):
        class x(VariableX[CenterPos]):
            def compute_id(self) -> int:
                return (
                    Var.Corners.n_vars()
                    + Var.Edges.n_vars()
                    + self.pos
                    + self.idx * 6
                    + self.t * 48
                    + 1
                )

            @staticmethod
            def parent() -> type["Var.Centers"]:
                return Var.Centers

            @staticmethod
            def n_vars() -> int:
                return 48 * (Var.t_max + 1) * 0

            @staticmethod
            def from_int(var: int) -> Variable:
                var -= 1 + Var.Corners.n_vars() + Var.Edges.n_vars()

                return Var.Centers.x(
                    var % 6,  # type: ignore
                    (var // 6) % 6,  # type: ignore
                    (var // 48) % (Var.t_max + 1),
                )

        @staticmethod
        def n_vars() -> int:
            return Var.Centers.x.n_vars()

        @staticmethod
        def g(pos: CenterPos) -> tuple[int, int, int]:
            gamma = (Var.size - 2) ** 2
            alpha = pos // gamma # position sur la face
            beta = pos % gamma # face
            axis = alpha // 2  # Détermine l'axe principal
            i = alpha % 2  # Détermine la position sur l'axe
            
            a = 1 + beta % (Var.size - 2)
            b = 1 + beta // (Var.size - 2)
            pos_axis = i * (Var.size - 1)  # Soit 0, soit Var.size - 1
            
            coordinates = [a, b]
            coordinates.insert(axis, pos_axis)

            c_x, c_y, c_z = coordinates[0], coordinates[1], coordinates[2]

            return c_x, c_y, c_z  # Retourne exactement les 3 coordonnées
        


        @staticmethod
        def g_inv(c_x: int, c_y: int, c_z: int) -> CenterPos:
            gamma = (Var.size - 2)**2
            x_0 = 0
            x_max = gamma
            y_0 = 2 * gamma
            y_max = 3 * gamma
            z_0 = 4 * gamma
            z_max = 5 * gamma

            center_pos = 0 

            if c_x == 0:
                center_pos = x_0 + (c_y -1) + (Var.size - 2) * (c_z - 1)
            
            elif c_x == Var.size - 1:
                center_pos = x_max + (c_y -1) + (Var.size - 2) * (c_z - 1)

            elif c_y == 0:
                center_pos = y_0 + (c_x -1) + (Var.size - 2) * (c_z - 1)

            elif c_y == Var.size - 1:
                center_pos = y_max + (c_x -1) + (Var.size - 2) * (c_z - 1)

            elif c_z == 0:
                center_pos = z_0 + (c_x -1) + (Var.size - 2) * (c_y - 1)

            elif c_z == Var.size - 1:
                center_pos = z_max + (c_x -1) + (Var.size - 2) * (c_y - 1)

            return cast(CenterPos, center_pos)

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
                + self.t * 9 * (Var.size - 1)
                + 1
            )

        @staticmethod
        def n_vars() -> int:
            return 9 * (Var.t_max + 1) * (Var.size - 1)

        @staticmethod
        def from_int(var: int) -> Variable:
            var -= 1 + Var.Corners.n_vars() + Var.Edges.n_vars() + Var.Centers.n_vars()

            return Var.Actions(
                Var.faces[(var // 3) % 3],
                Var.directions[var % 3],
                (var // 9) % (Var.size - 1),
                (var // (9 * (Var.size - 1))),
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
