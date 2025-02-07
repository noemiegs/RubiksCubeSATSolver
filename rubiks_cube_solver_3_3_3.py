import subprocess
from typing import Iterable, cast
from itertools import product

from rubiks_cube_3_3_3 import (
    Direction,
    Face,
    RubiksCube,
    Middle,
    CornerPos,
    EdgePos,
    CenterPos,
    CornerOrientation,
    EdgeOrientation,
)


Clause = list[int]
NamedClause = tuple[str, Clause]
ActionTuple = tuple[Face | Middle, Direction]


class Var:
    faces = [Face.RIGHT, Face.BOTTOM, Face.BACK]
    middles = [Middle.FRONT, Middle.RIGHT]

    class Corners:
        @staticmethod
        def x(corner_pos: CornerPos, cube_id: CornerPos, t: int) -> int:
            return corner_pos + cube_id * 8 + t * 64 + 1

        @staticmethod
        def theta(
            corner_pos: CornerPos, corner_orientation: CornerOrientation, t: int
        ) -> int:
            return (
                64 * (RubiksCubeSolver.t_max + 1)
                + 144 * (RubiksCubeSolver.t_max + 1)
                + 48 * (RubiksCubeSolver.t_max + 1)
                + corner_pos
                + corner_orientation * 8
                + t * 24
                + 1
            )

    class Edges:
        @staticmethod
        def x(edge_pos: EdgePos, cube_id: EdgePos, t: int) -> int:
            return (
                64 * (RubiksCubeSolver.t_max + 1)
                + edge_pos
                + cube_id * 12
                + t * 144
                + 1
            )

        @staticmethod
        def theta(edge_pos: EdgePos, edge_orientation: EdgeOrientation, t: int) -> int:
            return (
                64 * (RubiksCubeSolver.t_max + 1)
                + 144 * (RubiksCubeSolver.t_max + 1)
                + 48 * (RubiksCubeSolver.t_max + 1)
                + 24 * (RubiksCubeSolver.t_max + 1)
                + edge_pos
                + edge_orientation * 12
                + t * 24
                + 1
            )

    class Centers:
        @staticmethod
        def x(center_pos: CenterPos, cube_id: CenterPos, t: int) -> int:
            return (
                64 * (RubiksCubeSolver.t_max + 1)
                + 144 * (RubiksCubeSolver.t_max + 1)
                + center_pos
                + cube_id * 6
                + t * 48
                + 1
            )

    class Actions(int):
        def __init__(self, face_or_middle: Face | Middle, direction: Direction, t: int):
            self.face_or_middle = face_or_middle
            self.direction = direction.value
            self.t = t

        def __repr__(self):
            return f"Action({self.face_or_middle}, {self.direction}, t={self.t})"

        @staticmethod
        def a_face(face: Face, direction: Direction, t: int) -> int:
            assert 0 < t <= RubiksCubeSolver.t_max, f"Invalid time: {t}"

            return (
                64 * (RubiksCubeSolver.t_max + 1)
                + 144 * (RubiksCubeSolver.t_max + 1)
                + 48 * (RubiksCubeSolver.t_max + 1)
                + 24 * (RubiksCubeSolver.t_max + 1)
                + 24 * (RubiksCubeSolver.t_max + 1)
                + Var.faces.index(face) * 3
                + direction.value
                + t * 9
                + 1
            )

        @staticmethod
        def a_middle(middle: Middle, direction: Direction, t: int) -> int:
            assert 0 < t <= RubiksCubeSolver.t_max, f"Invalid time: {t}"

            return (
                64 * (RubiksCubeSolver.t_max + 1)
                + 144 * (RubiksCubeSolver.t_max + 1)
                + 48 * (RubiksCubeSolver.t_max + 1)
                + 24 * (RubiksCubeSolver.t_max + 1)
                + 24 * (RubiksCubeSolver.t_max + 1)
                + 9 * (RubiksCubeSolver.t_max + 1)
                + Var.middles.index(middle) * 3
                + direction.value
                + t * 6
                + 1
            )

        @staticmethod
        def is_action(var: int) -> bool:
            return (64 + 144 + 48 + 24 + 24) * (RubiksCubeSolver.t_max + 1) < var

        @staticmethod
        def get_action_from(a: int) -> tuple[Face | Middle, Direction, int]:
            assert Var.Actions.is_action(a), f"Invalid action: {a}, {Var.decode(a)}"
            a -= (64 + 144 + 48 + 24 + 24) * (RubiksCubeSolver.t_max + 1) + 1
            if a < 9 * (RubiksCubeSolver.t_max + 1):
                t = a // 9
                a = a % 9
                face = a // 3
                direction = a % 3
                return Var.faces[face], Direction(direction), t
            else:
                t = a // 6
                a = a % 6
                middle = a // 3
                direction = a % 3
                return Var.middles[middle], Direction(direction), t

    @staticmethod
    def n_vars() -> int:
        return (64 + 144 + 48 + 24 + 24 + 9 + 6) * (RubiksCubeSolver.t_max + 1)

    @staticmethod
    def decode(var: int) -> str:
        # To change
        prefix = "not " if var < 0 else ""
        var = abs(var)

        if var <= 64 * (RubiksCubeSolver.t_max + 1):
            t = (var - 1) // 64
            cube_pos = (var - 1) % 64
            cube_id = cube_pos // 8
            cube_pos %= 8
            return prefix + f"x({cube_pos}, {cube_id}, {t})"

        if var <= (64 + 24) * (RubiksCubeSolver.t_max + 1):
            new_var = var - 64 * (RubiksCubeSolver.t_max + 1) - 1
            t = new_var // 24
            cube_pos = new_var % 24
            orientation = cube_pos // 8
            cube_pos %= 8
            return prefix + f"theta({cube_pos}, {orientation}, {t})"

        face, direction, t = Var.Actions.get_action_from(var)
        return prefix + f"a({Face(face).name}, {Direction(direction).name}, {t})"

    @staticmethod
    def g_corner(cube_pos: CornerPos) -> tuple[int, int, int]:
        # To change
        return cube_pos % 2, (cube_pos // 2) % 2, (cube_pos // 4) % 2

    @staticmethod
    def g_edge(cube_pos: EdgePos) -> tuple[int, int, int]:
        # To change
        return cube_pos % 2, (cube_pos // 2) % 2, (cube_pos // 4) % 2

    @staticmethod
    def g_center(cube_pos: CenterPos) -> tuple[int, int, int]:
        # To change
        return cube_pos % 2, (cube_pos // 2) % 2, (cube_pos // 4) % 2

    @staticmethod
    def will_rotate(c_x: int, c_y: int, c_z: int, face: Face) -> bool:
        if face == Face.RIGHT:
            return c_x == 1
        if face == Face.BOTTOM:
            return c_y == 1
        if face == Face.BACK:
            return c_z == 1
        return False

    @staticmethod
    def rotate_x_corner(
        face: Face, direction: Direction, cube_pos: CornerPos
    ) -> CornerPos:
        assert face in {Face.RIGHT, Face.BOTTOM, Face.BACK}, f"Invalid face: {face}"

        c_x, c_y, c_z = Var.g_corner(cube_pos)

        if not Var.will_rotate(c_x, c_y, c_z, face):
            return cube_pos

        def rotate_1_x_corner(face: Face, c_x: int, c_y: int, c_z: int) -> CornerPos:
            if face == Face.RIGHT:
                return cast(CornerPos, c_x + 2 * c_z + 4 * (1 - c_y))
            if face == Face.BOTTOM:
                return cast(CornerPos, (1 - c_z) + 2 * c_y + 4 * c_x)
            if face == Face.BACK:
                return cast(CornerPos, c_y + 2 * (1 - c_x) + 4 * c_z)
            raise ValueError(f"Invalid face: {face}")

        if direction == Direction.CLOCKWISE:
            return rotate_1_x_corner(face, c_x, c_y, c_z)
        if direction == Direction.HALF_TURN:
            return rotate_1_x_corner(
                face, *Var.g_corner(rotate_1_x_corner(face, c_x, c_y, c_z))
            )
        if direction == Direction.COUNTERCLOCKWISE:
            return rotate_1_x_corner(
                face,
                *Var.g_corner(
                    rotate_1_x_corner(
                        face, *Var.g_corner(rotate_1_x_corner(face, c_x, c_y, c_z))
                    )
                ),
            )

    @staticmethod
    def rotate_x_edge(face: Face, direction: Direction, edge_pos: EdgePos) -> EdgePos:
        # to change
        assert face in {Face.RIGHT, Face.BOTTOM, Face.BACK}, f"Invalid face: {face}"

        c_x, c_y, c_z = Var.g_edge(edge_pos)

        if not Var.will_rotate(c_x, c_y, c_z, face):
            return edge_pos

        def rotate_1_x_corner(face: Face, c_x: int, c_y: int, c_z: int) -> CornerPos:
            if face == Face.RIGHT:
                return cast(CornerPos, c_x + 2 * c_z + 4 * (1 - c_y))
            if face == Face.BOTTOM:
                return cast(CornerPos, (1 - c_z) + 2 * c_y + 4 * c_x)
            if face == Face.BACK:
                return cast(CornerPos, c_y + 2 * (1 - c_x) + 4 * c_z)
            raise ValueError(f"Invalid face: {face}")

        if direction == Direction.CLOCKWISE:
            return rotate_1_x_corner(face, c_x, c_y, c_z)
        if direction == Direction.HALF_TURN:
            return rotate_1_x_corner(
                face, *Var.g_corner(rotate_1_x_corner(face, c_x, c_y, c_z))
            )
        if direction == Direction.COUNTERCLOCKWISE:
            return rotate_1_x_corner(
                face,
                *Var.g_corner(
                    rotate_1_x_corner(
                        face, *Var.g_corner(rotate_1_x_corner(face, c_x, c_y, c_z))
                    )
                ),
            )

    @staticmethod
    def rotate_x_edge_middle(
        middle: Middle, direction: Direction, edge_pos: EdgePos
    ) -> EdgePos:
        # to change
        # assert middle in {Middle.FRONT, Middle.RIGHT}, f"Invalid middle: {middle}"

        c_x, c_y, c_z = Var.g_edge(edge_pos)

        # if not Var.will_rotate(c_x, c_y, c_z, face):
        #     return edge_pos

        def rotate_1_x_corner(
            middle: Middle, c_x: int, c_y: int, c_z: int
        ) -> CornerPos:
            if middle == Face.RIGHT:
                return cast(CornerPos, c_x + 2 * c_z + 4 * (1 - c_y))
            if middle == Face.BOTTOM:
                return cast(CornerPos, (1 - c_z) + 2 * c_y + 4 * c_x)
            if middle == Face.BACK:
                return cast(CornerPos, c_y + 2 * (1 - c_x) + 4 * c_z)
            raise ValueError(f"Invalid face: {middle}")

        if direction == Direction.CLOCKWISE:
            return rotate_1_x_corner(middle, c_x, c_y, c_z)
        if direction == Direction.HALF_TURN:
            return rotate_1_x_corner(
                middle, *Var.g_corner(rotate_1_x_corner(middle, c_x, c_y, c_z))
            )
        if direction == Direction.COUNTERCLOCKWISE:
            return rotate_1_x_corner(
                middle,
                *Var.g_corner(
                    rotate_1_x_corner(
                        middle, *Var.g_corner(rotate_1_x_corner(middle, c_x, c_y, c_z))
                    )
                ),
            )

    @staticmethod
    def rotate_theta_corner(
        face: Face,
        direction: Direction,
        corner_pos: CornerPos,
        orientation: CornerOrientation,
    ) -> CornerOrientation:
        if direction == Direction.HALF_TURN:
            return orientation

        if not Var.will_rotate(*Var.g_corner(corner_pos), face):
            return orientation

        def s(
            i: CornerOrientation, j: CornerOrientation, orientation: CornerOrientation
        ) -> CornerOrientation:
            if orientation == i:
                return j
            if orientation == j:
                return i
            return orientation

        if face == Face.RIGHT:
            return s(0, 2, orientation)
        if face == Face.BOTTOM:
            return s(0, 1, orientation)
        if face == Face.BACK:
            return s(1, 2, orientation)
        raise ValueError(f"Invalid face: {face}")

    @staticmethod
    def rotate_theta_edge_face(
        # to change
        face: Face,
        direction: Direction,
        corner_pos: CornerPos,
        orientation: CornerOrientation,
    ) -> CornerOrientation:
        if direction == Direction.HALF_TURN:
            return orientation

        if not Var.will_rotate(*Var.g_corner(corner_pos), face):
            return orientation

        def s(
            i: CornerOrientation, j: CornerOrientation, orientation: CornerOrientation
        ) -> CornerOrientation:
            if orientation == i:
                return j
            if orientation == j:
                return i
            return orientation

        if face == Face.RIGHT:
            return s(0, 2, orientation)
        if face == Face.BOTTOM:
            return s(0, 1, orientation)
        if face == Face.BACK:
            return s(1, 2, orientation)
        raise ValueError(f"Invalid face: {face}")

    @staticmethod
    def rotate_theta_edge_middle(
        # to change
        face: Face,
        direction: Direction,
        corner_pos: CornerPos,
        orientation: CornerOrientation,
    ) -> CornerOrientation:
        if direction == Direction.HALF_TURN:
            return orientation

        if not Var.will_rotate(*Var.g_corner(corner_pos), face):
            return orientation

        def s(
            i: CornerOrientation, j: CornerOrientation, orientation: CornerOrientation
        ) -> CornerOrientation:
            if orientation == i:
                return j
            if orientation == j:
                return i
            return orientation

        if face == Face.RIGHT:
            return s(0, 2, orientation)
        if face == Face.BOTTOM:
            return s(0, 1, orientation)
        if face == Face.BACK:
            return s(1, 2, orientation)
        raise ValueError(f"Invalid face: {face}")


class RubiksCubeSolver:
    t_max: int = 11

    def __init__(
        self, rubiks_cube: RubiksCube, t_max: int = 11, cnf_filename="rubiks_cube.cnf"
    ):
        RubiksCubeSolver.t_max = t_max

        self.rubiks_cube = rubiks_cube
        self.cnf_filename = cnf_filename

    def generate_clauses(self) -> list[NamedClause]:
        """
        Génère les clauses.
        """
        clauses: list[NamedClause] = self.generate_initial_clauses()
        cube_pos = cast(Iterable[CornerPos], range(8))
        orientations = cast(Iterable[CornerOrientation], range(3))

        # Etat final
        for id in cube_pos:
            clauses.append(
                (
                    f"Etat final, position du cube {id}",
                    [Var.Corners.x(id, id, self.t_max)],
                )
            )
            clauses.append(
                (
                    f"Etat final, orientation du cube {id}",
                    [Var.Corners.theta(id, 0, self.t_max)],
                )
            )

        for t in range(1, self.t_max + 1):
            # Ajout des clauses pour forcer une action par étape
            clauses.append(
                (
                    f"Action obligatoire à chaque étape, temps {t}",
                    [
                        Var.Actions.a_face(f, d, t)
                        for (f, d) in product(Var.faces, Direction)
                    ],
                )
            )

            for f, d in product(Var.faces, Direction):
                for f_prime, d_prime in product(Var.faces, Direction):
                    if (f, d) < (f_prime, d_prime):
                        clauses.append(
                            (
                                f"Interdiction de rotations multiples, temps {t}, face {f}, {f_prime} et direction {d}, {d_prime}",
                                [
                                    -Var.Actions.a_face(f, d, t),
                                    -Var.Actions.a_face(f_prime, d_prime, t),
                                ],
                            )
                        )

                for c in cube_pos:
                    c_prime = Var.rotate_x_corner(f, d, c)
                    action = Var.Actions.a_face(f, d, t)

                    # Transitions des positions
                    for id in cube_pos:
                        x_prime = Var.Corners.x(c_prime, id, t)
                        x = Var.Corners.x(c, id, t - 1)

                        clauses.append(
                            (
                                f"Transition des positions, id_cube {id}, case_cube {c}, face {f},  direction {d}, temps {t}, clause 1",
                                [x_prime, -x, -action],
                            )
                        )

                        clauses.append(
                            (
                                f"Transition des positions, id_cube {id}, case_cube {c}, face {f},  direction {d}, temps {t}, clause 2",
                                [-x_prime, x, -action],
                            )
                        )

                    # Transitions des rotations
                    for o in orientations:
                        theta_prime = Var.Corners.theta(
                            c_prime, Var.rotate_theta_corner(f, d, c, o), t
                        )
                        theta = Var.Corners.theta(c, o, t - 1)

                        clauses.append(
                            (
                                f"Transition des orientations, case_cube {c}, orientation {o}, face {f},  direction {d}, temps {t}, clause 1",
                                [theta_prime, -theta, -action],
                            )
                        )

                        clauses.append(
                            (
                                f"Transition des orientations, case_cube {c}, orientation {o}, face {f},  direction {d}, temps {t}, clause 2",
                                [-theta_prime, theta, -action],
                            )
                        )

        return clauses

    def generate_cnf_file(self, clauses: list[Clause]) -> None:
        """
        Génère le fichier CNF pour le problème.
        """
        # Écriture du fichier CNF
        with open(self.cnf_filename, "w") as f:
            f.write(f"p cnf {Var.n_vars()} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")

    def verify(
        self,
        vars: dict[int, bool],
        clauses: list[NamedClause],
        default_var_value: bool = False,
    ) -> tuple[bool, list[NamedClause]]:
        """
        Vérifie si les variables sont satisfaisantes pour les clauses.

        Retourne un tuple (booléen, liste de clauses insatisfaites).
        """
        unsat_clauses: list[NamedClause] = []

        for clause in clauses:
            if not any(
                vars.get(abs(var), default_var_value) == (var > 0) for var in clause[1]
            ):
                unsat_clauses.append(clause)

        return len(unsat_clauses) == 0, unsat_clauses

    def solve(self, clauses: list[Clause]) -> tuple[bool, list[str], list[ActionTuple]]:
        """
        Exécute Gophersat et récupère le résultat.
        """
        self.generate_cnf_file(clauses)

        result = subprocess.run(
            ["gophersat", "--verbose", "rubiks_cube.cnf"],
            capture_output=True,
            text=True,
        )

        return self.parse_output(result.stdout)

    def parse_output(self, output: str) -> tuple[bool, list[str], list[ActionTuple]]:
        """
        Analyse la sortie de Gophersat et retourne les actions et positions trouvées.
        """
        if "UNSATISFIABLE" in output:
            return False, [], []

        variables: list[int] = []

        for line in output.splitlines():
            if line.startswith("v "):  # Ligne contenant les variables SAT
                values = map(int, line[2:].strip().split())
                variables.extend([v for v in values if v > 0])

        actions: list[tuple[Face | Middle, Direction, int]] = [
            Var.Actions.get_action_from(a)
            for a in variables
            if Var.Actions.is_action(a)
        ]

        return (
            True,
            [Var.decode(v) for v in variables],
            [
                (face, direction)
                for face, direction, t in sorted(actions, key=lambda a: a[2])
            ],
        )

    def generate_initial_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []

        for cube_pos in range(8):
            cube_pos = cast(CornerPos, cube_pos)

            colors = self.rubiks_cube.get_colors_from_pos(Var.g_corner(cube_pos))
            real_cube_id, real_orientation = (
                self.rubiks_cube.colors_to_id_and_orientation(colors)
            )

            for cube_id in range(8):
                cube_id = cast(CornerPos, cube_id)

                sign = 1 if cube_id == real_cube_id else -1
                clauses.append(
                    (
                        "Initial state position",
                        [sign * Var.Corners.x(cube_pos, cube_id, 0)],
                    )
                )

            for orientation in range(3):
                orientation = cast(CornerOrientation, orientation)

                sign = 1 if orientation == real_orientation else -1
                clauses.append(
                    (
                        "Initial state orientation",
                        [sign * Var.Corners.theta(cube_pos, orientation, 0)],
                    )
                )

        return clauses

    def run(
        self, true_instance: dict[int, bool] | None = None
    ) -> tuple[bool, list[ActionTuple]]:
        """
        Gère tout le processus : génération du CNF, exécution du solveur et extraction du résultat.

        true_instance : dictionnaire des variables SAT à forcer à True (Pour debug uniquement).
        """
        clauses = self.generate_clauses()
        sat, result, actions = self.solve([clauses[1] for clauses in clauses])

        if true_instance is not None and not sat:
            sat, unsat_clauses = self.verify(true_instance, clauses)

            for unsat_clause in unsat_clauses[-8:]:
                print(unsat_clause[0])
                print([Var.decode(v) for v in unsat_clause[1]])
                print()

        for line in result:
            print(line)

        return sat, actions

    def find_optimal(
        self, t_min: int = -1, t_max: int = 11
    ) -> tuple[bool, list[ActionTuple]]:
        """
        Trouve la solution optimale.
        """

        sat: bool = False
        actions: list[ActionTuple] = []

        while t_min < t_max - 1:
            t = (t_min + t_max) // 2

            RubiksCubeSolver.t_max = t
            sat_, actions_ = self.run()

            if sat_:
                t_max = t
                sat = True
                actions = actions_

            else:
                t_min = t

        return sat, actions
