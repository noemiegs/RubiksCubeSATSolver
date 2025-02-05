import subprocess
from typing import cast
from itertools import product

from rubiks_cube import Direction, Face, RubiksCube, CubePos, Orientation


Clause = list[int]
NamedClause = tuple[str, Clause]
Action = tuple[Face, Direction]


class Var:
    faces = [Face.RIGHT, Face.BOTTOM, Face.BACK]

    @staticmethod
    def x(cube_pos: CubePos, cube_id: CubePos, t: int) -> int:
        return cube_pos + cube_id * 8 + t * 64 + 1

    @staticmethod
    def theta(cube_pos: CubePos, orientation: Orientation, t: int) -> int:
        return (
            64 * (RubiksCubeSolver.t_max + 1) + cube_pos + orientation * 8 + t * 24 + 1
        )

    @staticmethod
    def a(face: Face, direction: Direction, t: int) -> int:
        assert 0 < t <= RubiksCubeSolver.t_max, f"Invalid time: {t}"

        return (
            64 * (RubiksCubeSolver.t_max + 1)
            + 24 * (RubiksCubeSolver.t_max + 1)
            + Var.faces.index(face) * 3
            + direction.value
            + t * 9
            + 1
        )

    @staticmethod
    def n_vars() -> int:
        return (64 + 24 + 9) * (RubiksCubeSolver.t_max + 1)

    @staticmethod
    def decode(var: int) -> str:
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

        face, direction, t = Var.get_action_from(var)
        return prefix + f"a({Face(face).name}, {Direction(direction).name}, {t})"

    @staticmethod
    def is_action(var: int) -> bool:
        return (64 + 24) * (RubiksCubeSolver.t_max + 1) < var

    @staticmethod
    def get_action_from(a: int) -> tuple[Face, Direction, int]:
        assert Var.is_action(a), f"Invalid action: {a}, {Var.decode(a)}"

        a -= (64 + 24) * (RubiksCubeSolver.t_max + 1) + 1
        t = a // 9
        a = a % 9
        face = a // 3
        direction = a % 3
        return Var.faces[face], Direction(direction), t

    @staticmethod
    def g(cube_pos: CubePos) -> tuple[int, int, int]:
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
    def rotate_x(face: Face, direction: Direction, cube_pos: CubePos) -> CubePos:
        assert face in {Face.RIGHT, Face.BOTTOM, Face.BACK}, f"Invalid face: {face}"

        c_x, c_y, c_z = Var.g(cube_pos)

        if not Var.will_rotate(c_x, c_y, c_z, face):
            return cube_pos

        def rotate_1_x(face: Face, c_x: int, c_y: int, c_z: int) -> CubePos:
            if face == Face.RIGHT:
                return cast(CubePos, c_x + 2 * c_z + 4 * (1 - c_y))
            if face == Face.BOTTOM:
                return cast(CubePos, c_z + 2 * c_y + 4 * (1 - c_x))
            if face == Face.BACK:
                return cast(CubePos, c_y + 2 * (1 - c_x) + 4 * c_z)
            raise ValueError(f"Invalid face: {face}")

        if direction == Direction.CLOCKWISE:
            return rotate_1_x(face, c_x, c_y, c_z)
        if direction == Direction.HALF_TURN:
            return rotate_1_x(face, *Var.g(rotate_1_x(face, c_x, c_y, c_z)))
        if direction == Direction.COUNTERCLOCKWISE:
            return rotate_1_x(
                face, *Var.g(rotate_1_x(face, *Var.g(rotate_1_x(face, c_x, c_y, c_z))))
            )

    @staticmethod
    def rotate_theta(
        face: Face,
        direction: Direction,
        cube_pos: CubePos,
        orientation: Orientation,
    ) -> Orientation:
        if direction == Direction.HALF_TURN:
            return orientation

        if not Var.will_rotate(*Var.g(cube_pos), face):
            return orientation

        def s(i: Orientation, j: Orientation, orientation: Orientation) -> Orientation:
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
        self.rubiks_cube = rubiks_cube  # Cube à résoudre
        self.t_max = t_max  # Nombre de mouvements maximum
        self.cnf_filename = cnf_filename  # Fichier CNF
        self.var_mapping = {}  # Correspondance des variables SAT

    def generate_clauses(self) -> list[NamedClause]:
        """
        Génère les clauses.
        """
        clauses: list[NamedClause] = self.generate_initial_clauses()

        # Etat final
        for i in range(8):
            i = cast(CubePos, i)
            clauses.append(
                (f"Etat final, position du cube {i}", [Var.x(i, i, self.t_max)])
            )
            clauses.append(
                (f"Etat final, orientation du cube {i}", [Var.theta(i, 0, self.t_max)])
            )

        # Transitions des positions
        for t in range(1, self.t_max + 1):
            for id in range(8):
                id = cast(CubePos, id)
                for c in range(8):
                    c = cast(CubePos, c)
                    for f in [Face.RIGHT, Face.BOTTOM, Face.BACK]:
                        for d in Direction:
                            clauses.append(
                                (
                                    f"Transition des positions, id_cube {id}, case_cube {c}, face {f},  direction {d}, temps {t}, clause 1",
                                    [
                                        Var.x(Var.rotate_x(f, d, c), id, t),
                                        -Var.x(c, id, t - 1),
                                        -Var.a(f, d, t),
                                    ],
                                )
                            )

                            clauses.append(
                                (
                                    f"Transition des positions, id_cube {id}, case_cube {c}, face {f},  direction {d}, temps {t}, clause 2",
                                    [
                                        -Var.x(Var.rotate_x(f, d, c), id, t),
                                        Var.x(c, id, t - 1),
                                        -Var.a(f, d, t),
                                    ],
                                )
                            )

        # Transitions des rotations
        for t in range(1, self.t_max + 1):
            for c in range(8):
                c = cast(CubePos, c)
                for f in [Face.RIGHT, Face.BOTTOM, Face.BACK]:
                    for d in Direction:
                        for o in range(3):
                            o = cast(Orientation, o)
                            clauses.append(
                                (
                                    f"Transition des orientations, case_cube {c}, orientation {o}, face {f},  direction {d}, temps {t}, clause 1",
                                    [
                                        Var.theta(
                                            Var.rotate_x(f, d, c),
                                            Var.rotate_theta(f, d, c, o),
                                            t,
                                        ),
                                        -Var.theta(c, o, t - 1),
                                        -Var.a(f, d, t),
                                    ],
                                )
                            )

                            clauses.append(
                                (
                                    f"Transition des orientations, case_cube {c}, orientation {o}, face {f},  direction {d}, temps {t}, clause 2",
                                    [
                                        -Var.theta(
                                            Var.rotate_x(f, d, c),
                                            Var.rotate_theta(f, d, c, o),
                                            t,
                                        ),
                                        Var.theta(c, o, t - 1),
                                        -Var.a(f, d, t),
                                    ],
                                )
                            )

            for f, d in product([Face.RIGHT, Face.BOTTOM, Face.BACK], Direction):
                for f_prime, d_prime in product(
                    [Face.RIGHT, Face.BOTTOM, Face.BACK], Direction
                ):
                    if (f, d) < (f_prime, d_prime):
                        clauses.append(
                            (
                                f"Interdiction de rotations multiples, temps {t}, face {f}, {f_prime} et direction {d}, {d_prime}",
                                [-Var.a(f, d, t), -Var.a(f_prime, d_prime, t)],
                            )
                        )

            # Ajout des clauses pour forcer une action par étape
            clauses.append(
                (
                    f"Action obligatoire à chaque étape, temps {t}",
                    [
                        Var.a(f, d, t)
                        for (f, d) in product(
                            [Face.RIGHT, Face.BOTTOM, Face.BACK], Direction
                        )
                    ],
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

    def solve(self, clauses: list[Clause]) -> tuple[bool, list[str], list[Action]]:
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

    def parse_output(self, output: str) -> tuple[bool, list[str], list[Action]]:
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

        actions: list[tuple[Face, Direction, int]] = [
            Var.get_action_from(a) for a in variables if Var.is_action(a)
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
            cube_pos = cast(CubePos, cube_pos)

            colors = self.rubiks_cube.get_colors_from_pos(Var.g(cube_pos))
            real_cube_id, real_orientation = (
                self.rubiks_cube.colors_to_id_and_orientation(colors)
            )

            for cube_id in range(8):
                cube_id = cast(CubePos, cube_id)

                sign = 1 if cube_id == real_cube_id else -1
                clauses.append(
                    ("Initial state position", [sign * Var.x(cube_pos, cube_id, 0)])
                )

            for orientation in range(3):
                orientation = cast(Orientation, orientation)

                sign = 1 if orientation == real_orientation else -1
                clauses.append(
                    (
                        "Initial state orientation",
                        [sign * Var.theta(cube_pos, orientation, 0)],
                    )
                )

        return clauses

    def run(
        self, true_instance: dict[int, bool] | None = None
    ) -> tuple[bool, list[Action]]:
        """
        Gère tout le processus : génération du CNF, exécution du solveur et extraction du résultat.

        true_instance : dictionnaire des variables SAT à forcer à True (Pour debug uniquement).
        """
        clauses = self.generate_clauses()
        sat, result, actions = self.solve([clauses[1] for clauses in clauses])

        if true_instance is not None and not sat:
            sat, unsat_clauses = self.verify(true_instance, clauses)

            for unsat_clause in unsat_clauses[-5:]:
                print(unsat_clause[0])
                print([Var.decode(v) for v in unsat_clause[1]])
                print()

        for line in result:
            print(line)

        return sat, actions
