import subprocess
from typing import Iterable, cast
from itertools import product

from rubiks_cube_3_3_3 import CenterPos, Direction, EdgePos, RubiksCube, CornerPos, CornerOrientation
from variables import Var, Variable
from variables_abc import Clause, NamedClause


class RubiksCubeSolver:
    t_max: int = 11

    def __init__(
        self,
        rubiks_cube: RubiksCube,
        t_max: int = 11,
        cnf_filename="rubiks_cube.cnf",
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
        cube_edge = cast(Iterable[EdgePos], range(12))
        cube_centers = cast(Iterable[CenterPos], range(6))
        orientations = cast(Iterable[CornerOrientation], range(3))

        # Etat final
        
        # Corners
        for idx in cube_pos:
            clauses.append(
                (
                    f"Etat final Corners, position du cube {idx}",
                    [Var.Corners.x(idx, idx, self.t_max)],
                )
            )
            clauses.append(
                (
                    f"Etat final Corners, orientation du cube {idx}",
                    [Var.Corners.theta(idx, 0, self.t_max)],
                )
            )
        
        # Edges
        for idx in cube_edge:
            clauses.append(
                (
                    f"Etat final Edges, position du cube {idx}",
                    [Var.Edges.x(idx, idx, self.t_max)],
                )
            )
            clauses.append(
                (
                    f"Etat final Corners, orientation du cube {idx}",
                    [Var.Edges.theta(idx, 0, self.t_max)],
                )
            )
        
        # Centers
        for idx in cube_centers:
            clauses.append(
                (
                    f"Etat final Centers, position du cube {idx}",
                    [Var.Centers.x(idx, idx, self.t_max)],
                )
            )

        for t in range(1, self.t_max + 1):
            # Ajout des clauses pour forcer une action par étape
            clauses.append(
                (
                    f"Action obligatoire à chaque étape, temps {t}",
                    [
                        Var.Actions(f, d, 0, t)
                        for (f, d) in product(Var.faces, Direction)
                    ],
                )
            )

            for f, d, depth in product(Var.faces, Direction, Var.depths):
                for f_prime, d_prime, depth_prime in product(
                    Var.faces, Direction, Var.depths
                ):
                    if (f, d, depth) < (f_prime, d_prime, depth_prime):
                        clauses.append(
                            (
                                f"Interdiction de rotations multiples, temps {t}, face {f}, {f_prime} et direction {d}, {d_prime}",
                                [
                                    -Var.Actions(f, d, depth, t),
                                    -Var.Actions(f_prime, d_prime, depth_prime, t),
                                ],
                            )
                        )

                for c in cube_pos:
                    action = Var.Actions(f, d, 0, t)

                    # Transitions des positions
                    for idx in cube_pos:
                        x = Var.Corners.x(c, idx, t - 1)
                        x_prime = x.rotate(f, d, 0)

                        clauses.append(
                            (
                                f"Transition des positions, id_cube {idx}, case_cube {c}, face {f},  direction {d}, temps {t}, clause 1",
                                [x_prime, -x, -action],
                            )
                        )

                        clauses.append(
                            (
                                f"Transition des positions, id_cube {idx}, case_cube {c}, face {f},  direction {d}, temps {t}, clause 2",
                                [-x_prime, x, -action],
                            )
                        )

                    # Transitions des rotations
                    for o in orientations:
                        theta = Var.Corners.theta(c, o, t - 1)
                        theta_prime = theta.rotate(f, d, 0)

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
                f.write(" ".join(map(lambda var: var.id_repr(), clause)) + " 0\n")

    def verify(
        self, vars: list[Variable], clauses: list[NamedClause]
    ) -> tuple[bool, list[NamedClause]]:
        """
        Vérifie si les variables sont satisfaisantes pour les clauses.

        Retourne un tuple (booléen, liste de clauses insatisfaites).
        """
        unsat_clauses: list[NamedClause] = []
        var_map = {var.id: var.is_true for var in vars}

        for clause in clauses:
            if not any(var_map.get(var.id, False) == var.is_true for var in clause[1]):
                unsat_clauses.append(clause)

        return len(unsat_clauses) == 0, unsat_clauses

    def solve(
        self, clauses: list[Clause]
    ) -> tuple[bool, list[Variable], list[Var.Actions]]:
        """
        Exécute Gophersat et récupère le résultat.
        """
        self.generate_cnf_file(clauses)

        result = subprocess.run(
            ["gophersat", "--verbose", self.cnf_filename],
            capture_output=True,
            text=True,
        )

        return self.parse_output(result.stdout)

    def parse_output(
        self, output: str
    ) -> tuple[bool, list[Variable], list[Var.Actions]]:
        """
        Analyse la sortie de Gophersat et retourne les actions et positions trouvées.
        """
        if "UNSATISFIABLE" in output:
            return False, [], []

        variables: list[Variable] = []

        for line in output.splitlines():
            if line.startswith("v "):  # Ligne contenant les variables SAT
                values = map(int, line[2:].strip().split())
                variables.extend([Var.from_int(v) for v in values if v > 0])

        actions: list[Var.Actions] = [
            a for a in variables if isinstance(a, Var.Actions)
        ]

        return (
            True,
            variables,
            [action for action in sorted(actions, key=lambda a: a.t)],
        )

    def generate_initial_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []

        for cube_pos in range(8):
            cube_pos = cast(CornerPos, cube_pos)

            colors = self.rubiks_cube.get_colors_from_pos(Var.Corners.g(cube_pos))
            real_cube_idx, real_orientation = (
                self.rubiks_cube.colors_to_id_and_orientation(colors)
            )

            for cube_idx in range(8):
                cube_idx = cast(CornerPos, cube_idx)

                sign = 1 if cube_idx == real_cube_idx else -1
                clauses.append(
                    (
                        "Initial state position",
                        [sign * Var.Corners.x(cube_pos, cube_idx, 0)],
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
        self, t_max: int, true_instance: list[Variable] | None = None
    ) -> tuple[bool, list[Var.Actions]]:
        """
        Gère tout le processus : génération du CNF, exécution du solveur et extraction du résultat.

        true_instance : dictionnaire des variables SAT à forcer à True (Pour debug uniquement).
        """
        RubiksCubeSolver.t_max = t_max
        Var.t_max = t_max

        clauses = self.generate_clauses()
        sat, result, actions = self.solve([clauses[1] for clauses in clauses])

        if true_instance is not None and not sat:
            sat, unsat_clauses = self.verify(true_instance, clauses)

            for unsat_clause in unsat_clauses:
                print(unsat_clause[0])
                print(unsat_clause[1])
                print()

        for line in result:
            print(line)

        return sat, actions

    def find_optimal(
        self, t_min: int = -1, t_max: int = 11
    ) -> tuple[bool, list[Var.Actions]]:
        """
        Trouve la solution optimale.
        """

        sat: bool = False
        actions: list[Var.Actions] = []

        while t_min < t_max - 1:
            t = (t_min + t_max) // 2

            sat_, actions_ = self.run(t)

            if sat_:
                t_max = t
                sat = True
                actions = actions_

            else:
                t_min = t

        return sat, actions
