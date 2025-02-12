import subprocess
from itertools import product

from utils import Direction
from rubiks_cube import RubiksCube
from variables import Var, Variable
from variables_abc import (
    Clause,
    NamedClause,
    TOrientation,
    TPos,
    VariableTheta,
    VariableX,
)


class RubiksCubeSolver:
    def __init__(
        self,
        rubiks_cube: RubiksCube,
        cnf_filename="rubiks_cube.cnf",
    ):
        self.rubiks_cube = rubiks_cube
        self.cnf_filename = cnf_filename

    def generate_initial_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []

        for pos in Var.Corners.pos_range():
            x, theta = self.rubiks_cube.get_vars_from_corner_pos(pos, 0)

            for idx in Var.Corners.pos_range():
                sign = 1 if idx == x.idx else -1
                clauses.append(
                    (
                        "Initial state position",
                        [sign * Var.Corners.x(pos, idx, 0)],
                    )
                )

            for orientation in Var.Corners.theta.orientation_range():
                sign = 1 if orientation == theta.orientation else -1
                clauses.append(
                    (
                        "Initial state orientation",
                        [sign * Var.Corners.theta(pos, orientation, 0)],
                    )
                )
        
        for pos in Var.Edges.pos_range():
            x, theta = self.rubiks_cube.get_vars_from_edge_pos(pos, 0)

            for idx in Var.Edges.pos_range():
                sign = 1 if idx == x.idx else -1
                clauses.append(
                    (
                        "Initial state position",
                        [sign * Var.Edges.x(pos, idx, 0)],
                    )
                )

            for orientation in Var.Edges.theta.orientation_range():
                sign = 1 if orientation == theta.orientation else -1
                clauses.append(
                    (
                        "Initial state orientation",
                        [sign * Var.Edges.theta(pos, orientation, 0)],
                    )
                )
        
        for pos in Var.Centers.pos_range():
            x = self.rubiks_cube.get_vars_from_centers_pos(pos, 0)

            for idx in Var.Centers.pos_range():
                sign = 1 if idx == x.idx else -1
                clauses.append(
                    (
                        "Initial state position",
                        [sign * Var.Centers.x(pos, idx, 0)],
                    )
                )

        return clauses

    def generate_clauses_final_x(self, x: type[VariableX[TPos]]) -> list[NamedClause]:
        clauses: list[NamedClause] = []

        for idx in x.parent().pos_range():
            clauses.append(
                (
                    f"Etat final {x.parent().__name__}, position du cube {idx}",
                    [x(idx, idx, Variable.t_max)],
                )
            )

        return clauses

    def generate_clauses_final_theta(
        self, theta: type[VariableTheta]
    ) -> list[NamedClause]:
        clauses: list[NamedClause] = []

        for idx in theta.parent().pos_range():
            clauses.append(
                (
                    f"Etat final {theta.parent().__name__}, orientation du cube {idx}",
                    [theta(idx, 0, Variable.t_max)],
                )
            )

        return clauses

    def generate_clauses_transition_x(
        self, x: type[VariableX], action: Var.Actions
    ) -> list[NamedClause]:
        clauses: list[NamedClause] = []

        for c in x.parent().pos_range():
            for idx in x.parent().pos_range():
                var_x = x(c, idx, action.t - 1)
                var_x_prime = var_x.rotate(action.face, action.direction, action.depth)

                clauses.append(
                    (
                        f"Transition des positions {x.parent().__name__}, id_cube {idx}, case_cube {c}, {action.face}, {action.direction}, depth {action.depth}, temps {action.t}, clause 1",
                        [var_x_prime, -var_x, -action],
                    )
                )

                clauses.append(
                    (
                        f"Transition des positions {x.parent().__name__}, id_cube {idx}, case_cube {c}, {action.face}, {action.direction}, depth {action.depth}, temps {action.t}, clause 2",
                        [-var_x_prime, var_x, -action],
                    )
                )

        return clauses

    def generate_clauses_transition_theta(
        self, theta: type[VariableTheta[TPos, TOrientation]], action: Var.Actions
    ) -> list[NamedClause]:
        clauses: list[NamedClause] = []

        for c in theta.parent().pos_range():
            for o in theta.orientation_range():
                var_theta = theta(c, o, action.t - 1)
                var_theta_prime = var_theta.rotate(
                    action.face, action.direction, action.depth
                )

                clauses.append(
                    (
                        f"Transition des orientations {theta.parent().__name__}, case_cube {c}, orientation {o}, {action.face}, {action.direction}, depth {action.depth}, temps {action.t}, clause 1",
                        [var_theta_prime, -var_theta, -action],
                    )
                )

                clauses.append(
                    (
                        f"Transition des orientations {theta.parent().__name__}, case_cube {c}, orientation {o}, {action.face}, {action.direction}, depth {action.depth}, temps {action.t}, clause 2",
                        [-var_theta_prime, var_theta, -action],
                    )
                )

        return clauses

    def generate_clauses(self) -> list[NamedClause]:
        """
        Génère les clauses.
        """
        clauses: list[NamedClause] = self.generate_initial_clauses()

        ## Etat final
        # Corners
        clauses += self.generate_clauses_final_x(Var.Corners.x)
        clauses += self.generate_clauses_final_theta(Var.Corners.theta)
        # Edges
        clauses += self.generate_clauses_final_x(Var.Edges.x)
        clauses += self.generate_clauses_final_theta(Var.Edges.theta)
        # Centers
        clauses += self.generate_clauses_final_x(Var.Centers.x)

        ## Transition
        for t in range(1, Variable.t_max + 1):
            # Ajout des clauses pour forcer une action par étape
            clauses.append(
                (
                    f"Action obligatoire à chaque étape, temps {t}",
                    [
                        Var.Actions(f, d, depth, t)
                        for (f, d, depth) in product(Var.faces, Direction, Var.depths)
                    ],
                )
            )

            for f, d, depth in product(Var.faces, Direction, Var.depths):
                action = Var.Actions(f, d, depth, t)

                for f_prime, d_prime, depth_prime in product(
                    Var.faces, Direction, Var.depths
                ):
                    if (f, d, depth) < (f_prime, d_prime, depth_prime):
                        clauses.append(
                            (
                                f"Interdiction de rotations multiples, temps {t}, face {f}, {f_prime} et direction {d}, {d_prime}",
                                [
                                    -action,
                                    -Var.Actions(f_prime, d_prime, depth_prime, t),
                                ],
                            )
                        )

                # Corners
                clauses += self.generate_clauses_transition_x(Var.Corners.x, action)
                clauses += self.generate_clauses_transition_theta(
                    Var.Corners.theta, action
                )
                # Edges
                clauses += self.generate_clauses_transition_x(Var.Edges.x, action)
                clauses += self.generate_clauses_transition_theta(
                    Var.Edges.theta, action
                )
                # Centers
                clauses += self.generate_clauses_transition_x(Var.Centers.x, action)

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

    def run(
        self, t_max: int, true_instance: list[Variable] | None = None
    ) -> tuple[bool, list[Var.Actions]]:
        """
        Gère tout le processus : génération du CNF, exécution du solveur et extraction du résultat.

        true_instance : dictionnaire des variables SAT à forcer à True (Pour debug uniquement).
        """
        Variable.t_max = t_max

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
