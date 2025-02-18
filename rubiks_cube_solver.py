import subprocess
import io

from itertools import product
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2 import service_account
from tqdm import tqdm

import step as Step
from utils import Direction, Face
from rubiks_cube import RubiksCube
from variables import Var, Variable
from variables_abc import Clause, NamedClause, TIdx, TPos, VariableState


class RubiksCubeSolver:
    def __init__(
        self,
        rubiks_cube: RubiksCube,
        cnf_filename="rubiks_cube.cnf",
    ):
        self.rubiks_cube = rubiks_cube
        self.cnf_filename = cnf_filename

        Variable.cube_size = rubiks_cube.size[0]
        Var.depths = list(range(Variable.cube_size - 1))

    def generate_initial_clauses(
        self, cube: RubiksCube | None = None
    ) -> list[NamedClause]:
        if cube is None:
            cube = self.rubiks_cube

        clauses: list[NamedClause] = []

        for pos in Var.Corners.pos_range():
            idx, theta = cube.get_vars_from_corner_pos(pos)

            for var in Var.Corners.x.from_decoded(pos, idx, 0):
                clauses.append(("Initial state position", [var]))

            for orientation in Var.Corners.theta.idx_range():
                clauses.append(
                    (
                        "Initial state orientation",
                        [Var.Corners.theta(pos, orientation, 0, orientation == theta)],
                    )
                )

        for pos in Var.Edges.pos_range():
            idx, theta = cube.get_vars_from_edge_pos(pos)

            for var in Var.Edges.x.from_decoded(pos, idx, 0):
                clauses.append(("Initial state position", [var]))

            clauses.append(
                ("Initial state orientation", [Var.Edges.theta(pos, 0, 0, bool(theta))])
            )

        for pos in Var.Centers.pos_range():
            idx = cube.get_vars_from_center_pos(pos)

            for var in Var.Centers.x.from_decoded(pos, idx, 0):
                clauses.append(("Initial state position", [var]))

        return clauses

    def generate_clauses_transition(
        self, state: type[VariableState[TPos, TIdx]], action: Var.Actions
    ) -> list[NamedClause]:
        clauses: list[NamedClause] = []

        for c in state.pos_range():
            for idx in state.idx_range():
                var = state(c, idx, action.t - 1)
                var_prime = var.rotate(action.face, action.direction, action.depth)

                clauses.append(
                    (
                        f"Transition {state.__name__}, case_cube {c}, {action.face}, {action.direction}, depth {action.depth}, temps {action.t}, clause 1",
                        [var_prime, -var, -action],
                    )
                )
                clauses.append(
                    (
                        f"Transition {state.__name__}, case_cube {c}, {action.face}, {action.direction}, depth {action.depth}, temps {action.t}, clause 2",
                        [-var_prime, var, -action],
                    )
                )

        return clauses

    def generate_transitions_clauses(
        self, actions: set[tuple[Face, Direction, int]] | None = None
    ) -> list[NamedClause]:
        """
        Génère les clauses pour les transitions.
        """
        if actions is None:
            actions = {*product(Var.faces, Var.directions, Var.depths)}

        clauses: list[NamedClause] = []

        for t in range(1, Variable.t_max + 1):
            # Ajout des clauses pour forcer une action par étape
            clauses.append(
                (
                    f"Action obligatoire à chaque étape, temps {t}",
                    [Var.Actions(f, d, depth, t) for f, d, depth in actions],
                )
            )

            for f, d, depth in actions:
                action = Var.Actions(f, d, depth, t)

                for f_prime, d_prime, depth_prime in actions:
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

                clauses += self.generate_clauses_transition(Var.Corners.x, action)
                clauses += self.generate_clauses_transition(Var.Corners.theta, action)
                clauses += self.generate_clauses_transition(Var.Edges.x, action)
                clauses += self.generate_clauses_transition(Var.Edges.theta, action)
                clauses += self.generate_clauses_transition(Var.Centers.x, action)

        return clauses

    def upload_to_drive(
        self,
        clauses: list[Clause],
        folder_id="16CfwqqPviDAp7ECScSmXLh94nxyET5Am",
    ):
        creds = service_account.Credentials.from_service_account_file(
            "credentials.json", scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        service = build("drive", "v3", credentials=creds)

        cnf_file = io.BytesIO(self.generate_str(clauses).encode("utf-8"))

        file_metadata = {"name": self.cnf_filename, "parents": [folder_id]}
        media = MediaIoBaseUpload(cnf_file, mimetype="text/plain")

        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        print(
            f"File uploaded successfully! File ID: {file.get('id')} ({self.cnf_filename})"
        )

    def generate_str(self, clauses: list[Clause]) -> str:
        """
        Génère une chaîne de caractères pour les clauses.
        """
        s = f"p cnf {Var.n_vars()} {len(clauses)}\n"
        for clause in clauses:
            s += " ".join(map(lambda var: var.id_repr(), clause)) + " 0\n"
        return s

    def generate_cnf_file(self, clauses: list[Clause]) -> None:
        """
        Génère le fichier CNF pour le problème.
        """
        # Écriture du fichier CNF
        with open(self.cnf_filename, "w") as f:
            f.write(self.generate_str(clauses))

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

    def remove_name(self, clauses: list[NamedClause]) -> list[Clause]:
        return [clause[1] for clause in clauses]

    def generate_clauses(
        self, cube: RubiksCube, steps: list[Step.Step] | None = None
    ) -> list[NamedClause]:
        if steps is None:
            steps = [Step.Corners() + Step.Edges() + Step.Centers()]

        clauses: list[NamedClause] = []
        clauses += self.generate_initial_clauses(cube)
        clauses += self.generate_transitions_clauses(steps[-1].actions)

        for step in steps:
            clauses += step.generate_final_clauses()

        return clauses

    def remove_duplicates(self, actions: list[Var.Actions]) -> list[Var.Actions]:
        new_actions: list[Var.Actions] = []

        for action in actions:
            if len(new_actions) == 0:
                new_actions.append(
                    Var.Actions(
                        action.face,
                        action.direction,
                        action.depth,
                        len(new_actions) + 1,
                    )
                )
                continue

            prev_action = new_actions[-1]

            if action.face == prev_action.face and action.depth == prev_action.depth:
                new_direction = action.direction + prev_action.direction

                if new_direction == Direction.NONE:
                    new_actions.pop()

                else:
                    new_actions[-1] = Var.Actions(
                        action.face, new_direction, action.depth, len(new_actions) + 1
                    )
            else:
                new_actions.append(
                    Var.Actions(
                        action.face,
                        action.direction,
                        action.depth,
                        len(new_actions) + 1,
                    )
                )

        return new_actions

    def run(
        self, t_max: int, cube: RubiksCube, steps: list[Step.Step] | None = None
    ) -> tuple[bool, list[Var.Actions]]:
        """
        Gère tout le processus : génération du CNF, exécution du solveur et extraction du résultat.

        true_instance : dictionnaire des variables SAT à forcer à True (Pour debug uniquement).
        """
        if steps is None:
            steps = [Step.Corners() + Step.Edges() + Step.Centers()]

        last_t_max = Variable.t_max

        Variable.t_max = t_max

        clauses = self.generate_clauses(cube, steps)
        sat, variables, actions = self.solve([clauses[1] for clauses in clauses])

        Variable.t_max = last_t_max
        return sat, actions

    def find_optimal(self, steps: list[Step.Step] | None = None) -> list[Var.Actions]:
        """
        Trouve la solution optimale.
        """

        if steps is None:
            steps = [Step.Corners() + Step.Edges() + Step.Centers()]

        cube = self.rubiks_cube.copy()
        actions: list[Var.Actions] = []

        for step_idx in tqdm(range(len(steps))):
            step = steps[step_idx]
            sat_found: bool = False
            unsat_found: bool = False
            t = step.t_median()

            actions_this_step: list[Var.Actions] = []

            while not sat_found or not unsat_found:
                sat, actions_ = self.run(t, cube, steps[: step_idx + 1])

                print(f"Step {step}, t = {t}, sat = {sat}")

                if sat:
                    sat_found = True
                    actions_this_step = actions_
                    t -= 1

                else:
                    unsat_found = True
                    t += 1

            for action in actions_this_step:
                cube.rotate(action.face, action.direction, action.depth)

            actions += actions_this_step

        return self.remove_duplicates(actions)
