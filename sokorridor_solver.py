import subprocess


class SokorridorSolver:
    def __init__(
        self, T=15, C=11, cnf_filename="sokorridor.cnf", output_filename="output.txt"
    ):
        self.T = T  # Nombre de pas de temps
        self.C = C  # Nombre de cases
        self.cnf_filename = cnf_filename  # Fichier CNF
        self.output_filename = output_filename  # Fichier de sortie du solveur
        self.var_mapping = {}  # Correspondance des variables SAT

    def verify(
        self,
        vars: dict[int, bool],
        clauses: list[list[int]],
        default_var_value: bool = False,
    ) -> tuple[bool, list[tuple[list[int], list[int]]]]:
        """
        Vérifie si les variables sont satisfaisantes pour les clauses.
        """
        unsat_clauses = []

        for clause in clauses:
            if not any(
                vars.get(abs(var), default_var_value) == (var > 0) for var in clause
            ):
                unsat_clauses.append(
                    (
                        clause,
                        [
                            var
                            for var in clause
                            if vars.get(abs(var), default_var_value) != (var > 0)
                        ],
                    )
                )
        return len(unsat_clauses) == 0, unsat_clauses

    def generate_cnf_file(self):
        """
        Génère le fichier CNF pour le problème.
        """
        clauses = []

        # Numérotation des variables
        def var_w(c, t):
            return c + t * self.C + 1

        def var_b1(c, t):
            return self.C * self.T + c + t * self.C + 1

        def var_b2(c, t):
            return self.C * self.T * 2 + c + t * self.C + 1

        def var_do(a, t):
            action_offset = 3 * self.C * self.T
            actions = {"m,d": 0, "m,g": 1, "p,d": 2, "p,g": 3}
            return action_offset + t * 4 + actions[a] + 1

        # État initial
        for c in range(self.C):
            clauses.append([var_w(c, 0)] if c == 6 else [-var_w(c, 0)])
            clauses.append([var_b1(c, 0)] if c == 2 else [-var_b1(c, 0)])
            clauses.append([var_b2(c, 0)] if c == 9 else [-var_b2(c, 0)])

        # Objectif
        clauses.append([var_b1(0, self.T - 1)])
        clauses.append([var_b2(10, self.T - 1)])

        # Contraintes sur les actions
        for t in range(self.T):
            # Ne peut pas bouger en dehors des limites
            clauses.append([-var_do("m,g", t), -var_w(0, t)])
            clauses.append([-var_do("m,d", t), -var_w(self.C - 1, t)])

            # Ne peut pas pousser en dehors des limites
            clauses.append([-var_do("p,g", t), -var_w(0, t)])
            clauses.append([-var_do("p,g", t), -var_w(1, t)])
            clauses.append([-var_do("p,d", t), -var_w(self.C - 1, t)])
            clauses.append([-var_do("p,d", t), -var_w(self.C - 2, t)])

            # Ne peut pas faire deux actions en même temps
            clauses.append([-var_do("m,g", t), -var_do("m,d", t)])
            clauses.append([-var_do("m,g", t), -var_do("p,g", t)])
            clauses.append([-var_do("m,g", t), -var_do("p,d", t)])
            clauses.append([-var_do("m,d", t), -var_do("p,g", t)])
            clauses.append([-var_do("m,d", t), -var_do("p,d", t)])
            clauses.append([-var_do("p,g", t), -var_do("p,d", t)])

            for c in range(self.C):
                # Possible déplacement à gauche
                if c >= 1:
                    clauses.append([-var_do("m,g", t), -var_w(c, t), -var_b1(c - 1, t)])

                # Possible de pousser à gauche
                if c >= 2:
                    clauses.append([-var_do("p,g", t), -var_w(c, t), var_b1(c - 1, t)])
                    clauses.append([-var_do("p,g", t), -var_w(c, t), -var_b1(c - 2, t)])

                # Possible déplacement à droite
                if c < self.C - 1:
                    clauses.append([-var_do("m,d", t), -var_w(c, t), -var_b2(c + 1, t)])

                # Possible de pousser à droite
                if c < self.C - 2:
                    clauses.append([-var_do("p,d", t), -var_w(c, t), var_b2(c + 1, t)])
                    clauses.append([-var_do("p,d", t), -var_w(c, t), -var_b2(c + 2, t)])

                # Aller à gauche
                if c >= 1 and t < self.T - 1:
                    clauses.append(
                        [-var_w(c, t), -var_do("m,g", t), var_w(c - 1, t + 1)]
                    )
                    for c_2 in range(self.C):
                        if c_2 != c - 1:
                            clauses.append(
                                [-var_w(c, t), -var_do("m,g", t), -var_w(c_2, t + 1)]
                            )

                # Pousser à gauche
                if c >= 2 and t < self.T - 1:
                    clauses.append(
                        [-var_w(c, t), -var_do("p,g", t), var_b1(c - 2, t + 1)]
                    )
                    for c_2 in range(self.C):
                        if c_2 != c - 2:
                            clauses.append(
                                [-var_w(c, t), -var_do("p,g", t), -var_b1(c_2, t + 1)]
                            )

                # Aller à droite
                if c < self.C - 1 and t < self.T - 1:
                    clauses.append(
                        [-var_w(c, t), -var_do("m,d", t), var_w(c + 1, t + 1)]
                    )
                    for c_2 in range(self.C):
                        if c_2 != c + 1:
                            clauses.append(
                                [-var_w(c, t), -var_do("m,d", t), -var_w(c_2, t + 1)]
                            )

                # Pousser à droite
                if c < self.C - 1 and c < self.C - 2 and t < self.T - 1:
                    clauses.append(
                        [-var_w(c, t), -var_do("p,d", t), var_b2(c + 2, t + 1)]
                    )
                    for c_2 in range(self.C):
                        if c_2 != c + 2:
                            clauses.append(
                                [-var_w(c, t), -var_do("p,d", t), -var_b2(c_2, t + 1)]
                            )

                # b2 reste à sa place si toutes les conditions ne sont pas réunies pour qu'elle bouge
                if t < self.T - 1:
                    clauses.append([var_b2(c, t), -var_b2(c, t + 1), var_do("p,d", t)])
                    clauses.append([-var_b2(c, t), var_b2(c, t + 1), var_do("p,d", t)])

                # b1 reste à sa place si toutes les conditions ne sont pas réunies pour qu'elle bouge
                if t < self.T - 1:
                    clauses.append([var_b1(c, t), -var_b1(c, t + 1), var_do("p,g", t)])
                    clauses.append([-var_b1(c, t), var_b1(c, t + 1), var_do("p,g", t)])

                # w reste à sa place si toutes les conditions ne sont pas réunies pour qu'elle bouge
                if t < self.T - 1:
                    clauses.append(
                        [
                            var_w(c, t),
                            -var_w(c, t + 1),
                            var_do("m,d", t),
                            var_do("m,g", t),
                        ]
                    )
                    clauses.append(
                        [
                            -var_w(c, t),
                            var_w(c, t + 1),
                            var_do("m,d", t),
                            var_do("m,g", t),
                        ]
                    )

        # Écriture du fichier CNF
        num_vars = 3 * self.C * self.T + 4 * self.T
        with open(self.cnf_filename, "w") as f:
            f.write(f"p cnf {num_vars} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")

        return clauses

    def solve(self):
        """
        Exécute Gophersat et récupère le résultat.
        """
        result = subprocess.run(
            ["gophersat", "--verbose", self.cnf_filename],
            capture_output=True,
            text=True,
        )
        return result.stdout

    def parse_output(self, output):
        """
        Analyse la sortie de Gophersat et retourne les actions et positions trouvées.
        """
        if "UNSATISFIABLE" in output:
            return ["Aucun coloriage possible"]

        variables = []
        for line in output.splitlines():
            if line.startswith("v "):  # Ligne contenant les variables SAT
                values = map(int, line[2:].strip().split())
                variables.extend([v for v in values if v > 0])

        return self.decode_variables(variables)

    def decode_variables(self, variables):
        """
        Décode les variables satisfaites en actions et positions.
        """
        decoded_info = []
        for var in variables:
            if 1 <= var <= 165:
                t = (var - 1) // self.C
                c = (var - 1) % self.C
                decoded_info.append(f"Le personnage est sur la case {c} au temps {t}.")
            elif 166 <= var <= 330:
                t = (var - 166) // self.C
                c = (var - 166) % self.C
                decoded_info.append(f"Une caisse 1 est sur la case {c} au temps {t}.")
            elif 331 <= var <= 495:
                t = (var - 331) // self.C
                c = (var - 331) % self.C
                decoded_info.append(f"Une caisse 2 est sur la case {c} au temps {t}.")
            elif 496 <= var <= 554:
                t = (var - 496) // 4
                action = [
                    "se déplacer à droite",
                    "se déplacer à gauche",
                    "pousser à droite",
                    "pousser à gauche",
                ][(var - 496) % 4]
                decoded_info.append(f"Action au temps {t} : {action}.")
        return decoded_info

    def run(self):
        """
        Gère tout le processus : génération du CNF, exécution du solveur et extraction du résultat.
        """
        clauses = self.generate_cnf_file()

        # Numérotation des variables
        def var_w(c, t):
            return c + t * self.C + 1

        def var_b1(c, t):
            return self.C * self.T + c + t * self.C + 1

        def var_b2(c, t):
            return self.C * self.T * 2 + c + t * self.C + 1

        def var_do(a, t):
            action_offset = 3 * self.C * self.T
            actions = {"m,d": 0, "m,g": 1, "p,d": 2, "p,g": 3}
            return action_offset + t * 4 + actions[a] + 1

        true_instaces = {
            var_w(6, 0): True,
            var_w(5, 1): True,
            var_w(4, 2): True,
            var_w(3, 3): True,
            var_w(3, 4): True,
            var_w(2, 5): True,
            var_w(2, 6): True,
            var_b1(2, 0): True,
            var_b1(2, 1): True,
            var_b1(2, 2): True,
            var_b1(2, 3): True,
            var_b1(1, 4): True,
            var_b1(1, 5): True,
            var_b1(0, 6): True,
            var_b1(0, self.T - 1): True,
            var_b2(9, 0): True,
            # var_b2(10, self.T - 1): True
            var_do("m,g", 0): True,
            var_do("m,g", 1): True,
            var_do("m,g", 2): True,
            var_do("p,g", 3): True,
            var_do("m,g", 4): True,
            var_do("p,g", 5): True,
        }

        is_sat, unsat_clauses = self.verify(true_instaces, clauses)
        for unsat_clause in unsat_clauses:
            print(unsat_clause[1])
            print(
                [
                    ("NOT " if unsat_clause[0][i] < 0 else "") + m
                    for i, m in enumerate(
                        self.decode_variables([abs(v) for v in unsat_clause[0]])
                    )
                ]
            )

        output = self.solve()

        result = self.parse_output(output)

        for line in result:
            print(line)
        return result


# =====================
# EXÉCUTION DU SOLVEUR
# =====================
solver = SokorridorSolver()
solver.run()
