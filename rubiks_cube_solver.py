import subprocess
from typing import Literal, cast

from rubiks_cube import Direction, Face, RubiksCube


CubePos = Literal[0, 1, 2, 3, 4, 5, 6, 7]
Orientation = Literal[0, 1, 2]


class Var:
    @staticmethod
    def x(cube_pos: CubePos, cube_id: CubePos, t: int) -> int:
        return cube_pos + cube_id * 8 + t * 64 + 1

    @staticmethod
    def theta(cube_pos: CubePos, orientation: Orientation, t: int) -> int:
        return 64 * RubiksCubeSolver.t_max + cube_pos + orientation * 8 + t * 24 + 1

    @staticmethod
    def g(cube_pos: CubePos) -> tuple[int, int, int]:
        return cube_pos % 2, (cube_pos // 2) % 2, (cube_pos // 4) % 2

    @staticmethod
    def rotate_x(face: Face, direction: Direction, cube_pos: CubePos) -> CubePos:
        assert face in {Face.RIGHT, Face.BOTTOM, Face.BACK}, f"Invalid face: {face}"

        c_x, c_y, c_z = RubiksCubeSolver.Var.g(cube_pos)

        if face == Face.RIGHT and c_x == 0:
            return cube_pos
        if face == Face.BOTTOM and c_y == 0:
            return cube_pos
        if face == Face.BACK and c_z == 0:
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
            return rotate_1_x(face, *g(rotate_1_x(face, c_x, c_y, c_z)))
        if direction == Direction.COUNTERCLOCKWISE:
            return rotate_1_x(
                face, *g(rotate_1_x(face, *g(rotate_1_x(face, c_x, c_y, c_z))))
            )
    
    @staticmethod
    def rotate_theta(
        face: Face,
        direction: Direction,
        cube_pos: CubePos,
        orientation: Orientation,
    ) -> Orientation:
        assert face in {Face.RIGHT, Face.BOTTOM, Face.BACK}, f"Invalid face: {face}"

        if direction == Direction.HALF_TURN:
            return orientation

        def s(
            i: Orientation, j: Orientation, orientation: Orientation
        ) -> Orientation:
            if orientation == i:
                return j
            if orientation == j:
                return i
            return orientation


class RubiksCubeSolver:
    t_max: int
            
            

    def __init__(
        self, rubiks_cube: RubiksCube, t_max: int = 11, cnf_filename="rubiks_cube.cnf"
    ):
        self.rubiks_cube = rubiks_cube  # Cube à résoudre
        self.t_max = t_max  # Nombre de mouvements maximum
        self.cnf_filename = cnf_filename  # Fichier CNF
        self.var_mapping = {}  # Correspondance des variables SAT

    def generate_cnf_file(self):
        """
        Génère le fichier CNF pour le problème.
        """
        clauses = []

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
        ...

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
