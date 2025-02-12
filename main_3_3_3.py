from rubiks_cube import RubiksCube, Size
from rubiks_cube_solver_3_3_3 import RubiksCubeSolver
from utils import Face
from variables import Var
from variables_abc import Variable


def generate_true_instance(cube: RubiksCube, moves: list[str]) -> list[Variable]:
    cube = cube.copy()
    true_instance: list[Variable] = []

    assert len(moves) == Variable.t_max

    for t in range(len(moves)):
        update_true_instance(cube, true_instance, t)

        face, direction, depth = cube.parse_move(moves[t])

        cube.rotate(face, direction, depth)

        true_instance.append(Var.Actions(face, direction, depth, t + 1))
    update_true_instance(cube, true_instance, len(moves))

    return true_instance


def update_true_instance(cube: RubiksCube, true_instance: list[Variable], t: int):
    for pos in Var.Corners.pos_range():
        true_instance += list(cube.get_vars_from_corner_pos(pos, t))
    for pos in Var.Edges.pos_range():
        true_instance += list(cube.get_vars_from_edge_pos(pos, t))
    for pos in Var.Centers.pos_range():
        true_instance += [cube.get_vars_from_centers_pos(pos, t)]


def main(size: Size = (3, 3, 3)):
    Variable.t_max = 30
    Variable.cube_size = size[0]
    Var.depths = list(range(Variable.cube_size - 1))

    rubiks_cube = RubiksCube(size)
    moves = rubiks_cube.shuffle(faces=(Face.BACK, Face.RIGHT, Face.BOTTOM))

    solver = RubiksCubeSolver(rubiks_cube, "rubiks_cube.cnf")
    sat, actions = solver.run(Variable.t_max)

    print("SATISFIABLE" if sat else "UNSATISFIABLE")

    if sat:
        print(f"Solved in {len(actions)} moves")

        moves = [
            RubiksCube.move_to_str(action.face, action.direction, action.depth)
            for action in actions
        ]
        rubiks_cube.animate(RubiksCube.parse_moves(moves), speed=2)


if __name__ == "__main__":
    main()
