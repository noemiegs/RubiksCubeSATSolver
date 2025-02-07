from typing import cast
from rubiks_cube_3_3_3 import CornerPos, RubiksCube, Size, Face
from rubiks_cube_solver_3_3_3 import RubiksCubeSolver
from variables import Var
from variables_abc import Variable


def generate_true_instance(cube: RubiksCube, moves: list[str]) -> list[Variable]:
    cube = cube.copy()
    true_instance: list[Variable] = []

    for t in range(len(moves)):
        for cube_pos in range(8):
            cube_pos = cast(CornerPos, cube_pos)

            colors = cube.get_colors_from_pos(Var.Corners.g(cube_pos))
            cube_idx, orientation = cube.colors_to_id_and_orientation(colors)

            true_instance.append(Var.Corners.x(cube_pos, cube_idx, t))
            true_instance.append(Var.Corners.theta(cube_pos, orientation, t))

        face, direction, depth = cube.parse_move(moves[t])

        cube.rotate(face, direction, depth)

        true_instance.append(Var.Actions(face, direction, 0, t + 1))

    for cube_pos in range(8):
        cube_pos = cast(CornerPos, cube_pos)

        colors = cube.get_colors_from_pos(Var.Corners.g(cube_pos))
        cube_idx, orientation = cube.colors_to_id_and_orientation(colors)

        true_instance.append(Var.Corners.x(cube_pos, cube_idx, len(moves)))
        true_instance.append(Var.Corners.theta(cube_pos, orientation, len(moves)))

    return true_instance


def main(size: Size = (2, 2, 2)):
    Var.t_max = 11
    
    rubiks_cube = RubiksCube(size)
    moves = rubiks_cube.shuffle(Var.t_max, faces=(Face.BACK, Face.RIGHT, Face.BOTTOM))

    solver = RubiksCubeSolver(rubiks_cube, Var.t_max)
    sat, actions = solver.run(Var.t_max)

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
