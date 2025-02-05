from typing import cast
from rubiks_cube import Direction, RubiksCube, Size, CubePos, Face
from rubiks_cube_solver import RubiksCubeSolver, Var


def generate_true_instance(cube: RubiksCube, moves: list[str]) -> dict[int, bool]:
    true_instance: dict[int, bool] = {}

    for t in range(len(moves)):
        for cube_pos in range(8):
            cube_pos = cast(CubePos, cube_pos)

            colors = cube.get_colors_from_pos(Var.g(cube_pos))
            cube_id, orientation = cube.colors_to_id_and_orientation(colors)

            true_instance[Var.x(cube_pos, cube_id, t)] = True
            true_instance[Var.theta(cube_pos, orientation, t)] = True

        face, direction = cube.parse_move(moves[t])

        cube.rotate(face, direction)

        true_instance[Var.a(face, direction, t + 1)] = True

    for cube_pos in range(8):
        cube_pos = cast(CubePos, cube_pos)

        colors = cube.get_colors_from_pos(Var.g(cube_pos))
        cube_id, orientation = cube.colors_to_id_and_orientation(colors)

        true_instance[Var.x(cube_pos, cube_id, len(moves))] = True
        true_instance[Var.theta(cube_pos, orientation, len(moves))] = True

    return true_instance


def reverse_moves(moves: list[str]) -> list[str]:
    reverse_moves = []
    for move in moves[::-1]:
        face, direction = RubiksCube.parse_move(move)
        reverse_moves.append(f"{face.to_str()}{Direction.opposite(direction).to_str()}")
    return reverse_moves


def main(size: Size = (2, 2, 2)):
    rubiks_cube = RubiksCube(size)
    rubiks_cube.shuffle(faces=(Face.BACK, Face.RIGHT, Face.BOTTOM))

    solver = RubiksCubeSolver(rubiks_cube)
    sat, actions = solver.run()

    print("SATISFIABLE" if sat else "UNSATISFIABLE")

    if sat:
        for face, direction in actions:
            rubiks_cube.rotate(face, direction)

        rubiks_cube.show()


if __name__ == "__main__":
    main()
