from typing import cast
from rubiks_cube import Direction, RubiksCube, Size, CubePos, Face
from rubiks_cube_solver import RubiksCubeSolver, Var


def generate_true_instance(size: Size, moves: list[str]) -> dict[int, bool]:
    cube = RubiksCube(size)
    true_instance: dict[int, bool] = {}

    for t in range(len(moves)):
        for cube_pos in range(8):
            cube_pos = cast(CubePos, cube_pos)

            colors = cube.get_colors_from_pos(Var.g(cube_pos))
            cube_id, orientation = cube.colors_to_id_and_orientation(colors)

            true_instance[Var.x(cube_pos, cube_id, t)] = True
            true_instance[Var.theta(cube_pos, orientation, t)] = True

        face, direction = cube.parse_move(moves[t])

        if face in [Face.FRONT, Face.LEFT, Face.TOP]:
            face = Face.opposite(face)
            direction = Direction.opposite(direction)

        cube.rotate(face, direction)

        # print(Var.decode(Var.a(face, direction, t + 1)))
        true_instance[Var.a(face, direction, t + 1)] = True

    return true_instance


def reverse_moves(moves: list[str]) -> list[str]:
    reverse_moves = []
    for move in reversed(moves):
        face, direction = RubiksCube.parse_move(move)
        reverse_moves.append(f"{face.to_str()}{Direction.opposite(direction).to_str()}")
    return reverse_moves


def main(size: Size = (2, 2, 2)):
    rubiks_cube = RubiksCube(size)
    moves = rubiks_cube.shuffle(11)

    true_instance = generate_true_instance(size, reverse_moves(moves))

    solver = RubiksCubeSolver(rubiks_cube)
    sat, actions = solver.run(true_instance)

    print("SATISFIABLE" if sat else "UNSATISFIABLE")

    if sat:
        for face, direction in actions:
            rubiks_cube.rotate(face, direction)

        rubiks_cube.show()


if __name__ == "__main__":
    main()
