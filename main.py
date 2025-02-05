from typing import cast
from rubiks_cube import RubiksCube, Size, CubePos
from rubiks_cube_solver import RubiksCubeSolver, Var


def generate_true_instance(size: Size, moves: list[str]) -> dict[int, bool]:
    cube = RubiksCube(size)
    true_instance: dict[int, bool] = {}

    # for t in range(len(moves)):
    #     for cube_pos in range(8):
    #         cube_pos = cast(CubePos, cube_pos)

    #         colors = cube.get_colors_from_pos(Var.g(cube_pos))
    #         cube_id, orientation = cube.colors_to_id_and_orientation(colors)

    #         true_instance[Var.x(cube_pos, cube_id, t)] = True
    #         true_instance[Var.theta(cube_pos, orientation, t)] = True

    #     face, direction = cube.parse_move(moves[t])
    #     cube.rotate(face, direction)

    #     true_instance[Var.a(face, direction, t)] = True

    return true_instance


def main(size: Size = (2, 2, 2)):
    rubiks_cube = RubiksCube(size)
    moves = rubiks_cube.shuffle(10)

    true_instance = generate_true_instance(size, moves)

    solver = RubiksCubeSolver(rubiks_cube)
    actions = solver.run(true_instance)

    for face, direction in actions:
        rubiks_cube.rotate(face, direction)

    rubiks_cube.show()


if __name__ == "__main__":
    main()
