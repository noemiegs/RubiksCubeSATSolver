from rubiks_cube import RubiksCube
from utils import Size
from rubiks_cube_solver import RubiksCubeSolver
from variables import Var


def main(size: Size = (2, 2, 2)):
    rubiks_cube = RubiksCube(size)
    rubiks_cube.shuffle(faces=Var.faces)

    solver = RubiksCubeSolver(rubiks_cube)
    sat, actions = solver.run()

    print("SATISFIABLE" if sat else "UNSATISFIABLE")

    if sat:
        print(f"Solved in {len(actions)} moves")

        moves = [
            RubiksCube.move_to_str(face, direction, depth)
            for face, direction, depth in actions
        ]
        rubiks_cube.animate(RubiksCube.parse_moves(moves), speed=2)


if __name__ == "__main__":
    main()
