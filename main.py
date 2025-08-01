from rubiks_cube import RubiksCube
from rubiks_cube_solver import RubiksCubeSolver
import step as Step


def main(size: int = 3):
    rubiks_cube = RubiksCube((size, size, size))
    rubiks_cube.shuffle(replace_origin=True)

    solver = RubiksCubeSolver(rubiks_cube, "rubiks_cube.cnf")
    actions = solver.find_optimal(
        steps=[
            # Step.WhiteAndYellowCentersLine(0),
            # Step.WhiteAndYellowCentersLine(1),
            # Step.CentersNotWhiteOrYellow(),
            # Step.SameEdges(0),
            # Step.SameEdges(1),
            # Step.SameEdges(2),
            # Step.SameEdges(3),
            # Step.SameEdges(4),
            # Step.SameEdges(5),
            # Step.SameEdges(6),
            # Step.SameEdges(7),
            # Step.SameEdges(8),
            # Step.SameEdges(9),
            # Step.SameEdges(10),
            Step.Corners(),
            Step.EdgeOrientation() + Step.Centers(),
            Step.EdgePostionOnCircle(),
            Step.FirstEdgePosition()
            + Step.SecondEdgePosition()
            + Step.ThirdEdgePosition(),
        ],
    )

    print(f"Solved in {len(actions)} moves")
    moves = [
        RubiksCube.move_to_str(action.face, action.direction, action.depth)
        for action in actions
    ]

    rubiks_cube.animate(RubiksCube.parse_moves(moves), speed=5, recording=True)


if __name__ == "__main__":
    main()
