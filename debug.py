from rubiks_cube import RubiksCube
from rubiks_cube_solver import RubiksCubeSolver
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
        idx, o = cube.get_vars_from_corner_pos(pos)

        true_instance.extend(Var.Corners.x.from_decoded(pos, idx, t))
        true_instance.append(Var.Corners.theta(pos, o, t))

    for pos in Var.Edges.pos_range():
        idx, o = cube.get_vars_from_edge_pos(pos)
        true_instance.extend(Var.Edges.x.from_decoded(pos, idx, t))
        true_instance.append(Var.Edges.theta(pos, 0, t, bool(o)))

    for pos in Var.Centers.pos_range():
        idx = cube.get_vars_from_center_pos(pos)
        true_instance.extend(Var.Centers.x.from_decoded(pos, idx, t))


def main(size: int = 3):
    t_max = 11

    Variable.cube_size = size
    Var.depths = list(range(Variable.cube_size - 1))

    rubiks_cube = RubiksCube((size, size, size))
    moves = rubiks_cube.shuffle(t_max)

    solver = RubiksCubeSolver(rubiks_cube, "rubiks_cube.cnf")
    sat, actions = solver.run(t_max, rubiks_cube)

    print("SATISFIABLE" if sat else "UNSATISFIABLE")

    if sat:
        print(f"Solved in {len(actions)} moves")

        moves = [
            RubiksCube.move_to_str(action.face, action.direction, action.depth)
            for action in actions
        ]
        rubiks_cube.animate(RubiksCube.parse_moves(moves), speed=5, recording=True)
    else:
        true_instance = generate_true_instance(
            rubiks_cube, RubiksCube.reverse_moves(moves)
        )
        clauses = solver.generate_clauses(rubiks_cube, t_max)

        _, unsatclauses = solver.verify(true_instance, clauses)
        for clause in unsatclauses:
            print(clause)


if __name__ == "__main__":
    main()
