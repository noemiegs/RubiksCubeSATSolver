from abc import ABC, abstractmethod
from itertools import product
from typing import cast

import numpy as np

from utils import CornerPos, Direction, EdgeIdx, Face
from variables_abc import TIdx, TPos, VariableTheta, VariableX, NamedClause
from variables import Var, Variable


def generate_final_clauses_x(x: type[VariableX[TPos, TIdx]]) -> list[NamedClause]:
    clauses: list[NamedClause] = []

    for pos in x.pos_range():
        for var in x.from_decoded(pos, x.pos_to_idx(pos), Variable.t_max):
            clauses.append(
                (f"Etat final {x.__qualname__} x, position du cube {pos}", [var])
            )

    return clauses


def generate_final_clauses_theta(
    theta: type[VariableTheta[TPos, TIdx]],
) -> list[NamedClause]:
    clauses: list[NamedClause] = []

    for pos in theta.pos_range():
        clauses.append(
            (
                f"Etat final {theta.__qualname__} x, position du cube {pos}",
                [theta(pos, 0, Variable.t_max)],  # type: ignore
            )
        )

    return clauses


class Step(ABC):
    def __init__(self) -> None:
        self.actions: set[tuple[Face, Direction, int]] = {
            *product(Var.faces, Var.directions, Var.depths)
        }

    def t_median(self) -> int:
        return 11

    @abstractmethod
    def generate_final_clauses(self) -> list[NamedClause]: ...

    def __add__(self, other: "Step") -> "Step":
        return Combined(self, other)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class Combined(Step):
    def __init__(self, *steps: Step):
        self.steps = steps
        self.actions = set.union(*(step.actions for step in steps))

    def t_median(self) -> int:
        return round(np.median([step.t_median() for step in self.steps]))

    def generate_final_clauses(self) -> list[NamedClause]:
        return [
            clause for step in self.steps for clause in step.generate_final_clauses()
        ]

    def __repr__(self) -> str:
        return " + ".join(repr(step) for step in self.steps)


class Corners(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {*product(Var.faces, Var.directions, [0])}

    def t_median(self) -> int:
        return 8

    def generate_final_clauses(self) -> list[NamedClause]:
        return generate_final_clauses_x(Var.Corners.x) + generate_final_clauses_theta(
            Var.Corners.theta
        )


class Edges(Step):
    def generate_final_clauses(self) -> list[NamedClause]:
        return generate_final_clauses_x(Var.Edges.x) + generate_final_clauses_theta(
            Var.Edges.theta
        )


class Centers(Step):
    def t_median(self) -> int:
        return 2

    def generate_final_clauses(self) -> list[NamedClause]:
        return generate_final_clauses_x(Var.Centers.x)


class WhiteAndYellowCenters(Step):
    def t_median(self) -> int:
        return 7

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []

        for pos in Var.Centers.x.pos_range():
            idx = Var.Centers.x.pos_to_idx(pos)

            if idx < 4:
                continue

            for var in Var.Centers.x.from_decoded(pos, idx, Variable.t_max):
                clauses.append((f"Etat final Centers x, position du cube {pos}", [var]))

        return clauses


class WhiteAndYellowCentersLine(Step):
    def __init__(self, line_idx: int) -> None:
        self.line_idx = line_idx

        super().__init__()

    def t_median(self) -> int:
        return 5

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []

        for pos in Var.Centers.x.pos_range():
            idx = Var.Centers.x.pos_to_idx(pos)

            if idx < 4:
                continue

            coords = Var.Centers.g(pos)
            axis = np.argmax([p == 0 or p == Variable.cube_size - 1 for p in coords])
            other_coords = [p for i, p in enumerate(coords) if i != axis]

            if other_coords[0] - 1 != self.line_idx:
                continue

            for var in Var.Centers.x.from_decoded(pos, idx, Variable.t_max):
                clauses.append((f"Etat final Centers x, position du cube {pos}", [var]))

        return clauses


class CentersNotWhiteOrYellow(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = (
            {*product(Var.faces, [Direction.HALF_TURN], [1, 2])}
            | {
                *product(
                    [Face.BACK],
                    [Direction.CLOCKWISE, Direction.COUNTERCLOCKWISE],
                    [1, 2],
                )
            }
            | {*product([Face.BOTTOM, Face.RIGHT], Var.directions, [0])}
        )

    def t_median(self) -> int:
        return 8

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []

        for pos in Var.Centers.x.pos_range():
            idx = Var.Centers.x.pos_to_idx(pos)

            if idx >= 4:
                continue

            for var in Var.Centers.x.from_decoded(pos, idx, Variable.t_max):
                clauses.append((f"Etat final Centers x, position du cube {pos}", [var]))

        return clauses


class SameEdges(Step):
    def __init__(self, edge_idx: EdgeIdx) -> None:
        self.edge_idx = edge_idx
        super().__init__()

    def t_median(self) -> int:
        return 5

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for i in range(Variable.cube_size - 2):
            for j in range(i):
                for idx in Var.Edges.x.idx_range():
                    var_i = Var.Edges.x(
                        self.edge_idx * (Variable.cube_size - 2) + i,
                        idx,
                        Variable.t_max,
                    )
                    var_j = Var.Edges.x(
                        self.edge_idx * (Variable.cube_size - 2) + j,
                        idx,
                        Variable.t_max,
                    )

                    clauses.append(
                        (
                            f"Etat final SameEdges x, edge {self.edge_idx}, i={i}, j={j} clause 1",
                            [var_i, -var_j],
                        )
                    )
                    clauses.append(
                        (
                            f"Etat final SameEdges x, edge {self.edge_idx}, i={i}, j={j} clause 2",
                            [-var_i, var_j],
                        )
                    )

        return clauses


class EdgeOrientation(Step):
    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for pos in Var.Edges.pos_range():
            clauses.append(
                (
                    f"Etat final EdgeOrientation theta, edge {pos}",
                    [Var.Edges.theta(pos, 0, Variable.t_max)],
                )
            )
        return clauses


class EdgePostionOnCircle(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {*product(Var.faces, [Direction.HALF_TURN], [1])} | {
            *product(
                [Face.BACK, Face.BOTTOM],
                [Direction.CLOCKWISE, Direction.COUNTERCLOCKWISE],
                [0],
            )
        }

    def t_median(self) -> int:
        return 11

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for pos in [0, 1, 2, 3]:
            clauses.append(
                (
                    f"Etat final EdgePosition x, edge {pos}",
                    [-Var.Edges.x(pos, 0, Variable.t_max)],
                )
            )
            clauses.append(
                (
                    f"Etat final EdgePosition x, edge {pos}",
                    [-Var.Edges.x(pos, 1, Variable.t_max)],
                )
            )

        for pos in [4, 5, 6, 7]:
            clauses.append(
                (
                    f"Etat final EdgePosition x, edge {pos}",
                    [-Var.Edges.x(pos, 0, Variable.t_max)],
                )
            )
            clauses.append(
                (
                    f"Etat final EdgePosition x, edge {pos}",
                    [Var.Edges.x(pos, 1, Variable.t_max)],
                )
            )

        for pos in [8, 9, 10, 11]:
            clauses.append(
                (
                    f"Etat final EdgePosition x, edge {pos}",
                    [Var.Edges.x(pos, 0, Variable.t_max)],
                )
            )
            clauses.append(
                (
                    f"Etat final EdgePosition x, edge {pos}",
                    [-Var.Edges.x(pos, 1, Variable.t_max)],
                )
            )

        return clauses


class FirstEdgePosition(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {*product(Var.faces, [Direction.HALF_TURN], [0])} | {
            *product(Var.faces, Var.directions, [1])
        }

    def t_median(self) -> int:
        return 8

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for pos in [0, 1, 2, 3]:
            for var in Var.Edges.x.from_decoded(
                pos, Var.Edges.x.pos_to_idx(pos), Variable.t_max
            ):
                clauses.append((f"Etat final EdgePosition x, edge {pos}", [var]))

        return clauses


class SecondEdgePosition(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {*product(Var.faces, [Direction.HALF_TURN], [0])} | {
            *product(Var.faces, Var.directions, [1])
        }

    def t_median(self) -> int:
        return 8

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for pos in [4, 5, 6, 7]:
            for var in Var.Edges.x.from_decoded(
                pos, Var.Edges.x.pos_to_idx(pos), Variable.t_max
            ):
                clauses.append((f"Etat final EdgePosition x, edge {pos}", [var]))

        return clauses


class ThirdEdgePosition(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {*product(Var.faces, [Direction.HALF_TURN], [0])} | {
            *product(Var.faces, Var.directions, [1])
        }

    def t_median(self) -> int:
        return 8

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for pos in [8, 9, 10, 11]:
            for var in Var.Edges.x.from_decoded(
                pos, Var.Edges.x.pos_to_idx(pos), Variable.t_max
            ):
                clauses.append((f"Etat final EdgePosition x, edge {pos}", [var]))

        return clauses


class WhiteCross(Step):
    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for var in Var.Centers.x.from_decoded(
            4, Var.Centers.x.pos_to_idx(4), Variable.t_max
        ):
            clauses.append(("Etat final WhiteCrossStep x, center", [var]))

        for pos in [0, 1, 4, 5]:
            clauses += [
                cast(NamedClause, (f"Etat final WhiteCrossStep x, edge {pos}", [var]))
                for var in Var.Edges.x.from_decoded(
                    pos, Var.Edges.x.pos_to_idx(pos), Variable.t_max
                )
            ]

            clauses.append(
                (
                    f"Etat final WhiteCrossStep theta, edge {pos}",
                    [Var.Edges.theta(pos, 0, Variable.t_max)],
                )
            )
        return clauses


class WhiteCorners(Step):
    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for pos in [0, 1, 2, 3]:
            pos = cast(CornerPos, pos)

            for var in Var.Corners.x.from_decoded(
                pos, Var.Corners.x.pos_to_idx(pos), Variable.t_max
            ):
                clauses.append((f"Etat final WhiteCorners x, corner {pos}", [var]))

            clauses.append(
                (
                    f"Etat final WhiteCorners theta, corner {pos}",
                    [Var.Corners.theta(pos, 0, Variable.t_max)],
                )
            )
        return clauses


class SecondCrownCenters(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {*product([Face.BACK], Var.directions, [1])}

    def generate_final_clauses(self) -> list[tuple[str, list[Variable]]]:
        clauses: list[tuple[str, list[Variable]]] = []
        for pos in [0, 1, 2, 3]:
            for var in Var.Centers.x.from_decoded(
                pos, Var.Centers.x.pos_to_idx(pos), Variable.t_max
            ):
                clauses.append((f"Etat final SecondCrown x, centers {pos}", [var]))

        return clauses


class SecondCrownEdge(Step):
    def __init__(self, pos: int) -> None:
        self.pos = pos
        super().__init__()

    def generate_final_clauses(self) -> list[tuple[str, list[Variable]]]:
        clauses: list[NamedClause] = []
        for var in Var.Edges.x.from_decoded(
            self.pos, Var.Edges.x.pos_to_idx(self.pos), Variable.t_max
        ):
            clauses.append((f"Etat final SecondCrown x, edge {self.pos}", [var]))

        clauses.append(
            (
                f"Etat final SecondCrown theta, edge {self.pos}",
                [Var.Edges.theta(self.pos, 0, Variable.t_max)],
            )
        )

        return clauses


class SecondCrownEdges(Step):
    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for pos in [8, 9, 10, 11]:
            for var in Var.Edges.x.from_decoded(
                pos, Var.Edges.x.pos_to_idx(pos), Variable.t_max
            ):
                clauses.append((f"Etat final SecondCrown x, edge {pos}", [var]))

            clauses.append(
                (
                    f"Etat final SecondCrown theta, edge {pos}",
                    [Var.Edges.theta(pos, 0, Variable.t_max)],
                )
            )
        return clauses


class YellowCross(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {
            *product(Var.faces, [Direction.CLOCKWISE, Direction.COUNTERCLOCKWISE], [0]),
        }

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for var in Var.Centers.x.from_decoded(
            5, Var.Centers.x.pos_to_idx(5), Variable.t_max
        ):
            clauses.append(("Etat final WhiteCrossStep x, center", [var]))

        for pos in [2, 3, 6, 7]:
            for var in Var.Edges.x.from_decoded(
                pos, Var.Edges.x.pos_to_idx(pos), Variable.t_max
            ):
                clauses.append((f"Etat final WhiteCrossStep x, edge {pos}", [var]))

            clauses.append(
                (
                    f"Etat final WhiteCrossStep theta, edge {pos}",
                    [Var.Edges.theta(pos, 0, Variable.t_max)],
                )
            )
        return clauses


class YellowLine(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {
            *product(Var.faces, [Direction.CLOCKWISE, Direction.COUNTERCLOCKWISE], [0]),
        }

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for var in Var.Centers.x.from_decoded(
            5, Var.Centers.x.pos_to_idx(5), Variable.t_max
        ):
            clauses.append(("Etat final WhiteCrossStep x, center", [var]))

        for pos in [2, 3]:
            for var in Var.Edges.x.from_decoded(
                pos, Var.Edges.x.pos_to_idx(pos), Variable.t_max
            ):
                clauses.append((f"Etat final WhiteCrossStep x, edge {pos}", [var]))

            clauses.append(
                (
                    f"Etat final WhiteCrossStep theta, edge {pos}",
                    [Var.Edges.theta(pos, 0, Variable.t_max)],
                )
            )
        return clauses


class OtherYellowLine(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {
            *product(Var.faces, [Direction.CLOCKWISE, Direction.COUNTERCLOCKWISE], [0]),
        }

    def t_median(self) -> int:
        return 11

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for pos in [6, 7]:
            for var in Var.Edges.x.from_decoded(
                pos, Var.Edges.x.pos_to_idx(pos), Variable.t_max
            ):
                clauses.append((f"Etat final WhiteCrossStep x, edge {pos}", [var]))

            clauses.append(
                (
                    f"Etat final WhiteCrossStep theta, edge {pos}",
                    [Var.Edges.theta(pos, 0, Variable.t_max)],
                )
            )
        return clauses


class FinalCrownCornersPosition(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {
            *product(Var.faces, [Direction.CLOCKWISE, Direction.COUNTERCLOCKWISE], [0]),
        }

    def t_median(self) -> int:
        return 12

    def generate_final_clauses(self) -> list[tuple[str, list[Variable]]]:
        clauses: list[NamedClause] = []
        for pos in [4, 5, 6, 7]:
            pos = cast(CornerPos, pos)
            for var in Var.Corners.x.from_decoded(
                pos, Var.Corners.x.pos_to_idx(pos), Variable.t_max
            ):
                clauses.append((f"Etat final SecondCrown x, corner {pos}", [var]))

        return clauses


class FinalCrownCornerOrientation(Step):
    def __init__(self, idx: CornerPos) -> None:
        self.idx: CornerPos = idx
        super().__init__()

        self.actions = {
            *product(Var.faces, [Direction.CLOCKWISE, Direction.COUNTERCLOCKWISE], [0]),
        }

    def t_median(self) -> int:
        return 11

    def generate_final_clauses(self) -> list[tuple[str, list[Variable]]]:
        return [
            (
                f"Etat final SecondCrown theta, corner {self.idx}",
                [Var.Corners.theta(self.idx, 0, Variable.t_max)],
            )
        ]
