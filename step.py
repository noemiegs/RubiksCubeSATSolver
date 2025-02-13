from abc import ABC, abstractmethod
from itertools import product
from typing import cast

from utils import CornerPos, Direction, Face
from variables_abc import VariableTheta, VariableX, NamedClause
from variables import Var, Variable


def generate_final_clauses_x(x: type[VariableX]) -> list[NamedClause]:
    clauses: list[NamedClause] = []

    for idx in x.parent().pos_range():
        clauses.append(
            (
                f"Etat final {x.__class__.__qualname__} x, position du cube {idx}",
                [x(idx, idx, Variable.t_max)],
            )
        )

    return clauses


def generate_final_clauses_theta(theta: type[VariableTheta]) -> list[NamedClause]:
    clauses: list[NamedClause] = []

    for idx in theta.parent().pos_range():
        clauses.append(
            (
                f"Etat final {theta.__class__.__qualname__} theta, position du cube {idx}",
                [theta(idx, 0, Variable.t_max)],
            )
        )

    return clauses


class Step(ABC):
    def __init__(self) -> None:
        self.actions: set[tuple[Face, Direction, int]] = {
            *product(Var.faces, Direction, Var.depths)
        }

    @abstractmethod
    def generate_final_clauses(self) -> list[NamedClause]: ...

    def __add__(self, other: "Step") -> "Step":
        return Combined(self, other)


class Combined(Step):
    def __init__(self, *steps: Step):
        self.steps = steps
        self.actions = set.union(*(step.actions for step in steps))

    def generate_final_clauses(self) -> list[NamedClause]:
        return [
            clause for step in self.steps for clause in step.generate_final_clauses()
        ]


class Corners(Step):
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
    def generate_final_clauses(self) -> list[NamedClause]:
        return generate_final_clauses_x(Var.Centers.x)


class WhiteCross(Step):
    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        clauses.append(
            (
                "Etat final WhiteCrossStep x, center",
                [Var.Centers.x(4, 4, Variable.t_max)],
            )
        )
        for idx in [0, 1, 4, 5]:
            clauses.append(
                (
                    f"Etat final WhiteCrossStep x, edge {idx}",
                    [Var.Edges.x(idx, idx, Variable.t_max)],
                )
            )
            clauses.append(
                (
                    f"Etat final WhiteCrossStep theta, edge {idx}",
                    [Var.Edges.theta(idx, 0, Variable.t_max)],
                )
            )
        return clauses


class WhiteCorners(Step):
    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for idx in [0, 1, 2, 3]:
            idx = cast(CornerPos, idx)

            clauses.append(
                (
                    f"Etat final WhiteCorners x, corner {idx}",
                    [Var.Corners.x(idx, idx, Variable.t_max)],
                )
            )
            clauses.append(
                (
                    f"Etat final WhiteCorners theta, corner {idx}",
                    [Var.Corners.theta(idx, 0, Variable.t_max)],
                )
            )
        return clauses


class SecondCrownCenters(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {*product([Face.BACK], Direction, [1])}

    def generate_final_clauses(self) -> list[tuple[str, list[Variable]]]:
        clauses: list[tuple[str, list[Variable]]] = []
        for idx in [0, 1, 2, 3]:
            clauses.append(
                (
                    f"Etat final SecondCrown x, centers {idx}",
                    [Var.Centers.x(idx, idx, Variable.t_max)],
                )
            )

        return clauses


class SecondCrownEdge(Step):
    def __init__(self, idx: int) -> None:
        self.idx = idx
        super().__init__()

    def generate_final_clauses(self) -> list[tuple[str, list[Variable]]]:
        return [
            (
                f"Etat final SecondCrown x, corner {self.idx}",
                [Var.Edges.x(self.idx, self.idx, Variable.t_max)],
            ),
            (
                f"Etat final SecondCrown theta, corner {self.idx}",
                [Var.Edges.theta(self.idx, 0, Variable.t_max)],
            ),
        ]


class YellowCross(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {
            *product(Var.faces, [Direction.CLOCKWISE, Direction.COUNTERCLOCKWISE], [0]),
        }

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        clauses.append(
            (
                "Etat final WhiteCrossStep x, center",
                [Var.Centers.x(5, 5, Variable.t_max)],
            )
        )
        for idx in [2, 3, 6, 7]:
            clauses.append(
                (
                    f"Etat final WhiteCrossStep x, edge {idx}",
                    [Var.Edges.x(idx, idx, Variable.t_max)],
                )
            )
            clauses.append(
                (
                    f"Etat final WhiteCrossStep theta, edge {idx}",
                    [Var.Edges.theta(idx, 0, Variable.t_max)],
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
        clauses.append(
            (
                "Etat final WhiteCrossStep x, center",
                [Var.Centers.x(5, 5, Variable.t_max)],
            )
        )
        for idx in [2, 3]:
            clauses.append(
                (
                    f"Etat final WhiteCrossStep x, edge {idx}",
                    [Var.Edges.x(idx, idx, Variable.t_max)],
                )
            )
            clauses.append(
                (
                    f"Etat final WhiteCrossStep theta, edge {idx}",
                    [Var.Edges.theta(idx, 0, Variable.t_max)],
                )
            )
        return clauses


class OtherYellowLine(Step):
    def __init__(self) -> None:
        super().__init__()

        self.actions = {
            *product(Var.faces, [Direction.CLOCKWISE, Direction.COUNTERCLOCKWISE], [0]),
        }

    def generate_final_clauses(self) -> list[NamedClause]:
        clauses: list[NamedClause] = []
        for idx in [6, 7]:
            clauses.append(
                (
                    f"Etat final WhiteCrossStep x, edge {idx}",
                    [Var.Edges.x(idx, idx, Variable.t_max)],
                )
            )
            clauses.append(
                (
                    f"Etat final WhiteCrossStep theta, edge {idx}",
                    [Var.Edges.theta(idx, 0, Variable.t_max)],
                )
            )
        return clauses


class FinalCrownCorners(Step):
    def __init__(self, idx: CornerPos) -> None:
        self.idx: CornerPos = idx
        super().__init__()

        self.actions = {
            *product(Var.faces, [Direction.CLOCKWISE, Direction.COUNTERCLOCKWISE], [0]),
        }

    def generate_final_clauses(self) -> list[tuple[str, list[Variable]]]:
        return [
            (
                f"Etat final SecondCrown x, corner {self.idx}",
                [Var.Corners.x(self.idx, self.idx, Variable.t_max)],
            ),
            (
                f"Etat final SecondCrown theta, corner {self.idx}",
                [Var.Corners.theta(self.idx, 0, Variable.t_max)],
            ),
        ]


All = Corners() + Edges() + Centers()
