from enum import Enum
from itertools import pairwise
import numpy as np
import pygame
from math import cos, sin, radians
import random
from typing import Literal, cast
from colorama import Fore, Style


WIDTH, HEIGHT = 600, 600
X, Y, Z = 0, 1, 2

CubePos = Literal[0, 1, 2, 3, 4, 5, 6, 7]
Orientation = Literal[0, 1, 2]
Size = tuple[int, int, int]


class Color(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2
    YELLOW = 3
    ORANGE = 4
    WHITE = 5

    def to_rgb(self) -> tuple[int, int, int]:
        return {
            Color.RED: (255, 0, 0),
            Color.BLUE: (0, 0, 255),
            Color.GREEN: (0, 255, 0),
            Color.YELLOW: (255, 255, 0),
            Color.ORANGE: (255, 165, 0),
            Color.WHITE: (255, 255, 255),
        }[self]


class Direction(Enum):
    CLOCKWISE = 0
    HALF_TURN = 1
    COUNTERCLOCKWISE = 2

    @staticmethod
    def from_str(s: str) -> "Direction":
        return {
            "": Direction.CLOCKWISE,
            "2": Direction.HALF_TURN,
            "'": Direction.COUNTERCLOCKWISE,
        }[s]

    @staticmethod
    def opposite(direction: "Direction") -> "Direction":
        return {
            Direction.CLOCKWISE: Direction.COUNTERCLOCKWISE,
            Direction.HALF_TURN: Direction.HALF_TURN,
            Direction.COUNTERCLOCKWISE: Direction.CLOCKWISE,
        }[direction]

    def to_str(self) -> str:
        return {
            Direction.CLOCKWISE: "",
            Direction.HALF_TURN: "2",
            Direction.COUNTERCLOCKWISE: "'",
        }[self]

    def __lt__(self, other: "Direction") -> bool:
        return self.value < other.value


class Face(Enum):
    FRONT = 0
    BACK = 1
    LEFT = 2
    RIGHT = 3
    TOP = 4
    BOTTOM = 5

    def get_vertices_idx(self) -> list[int]:
        return {
            Face.FRONT: [0, 1, 3, 2],
            Face.BACK: [4, 5, 7, 6],
            Face.LEFT: [0, 4, 6, 2],
            Face.RIGHT: [1, 5, 7, 3],
            Face.TOP: [0, 1, 5, 4],
            Face.BOTTOM: [2, 3, 7, 6],
        }[self]

    @staticmethod
    def from_str(s: str) -> "Face":
        return {
            "F": Face.FRONT,
            "B": Face.BACK,
            "L": Face.LEFT,
            "R": Face.RIGHT,
            "U": Face.TOP,
            "D": Face.BOTTOM,
        }[s]

    @staticmethod
    def opposite(face: "Face") -> "Face":
        return {
            Face.FRONT: Face.BACK,
            Face.BACK: Face.FRONT,
            Face.LEFT: Face.RIGHT,
            Face.RIGHT: Face.LEFT,
            Face.TOP: Face.BOTTOM,
            Face.BOTTOM: Face.TOP,
        }[face]

    def to_str(self) -> str:
        return {
            Face.FRONT: "F",
            Face.BACK: "B",
            Face.LEFT: "L",
            Face.RIGHT: "R",
            Face.TOP: "U",
            Face.BOTTOM: "D",
        }[self]

    def __lt__(self, other: "Face") -> bool:
        return self.value < other.value


class RubiksCube:
    """
    ### Fonctions :
    - rotate(face: Face, direction: Direction)
    - shuffle(n: int = 100)
    - apply_rotations(self, rotations: list[str]): Apply a list of rotations using standard notation
        - example : ["U'", "F2", "B", "L'", "R2"]
        - Up (anticlockwise) / Front (2x) / Back (clockwise) / Left (anticlockwise) / Right (2x)
    - solve()
        - solve it and return the list of rotations using standard notation
        !
        - NOT IMPLEMENTED YET

    - show()
        - run a simulator to view the rubik's cube in 3D


    """

    def __init__(self, size: Size) -> None:
        self.size = size
        self.faces = {
            Face.FRONT: np.full((size[0], size[1]), Color.WHITE.value, dtype=np.int8),
            Face.BACK: np.full((size[0], size[1]), Color.YELLOW.value, dtype=np.int8),
            Face.LEFT: np.full((size[2], size[1]), Color.BLUE.value, dtype=np.int8),
            Face.RIGHT: np.full((size[2], size[1]), Color.GREEN.value, dtype=np.int8),
            Face.TOP: np.full((size[0], size[2]), Color.RED.value, dtype=np.int8),
            Face.BOTTOM: np.full((size[0], size[2]), Color.ORANGE.value, dtype=np.int8),
        }

    def get_colors_from_pos(
        self, pos: tuple[int, int, int]
    ) -> tuple[Color, Color, Color]:
        """
        Get the colors of the cube at position pos (front/back, left/right, up/down)
        """
        return (
            Color(
                self.faces[Face.FRONT if pos[2] == 0 else Face.BACK][
                    pos[0] if pos[2] == 0 else (1 - pos[0]), pos[1]
                ]
            ),
            Color(
                self.faces[Face.LEFT if pos[0] == 0 else Face.RIGHT][
                    (1 - pos[2]) if pos[0] == 0 else pos[2], pos[1]
                ]
            ),
            Color(
                self.faces[Face.TOP if pos[1] == 0 else Face.BOTTOM][
                    pos[0], (1 - pos[2]) if pos[1] == 0 else pos[2]
                ]
            ),
        )

    def colors_to_id_and_orientation(
        self, colors: tuple[Color, Color, Color]
    ) -> tuple[CubePos, Orientation]:
        """
        Convert the colors of a corner piece to its id and orientation

        colors: tuple of 3 colors of the corner piece (front/back, left/right, up/down)
        """
        cube_pos = (
            int(Color.GREEN in colors)
            + 2 * int(Color.ORANGE in colors)
            + 4 * int(Color.YELLOW in colors)
        )
        orientation = np.argmax(
            [color in (Color.WHITE, Color.YELLOW) for color in colors]
        )

        return cast(CubePos, cube_pos), cast(Orientation, orientation)

    def _up_face_and_slice(self, face: Face) -> tuple[Face, slice]:
        return {
            Face.FRONT: (Face.TOP, np.s_[:, self.size[2] - 1]),
            Face.BACK: (Face.TOP, np.s_[::-1, 0]),
            Face.LEFT: (Face.TOP, np.s_[0, :]),
            Face.RIGHT: (Face.TOP, np.s_[self.size[0] - 1, ::-1]),
            Face.TOP: (Face.BACK, np.s_[::-1, 0]),
            Face.BOTTOM: (Face.FRONT, np.s_[:, self.size[1] - 1]),
        }[face]

    def _bottom_face_and_slice(self, face: Face) -> tuple[Face, slice]:
        return {
            Face.FRONT: (Face.BOTTOM, np.s_[::-1, 0]),
            Face.BACK: (Face.BOTTOM, np.s_[:, self.size[2] - 1]),
            Face.LEFT: (Face.BOTTOM, np.s_[0, :]),
            Face.RIGHT: (Face.BOTTOM, np.s_[self.size[0] - 1, ::-1]),
            Face.TOP: (Face.FRONT, np.s_[::-1, 0]),
            Face.BOTTOM: (Face.BACK, np.s_[:, self.size[1] - 1]),
        }[face]

    def _left_face_and_slice(self, face: Face) -> tuple[Face, slice]:
        return {
            Face.FRONT: (Face.LEFT, np.s_[self.size[2] - 1, ::-1]),
            Face.BACK: (Face.RIGHT, np.s_[self.size[2] - 1, ::-1]),
            Face.LEFT: (Face.BACK, np.s_[self.size[0] - 1, ::-1]),
            Face.RIGHT: (Face.FRONT, np.s_[self.size[0] - 1, ::-1]),
            Face.TOP: (Face.LEFT, np.s_[::-1, 0]),
            Face.BOTTOM: (Face.LEFT, np.s_[:, self.size[1] - 1]),
        }[face]

    def _right_face_and_slice(self, face: Face) -> tuple[Face, slice]:
        return {
            Face.FRONT: (Face.RIGHT, np.s_[0, :]),
            Face.BACK: (Face.LEFT, np.s_[0, :]),
            Face.LEFT: (Face.FRONT, np.s_[0, :]),
            Face.RIGHT: (Face.BACK, np.s_[0, :]),
            Face.TOP: (Face.RIGHT, np.s_[::-1, 0]),
            Face.BOTTOM: (Face.RIGHT, np.s_[:, self.size[1] - 1]),
        }[face]

    def _rotate_clockwise(self, face: Face) -> None:
        self.faces[face] = np.rot90(self.faces[face], k=1)

        up_face, up_slice = self._up_face_and_slice(face)
        bottom_face, bottom_slice = self._bottom_face_and_slice(face)
        left_face, left_slice = self._left_face_and_slice(face)
        right_face, right_slice = self._right_face_and_slice(face)

        up_color = self.faces[up_face][up_slice].copy()
        self.faces[up_face][up_slice] = self.faces[left_face][left_slice]
        self.faces[left_face][left_slice] = self.faces[bottom_face][bottom_slice]
        self.faces[bottom_face][bottom_slice] = self.faces[right_face][right_slice]
        self.faces[right_face][right_slice] = up_color

    def _rotate_half_turn(self, face: Face) -> None:
        self.faces[face] = self.faces[face][::-1, ::-1]

        up_face, up_slice = self._up_face_and_slice(face)
        bottom_face, bottom_slice = self._bottom_face_and_slice(face)
        left_face, left_slice = self._left_face_and_slice(face)
        right_face, right_slice = self._right_face_and_slice(face)

        up_color = self.faces[up_face][up_slice].copy()
        self.faces[up_face][up_slice] = self.faces[bottom_face][bottom_slice]
        self.faces[bottom_face][bottom_slice] = up_color
        left_color = self.faces[left_face][left_slice].copy()
        self.faces[left_face][left_slice] = self.faces[right_face][right_slice]
        self.faces[right_face][right_slice] = left_color

    def can_rotate(self, face: Face, direction: Direction) -> bool:
        if direction == Direction.HALF_TURN:
            return True

        if face in (Face.FRONT, Face.BACK):
            return self.size[0] == self.size[1]
        if face in (Face.LEFT, Face.RIGHT):
            return self.size[1] == self.size[2]
        if face in (Face.TOP, Face.BOTTOM):
            return self.size[0] == self.size[2]

    def rotate(self, face: Face, direction: Direction) -> None:
        assert self.can_rotate(face, direction), "Cannot rotate face"

        if direction == Direction.HALF_TURN:
            self._rotate_half_turn(face)
            return

        for i in range(direction.value + 1):
            self._rotate_clockwise(face)

    def shuffle(
        self,
        n: int = 100,
        faces: tuple[Face, ...] = tuple(Face),
        directions: tuple[Direction, ...] = tuple(Direction),
    ) -> list[str]:
        moves_str: list[str] = []

        for _ in range(n):
            face = random.choice(faces)
            direction = random.choice(directions)
            while not self.can_rotate(face, direction):
                face = random.choice(faces)
                direction = random.choice(directions)

            self.rotate(face, direction)

            moves_str.append(face.to_str() + direction.to_str())

        return moves_str

    @staticmethod
    def reverse_moves(moves: list[str]) -> list[str]:
        reverse_moves = []
        for face, direction in RubiksCube.parse_moves(moves)[::-1]:
            reverse_moves.append(
                RubiksCube.move_to_str(face, Direction.opposite(direction))
            )
        return reverse_moves

    @staticmethod
    def move_to_str(face: Face, direction: Direction) -> str:
        return f"{face.to_str()}{direction.to_str()}"

    @staticmethod
    def parse_move(move: str) -> tuple[Face, Direction]:
        return Face.from_str(move[0]), Direction.from_str(move[1:])

    @staticmethod
    def parse_moves(moves: list[str]) -> list[tuple[Face, Direction]]:
        return [RubiksCube.parse_move(move) for move in moves]

    def apply_rotations(self, rotations: list[str]) -> None:
        for rotation in rotations:
            self.rotate(*self.parse_move(rotation))

    def _draw_face(
        self,
        screen: pygame.surface.Surface,
        coords: list[tuple[float, float, float]],
        color: Color,
    ) -> None:
        def project_3d_to_2d(point: tuple[float, float, float], scale: float = 200):
            """Convert 3D point to 2D screen coordinates with perspective projection"""
            x, y, z = point
            factor = scale / (z + 4)  # Simple perspective division
            screen_x = int(WIDTH / 2 + x * factor)
            screen_y = int(HEIGHT / 2 + y * factor)
            return screen_x, screen_y

        screen_positions = [project_3d_to_2d(coord) for coord in coords]
        pygame.draw.polygon(screen, color.to_rgb(), screen_positions)
        pygame.draw.polygon(screen, (0, 0, 0), screen_positions, 2)

    def _draw(
        self,
        screen: pygame.surface.Surface,
        angle_x: float,
        angle_y: float,
        rotating_face: Face | None = None,
        rotating_angle: float = 0,
    ) -> None:
        """Draw the 3D Rubik's Cube"""

        def rotate_x(
            point: tuple[float, float, float], angle: float
        ) -> tuple[float, float, float]:
            """Rotate a point in 3D space around X-axis"""
            x, y, z = point
            angle = radians(angle)
            y, z = y * cos(angle) - z * sin(angle), y * sin(angle) + z * cos(angle)
            return x, y, z

        def rotate_y(
            point: tuple[float, float, float], angle: float
        ) -> tuple[float, float, float]:
            """Rotate a point in 3D space around Y-axis"""
            x, y, z = point
            angle = radians(angle)
            x, z = x * cos(angle) + z * sin(angle), -x * sin(angle) + z * cos(angle)
            return x, y, z

        def rotate_z(
            point: tuple[float, float, float], angle: float
        ) -> tuple[float, float, float]:
            """Rotate a point in 3D space around Z-axis"""
            x, y, z = point
            angle = radians(angle)
            x, y = x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle)
            return x, y, z

        faces: list[tuple[list[tuple[float, float, float]], int]] = []

        for cube_x, (x0, x1) in enumerate(
            pairwise(np.linspace(-1, 1, self.size[X] + 1))
        ):
            for cube_y, (y0, y1) in enumerate(
                pairwise(np.linspace(-1, 1, self.size[Y] + 1))
            ):
                for cube_z, (z0, z1) in enumerate(
                    pairwise(np.linspace(-1, 1, self.size[Z] + 1))
                ):
                    points = [
                        (x0, y0, z0),
                        (x1, y0, z0),
                        (x0, y1, z0),
                        (x1, y1, z0),
                        (x0, y0, z1),
                        (x1, y0, z1),
                        (x0, y1, z1),
                        (x1, y1, z1),
                    ]

                    if rotating_face is not None:
                        if rotating_face == Face.LEFT and cube_x == 0:
                            points = [
                                rotate_x(point, rotating_angle) for point in points
                            ]
                        if rotating_face == Face.RIGHT and cube_x == self.size[X] - 1:
                            points = [
                                rotate_x(point, rotating_angle) for point in points
                            ]
                        if rotating_face == Face.TOP and cube_y == 0:
                            points = [
                                rotate_y(point, rotating_angle) for point in points
                            ]
                        if rotating_face == Face.BOTTOM and cube_y == self.size[Y] - 1:
                            points = [
                                rotate_y(point, rotating_angle) for point in points
                            ]
                        if rotating_face == Face.FRONT and cube_z == 0:
                            points = [
                                rotate_z(point, rotating_angle) for point in points
                            ]
                        if rotating_face == Face.BACK and cube_z == self.size[Z] - 1:
                            points = [
                                rotate_z(point, rotating_angle) for point in points
                            ]

                    points = [rotate_x(point, angle_x) for point in points]
                    points = [rotate_y(point, angle_y) for point in points]

                    if cube_x == 0:
                        faces.append(
                            (
                                [points[idx] for idx in Face.LEFT.get_vertices_idx()],
                                self.faces[Face.LEFT][-cube_z - 1, cube_y],
                            )
                        )
                    if cube_x == self.size[X] - 1:
                        faces.append(
                            (
                                [points[idx] for idx in Face.RIGHT.get_vertices_idx()],
                                self.faces[Face.RIGHT][cube_z, cube_y],
                            )
                        )
                    if cube_y == 0:
                        faces.append(
                            (
                                [points[idx] for idx in Face.TOP.get_vertices_idx()],
                                self.faces[Face.TOP][cube_x, -cube_z - 1],
                            )
                        )
                    if cube_y == self.size[Y] - 1:
                        faces.append(
                            (
                                [points[idx] for idx in Face.BOTTOM.get_vertices_idx()],
                                self.faces[Face.BOTTOM][cube_x, cube_z],
                            )
                        )
                    if cube_z == 0:
                        faces.append(
                            (
                                [points[idx] for idx in Face.FRONT.get_vertices_idx()],
                                self.faces[Face.FRONT][cube_x, cube_y],
                            )
                        )
                    if cube_z == self.size[Z] - 1:
                        faces.append(
                            (
                                [points[idx] for idx in Face.BACK.get_vertices_idx()],
                                self.faces[Face.BACK][-cube_x - 1, cube_y],
                            )
                        )

        faces_ordered = sorted(
            faces,
            key=lambda face: sum(coord[Z] for coord in face[0]),
            reverse=True,
        )

        for coords, color in faces_ordered:
            self._draw_face(screen, coords, Color(color))

    def show(
        self,
        screen_size: tuple[int, int] = (600, 600),
        background_color: tuple[int, int, int] = (30, 30, 30),
    ) -> None:
        pygame.init()
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("3D Rubik's Cube")
        clock = pygame.time.Clock()

        running = True
        angle_x, angle_y = 30, 30
        rotating = False
        last_mouse_pos = None

        while running:
            screen.fill(background_color)

            self._draw(screen, angle_x, angle_y)

            pygame.display.flip()
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    rotating = True
                    last_mouse_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP:
                    rotating = False
                    last_mouse_pos = None

            if rotating:
                current_mouse_pos = pygame.mouse.get_pos()
                if last_mouse_pos:
                    dx = last_mouse_pos[0] - current_mouse_pos[0]
                    dy = last_mouse_pos[1] - current_mouse_pos[1]

                    angle_x -= dy * 0.5
                    angle_y += dx * 0.5

                    last_mouse_pos = current_mouse_pos

        pygame.quit()

    def animate(
        self,
        rotations: list[tuple[Face, Direction]],
        screen_size: tuple[int, int] = (600, 600),
        background_color: tuple[int, int, int] = (30, 30, 30),
        speed: float = 1,
    ) -> None:
        pygame.init()
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("3D Rubik's Cube")
        clock = pygame.time.Clock()

        running = True
        angle_x, angle_y = 30, 30
        rotating = False
        last_mouse_pos = None

        rotating_face = None
        rotating_angle = 0
        rotation_idx = 0
        angle_direction = 1
        target_angle = 0

        while running:
            screen.fill(background_color)

            if rotation_idx < len(rotations):
                face, direction = rotations[rotation_idx]

                if rotating_face is None:
                    rotating_face = face
                    rotating_angle = 0

                    angle_direction = {
                        Direction.CLOCKWISE: 1,
                        Direction.HALF_TURN: 2,
                        Direction.COUNTERCLOCKWISE: -1,
                    }[direction] * {
                        Face.FRONT: 1,
                        Face.BACK: -1,
                        Face.LEFT: 1,
                        Face.RIGHT: -1,
                        Face.TOP: 1,
                        Face.BOTTOM: -1,
                    }[face]

                    target_angle = 90 * angle_direction

            if rotating_face is not None:
                rotating_angle += speed * angle_direction

                if abs(rotating_angle) >= abs(target_angle):
                    self.rotate(face, direction)  # type: ignore
                    rotating_face = None
                    rotation_idx += 1

            self._draw(screen, angle_x, angle_y, rotating_face, rotating_angle)

            pygame.display.flip()
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    rotating = True
                    last_mouse_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP:
                    rotating = False
                    last_mouse_pos = None

            if rotating:
                current_mouse_pos = pygame.mouse.get_pos()
                if last_mouse_pos:
                    dx = last_mouse_pos[0] - current_mouse_pos[0]
                    dy = last_mouse_pos[1] - current_mouse_pos[1]

                    angle_x -= dy * 0.5
                    angle_y += dx * 0.5

                    last_mouse_pos = current_mouse_pos

        pygame.quit()

    def copy(self) -> "RubiksCube":
        new_cube = RubiksCube(self.size)
        new_cube.faces = {face: self.faces[face].copy() for face in self.faces}
        return new_cube

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RubiksCube):
            return False

        return all(
            np.array_equal(self.faces[face], other.faces[face]) for face in self.faces
        )

    def __str__(self) -> str:
        colors = [
            Fore.RED,
            Fore.MAGENTA,
            Fore.GREEN,
            Fore.YELLOW,
            Fore.BLUE,
            Fore.WHITE,
        ]

        s = ""
        for y in range(self.size[X]):
            s += "   "
            for x in range(self.size[X]):
                s += f"{colors[self.faces[Face.TOP][x, y].item()]}@"
            s += "\n"
        for y in range(self.size[Y]):
            for z in range(self.size[Z]):
                s += f"{colors[self.faces[Face.LEFT][z, y].item()]}@"
            for x in range(self.size[X]):
                s += f"{colors[self.faces[Face.FRONT][x, y].item()]}@"
            for z in range(self.size[Z]):
                s += f"{colors[self.faces[Face.RIGHT][z, y].item()]}@"
            for x in range(self.size[X]):
                s += f"{colors[self.faces[Face.BACK][x, y].item()]}@"
            s += "\n"
        for y in range(self.size[X]):
            s += "   "
            for x in range(self.size[X]):
                s += f"{colors[self.faces[Face.BOTTOM][x, y].item()]}@"
            s += "\n"

        s += Style.RESET_ALL

        return s


def main():
    CUBE_SIZE = (3, 3, 3)

    cube = RubiksCube(CUBE_SIZE)
    moves = cube.shuffle()

    cube.animate(RubiksCube.parse_moves(RubiksCube.reverse_moves(moves)), speed=25)


if __name__ == "__main__":
    main()
