import numpy as np
import pygame
import random
import re
import imageio

from itertools import pairwise
from math import cos, sin, radians
from typing import cast
from colorama import Fore, Style
from utils import (
    CenterIdx,
    CenterPos,
    Color,
    Direction,
    EdgeIdx,
    EdgeOrientation,
    Face,
    Size,
    CornerPos,
    CornerOrientation,
    EdgePos,
)

from variables import Var
from variables_abc import Variable


WIDTH, HEIGHT = 600, 600
X, Y, Z = 0, 1, 2


class RubiksCube:
    """
    ### Fonctions :
    - rotate(face: Face, direction: Direction)
    - shuffle(n: int = 100)
    - apply_rotations(self, rotations: list[str]): Apply a list of rotations using standard notation
        - example : ["U'", "F2", "B", "L'", "R2"]
        - Up (anticlockwise) / Front (2x) / Back (clockwise) / Left (anticlockwise) / Right (2x)
    - show()
        - run a simulator to view the rubik's cube in 3D
    - animate(rotations: list[str], speed: float = 1)
        - run a simulator to view the rubik's cube in 3D with animations
    """

    def __init__(self, size: Size) -> None:
        self.size = size
        self.faces = {
            Face.FRONT: np.full((size[X], size[Y]), Color.WHITE.value, dtype=np.int8),
            Face.BACK: np.full((size[X], size[Y]), Color.YELLOW.value, dtype=np.int8),
            Face.LEFT: np.full((size[Z], size[Y]), Color.BLUE.value, dtype=np.int8),
            Face.RIGHT: np.full((size[Z], size[Y]), Color.GREEN.value, dtype=np.int8),
            Face.TOP: np.full((size[X], size[Z]), Color.RED.value, dtype=np.int8),
            Face.BOTTOM: np.full((size[X], size[Z]), Color.ORANGE.value, dtype=np.int8),
        }

    def get_vars_from_corner_pos(
        self, pos: CornerPos
    ) -> tuple[CornerPos, CornerOrientation]:
        x, y, z = Var.Corners.g(pos)

        colors = [
            Color(
                self.faces[Face.FRONT if z == 0 else Face.BACK][
                    x if z == 0 else (-x - 1), y
                ]
            ),
            Color(
                self.faces[Face.LEFT if x == 0 else Face.RIGHT][
                    (-z - 1) if x == 0 else z, y
                ]
            ),
            Color(
                self.faces[Face.TOP if y == 0 else Face.BOTTOM][
                    x, (-z - 1) if y == 0 else z
                ]
            ),
        ]

        idx = Var.Corners.g_inv(
            int(Color.GREEN in colors) * (self.size[X] - 1),
            int(Color.ORANGE in colors) * (self.size[Y] - 1),
            int(Color.YELLOW in colors) * (self.size[Z] - 1),
        )
        orientation = cast(
            CornerOrientation,
            np.argmax([color in (Color.WHITE, Color.YELLOW) for color in colors]),
        )

        return idx, orientation

    def get_vars_from_edge_pos(self, pos: EdgePos) -> tuple[EdgeIdx, EdgeOrientation]:
        coords = list(Var.Edges.g(pos))

        axis = np.argmax([0 < p < self.size[X] - 1 for p in coords]).item()
        for i in range(len(coords)):
            if i != axis:
                coords[i] = -int(coords[i] != 0)

        match coords:
            case [x, 0, 0]:
                colors = [
                    Color(self.faces[Face.TOP][x, -1]),
                    Color(self.faces[Face.FRONT][x, 0]),
                ]
            case [x, -1, 0]:
                colors = [
                    Color(self.faces[Face.BOTTOM][x, 0]),
                    Color(self.faces[Face.FRONT][x, -1]),
                ]
            case [x, 0, -1]:
                colors = [
                    Color(self.faces[Face.TOP][x, 0]),
                    Color(self.faces[Face.BACK][-x - 1, 0]),
                ]
            case [x, -1, -1]:
                colors = [
                    Color(self.faces[Face.BOTTOM][x, -1]),
                    Color(self.faces[Face.BACK][-x - 1, -1]),
                ]
            case [0, y, 0]:
                colors = [
                    Color(self.faces[Face.LEFT][-1, y]),
                    Color(self.faces[Face.FRONT][0, y]),
                ]
            case [-1, y, 0]:
                colors = [
                    Color(self.faces[Face.RIGHT][0, y]),
                    Color(self.faces[Face.FRONT][-1, y]),
                ]
            case [0, y, -1]:
                colors = [
                    Color(self.faces[Face.LEFT][0, y]),
                    Color(self.faces[Face.BACK][-1, y]),
                ]
            case [-1, y, -1]:
                colors = [
                    Color(self.faces[Face.RIGHT][-1, y]),
                    Color(self.faces[Face.BACK][0, y]),
                ]
            case [0, 0, z]:
                colors = [
                    Color(self.faces[Face.TOP][0, -z - 1]),
                    Color(self.faces[Face.LEFT][-z - 1, 0]),
                ]
            case [-1, 0, z]:
                colors = [
                    Color(self.faces[Face.TOP][-1, -z - 1]),
                    Color(self.faces[Face.RIGHT][z, 0]),
                ]
            case [0, -1, z]:
                colors = [
                    Color(self.faces[Face.BOTTOM][0, z]),
                    Color(self.faces[Face.LEFT][-z - 1, -1]),
                ]
            case [-1, -1, z]:
                colors = [
                    Color(self.faces[Face.BOTTOM][-1, z]),
                    Color(self.faces[Face.RIGHT][z, -1]),
                ]
            case _:
                raise ValueError(f"Invalid edge position: {pos}")

        orientation = cast(EdgeOrientation, int(colors[0].value < colors[1].value))
        idx = None
        # X Axis
        if Color.BLUE not in colors and Color.GREEN not in colors:
            idx = int(Color.ORANGE in colors) + 2 * int(Color.YELLOW in colors)
        # Y Axis
        if Color.RED not in colors and Color.ORANGE not in colors:
            idx = 4 + int(Color.GREEN in colors) + 2 * int(Color.YELLOW in colors)
        # Z Axis
        if Color.WHITE not in colors and Color.YELLOW not in colors:
            idx = 8 + int(Color.GREEN in colors) + 2 * int(Color.ORANGE in colors)

        assert idx is not None, f"Invalid edge colors: {colors}"

        return cast(EdgeIdx, idx), orientation

    def get_vars_from_center_pos(self, pos: CenterPos) -> CenterIdx:
        x, y, z = Var.Centers.g(pos)

        color = None
        if x == 0:
            color = Color(self.faces[Face.LEFT][-z - 1, y])
        if x == self.size[X] - 1:
            color = Color(self.faces[Face.RIGHT][z, y])
        if y == 0:
            color = Color(self.faces[Face.TOP][x, -z - 1])
        if y == self.size[Y] - 1:
            color = Color(self.faces[Face.BOTTOM][x, z])
        if z == 0:
            color = Color(self.faces[Face.FRONT][x, y])
        if z == self.size[Z] - 1:
            color = Color(self.faces[Face.BACK][-x - 1, y])
        assert color is not None, f"Invalid center position: {pos}"

        return cast(
            CenterIdx,
            {
                Color.BLUE: 0,
                Color.GREEN: 1,
                Color.RED: 2,
                Color.ORANGE: 3,
                Color.WHITE: 4,
                Color.YELLOW: 5,
            }[color],
        )

    def __up_face_and_slice(self, face: Face, depth: int) -> tuple[Face, slice]:
        return {
            Face.FRONT: (Face.TOP, np.s_[:, self.size[Z] - 1 - depth]),
            Face.BACK: (Face.TOP, np.s_[::-1, depth]),
            Face.LEFT: (Face.TOP, np.s_[depth, :]),
            Face.RIGHT: (Face.TOP, np.s_[self.size[X] - 1 - depth, ::-1]),
            Face.TOP: (Face.BACK, np.s_[::-1, depth]),
            Face.BOTTOM: (Face.FRONT, np.s_[:, self.size[Y] - 1 - depth]),
        }[face]

    def __bottom_face_and_slice(self, face: Face, depth: int) -> tuple[Face, slice]:
        return {
            Face.FRONT: (Face.BOTTOM, np.s_[::-1, depth]),
            Face.BACK: (Face.BOTTOM, np.s_[:, self.size[Z] - 1 - depth]),
            Face.LEFT: (Face.BOTTOM, np.s_[depth, :]),
            Face.RIGHT: (Face.BOTTOM, np.s_[self.size[X] - 1 - depth, ::-1]),
            Face.TOP: (Face.FRONT, np.s_[::-1, depth]),
            Face.BOTTOM: (Face.BACK, np.s_[:, self.size[Y] - 1 - depth]),
        }[face]

    def __left_face_and_slice(self, face: Face, depth: int) -> tuple[Face, slice]:
        return {
            Face.FRONT: (Face.LEFT, np.s_[self.size[Z] - 1 - depth, ::-1]),
            Face.BACK: (Face.RIGHT, np.s_[self.size[Z] - 1 - depth, ::-1]),
            Face.LEFT: (Face.BACK, np.s_[self.size[X] - 1 - depth, ::-1]),
            Face.RIGHT: (Face.FRONT, np.s_[self.size[X] - 1 - depth, ::-1]),
            Face.TOP: (Face.LEFT, np.s_[::-1, depth]),
            Face.BOTTOM: (Face.LEFT, np.s_[:, self.size[Y] - 1 - depth]),
        }[face]

    def __right_face_and_slice(self, face: Face, depth: int) -> tuple[Face, slice]:
        return {
            Face.FRONT: (Face.RIGHT, np.s_[depth, :]),
            Face.BACK: (Face.LEFT, np.s_[depth, :]),
            Face.LEFT: (Face.FRONT, np.s_[depth, :]),
            Face.RIGHT: (Face.BACK, np.s_[depth, :]),
            Face.TOP: (Face.RIGHT, np.s_[::-1, depth]),
            Face.BOTTOM: (Face.RIGHT, np.s_[:, self.size[Y] - 1 - depth]),
        }[face]

    def __rotate_clockwise(self, face: Face, depth: int) -> None:
        if depth == 0:
            self.faces[face] = np.rot90(self.faces[face], k=1)

        up_face, up_slice = self.__up_face_and_slice(face, depth)
        bottom_face, bottom_slice = self.__bottom_face_and_slice(face, depth)
        left_face, left_slice = self.__left_face_and_slice(face, depth)
        right_face, right_slice = self.__right_face_and_slice(face, depth)

        up_color = self.faces[up_face][up_slice].copy()
        self.faces[up_face][up_slice] = self.faces[left_face][left_slice]
        self.faces[left_face][left_slice] = self.faces[bottom_face][bottom_slice]
        self.faces[bottom_face][bottom_slice] = self.faces[right_face][right_slice]
        self.faces[right_face][right_slice] = up_color

    def __rotate_half_turn(self, face: Face, depth: int) -> None:
        if depth == 0:
            self.faces[face] = self.faces[face][::-1, ::-1]

        up_face, up_slice = self.__up_face_and_slice(face, depth)
        bottom_face, bottom_slice = self.__bottom_face_and_slice(face, depth)
        left_face, left_slice = self.__left_face_and_slice(face, depth)
        right_face, right_slice = self.__right_face_and_slice(face, depth)

        up_color = self.faces[up_face][up_slice].copy()
        self.faces[up_face][up_slice] = self.faces[bottom_face][bottom_slice]
        self.faces[bottom_face][bottom_slice] = up_color
        left_color = self.faces[left_face][left_slice].copy()
        self.faces[left_face][left_slice] = self.faces[right_face][right_slice]
        self.faces[right_face][right_slice] = left_color

    def can_rotate(self, face: Face, direction: Direction, depth: int) -> bool:
        if depth < 0:
            return False

        if face in (Face.FRONT, Face.BACK) and depth >= self.size[Z] - 1:
            return False
        if face in (Face.LEFT, Face.RIGHT) and depth >= self.size[X] - 1:
            return False
        if face in (Face.TOP, Face.BOTTOM) and depth >= self.size[Y] - 1:
            return False

        if direction == Direction.HALF_TURN:
            return True

        if face in (Face.FRONT, Face.BACK):
            return self.size[X] == self.size[Y]
        if face in (Face.LEFT, Face.RIGHT):
            return self.size[Y] == self.size[Z]
        if face in (Face.TOP, Face.BOTTOM):
            return self.size[X] == self.size[Z]

    def rotate(self, face: Face, direction: Direction, depth: int) -> None:
        assert self.can_rotate(face, direction, depth), "Cannot rotate face"

        if direction == Direction.HALF_TURN:
            self.__rotate_half_turn(face, depth)
            return

        for _ in range(direction.value):
            self.__rotate_clockwise(face, depth)

    def shuffle(self, n: int = 100, replace_origin: bool = False) -> list[str]:
        faces = list(Face)
        directions = Direction.not_none()
        depths = list(range(max(self.size)))

        moves_str: list[str] = []
        last_face: Face | None = None
        last_depth: int | None = None

        for _ in range(n):
            face = random.choice(faces)
            direction = random.choice(directions)
            depth = random.choice(depths)

            while not self.can_rotate(face, direction, depth) or (
                face == last_face and depth == last_depth
            ):
                face = random.choice(faces)
                direction = random.choice(directions)
                depth = random.choice(depths)

            last_face = face
            last_depth = depth
            self.rotate(face, direction, depth)

            moves_str.append(RubiksCube.move_to_str(face, direction, depth))

        if replace_origin:
            self.replace_origin()

        return moves_str

    def rotate_whole_cube(self, face: Face, direction: Direction) -> None:
        self.rotate(face.opposite(), direction.opposite(), 0)
        for i in range(self.size[X] - 1):
            self.rotate(face, direction, i)

    def __origin(self) -> tuple[CornerPos, CornerOrientation]:
        for pos in Var.Corners.pos_range():
            idx, o = self.get_vars_from_corner_pos(pos)
            if idx == 0:
                return pos, o
        return self.get_vars_from_corner_pos(0)

    def replace_origin(self) -> None:
        Variable.cube_size = self.size[X]

        pos, o = self.__origin()
        if pos == 0 and o == 0:
            return

        x, y, z = Var.Corners.g(pos)

        if o == 0 and z != 0:
            self.rotate_whole_cube(Face.BOTTOM, Direction.HALF_TURN)
        elif o == 1:
            self.rotate_whole_cube(
                Face.BOTTOM,
                Direction.CLOCKWISE if x == 0 else Direction.COUNTERCLOCKWISE,
            )
        elif o == 2:
            self.rotate_whole_cube(
                Face.RIGHT,
                Direction.COUNTERCLOCKWISE if y == 0 else Direction.CLOCKWISE,
            )

        pos, o = self.__origin()
        assert o == 0, f"Origin not oriented correctly: {o}"

        x, y, z = Var.Corners.g(pos)

        assert z == 0, f"Origin not placed correctly: {pos}"

        match (x == 0, y == 0):
            case (False, False):
                self.rotate_whole_cube(Face.FRONT, Direction.HALF_TURN)
            case (True, False):
                self.rotate_whole_cube(Face.FRONT, Direction.CLOCKWISE)
            case (False, True):
                self.rotate_whole_cube(Face.FRONT, Direction.COUNTERCLOCKWISE)

        pos, o = self.__origin()
        assert pos == 0, f"Origin not placed correctly: {pos}"

    @staticmethod
    def reverse_moves(moves: list[str]) -> list[str]:
        reverse_moves = []
        for face, direction, depth in RubiksCube.parse_moves(moves)[::-1]:
            reverse_moves.append(
                RubiksCube.move_to_str(face, direction.opposite(), depth)
            )
        return reverse_moves

    @staticmethod
    def move_to_str(face: Face, direction: Direction, depth: int) -> str:
        return f"{face.to_str()}{direction.to_str()}" + "{" + str(depth) + "}"

    @staticmethod
    def parse_move(move: str) -> tuple[Face, Direction, int]:
        face = Face.from_str(move[0])
        move = move[1:]
        depth = re.search(r"{(\d+)}", move)

        if depth:
            depth = int(depth.group(1))
            move = move.replace("{" + str(depth) + "}", "")
        else:
            depth = 0

        return face, Direction.from_str(move), depth

    @staticmethod
    def parse_moves(moves: list[str]) -> list[tuple[Face, Direction, int]]:
        return [RubiksCube.parse_move(move) for move in moves]

    def apply_rotations(self, rotations: list[str]) -> None:
        for rotation in rotations:
            self.rotate(*self.parse_move(rotation))

    def __draw_face(
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

    def __draw(
        self,
        screen: pygame.surface.Surface,
        angle_x: float,
        angle_y: float,
        rotating_face: Face | None = None,
        rotating_depth: int = 0,
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

        def rotate_points_on_face(
            points: list[tuple[float, float, float]],
            face: Face | None,
            depth: int,
            angle: float,
            x: int,
            y: int,
            z: int,
        ) -> list[tuple[float, float, float]]:
            if face is None:
                return points

            if face == Face.LEFT and x == depth:
                return [rotate_x(point, angle) for point in points]
            if face == Face.RIGHT and x == self.size[X] - 1 - depth:
                return [rotate_x(point, angle) for point in points]
            if face == Face.TOP and y == depth:
                return [rotate_y(point, angle) for point in points]
            if face == Face.BOTTOM and y == self.size[Y] - 1 - depth:
                return [rotate_y(point, angle) for point in points]
            if face == Face.FRONT and z == depth:
                return [rotate_z(point, angle) for point in points]
            if face == Face.BACK and z == self.size[Z] - 1 - depth:
                return [rotate_z(point, angle) for point in points]

            return points

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

                    points = rotate_points_on_face(
                        points,
                        rotating_face,
                        rotating_depth,
                        rotating_angle,
                        cube_x,
                        cube_y,
                        cube_z,
                    )
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
            self.__draw_face(screen, coords, Color(color))

    def show(
        self,
        screen_size: tuple[int, int] = (600, 600),
        background_color: tuple[int, int, int] = (30, 30, 30),
    ) -> None:
        self.animate([], screen_size, background_color, speed=0)

    def animate(
        self,
        rotations: list[tuple[Face, Direction, int]],
        screen_size: tuple[int, int] = (600, 600),
        background_color: tuple[int, int, int] = (30, 30, 30),
        speed: float = 1,
        fps: int = 30,
        recording: bool = False,
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
        rotating_depth = 0
        rotating_angle = 0
        rotation_idx = 0
        angle_direction = 1
        target_angle = 0

        frames = []
        record_finished = False

        while running:
            screen.fill(background_color)

            if rotation_idx < len(rotations):
                face, direction, depth = rotations[rotation_idx]

                if rotating_face is None:
                    rotating_face = face
                    rotating_depth = depth
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

            else:
                record_finished = True

            if rotating_face is not None:
                rotating_angle += speed * angle_direction

                if abs(rotating_angle) >= abs(target_angle):
                    self.rotate(face, direction, rotating_depth)  # type: ignore
                    rotating_face = None
                    rotation_idx += 1

            self.__draw(
                screen, angle_x, angle_y, rotating_face, rotating_depth, rotating_angle
            )

            if recording and not record_finished:
                frame = pygame.surfarray.array3d(pygame.display.get_surface())
                frame = np.transpose(frame, (1, 0, 2))
                frames.append(frame)

            pygame.display.flip()
            clock.tick(fps)

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

        if recording and frames:
            imageio.mimsave("output.gif", frames, fps=fps)

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
    print(moves)
    cube.animate(RubiksCube.parse_moves(RubiksCube.reverse_moves(moves)), speed=1)


if __name__ == "__main__":
    main()
