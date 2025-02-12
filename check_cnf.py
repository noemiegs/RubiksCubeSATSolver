from variables import Var
from variables_abc import Variable


def compare_files(file1_path, file2_path):
    with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
        for line_num, (line1, line2) in enumerate(list(zip(file1, file2))[1:], start=1):
            if line1 != line2:
                return line_num, line1, line2
    return None


file1_path = "rubiks_cube.cnf"
file2_path = "rubiks_cube_3_3_3.cnf"

result = compare_files(file1_path, file2_path)
if result:
    line_num, line1, line2 = result
    print(f"Files differ at line {line_num}:\n\nTrue: {line1}Fake: {line2}")
else:
    print("Files are identical")

Variable.t_max = 11
print(Var.from_int(1164))
print(Var.from_int(17935))
print(Var.from_int(36646))
