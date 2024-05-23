import os
import random


def generate_map(threshold=0.5):
    input_file = "worlds/door.wbt"
    output_file = "worlds/generated_map.wbt"

    with open(input_file, 'r') as f:
        lines = f.readlines()

    adjusted_lines = []
    adjust_next_line = False
    for line in lines:
        if adjust_next_line:
            parts = line.split()
            # Adjust the translation values
            x_value = random.uniform(-threshold, threshold)
            y_value = random.uniform(-threshold, threshold)
            if 0 <= float(parts[1]) + x_value <= 2 and 0 <= float(parts[2]) + y_value <= 2:
                parts[1] = str(float(parts[1]) + x_value)
                parts[2] = str(float(parts[2]) + y_value)
            adjusted_lines.append(" ".join(parts))
            adjust_next_line = False
        else:
            adjusted_lines.append(line)
            if line.strip().startswith("Solid"):
                adjust_next_line = True

    with open(output_file, 'w') as f:
        f.writelines(adjusted_lines)
