import json
import re

from pyGameWorld import ToolPicker
from src.utils import draw_data


def extract_and_draw(data_file_name, json_dir, group_name):
    
    # Extract data points for the specified level from the SQL file
    data, level_names = extract_data_from_sql(data_file_name)
    
    level_name_set = sorted(set(level_names))
    
    for level_name in level_name_set:
        # Load the basic trial data
        with open(json_dir + level_name + '.json', 'r') as f:
            btr = json.load(f)

        tp = ToolPicker(btr)  # Initialize ToolPicker with loaded data
        data_points, group_colors, group_sizes = extract_data_for_level(data, level_name)

        # Draw on the toolPicker (tp)
        draw_on_tool_picker(tp, data_points, json_dir, level_name, group_colors, group_sizes, group_name)

    # Save the file (if needed, implement save logic here)

def extract_data_from_sql(data_file_name):
    data_points = []
    group_colors = []
    group_sizes = []
    level_names = []
    
    with open(data_file_name, 'r') as f:
        sql_content = f.read()
        
        # Use regex to find all INSERT statements for the Results table
        insert_pattern = r"INSERT INTO `Results` .*?;"
        inserts = re.findall(insert_pattern, sql_content, re.DOTALL)

        for insert in inserts:
            # Extract values from the INSERT statement
            values_pattern = r"\((.*?)\)"
            values = re.findall(values_pattern, insert)

            for value in values[1:]:
                fields = value.split(',')
                trial = fields[2].strip().strip("'")  # Get the trial name
                obj = fields[4].strip().strip("'")
                pos_x = float(fields[5].strip())  # Get pos_x
                pos_y = float(fields[6].strip())  # Get pos_y
                level_names.append(trial)
                data_points.append((trial, pos_x, pos_y, obj, fields[10].strip()))

    return data_points, level_names

def extract_data_for_level(data_points, level_name):
    filtered_data_points = []
    group_colors = []
    group_sizes = []
    
    for trial, pos_x, pos_y, obj, success in data_points:
        if trial == level_name:
            filtered_data_points.append((pos_x, pos_y))
            color = (0, 0, 0)
            if success == '1':
                if obj == 'obj1':
                    color = (255, 0, 255)
                elif obj == 'obj2':
                    color = (255, 255, 0)
                elif obj == 'obj3':
                    color = (0, 255, 255)
            if success == '0':
                if obj == 'obj1':
                    color = (128, 0, 128)  # Lighter version of obj1 color
                elif obj == 'obj2':
                    color = (128, 128, 0)  # Lighter version of obj2 color
                elif obj == 'obj3':
                    color = (0, 128, 128)  # Lighter version of obj3 color
            size = 5
            group_colors.append(color)
            group_sizes.append(size)

    return filtered_data_points, group_colors, group_sizes

def draw_on_tool_picker(tp, img_poss, json_dir, tnm, group_colors, group_sizes, group_name):
    # Placeholder function to represent drawing logic
    draw_data(tp, img_poss, 'human_data/img/'+tnm+'_'+group_name+'.png', group_colors, group_sizes)

# Example usage (to be replaced with actual user input)
extract_and_draw('human_data/Results_1219.sql', 'environment/Trials/Strategy_Selected/', 'group_2')




