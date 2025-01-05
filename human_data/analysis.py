import json
import re

from pyGameWorld import ToolPicker
from src.utils import draw_data

import pandas as pd

def import_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    data_list = data.values.tolist()
    return data_list



def extract_and_draw(data_file_name, json_dir, group_name):
    
    # Extract data points for the specified level from the SQL file
    data, level_names, _ = extract_data_from_sql(data_file_name)
    
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

def extract_and_build_csv(data_file_name, json_dir, group_name):
    data, level_names, participants = extract_data_from_sql(data_file_name)
    with open('human_data/csv/'+group_name+'.csv', 'w') as f:
        f.write('participant,level,obj,x,y,success\n')
        for i, datum in enumerate(data):
            f.write(participants[i] + ',' + str(datum[0]) + ',' + str(datum[1]) + ',' + str(datum[2]) + ',' + str(datum[3]) + ',' + str(datum[4]) + '\n')

def extract_data_from_sql(data_file_name, include_group=False):
    data_points = []
    group_colors = []
    group_sizes = []
    level_names = []
    participants = []
    group_names = []
    
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
                success = fields[10].strip()
                time = float(fields[7].strip())/1000
                level_names.append(trial)
                data_points.append((trial, pos_x, pos_y, obj, success, time))
                participants.append(fields[0].strip().strip("'"))

    return data_points, level_names, participants

def extract_data_for_level(data_points, level_name):
    filtered_data_points = []
    group_colors = []
    group_sizes = []
    
    for trial, pos_x, pos_y, obj, success, time in data_points:
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
                    color = (160, 0, 128)  # Lighter version of obj1 color
                elif obj == 'obj2':
                    color = (160, 160, 0)  # Lighter version of obj2 color
                elif obj == 'obj3':
                    color = (0, 128, 160)  # Lighter version of obj3 color
            size = 5
            group_colors.append(color)
            group_sizes.append(size)

    return filtered_data_points, group_colors, group_sizes

def extract_data_csv_each_participant(csv_data):
    group_colors = {}
    group_sizes = {}
    
    data_dict = {}
    
    for participant,level,x,y,obj,success,group in csv_data:
        data_dict.setdefault(participant+'-'+group+'-'+level, []).append((x, y))
        color = (0, 0, 0)
        if success == 1:
            if obj == 'obj1':
                color = (255, 0, 255)
            elif obj == 'obj2':
                color = (255, 255, 0)
            elif obj == 'obj3':
                color = (0, 255, 255)
        if success == 0:
            if obj == 'obj1':
                color = (160, 0, 128)  # Lighter version of obj1 color
            elif obj == 'obj2':
                color = (160, 160, 0)  # Lighter version of obj2 color
            elif obj == 'obj3':
                color = (0, 128, 160)  # Lighter version of obj3 color
        size = 1 + 2*len(data_dict[participant+'-'+group+'-'+level])
        group_colors.setdefault(participant+'-'+group+'-'+level, []).append(color)
        group_sizes.setdefault(participant+'-'+group+'-'+level, []).append(size)

    return data_dict, group_colors, group_sizes

def draw_data_each_participant(data_dict, group_colors, group_sizes):
    for img_name in data_dict.keys():
        print(img_name)
        level = img_name.split('-')[-1]
        with open('environment/Trials/Strategy_Selected/'+level+'.json', 'r') as f:
            btr = json.load(f)
            tp = ToolPicker(btr)
        draw_on_tool_picker(tp, data_dict[img_name], 'environment/Trials/Strategy_Selected/', level, group_colors[img_name], group_sizes[img_name], 'participants', img_name)

def draw_on_tool_picker(tp, img_poss, json_dir, tnm, group_colors, group_sizes, group_name, img_name = None):
    img_name = tnm if img_name is None else img_name
    # Placeholder function to represent drawing logic
    import os

    # Create a new directory with the group name
    group_name_dir = 'human_data/img/' + group_name
    if not os.path.exists(group_name_dir):
        os.makedirs(group_name_dir)

    draw_data(tp, img_poss, group_name_dir+'/'+img_name+'.png', group_colors, group_sizes)
    # if "Comp" in tnm:
    #     draw_data(tp, img_poss, 'human_data/img/'+img_name+'.png', group_colors, group_sizes)

    
from scipy.stats import chisquare
from scipy.stats import chi2_contingency

def calculate_chi_square_for_group(trial_name, group_data):
    observed = []
    expected = []

    for group_name in ['group_1', 'group_2']:
        O_1 = sum(1 for participant, trial, _, _, _, success, group in group_data if trial == trial_name and success == 1 and group == group_name)
        level_count = set(participant for participant, trial, _, _, _, success, group in group_data if trial == trial_name and group == group_name)
        total = sum(1 for level in level_count)
        O_2 = total - O_1
        observed.append([O_1, O_2])
    
    print(observed)

    # Calculate expected values
    total_success = sum([observed[i][0] for i in range(len(observed))])
    total_trials = sum([observed[i][0]+observed[i][1] for i in range(len(observed))])
    
    for i in range(len(observed)):
        expected_success = (total_success / total_trials) * (observed[i][0]+observed[i][1])
        expected.append([expected_success, observed[i][0]+observed[i][1] - expected_success])
        # expected.append([expected_success, observed[i][-1] - expected_success])
    
    print(expected)

    # Chi-square calculation
    chi_square_0 = ((observed[0][0] - expected[0][0]) ** 2) / expected[0][0] + ((observed[0][1] - expected[0][1]) ** 2) / expected[0][1]
    chi_square_1 = ((observed[1][0] - expected[1][0]) ** 2) / expected[1][0] + ((observed[1][1] - expected[1][1]) ** 2) / expected[1][1]
    print(chi_square_0, chi_square_1)
    
    

    
    chi2, p_value, dof, expected = chi2_contingency(observed, correction=False)
    print(chi2, p_value, dof, expected)
    return chi2, p_value


from collections import defaultdict

def group_data_by_index(data_points):
    grouped_data = defaultdict(list)
    for trial, pos_x, pos_y, obj, success, _, group in data_points:
        grouped_data[group].append((trial, pos_x, pos_y, obj, success))
    return grouped_data

    
    

file_name = '1219_2'

# extract_and_draw('human_data/Results_'+file_name+'.sql', 'environment/Trials/Strategy_Selected/', file_name)

# extract_and_build_csv('human_data/Results_'+file_name+'.sql', 'environment/Trials/Strategy_Selected/', file_name)
group_data = import_data_from_csv('human_data/csv/data.csv')

data_dict, group_colors, group_sizes = extract_data_csv_each_participant(group_data)

draw_data_each_participant(data_dict, group_colors, group_sizes)

# calculate_chi_square_for_group('Comp_SoCloseLaunch', group_data)





