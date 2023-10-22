import time
import math
import pandas as pd
import tobii_research as tr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

global global_gaze_data

def get_tracker():
  all_eyetrackers = tr.find_all_eyetrackers()

  for tracker in all_eyetrackers:
    print("Model: " + tracker.model)
    print("Serial number: " + tracker.serial_number) 
    print(f"Can stream eye images: {tr.CAPABILITY_HAS_EYE_IMAGES in tracker.device_capabilities}")
    print(f"Can stream gaze data: {tr.CAPABILITY_HAS_GAZE_DATA in tracker.device_capabilities}")
    return tracker

def gaze_data_callback(gaze_data):
  global global_gaze_data
  global_gaze_data = gaze_data
  
def gaze_data(eyetracker, wait_time=5):
  global global_gaze_data

  # print("Getting data...")
  eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

  time.sleep(wait_time)

  eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)

  return global_gaze_data

def eye_img_data_callback(eye_data):
  global global_eye
  global_eye = eye_data
  
def eye_img_data(eyetracker, wait_time=0.5):
  global global_eye

  eyetracker.subscribe_to(tr.EYETRACKER_EYE_IMAGES, eye_img_data_callback, as_dictionary=True)

  time.sleep(wait_time)

  eyetracker.unsubscribe_from(tr.EYETRACKER_EYE_IMAGES, eye_img_data_callback)

  return global_eye

def combine_dicts_with_labels(dict_list):
    combined_dict = {}
    for i, dictionary in enumerate(dict_list, start=1):
        label = f"timestep_{i}"
        combined_dict[label] = dictionary

    return combined_dict

def build_dataset(tracker, label, add_on = False, df_orig = pd.DataFrame(), 
                  time_step_sec = 0.5, tot_time_min = 0.1):
    
    global global_gaze_data
    
    intervals = math.ceil((tot_time_min * 60) / time_step_sec)
    dict_list = []
    
    for _ in range(intervals):
        data = gaze_data(tracker, time_step_sec)
        dict_list.append(data)
    
    tot_dict = combine_dicts_with_labels(dict_list)
    df = pd.DataFrame(tot_dict).T
    df['type'] = label
        
    if add_on:
        df_new = pd.concat([df_orig, df])
        df_new = df_new.reset_index(drop=True)
        return df_new
    
    else:
        return df, dict_list
    
def process_dataframe(df):
    validity_columns = [col for col in df.columns if 'validity' in col]
    df = df[~(df[validity_columns] == 0).any(axis=1)]

    columns_to_drop = ['Unnamed: 0', 'device_time_stamp'] # 'Unnamed: 0', 
    
    df = df.drop(columns=columns_to_drop)
    df = df.drop(columns=validity_columns)

    return df

def get_gazepoints(df, side = "left"):
    
    col_name = side + "_gaze_point_on_display_area"

    coordinate_pattern = r'\((-?\d+\.\d+), (-?\d+\.\d+)\)'

    df['coordinates'] = df[col_name].str.findall(coordinate_pattern)
    
    x_col = 'x_' + side
    y_col = 'y_' + side
    df[[x_col, y_col]] = pd.DataFrame(df['coordinates'].apply(lambda x: x[0] if x else [None, None]).tolist(), index=df.index)

    df = df.drop(columns = ['coordinates', col_name])
    
    columns_to_drop = [col for col in df.columns if 'coordinate' in col and side in col]
    df = df.drop(columns=columns_to_drop)
    
    return df

def convert_columns_to_float(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
        
def calculate_grouped_statistics(group):
    output_df = pd.DataFrame()

    avg_left_pupil_diameter = group['left_pupil_diameter'].mean()
    avg_right_pupil_diameter = group['right_pupil_diameter'].mean()
    avg_x_left = group['x_left'].mean()
    avg_x_right = group['x_right'].mean()
    avg_y_left = group['y_left'].mean()
    avg_y_right = group['y_right'].mean()

    max_x_velocity = max(group['x_left'].diff().max(), group['x_right'].diff().max())
    max_y_velocity = max(group['y_left'].diff().max(), group['y_right'].diff().max())

    output_df = output_df.append({
        'avg_left_pupil_diameter': avg_left_pupil_diameter,
        'avg_right_pupil_diameter': avg_right_pupil_diameter,
        'avg_x_left': avg_x_left,
        'avg_x_right': avg_x_right,
        'avg_y_left': avg_y_left,
        'avg_y_right': avg_y_right,
        'max_x_velocity': max_x_velocity,
        'max_y_velocity': max_y_velocity
    }, ignore_index=True)

    return output_df

def plot_circles(radius1, radius2):
    fig, ax = plt.subplots()

    circle1 = Circle((3, 1), radius1, fill=True, color='#415A6A', linewidth=2)
    circle2 = Circle((10, 1), radius2, fill=True, color='#51416A', linewidth=2)

    ax.add_patch(circle1)
    ax.add_patch(circle2)

    ax.set_xlim(-2, 15)
    ax.set_ylim(-6, 6)

    ax.set_aspect('equal')

    ax.text(0, -4, f"Left diameter: {radius1 * 2}", fontsize=12, color='#415A6A')
    ax.text(0, -5, f"Right diameter: {radius2 * 2}", fontsize=12, color='#51416A')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    desired_width = 350
    fig_width, fig_height = fig.get_size_inches()
    dpi = int(desired_width / fig_width)
    
    plt.savefig("radius_eyes.png", dpi=dpi)
    
def plot_gazepoints(x_left, y_left, x_right, y_right):
    plt.scatter(float(x_left), float(y_left), color = "#415A6A", label = "Left", s=20)
    plt.scatter(float(x_right), float(y_right), color = "#51416A", label = "Right", s=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])

    plt.savefig("gaze_grid.png")