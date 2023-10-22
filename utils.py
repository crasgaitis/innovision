import time
import math
import pandas as pd
import tobii_research as tr

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

    columns_to_drop = ['device_time_stamp'] # 'Unnamed: 0', 
    
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