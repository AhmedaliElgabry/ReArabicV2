import os
import moviepy.editor as mp



def get_image_files_in_folder(folder_path):
    # List all files in the given folder
    all_files = os.listdir(folder_path)
    
    # Filter out non-image files
    image_files = ["slidesImages\\"+f for f in all_files if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff'))]
    
    return image_files


from PIL import Image
import moviepy.editor as mp
import numpy as np

def insert_images_into_video(video_path, image_paths, insert_times, image_durations):
    if len(image_paths) != len(insert_times) or len(image_paths) != len(image_durations):
        raise ValueError("The lengths of image_paths, insert_times, and image_durations must all be the same.")

    video = mp.VideoFileClip(video_path).without_audio()
    clips = []

    last_end_original = 0
    last_end_new = 0

    for i, img_path in enumerate(image_paths):
        if insert_times[i] < last_end_new:
            raise ValueError(f"The insert time {insert_times[i]} is before the end time {last_end_new} of the previous clip.")

        video_duration_needed = insert_times[i] - last_end_new
        video_part_before_image = video.subclip(last_end_original, last_end_original + video_duration_needed)

        clips.append(video_part_before_image)

        img = Image.open(img_path)
        img_w, img_h = img.size

        blank_image = Image.new("RGB", (video.size[0], video.size[1]), "black")

        x_offset = (video.size[0] - img_w) // 2
        y_offset = (video.size[1] - img_h) // 2

        blank_image.paste(img, (x_offset, y_offset))

        img_clip = mp.ImageClip(np.array(blank_image)).set_duration(image_durations[i])
        clips.append(img_clip)

        last_end_original += video_duration_needed
        last_end_new += video_duration_needed + image_durations[i]

    clips.append(video.subclip(last_end_original))

    final_video = mp.concatenate_videoclips(clips).without_audio()

    # Use a faster codec and reduce quality
    final_video.write_videofile('output_video.mp4', codec='mpeg4', bitrate="1000k")



# Paths for the original and final output videos
video_path =  r"C:\Users\fatma taha\Desktop\programmingProjects\FlaskProject\videos\Three Minute Thesis (3MT) 2011 Winner - Matthew Thompson.mp4"
insert_times = [5.901, 15.001, 25.841, 35.841]
image_durations =  [4.099, 4.999, 4.84, 4.159]
image_files=  ['slidesImages\\slide_1.png', 'slidesImages\\slide_2.png', 'slidesImages\\slide_3.png', 'slidesImages\\slide_4.png']

 
# image_files=get_image_files_in_folder(r"slidesImages")


#insert_images_into_video(video_path, image_files, insert_times, image_durations)




from moviepy.editor import VideoFileClip
import math
import os


video_path=r"videos\Good Vibes Music 🌻 Top 100 Chill Out Songs Playlist  New Tiktok Songs With Lyrics.mp4"
videoName=r"Good Vibes Music 🌻 Top 100 Chill Out Songs Playlist  New Tiktok Songs With Lyrics.mp4"
subclip_paths=[]

# with VideoFileClip(video_path) as clip:
#         video_duration = clip.duration  # duration in seconds

# print(f"Video duration: {video_duration} seconds")

# if video_duration > 1800:  # 30 minutes = 1800 seconds
#     num_subclips = math.ceil(video_duration / 1800)  # calculate the number of 30-min subclips needed
#     for i in range(num_subclips):
#         start = i * 1800
#         end = (i + 1) * 1800 if (i + 1) * 1800 < video_duration else video_duration
#         subclip_path = f"videos/subclip_{i}_{videoName}"
        
#         with VideoFileClip(video_path).subclip(start, end) as subclip:
#             subclip.write_videofile(subclip_path)

#         subclip_paths.append(subclip_path)  # Append the path to the list

# import subprocess

# result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 
#                                      'format=duration', '-of', 
#                                      'default=noprint_wrappers=1:nokey=1', video_path], 
#                                     stdout=subprocess.PIPE,
#                                     stderr=subprocess.STDOUT)
# video_duration = float(result.stdout)

# print(f"Video duration: {video_duration} seconds")

# if video_duration > 1800:  # 30 minutes = 1800 seconds
#     num_subclips = math.ceil(video_duration / 1800)
#     for i in range(num_subclips):
#         start = i * 1800
#         end = (i + 1) * 1800 if (i + 1) * 1800 < video_duration else video_duration
#         subclip_path = f"videos/subclip_{i}_{videoName}"


#         subprocess.run(['ffmpeg', '-ss', str(start), '-to', str(end), 
#                         '-i', video_path, '-c', 'copy', subclip_path])
#         subclip_paths.append(subclip_path)




# import re

# def extract_time_steps(vtt_list):
#     time_steps = []
#     for entry in vtt_list:
#         match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', entry)
#         if match:
#             time_steps.append(match.group(0))
#     return time_steps



# def split_into_sublists(lst, size):
#     sublists = []
#     for i in range(0, len(lst), size):
#         sublist = lst[i:i + size]
#         sublists.append(sublist)
#     return sublists



# from moviepy.editor import VideoFileClip

# def split_video( video_path,time_steps):
#     # Load the video
#     video = VideoFileClip(video_path)
    
#     # To store new video paths
#     new_video_paths = []
    
#     for i, (start_time, end_time) in enumerate(time_steps):
#         # Convert time in 'HH:MM:SS:FFF' to seconds
#         start_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(":"))))
#         end_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(":"))))
        
#         # Cut the video
#         new_video = video.subclip(start_seconds, end_seconds)
        
#         # Write the result to a file
#         new_video_path = f"{video_path.split('.')[0]}_part_{i + 1}.mp4"
#         new_video.write_videofile(new_video_path)
        
#         # Append the new path to the list
#         new_video_paths.append(new_video_path)

#     return new_video_paths




# def prepareVideoSegments(videopath,segmantslist,num_lines):
#     time_steps = extract_time_steps(segmantslist)
#     sublists=split_into_sublists(time_steps,num_lines)
#     videocuts=[]
#     for list in sublists:
#         # Start and end times from the whole list
#         start_time = list[0].split(" --> ")[0]
#         end_time = list[-1].split(" --> ")[1]

#         steps=(start_time,end_time)
#         videocuts.append(steps)

#     videosSegments=split_video(videopath,videocuts)

#     return videosSegments


import re
from moviepy.editor import VideoFileClip

def extract_time_steps(vtt_list):
    """
    Extracts the time steps from a list of VTT formatted strings.
    
    Parameters:
        vtt_list (list): List of VTT formatted strings.
    
    Returns:
        list: List of extracted time steps in 'HH:MM:SS:FFF --> HH:MM:SS:FFF' format.
    """
    # Initialize an empty list to hold time steps
    time_steps = []
    
    # Loop through each VTT entry to find the time steps
    for entry in vtt_list:
        match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', entry)
        if match:
            time_steps.append(match.group(0))
            
    return time_steps

def split_into_sublists(lst, size):
    """
    Splits a list into multiple sub-lists each containing 'size' elements.
    
    Parameters:
        lst (list): Original list to be split.
        size (int): Size of each sub-list.
        
    Returns:
        list: List of sub-lists.
    """
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def split_video(video_path, time_steps):
    """
    Splits a video file into multiple segments based on given time steps.
    
    Parameters:
        video_path (str): File path of the video to be split.
        time_steps (list): List of tuples containing start and end times.
        
    Returns:
        list: List of file paths for the newly created video segments.
    """
    try:
        # Load the video from the given file path, without audio
        video = VideoFileClip(video_path).without_audio()
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
    # Initialize list to store paths of new video segments
    new_video_paths = []

    # Loop through the list of time steps and split the video accordingly
    for i, (start_time, end_time) in enumerate(time_steps):
        # Convert time strings to seconds
        start_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(":"))))
        end_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(":"))))
        
        # Cut the video segment, without audio
        new_video = video.subclip(start_seconds, end_seconds).without_audio()
        
        # Define the file path for the new segment
        new_video_path = f"{video_path.rsplit('.', 1)[0]}_part_{i + 1}.mp4"
        
        # Save the new video segment to disk, using a faster codec and lower bitrate
        new_video.write_videofile(new_video_path, codec='mpeg4', bitrate="1000k")
        
        # Store the new video path
        new_video_paths.append(new_video_path)

    return new_video_paths

def prepare_video_segments(video_path, segments_list, num_lines):
    """
    Prepares segmented videos based on a list of text segments and a line count limit per segment.
    
    Parameters:
        video_path (str): File path of the original video.
        segments_list (list): List of VTT formatted strings.
        num_lines (int): Maximum line count per text segment.
        
    Returns:
        list: List of file paths for the newly created video segments.
    """
    # Extract time steps from the VTT segments
    time_steps = extract_time_steps(segments_list)
    
    # Split the list of time steps into sub-lists based on the line count limit
    sublists = split_into_sublists(time_steps, num_lines)
    
    # Prepare start and end times for each video segment
    video_cuts = [(lst[0].split(" --> ")[0], lst[-1].split(" --> ")[1]) for lst in sublists]
    
    # Call function to actually split the video
    return split_video(video_path, video_cuts)



# Your sample VTT content list
vtt_list = [
    '00:00:00.000 --> 00:00:02.560\r\nفي حد تاني بيكره لما الناس تقولك انت بتعيش مرة واحدة بس؟',
    '00:00:00.000 --> 00:00:02.560\r\nفي حد تاني بيكره لما الناس تقولك انت بتعيش مرة واحدة بس؟',
    '00:00:03.080 --> 00:00:05.900\r\nشكراً على المعلومة الجديدة دي وبصراحة',
    '00:00:05.901 --> 00:00:10.000\r\nفي القسم الأول، يعبر الشخص عن استيائه من الأقوال الشائعة',
    '00:00:10.001 --> 00:00:13.040\r\nالحمد لله لأني مش عايز أعيشها تاني، أو لما يقولوا اعمل حاجة كل يوم تخوفك',
    '00:00:13.041 --> 00:00:15.000\r\nكل اللي أنا بعمله بيخوفني. ده اسمه قلق',
    '00:00:15.001 --> 00:00:20.000\r\nفي القسم الثاني، يتحدث الشخص عن مشاعر القلق التي يعيشها يومياً.',
    '00:00:20.001 --> 00:00:23.840\r\nأنا بعيش حياتي على الحافة، حافة الانهيار التام',
    '00:00:23.841 --> 00:00:25.840\r\nبس دي حافة في النهاية، أو الأسوأ لما يقولوا انت قادر على التعامل مع الأمور، انت قوي',
    '00:00:25.841 --> 00:00:30.000\r\nفي القسم الثالث، يصف الشخص حياته كأنها على حافة الانهيار التام',
    '00:00:30.001 --> 00:00:33.840\r\nالسبب الوحيد إني قوي هو إن الجسم بيستخدم 47 عضلة لما بتبكي',
    '00:00:33.841 --> 00:00:35.840\r\nعشان كده مش بتشوفني في الجيم. شكراً',
    '00:00:35.841 --> 00:00:40.000\r\nفي القسم الرابع والأخير، يقدم الشخص تفسيراً ساخراً لقوته'
]
video_path = r"C:\Users\fatma taha\Desktop\programmingProjects\FlaskProject\output_video.mp4"
num_lines = 6
videosPaths=prepare_video_segments(video_path, vtt_list, num_lines)
videosPaths='||'.join(videosPaths)

print(videosPaths.split('||'))





# import openai
# def get_gpt_discriptions(vvt):

#     openai.api_key = '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'
#    # Prepare the system and user messages
#     messages = [
#         {
#         '''
#        Task Prompt:
#         Context:
#         As a content creator, your job is to enhance a given VTT file . Divide the VTT content into four sections and provide a summary for each section in VTT format.

#         Special Goals:
#         1. Divide the given VTT content into four equal parts.
#         2. Generate a summary for each of the four sections.
#         3. Merge these summaries back into the original VTT file, ensuring the timing aligns appropriately.

#         Format:
#         1. "VTT_File": the modified VTT file content, which includes both the original captions and the newly-added summaries statring by summary.

#         '''
#         },
#         {"role": "user", "content": vvt},
#     ]

#     # Generate the API request for Chat Completion
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=messages,
#         temperature=0,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0,
#         functions=[
#             {
#                 "name": "descriptions_v2",
#                 "description": "This function enhances WebVTT captions by dividing the content into four segments and adding descriptive summaries to each. The summaries are integrated into the original WebVTT timeline.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "VTT_File": {
#                             "type": "string",
#                             "description": "The modified WebVTT file containing both the original content and the added summaries."
#                         }
#                     },
#                     "required": ["finalVtt"]
#                 }
#             }
#         ],
#         function_call={"name": "descriptions_v2"}
#     )

#     # Extract the relevant information from the response
#     # reply_content = response.choices[0].message
#     # response_options = reply_content.to_dict()['function_call']['arguments']
#     # tokens = response.usage.total_tokens

#     # Compile the output
#     # output = {'response': response_options, 'tokens': tokens}

#     return response
import openai
import json

def get_gpt_descriptions(vvt):
    # Set API key (ideally use environment variables for security)
    openai.api_key = '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'

    # Define the system and user messages
    messages = [
        {
            "role": "system",
            "content": '''
                Task Prompt:
                Context:
                As a content creator, your job is to enhance a given VTT file. 
                Divide the VTT content into four sections and provide a summary for each section in VTT format.

                Special Goals:
                1. Divide the given VTT content into four equal parts.
                2. Generate a summary for each of the four sections.
                3. Merge these summaries back into the original VTT file, ensuring the timing aligns appropriately.

                Format:
                1. "VTT_File": the modified VTT file content, which includes both the original captions and the newly-added summaries starting by summary.
            '''
        },
        {"role": "user", "content": vvt},
    ]

    # Make the API request for Chat Completion
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            functions=[
                {
                    "name": "descriptions_v2",
                    "description": "This function enhances WebVTT captions by dividing the content into four segments and adding descriptive summaries to each. The summaries are integrated into the original WebVTT timeline.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "VTT_File": {
                                "type": "string",
                                "description": "The modified WebVTT file containing both the original content and the added summaries."
                            }
                        },
                        "required": ["VTT_File"]
                    }
                }
            ],
            function_call={"name": "descriptions_v2"})

        # Extract the relevant information from the response
        reply_content = response.choices[0].message
        response_options = reply_content.to_dict()['function_call']['arguments']
        tokens = response.usage.total_tokens

        # Compile the output
        output = {'response': response_options, 'tokens': tokens}
        return output


    except TypeError as e:
        print(f"Error: {e}")
        return None





import subprocess
import os

def split_video(video_path, time_steps):
    """
    Splits a video file into multiple segments based on given time steps.
    
    Parameters:
        video_path (str): File path of the video to be split.
        time_steps (list): List of tuples containing start and end times.
        
    Returns:
        list: List of file paths for the newly created video segments.
    """
    try:
        # Initialize list to store paths of new video segments
        new_video_paths = []
        
        # Loop through the list of time steps and split the video accordingly
        for i, (start_time, end_time) in enumerate(time_steps):
            # Create new video file path
            filename, file_extension = os.path.splitext(video_path)
            new_video_path = f"{filename}_part_{i + 1}{file_extension}"
            
            # Run ffmpeg command
            command = [
                'ffmpeg',
                '-ss', start_time,    # Start time
                '-to', end_time,      # End time
                '-i', video_path,     # Input file path
                '-c', 'copy',         # Copy streams without re-encoding
                new_video_path        # Output file path
            ]
            subprocess.run(command)
            
            new_video_paths.append(new_video_path)
        
        return new_video_paths

    except Exception as e:
        print(f"An error occurred: {e}")
        return []



