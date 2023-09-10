import subprocess
import openai 
import time
import os 
from moviepy.editor import VideoFileClip,AudioFileClip, concatenate_audioclips
from datetime import timedelta
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
import imgkit  # <-- Add this line to req.txt 
import platform   # <-- Add this line to req.txt 
from pytube import YouTube
import math
import re
import moviepy.editor as mp
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
from datetime import datetime, timedelta
import json
import threading
import queue
import shutil


model="gpt-3.5-turbo-32k"
openai.api_key = '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'






def generate_combined_vtt(vtt_content):
    openai.api_key = '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'

    # Construct the message prompt for the model
    messages = [
        {
            "role": "system",
            "content": 
            '''
            You are a skilled assistant with a dual-task. Your primary responsibility is to translate and your secondary is to reformat and enrich the content.

            Translation Task:
            Translate the provided English VTT captions into Egyptian Arabic ammiya (بالعامية المصرية). Ensure the translation is fluent, clear, and accurate while preserving the timing cues and formatting.

            Reformatting & Enrichment Task:
            1. Divide the provided VTT content into four equal segments.
            2. Generate a description in "العاميه المصريه" for each segment in the VTT format. This description should be at least three sentences.
            3. Ensure that the ending timestamp of a segment becomes the starting timestamp for the description. After the description, the following segment should begin.
            4. Ensure all content, both original and generated descriptions, are in "العاميه المصريه".

            Format:
            You should provide the translated captions in WebVTT format with the corresponding Egyptian Arabic translation. Ensure to match the original timing cues and formatting. The final output should merge original VTT content with generated descriptions.

            Your ultimate goal is to produce a VTT file that's been translated to Egyptian Arabic and is interspersed with descriptive segments.
            '''
        },
        {"role": "user", "content": vtt_content}
    ]

    # Send the message to OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        combined_vtt = response.choices[0].message['content']
        return combined_vtt

    except Exception as e:
        print(f"Error: {e}")
        return ""


def save_html_to_img(slide_data_list):
    # Configurations for imgkit
    wkhtmltoimage_path = r'wkhtmltox\wkhtmltoimage.exe''
    
    config = imgkit.config(wkhtmltoimage=wkhtmltoimage_path)

    for index, slide_data in enumerate(slide_data_list):
        slide_title = slide_data["Slide"]["Title"]
        points = slide_data["Slide"]["Points"]
        formatted_points = "".join([f"<li>{point}</li>" for point in points])

        # Enhanced styling for the slide in HTML format
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Slide {index + 1}</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    padding: 60px;
                    text-align: center;
                    background: linear-gradient(to bottom, #F7F8FC, #D7DAE5);
                    color: #333;
                    font-size: 18px;
                    line-height: 1.6;
                }}
                div.slide-container {{
                    background-color: #fff;
                    padding: 60px 40px;
                    border-radius: 15px;
                    box-shadow: 0px 0px 30px rgba(0, 0, 0, 0.1);
                    max-width: 900px;
                    margin: 0 auto;
                    position: relative;
                }}
                h1 {{
                    font-size: 36px;
                    border-bottom: 2px solid #333;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                    font-weight: bold;
                    text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                }}
                p, ul {{
                    text-align: justify;
                    margin: 0 auto;
                    max-width: 750px;
                }}
                ul {{
                    list-style-type: disc;
                    padding-left: 25px;
                }}
                li {{
                    margin-bottom: 15px;
                }}
                div.footer {{
                    position: absolute;
                    bottom: 20px;
                    right: 30px;
                    font-size: 14px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="slide-container">
                <h1>{slide_title}</h1>
                <ul>
                    {formatted_points}
                </ul>
            </div>
        </body>
        </html>
        """

        # Save the slide as an image
        filename = f'slidesImages/slide_{index + 1}.png'
        imgkit.from_string(html_content, filename, config=config)
        
    print("Slides saved as images!")

def get_gpt_vtt(vvt, start_timestamp="00:05:00.000"):
    messages = [
                  {
              "role": "system",
              "content": '''
              You are a content creator tasked with creating descriptive metadata for video content. Your specialty is generating WebVTT (Web Video Text Tracks) files that serve as an overlay description for videos.

              Your task is to:
              1. Create a WebVTT file that provides a descriptive overlay for the provided text content, beginning from the timestamp {start_timestamp}. 

              Return:
              - A JSON object that includes the WebVTT-formatted description. The object should contain a single key called "VTT_File" that stores the WebVTT content.
              '''
          }
          ,
        {
            "role": "user",
            "content": f"Text: {vvt}; Start Time: {start_timestamp}"
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        functions=[
            {
                "name": "createVTTFile_with_starting_timestep_ara",
                "description": "Generate a WebVTT file  starting from a specific time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "VTT_File": {
                            "type": "string",
                            "description": "WebVTT content , starting from the specified time."
                        }
                    },
                    "required": ["VTT_File"]
                }
            }
        ],
        function_call={
            "name": "createVTTFile_with_starting_timestep_ara",
        }
    )

    reply_content = response.choices[0].message
    response_options = reply_content.to_dict()['function_call']['arguments']
    tokens = response.usage.total_tokens
    output = {'response': response_options, 'tokens': tokens}
    
    return output


def generate_slides_and_vtt(index, topic, output_list):
    task_prompt = '''
    Task Prompt:
    Context:
    As a content creator, your job is to create one slide on a specific topic. the slide should have a title and 3-4 points that illustrate the concept well.

    Special Goals:
    1. Develop informative slide on a specific topic with a title and 3-4 points for the slide in english.

    Format:
    Present the information in a JSON format with the following key-value pairs:
    1. "Slide": An object contains "Title" in english and "Points" keys. "Title" is the title of the slide, and "Points" is an array of 3-4 points that illustrate the concept well in english.
    '''

    functions = [
        {
            "name": "get_slide",
            "description": "Generates slide for a YouTube video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Topic": {
                        "type": "string",
                        "description": "The topic for which the slide and VTT file should be created. in english"
                    },
                    "Slide": {
                            "type": "object",
                            "properties": {
                                "Title": {
                                    "type": "string",
                                    "description": "The title of the slide.in english"
                                },
                                "Points": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "description": "A point that illustrates the slide's concept.in english"
                                    }
                                }
                            },
                            "description": "An object for the slide continus its information."

                        },
                    },
                    
                },
                "required": ["Topic", "Slide"],
            },
        
    ]

    # This line is based on your hypothetical API construct
    completion = openai.ChatCompletion.create(
        model="gpt-4",  # This is hypothetical; substitute with an actual available model
        messages=[
            {"role": "system", "content": task_prompt},
            {"role": "user", "content": topic}
        ],
        functions=functions,
        function_call={"name": "get_slide"}
    )

    # Process the hypothetical response
    reply_content = completion.choices[0].message
    response_options = reply_content.to_dict()['function_call']['arguments']
    tokens = completion.usage.total_tokens
    output = {'response': response_options, 'tokens': tokens}
    output_list[index] = json.loads(output.get('response'),strict=False)

    return output




def generate_slide_description_vtt(slide_content):
    # Setup the API key
    openai.api_key = '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'

    messages = [
        {
            "role": "system",
            "content":
            '''
            Your task is to produce a descriptive summary for the summarized PowerPoint slide provided in the input.

            Objective:
            Generate a description for the provided slide content.

            Format:
            It is imperative that you provide the description specifically in WebVTT (VTT) format without cue identifiers. The language used should be العاميه المصريه. Ensure the description adheres strictly to the VTT format.
            '''
        },
        {"role": "user", "content": slide_content},
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,

            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    except Exception as e:
        print(f"Error: {e}")
        return ""

    # Taking the returned VTT formatted description
    vtt_format_output = response.choices[0].message['content']
    return vtt_format_output

def extract_audio(video_path):
    """
    Extract audio from video using FFmpeg.

    :param video_path: path to the input video file
    """
    # FFmpeg command to extract audio
    print(video_path)
    timestamp = int(time.time())
   
    output_path =  f'output_{timestamp}.mp3'

    cmd = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", output_path]

    subprocess.run(cmd)
    print(' the runing paths is',cmd)

    return output_path

def time_to_seconds(time_str):
        h, m, s = map(float, time_str.split(':'))
        return timedelta(hours=h, minutes=m, seconds=s).total_seconds()




def whisperModel(file):
    openai.api_key = "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"

    # Convert the OGG audio to MP3 using ogg2mp3() function
    with open(file, "rb") as audio_file:
        # Call the OpenAI API to transcribe the audio using Whisper API
        whisper_response = openai.Audio.transcribe(
            file=audio_file,
            model="whisper-1",
            language="EN",
            temperature=0.1,
            response_format="vtt"
        )
        text = r'{}'.format(whisper_response)

        return text



def replace_audio_in_video_option_one(video_path, voice_path, output_directory):
    """
    Replace the audio of a video with a provided voiceover audio. If the voiceover audio is shorter 
    than the video, it pads the audio with silence to match the video's duration. The function saves 
    the output video with a unique name based on the current timestamp and returns that filename.

    Parameters:
    - video_path (str): Path to the input video file.
    - voice_path (str): Path to the voiceover audio file.
    - output_directory (str): Directory where the resulting video will be saved.

    Returns:
    - str: file path  of the generated output video file.
    """
    # Load video and voiceover
    print('video path')
    video_clip = VideoFileClip(video_path)
    voiceover_audio = AudioFileClip(voice_path)

    # If voiceover_audio is shorter than video_clip, add silence to the audio
    if voiceover_audio.duration < video_clip.duration:
        silence_duration = video_clip.duration - voiceover_audio.duration
        
        # Create a silent audio clip. The np.zeros() function generates the silence.
        silence = AudioArrayClip(np.zeros((int(silence_duration * voiceover_audio.fps), 2)), fps=voiceover_audio.fps)
        
        # Concatenate original audio with silence
        voiceover_audio = concatenate_audioclips([voiceover_audio, silence])

    # Ensure voiceover_audio duration matches video_clip duration
    voiceover_audio = voiceover_audio.subclip(0, video_clip.duration)

    # Set the video's audio to the voiceover
    output_filename = f"output_{int(time.time())}.mp4" # Using the current timestamp to create a unique filename
    output_path = os.path.join(output_directory, output_filename)
    final_video = video_clip.set_audio(voiceover_audio)


    # Export the resulting video
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')


    return output_path


def replace_audio_in_video(video_path, voice_path, output_directory):

    """
    Replace the audio of a video with a provided voiceover audio. If the voiceover audio is shorter 
    than the video, it pads the audio with silence to match the video's duration. The function saves 
    the output video with a unique name based on the current timestamp and returns that filename.

    Parameters:
    - video_path (str): Path to the input video file.
    - voice_path (str): Path to the voiceover audio file.
    - output_directory (str): Directory where the resulting video will be saved.

    Returns:
    - str: file path  of the generated output video file.
    """
    # Load video and voiceover
    video_clip = VideoFileClip(video_path)
    voiceover_audio = AudioFileClip(voice_path)

    # If voiceover_audio is shorter than video_clip, add silence to the audio
    if voiceover_audio.duration < video_clip.duration:
        silence_duration = video_clip.duration - voiceover_audio.duration
        
        # Create a silent audio clip. The np.zeros() function generates the silence.
        silence = AudioArrayClip(np.zeros((int(silence_duration * voiceover_audio.fps), 2)), fps=voiceover_audio.fps)
        
        # Concatenate original audio with silence
        voiceover_audio = concatenate_audioclips([voiceover_audio, silence])

    # Ensure voiceover_audio duration matches video_clip duration
    voiceover_audio = voiceover_audio.subclip(0, video_clip.duration)

    # Set the video's audio to the voiceover
    output_filename = f"output_{int(time.time())}.mp4" # Using the current timestamp to create a unique filename
    output_path = os.path.join(output_directory, output_filename)
    final_video = video_clip.set_audio(voiceover_audio)

    # Export the resulting video
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

    return output_filename









def get_gpt_discriptions(vvt):

    openai.api_key = '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'
    messages=[
    {
        "role": "system",
        "content": """
            Consider the VTT content provided below. Your task is to:

            1. Divide the content into exactly four roughly equal sections based on the number of captions.
            2. After each section, provide a detailed and satisfying description in VTT format that summarizes and provides insight into the content. This means there will be a total of four descriptions added.
            3. Adjust the timing of the subsequent original captions after adding each description. Ensure that there is no overlap or gap in timing between the description and the next caption.
            4. Ensure that the final output is in العاميه المصريه.
            5. Ensure the `starttime` and `durations` lists both have a length of exactly 4, representing the four added descriptive segments.

            For clarity, see this example:

            Input VTT:
            00:00:01.000 --> 00:00:03.000
            A lion prowls the savannah.
            00:00:03.001 --> 00:00:05.000
            Zebras graze nearby, unaware of the danger.

            Expected Output:
            finalVtt:
            00:00:01.000 --> 00:00:03.000
            A lion prowls the savannah.
            00:00:03.001 --> 00:00:05.000
            Zebras graze nearby, unaware of the danger.
            00:00:05.001 --> 00:00:09.000
            In this intense scene, a lion, the king of the savannah, is on the hunt. Oblivious zebras, with their striking patterns, continue grazing, underscoring the harsh reality of nature's food chain.
            00:00:09.001 --> 00:00:11.000
            [Next Original Caption After Adjustment]

            starttime: [00:00:05, ...], (length = 4)
            durations: [3.99, ...],   (length = 4)

            Apply this process to the VTT content below and output the modified VTT with descriptions.
        """
    },
    {"role": "user", "content": vvt},
]

    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=messages,
      temperature=0,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
       functions=[
        {
              "name": "descriptions",
              "description": "This endpoint provides enhanced VTT content by dividing the input into sections and appending descriptive segments after each section.",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "finalVtt": {
                          "type": "string",
                          "description": "The modified VTT content which includes both the original segments and the newly appended descriptive segments. in العاميه المصريه "
                      },
                     "starttime": {
                      "type": "array",
                      "items": {
                          "type": "string",
                          "description": "A timestamp marking the start of an appended descriptive segment. This array collects these timestamps for each descriptive insertion."
                      }
                  },
                  "durations": {
                      "type": "array",
                      "items": {
                          "type": "string",
                          "description": "Represents the duration for which an appended descriptive segment plays. This array compiles these durations for every descriptive insertion."
                      }
                  }

                  },
                  "required": ["finalVtt", "starttime", "durations"]
              }
          }

            ],
        function_call={"name": "descriptions"},
        )

    reply_content = response.choices[0].message
    response_options = reply_content.to_dict()['function_call']['arguments']
    tokens=response.usage.total_tokens
    output={'response':response_options,'tokens':tokens}
    return output










def get_image_files_in_folder(folder_path):
    # List all files in the given folder
    all_files = os.listdir(folder_path)
    
    # Filter out non-image files
    image_files = ["slidesImages\\"+f for f in all_files if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff'))]
    
    return image_files


def insert_images_into_video(video_path, image_paths, insert_times, image_durations):
    if len(image_paths) != len(insert_times) or len(image_paths) != len(image_durations):
        raise ValueError("The lengths of image_paths, insert_times, and image_durations must all be the same.")

    video = mp.VideoFileClip(video_path).without_audio()
    clips = []

    last_end_original = 0
    last_end_new = 0
    print(image_paths)
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



def split_and_adjust_video_moviepy(input_path, durations, required_length):
    # Load video
    clip = VideoFileClip(input_path)

    # Ensure the length of durations and required_length are the same
    if len(durations) != len(required_length):
        raise ValueError("The durations and required_length lists must have the same length.")

    adjusted_clips = []
    start_time = 0

    for i in range(len(durations)):
        # Calculate the end time for this segment
        end_time = start_time + durations[i]

        # Extract the segment from the main video
        segment = clip.subclip(start_time, end_time)

        # Adjust the segment speed to match the required length
        factor = durations[i] / required_length[i]
        adjusted_segment = segment.fx(vfx.speedx, factor)

        adjusted_clips.append(adjusted_segment)

        # Update the start time for the next segment
        start_time = end_time

    # Combine the adjusted clips
    final_clip = concatenate_videoclips(adjusted_clips, method="compose")

    # Save the final output
    final_clip.write_videofile(r"static\videos\final_output.mp4")




def download_video(url):
    yt = YouTube(url)
    # Get the highest resolution stream (note: this is without audio)
    video_stream = yt.streams.get_highest_resolution()
    
    save_path = 'videos'
    custom_filename = "my_custom_video_name.mp4"  # Replace this with your custom name
    
    # Construct the complete path of the video
    video_file_path = f"{save_path}/{custom_filename}"
    
    # Download the video
    video_stream.download(output_path=save_path, filename=custom_filename)
    
    print('The video path is', video_file_path)
    print(f"Downloaded: {yt.title}")

    return video_file_path, custom_filename







def timestamp_to_seconds(timestamp):
    hours, minutes, seconds_milliseconds = timestamp.split(':')
    seconds, milliseconds = seconds_milliseconds.split('.')
    
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    
    # Convert milliseconds to seconds and add it to total_seconds
    total_seconds += int(milliseconds) / 1000.0
    
    return total_seconds





def dividingVideos(video_path, video_name):
    """
    Divide a video into 30-minute (or less) subclips.

    Parameters:
    video_path (str): The file path of the source video.
    video_name (str): The name of the source video.

    Returns:
    list: A list of paths to the created subclips or the original video. 
          Returns an empty list if an error occurs.
    """
    # Initialize an empty list to store paths of the subclips
    subclip_paths = []

    # Get the video duration using ffprobe
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                                 'format=duration', '-of',
                                 'default=noprint_wrappers=1:nokey=1', video_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        video_duration = float(result.stdout)
    except Exception as e:
        print(f"Error in getting video duration: {e}")
        return []

    print(f"Video duration: {video_duration} seconds")

    # If video duration is less than or equal to 30 minutes, return the original path
    if video_duration <= 1800:
        return [video_path]

    # Otherwise, split the video into subclips
    else:
        num_subclips = math.ceil(video_duration / 1800)
        for i in range(num_subclips):
            start = i * 1800
            end = (i + 1) * 1800 if (i + 1) * 1800 < video_duration else video_duration
            subclip_path = f"videos/subclip_{i}_{video_name}"

            try:
                subprocess.run(['ffmpeg', '-ss', str(start), '-to', str(end),
                                '-i', video_path, '-c', 'copy', subclip_path])
                subclip_paths.append(subclip_path)
            except Exception as e:
                print(f"Error in creating subclip: {e}")

        return subclip_paths





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

def split_video( video_path,time_steps):
    # Load the video
    video = VideoFileClip(video_path)
    
    # To store new video paths
    new_video_paths = []
    
    for i, (start_time, end_time) in enumerate(time_steps):
        # Convert time in 'HH:MM:SS:FFF' to seconds
        start_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(":"))))
        end_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(":"))))
        
        # Cut the video
        new_video = video.subclip(start_seconds, end_seconds)
        
        # Write the result to a file
        new_video_path = f"static/videos/{video_path.split('/')[-1].split('.')[0]}_part_{i + 1}.mp4"
        new_video.write_videofile(new_video_path)
        
        # Append the new path to the list
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






def get_last_timestep_from_vtt(vtt_str):
    # Regular expression to match WebVTT time steps
    pattern = r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})"
    
    # Use re.findall to find all matching time steps in the VTT string
    matches = re.findall(pattern, vtt_str)
    
    # Get the end time of the last match
    if matches:
        last_end_time = matches[-1][1]
        return last_end_time
    else:
        return "No time steps found"


def get_vtt_duration(vtt_str):
    # Regular expression to match WebVTT time steps
    pattern = r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})"
    
    # Use re.findall to find all matching time steps in the VTT string
    matches = re.findall(pattern, vtt_str)
    
    # Initialize total duration
    total_duration = timedelta()
    
    # Define a function to convert HH:MM:SS.SSS to timedelta
    def str_to_timedelta(timestr):
        hours, minutes, seconds = map(float, timestr.split(":"))
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
    # Calculate total duration
    for start, end in matches:
        start_time = str_to_timedelta(start)
        end_time = str_to_timedelta(end)
        duration = end_time - start_time
        total_duration += duration
    
    # Convert total duration to HH:MM:SS.SS format
    total_seconds = total_duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    formatted_duration = f"{hours:02}:{minutes:02}:{seconds:05.2f}"
    
    return formatted_duration


def merge_vtt_files(vtt1_str, vtt2_str):
    # Remove 'WEBVTT' and any whitespace from the start of the second VTT string
    vtt2_str_modified = vtt2_str.replace("WEBVTT", "").lstrip()
    vtt2_str_modified=vtt2_str_modified.replace('\\n','\n')
    # Concatenate the two VTT strings
    # Remove the numbers and associated newline characters
    vtt2_str_modified = re.sub(r'^\d+\n', '', vtt2_str_modified, flags=re.MULTILINE)
    merged_vtt = vtt1_str.rstrip() + "\n\n" + vtt2_str_modified
    
    return merged_vtt

def createVtt(summary,orignalvtt,lasttimestep):

    gptresponde=get_gpt_vtt(summary,lasttimestep)

    response=gptresponde.get('response')

    disc=json.loads(response,strict=False)

    disc=disc.get('VTT_File')

    updatedvtt=merge_vtt_files( orignalvtt,disc)

    return (updatedvtt,get_vtt_duration(disc))



def split_with_ffmpeg(video_path):
    """
    Splits a video into segments using ffmpeg.
    
    Parameters:
    video_path (str): The path to the video file to be split.

    Returns:
    list: A list of paths to the video segments created.
    """
    
    # Command to get the total duration of the video using ffprobe
    cmd = f'ffprobe -i "{video_path}" -show_entries format=duration -v quiet -of csv="p=0"'
    
    # Debugging: Print the command to the console
    print(f"Running command: {cmd}")
    
    # Execute the command and get the total duration
    try:
        total_duration = float(subprocess.check_output(cmd, shell=True).decode('utf-8').strip())
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return None

    # Set the duration of each segment (in seconds)
    step = 500  # 15 minutes in seconds
    
    # Calculate the number of segments
    n_segments = math.ceil(total_duration / step)
    
    # Initialize list to store paths to the created video segments
    videoPaths = []

    # Loop to create video segments
    for i in range(n_segments):
        start = i * step
        segment_path = f'videoParts\\segment_{i+1}.mp4'
        
        # ffmpeg command to create a segment
        cmd = (
            f'ffmpeg -y -ss {start} -i "{video_path}" -c copy -t {step} '
            f'{segment_path}'
        )
        
        # Add the path of the new segment to the list
        videoPaths.append(segment_path)

        # Debugging: Print the command to the console
        print(f"Running command: {cmd}")

        # Execute the ffmpeg command
        subprocess.run(cmd, shell=True)

    # Return the list of created video segments
    return videoPaths


def get_video_duration(video_path):
    """
    Retrieves the total duration of a video using ffprobe.
    
    Parameters:
    video_path (str): The path to the video file whose duration needs to be found.
    
    Returns:
    float: The total duration of the video in seconds. Returns None if an error occurs.
    """
    
    # Construct the ffprobe command to get the video duration
    cmd = f'ffprobe -i "{video_path}" -show_entries format=duration -v quiet -of csv="p=0"'
    
    # Debugging: Print the ffprobe command to the console
    print(f"Running ffprobe command: {cmd}")

    # Try to execute the command and parse the total duration
    try:
        # Run the command and collect the output
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        
        # Convert the output to a floating-point number
        total_duration = float(output)
        
        return total_duration

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to get the video duration: {e}")
        return None
    



def split_webvtt(content):
    """
    Splits a WebVTT content into segments of 15 minutes.
    
    Parameters:
    content (str): The content of the WebVTT file as a string.
    
    Returns:
    list: A list of strings where each string is the content of a 15-minute segment.
    """
    
    segments = []  # List to hold each 15-min segment
    current_segment = []  # Temporary list to accumulate lines for the current segment
    current_time = timedelta()  # Time counter for the current segment
    time_limit = timedelta(minutes=5)  # 15-min time limit for each segment
    
    # Split the entire content by double newline to separate each entry
    lines = content.split('\n\n')
    
    i = 0  # Line counter
    while i < len(lines):
        line = lines[i]
        
        # If we find a timing line (which contains the '-->' symbol)
        if '-->' in line:
            # Extract timing information
            time_line = line.split('\n')[0]
            start_time_str, end_time_str = time_line.split('-->')
            
            # Convert the end time to a timedelta object
            end_time = datetime.strptime(end_time_str.strip(), "%H:%M:%S.%f").time()
            end_time = timedelta(hours=end_time.hour, minutes=end_time.minute,
                                 seconds=end_time.second, microseconds=end_time.microsecond)
            
            # Check if adding this subtitle entry would exceed the 15-min limit
            if end_time - current_time >= time_limit:
                segments.append('\n\n'.join(current_segment))
                current_segment = []
                current_time = end_time
            
            # Add the current entry to the current_segment list
            current_segment.append(lines[i])
        
        i += 1  # Move to the next line
    
    # Add any remaining content to the last segment
    if current_segment:
        segments.append('\n\n'.join(current_segment))
    
    return segments





# def generateExcelReport(all_content, translation, seo, timeline, englishsummary, arabicsummary, summaryvtt,excel_file_path):
#     # Create a dictionary to hold your data
#     data = {
#         "Original Text": [all_content],
#         "Translation": [translation],
#         "SEO": [seo],
#         "Timeline": [timeline],
#         "English Summary": [englishsummary],
#         "Arabic Summary": [arabicsummary],
#         "Generated Vtt For The Summary":[summaryvtt]
#     }

#     # Convert the dictionary to a Pandas DataFrame
#     df = pd.DataFrame(data)

#     # Save the DataFrame to an Excel file
#     df.to_excel(excel_file_path, index=False)


def extract_text_from_vtt(vtt):
    """
    Extracts subtitle text and their corresponding timestamps from a VTT formatted string.

    Args:
    - vtt (str): The VTT formatted string containing subtitles.

    Returns:
    - tuple: A tuple containing two lists:
             1. List of timestamps.
             2. List of extracted subtitles.
    """
    
    # Remove any leading or trailing whitespace
    cleaned_vtt = vtt.strip()
    
    # Split the VTT content into distinct subtitle blocks using double newline as a delimiter
    blocks = cleaned_vtt.split("\n\n")
    
    # Extract the timestamps and text.
    timestamps = [block.splitlines()[0] for block in blocks if block.splitlines()[0] != "WEBVTT"]

    # Remove semicolons from the text and then join the cleaned subtitles with semicolons.
    subtitle_texts = ['\n'.join(block.splitlines()[1:]).replace(';', '') for block in blocks if block.splitlines()[0] != "WEBVTT"]
    
    return timestamps,subtitle_texts





def get_gpt_translation(input):
   
   
    messages = [
    {
        "role": "system",
        "content": '''
        You are a skilled translator with a specific task. You will be given a list of English sentences. For each and every sentence in the list:
        1) Translate it into Egyptian Arabic ammiya (بالعامية المصرية).
        2) Ensure the translated list is of the SAME length as the input list.
        3) Provide a summary of the entire content in Egyptian Arabic ammiya (بالعامية المصرية).

        Example: 
        Input: ["Hello", "How are you?", "Thank you"]
        Expected Output: ["مرحبا", "ازيك؟", "شكراً"]

        Note: If you can't translate a sentence, return the original sentence in the output list. Do NOT skip any sentence.
        '''
    },
    {
        "role": "user",
        "content": input
    }
]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    functions = [
    {
        "name": "translation_list_with_example_v5",  # Incremented version for clarity
        "description": "Translate EACH item from the given list of English content to Egyptian Arabic ammiya (بالعامية المصرية). The returned translation list MUST have the same length as the input list, ensuring a direct 1:1 correspondence. If a translation for a specific item isn't possible, the original item should be returned in its place.",
        "parameters": {
            "type": "object",
            "properties": {
                "translation": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of translated items. Each translated item should directly correspond to the respective item in the input list. If translation isn't possible for an item, the original should be retained."
                },

                "summary": {
                    "type":"string",
                    "description": "A concise summary of the entire content in Egyptian Arabic ammiya (بالعامية المصرية)."
                }
            },
            "required": ["translation", "summary"]
        }
    }
]

,

        function_call={"name": "translation_list_with_example_v5"},
    )

    reply_content = response.choices[0].message
    response_options = reply_content.to_dict()['function_call']['arguments']
    tokens = response.usage
    output = {'response': response_options, 'tokens': tokens}
    return output





def threaded_translation(index, sublist, output_dict):
    sublist_string = json.dumps(sublist)
    translation_output = get_gpt_translation(sublist_string)

    response = translation_output['response']
    response_data = json.loads(response, strict=False)

    translated_data = response_data.get('translation', [])
    summary_data = response_data.get('summary', "")

    # Append empty strings to translated_data until its length matches sublist
    while len(translated_data) < len(sublist):
        translated_data.append("")

    output_dict[index] = (translated_data, summary_data)


def translate_vtt_content(original_text):
    timeline, text = extract_text_from_vtt(original_text)
    sublist_size = 50
    sublists = [text[i:i + sublist_size] for i in range(0, len(text), sublist_size)]
    timeline_list=[timeline[i:i + sublist_size] for i in range(0, len(timeline), sublist_size)]
    # Capture the corresponding last timestamp for each sublist
    sublist_last_timestamps = [timeline[min(i + sublist_size - 1, len(timeline) - 1)] for i in range(0, len(timeline), sublist_size)]
    sublist_last_timestamps_endtimes = [i.split('-->')[1].strip() for i in sublist_last_timestamps]
    # Use a dictionary to hold the results with the index as the key
    output_dict = {}

    threads = []
    for i, sublist in enumerate(sublists):
        # Pass the index to the threaded function
        t = threading.Thread(target=threaded_translation, args=(i, sublist, output_dict))
        t.start()
        threads.append(t)

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Sort results based on the index and reassemble them
    all_translations = []
    summary = []
    for i in range(len(sublists)):
        translated_data, summary_data = output_dict[i]
        summary.append(summary_data)
        combined_list = [f"{x}\n{y}" for x, y in zip(timeline_list[i], translated_data)]
        all_translations.append(combined_list)

    return (all_translations, summary, sublist_last_timestamps_endtimes)


def threaded_create_vtt(index, summary, original_vtt, last_timestep, results):
    translation_model_output, durations_output = createVtt(summary, original_vtt, last_timestep)
    results[index] = (translation_model_output, durations_output)

def process_translation_list(translation_list, arabicsummary_list, lasttimes):
    # A list to hold thread instances
    threads = []
    
    # A dictionary to store the results from threads
    results = {}
    
    # Start threads to process the VTT creation
    for i in range(len(translation_list)):
        t = threading.Thread(target=threaded_create_vtt, args=(i, arabicsummary_list[i], '\n\n'.join(translation_list[i]), lasttimes[i], results))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Extract results from the results dictionary and maintain original order
    durations = []
    all_vtt_content = ''
    for i in range(len(translation_list)):
        translation_model_output, durations_output = results[i]
        durations.append(durations_output)
        all_vtt_content += '\n\n' + translation_model_output

    return all_vtt_content, durations



def generate_slides(topics):
    threads = []
    output_list = [None] * len(topics)
    
    for i, topic in enumerate(topics):
        t = threading.Thread(target=generate_slides_and_vtt, args=(i, topic, output_list))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return output_list


def delete_all_elements_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
