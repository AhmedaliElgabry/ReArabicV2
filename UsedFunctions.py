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
from html2image import Html2Image
import os
import subprocess
import concurrent.futures

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
    
    hti = Html2Image()
    imagePaths=[]
    for index, slide_data in enumerate(slide_data_list):
        slide_title = slide_data["Slide"]["Title"]
        points = slide_data["Slide"]["Points"]
        formatted_points = "".join([f"<li>{point}</li>" for point in points])

        html_content = f"""
            <!DOCTYPE html>
            <html>

            <head>
                <title>Slide {index + 1}</title>
                <style>
                  body {{
                    font-family: 'Arial', sans-serif;
                    background: linear-gradient(to bottom, #F7F8FC, #D7DAE5);
                    color: #333;
                    font-size: 18px;
                    line-height: 1.6;
                    height: 100vh; /* Full height of the viewport */
                    margin: 0; /* Removing default margin */
                    display: flex; /* Making body a flex container */
                    justify-content: center; /* Centering content horizontally */
                    align-items: center; /* Centering content vertically */
                }}

                div.slide-container{{
                    background-color: #fff;
                    padding: 100px 60px;
                    border-radius: 15px;
                    box-shadow: 0px 0px 30px rgba(0, 0, 0, 0.1);
                    max-width: 2000px;
                    min-width:1500px;
                    min-height:500px;
                }}

                  h1 {{
                        font-size: 36px;
                        border-bottom: 2px solid #333;
                        padding-bottom: 20px;
                        margin-bottom: 30px;
                        font-weight: bold;
                        text-align: center;

                        text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                    }}

                   
                    ul li{{
                                            font-size: 30px;
                                            margin-bottom:5px;

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
        
        hti.screenshot(html_str=html_content, save_as= f'slide_{index + 1}.png')
        imagePaths.append(f'slide_{index + 1}.png')
    print("Slides saved as images!")
    return imagePaths
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



import subprocess
import os

def replace_audio_in_video_ffmpeg(video_path, voice_path, output_directory):
    """
    Replace the audio of a video with a provided voiceover audio using FFmpeg. If the voiceover audio is shorter 
    than the video, it pads the audio with silence to match the video's duration. The function saves 
    the output video with a unique name based on the current timestamp and returns that filename.

    Parameters:
    - video_path (str): Path to the input video file.
    - voice_path (str): Path to the voiceover audio file.
    - output_directory (str): Directory where the resulting video will be saved.

    Returns:
    - str: File path of the generated output video file.
    """

    # Get the durations of the video and audio
    get_video_duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    get_audio_duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", voice_path]
    
    video_duration = float(subprocess.check_output(get_video_duration_cmd).decode('utf-8').strip())
    audio_duration = float(subprocess.check_output(get_audio_duration_cmd).decode('utf-8').strip())
    
    # If the voiceover audio is shorter than the video duration, pad it with silence.
    padded_voice_path = voice_path
    if audio_duration < video_duration:
        silence_duration = video_duration - audio_duration
        padded_voice_path = os.path.join(output_directory, "padded_audio.aac")
        
        # FFmpeg command to pad audio with silence
        pad_audio_cmd = [
            "ffmpeg", "-y", 
            "-i", voice_path,
            "-af", f"apad=pad_dur={silence_duration}", 
            "-c:a", "aac",
            padded_voice_path
        ]
        subprocess.run(pad_audio_cmd, check=True)
    
    # Combine video from video_path with audio from padded_voice_path
    output_filename = "final_output_voice.mp4"
    output_path = os.path.join(output_directory, output_filename)
        
    # FFmpeg command to replace audio in video
    combine_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", padded_voice_path,
        "-c:v", "copy",       # Copy video codec
        "-c:a", "aac",        # Encode audio to AAC
        "-strict", "experimental",
        "-map", "0:v:0",      # Map video stream from first input
        "-map", "1:a:0",      # Map audio stream from second input
        output_path
    ]
    subprocess.run(combine_cmd, check=True)
    
    # Remove padded audio file if it was created
    if padded_voice_path != voice_path:
        os.remove(padded_voice_path)

    # Re-encode the video for better compatibility
    reencoded_output_filename = "reencoded_output_voice.mp4"
    reencoded_output_path = os.path.join(output_directory, reencoded_output_filename)
    reencode_cmd = [
        "ffmpeg", "-y",
        "-i", output_path,
        "-c:v", "libx264",    # Encode video as H.264
        "-preset", "medium",  # Video quality/encoding speed tradeoff
        "-b:v", "2M",         # Video bitrate
        "-c:a", "aac",        # Encode audio as AAC
        "-b:a", "128k",       # Audio bitrate
        "-strict", "-2",
        reencoded_output_path
    ]
    subprocess.run(reencode_cmd, check=True)
    
    # Optionally, remove the initial combined video
    os.remove(output_path)

    return reencoded_output_path

def replace_audio_in_video_option_ffmpeg(video_path, voice_path, output_directory):
    """
    Replace the audio of a video with a provided voiceover audio using FFmpeg. If the voiceover audio is shorter 
    than the video, it pads the audio with silence to match the video's duration. The function saves 
    the output video with a unique name based on the current timestamp and returns that filename.

    Parameters:
    - video_path (str): Path to the input video file.
    - voice_path (str): Path to the voiceover audio file.
    - output_directory (str): Directory where the resulting video will be saved.

    Returns:
    - str: File path of the generated output video file.
    """
    
    # Get the durations of the video and audio
    get_video_duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    get_audio_duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", voice_path]
    
    video_duration = float(subprocess.check_output(get_video_duration_cmd).decode('utf-8').strip())
    audio_duration = float(subprocess.check_output(get_audio_duration_cmd).decode('utf-8').strip())
    
    # If the voiceover audio is shorter than the video duration, pad it with silence.
    padded_voice_path = voice_path
    if audio_duration < video_duration:
        silence_duration = video_duration - audio_duration
        padded_voice_path = os.path.join(output_directory, "padded_audio.aac")
        
        # FFmpeg command to pad audio with silence
        pad_audio_cmd = [
            "ffmpeg", "-y", 
            "-i", voice_path,
            "-af", f"apad=pad_dur={silence_duration}", 
            "-c:a", "aac",
            padded_voice_path
        ]
        subprocess.run(pad_audio_cmd, check=True)
    
    # Combine video from video_path with audio from padded_voice_path
    output_filename = "final_output_voice.mp4"
    output_path = os.path.join(output_directory, output_filename)
    
    # FFmpeg command to replace audio in video
    combine_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", padded_voice_path,
        "-c:v", "copy",       # Copy video codec
        "-c:a", "aac",        # Encode audio to AAC
        "-strict", "experimental",
        "-map", "0:v:0",      # Map video stream from first input
        "-map", "1:a:0",      # Map audio stream from second input
        output_path
    ]
    subprocess.run(combine_cmd, check=True)
    
    # Remove padded audio file if it was created
    if padded_voice_path != voice_path:
        os.remove(padded_voice_path)
    
    return output_path
     






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


def insert_images_into_video_ffmpeg(video_path, image_paths, insert_times, image_durations):
    if len(image_paths) != len(insert_times) or len(image_paths) != len(image_durations):
        raise ValueError("The lengths of image_paths, insert_times, and image_durations must all be the same.")
    
    # Query the main video for resolution and frame rate
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'csv=p=0', video_path]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    width, height, frame_rate = output.split(',')

    filter_complex = []
    inputs = []
    outputs = []
    index = 0

    last_end = 0
    total_image_duration = 0  # Total duration of images added

    for i, (img_path, insert_time, duration) in enumerate(zip(image_paths, insert_times, image_durations)):
        adjusted_insert_time = insert_time + total_image_duration
        if adjusted_insert_time < last_end:
            raise ValueError(f"The adjusted insert time {adjusted_insert_time} is before the end time {last_end} of the previous clip.")
        
        # Video segment input and filter
        inputs.extend(['-i', video_path])
        filter_complex.append(f"[{index}:v]trim=start={last_end}:end={adjusted_insert_time},setpts=PTS-STARTPTS[v{i*2}];")
        outputs.append(f"[v{i*2}]")
        index += 1

        # Image to video input and filter
        inputs.extend(['-loop', '1', '-i', img_path])
        filter_complex.append(f"[{index}:v]trim=start=0:end={duration},setpts=PTS-STARTPTS,scale={width}:{height},fps={frame_rate}[v{i*2+1}];")
        outputs.append(f"[v{i*2+1}]")
        index += 1

        last_end = adjusted_insert_time
        total_image_duration += duration  # Update the total duration

    # Last video segment input and filter
    inputs.extend(['-i', video_path])
    filter_complex.append(f"[{index}:v]trim=start={last_end},setpts=PTS-STARTPTS[v{len(image_paths)*2}];")
    outputs.append(f"[v{len(image_paths)*2}]")

    # Prepare final filter string
    filter_string = ''.join(filter_complex) + ''.join(outputs) + f"concat=n={len(outputs)}:v=1:a=0[outv]"

    # Final FFmpeg command
    cmd = ['ffmpeg', '-y']
    cmd.extend(inputs)
    cmd.extend(['-filter_complex', filter_string, '-map', '[outv]', '-c:v', 'mpeg4', '-b:v', '1000k', 'output_video.mp4'])
    subprocess.call(cmd)


from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx

def split_and_adjust_video_ffmpeg(input_path, durations, required_length, output_path):
    if len(durations) != len(required_length):
        raise ValueError("The durations and required_length lists must have the same length.")
    if any(d < 0 for d in durations) or any(r < 0 for r in required_length):
        raise ValueError("Durations and required lengths must be positive values.")
    
    video_duration = get_video_duration(input_path)
    if sum(durations) > video_duration:
        print(f"Warning: Specified durations exceed video length. Adjusting the last segment to match the video's end.")
        durations[-1] = video_duration - sum(durations[:-1])

    temp_segments = []
    start_time = 0

    for i, (dur, req_len) in enumerate(zip(durations, required_length)):
        end_time = start_time + dur
        factor = dur / req_len
        
        temp_segment_path = f"static\\videos\\temp_segment_{i}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-vf", f"setpts={factor}*PTS",
            "-an",  # No audio for simplicity. Remove this if you want to keep the audio.
            temp_segment_path
        ]
        subprocess.run(cmd, check=True)
        temp_segments.append(temp_segment_path)
        start_time = end_time

    # Concatenate all the segments
    concat_list_path = os.path.join(os.path.dirname(output_path), "concat_list.txt")
    with open(concat_list_path, 'w') as f:
        for seg in temp_segments:
            # Use only the filename, not the full path
            f.write(f"file '{os.path.basename(seg)}'\n")


    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        output_path
    ]
    subprocess.run(concat_cmd, check=True)

    # Cleanup temporary files
    for seg in temp_segments:
        os.remove(seg)
    os.remove(concat_list_path)






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
    # Using compiled regex pattern
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})')
    time_steps = [match.group(0) for entry in vtt_list if (match := pattern.search(entry))]
    return time_steps

def split_into_sublists(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]



def create_video_segment(idx, video_path, start_time, end_time, output_dir):
    segment_path = os.path.join(output_dir, f"segment_{idx}.mp4")
    
    ffmpeg_command = [
    "ffmpeg",
    "-y",
    "-i", video_path,
    "-ss", start_time,
    "-to", end_time,
    "-c:v", "libx264",
    "-preset", "ultrafast", 
    "-an",  # This flag removes the audio track
    "-crf", "32",

    segment_path
]

    try:
        subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return segment_path
    except subprocess.CalledProcessError as e:
        # Printing the FFmpeg error message for more clarity
        print(f"Error splitting segment {idx}: {e}\nFFmpeg Error Message: {e.stderr.decode('utf-8')}")
        return None

def prepare_video_segments(video_path, segments_list, num_lines):
    time_steps = extract_time_steps(segments_list)
    sublists = split_into_sublists(time_steps, num_lines)
    video_cuts = [(lst[0].split(" --> ")[0], lst[-1].split(" --> ")[1]) for lst in sublists]
    output_dir = r"static\videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parallel processing of video segments
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(create_video_segment, idx, video_path, start_time, end_time, output_dir) for idx, (start_time, end_time) in enumerate(video_cuts)]
        segment_paths = [future.result() for future in futures if future.result() is not None]
    
    return segment_paths



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
    Groups every five subtitle lines into one element in the resulting list.

    Args:
    - vtt (str): The VTT formatted string containing subtitles.

    Returns:
    - tuple: A tuple containing two lists:
             1. List of timestamps.
             2. List of extracted subtitles (grouped by five).
    """
    
    # Remove any leading or trailing whitespace
    cleaned_vtt = vtt.strip()
    
    # Split the VTT content into distinct subtitle blocks using double newline as a delimiter
    blocks = cleaned_vtt.split("\n\n")
    
    # Extract the timestamps and text.
    timestamps = [block.splitlines()[0] for block in blocks if block.splitlines()[0] != "WEBVTT"]

    # Remove semicolons from the text and then join the cleaned subtitles with semicolons.
    subtitle_texts = ['\n'.join(block.splitlines()[1:]).replace(';', '') for block in blocks if block.splitlines()[0] != "WEBVTT"]

    # Group every five subtitles into one element in the resulting list
    grouped_timestamps = []
    grouped_texts = []

    for i in range(0, len(timestamps), 5):
        # Get the start timestamp of the first subtitle and the end timestamp of the fifth (or last) subtitle in the group
        start_time = timestamps[i].split(" --> ")[0]
        end_time = timestamps[min(i+4, len(timestamps)-1)].split(" --> ")[1]
        grouped_timestamps.append(f"{start_time} --> {end_time}")
        
        # Group the texts
        grouped_text = '\n'.join(subtitle_texts[i:i+5])
        grouped_texts.append(grouped_text)
    
    return grouped_timestamps, grouped_texts



def get_gpt_translation(input):
   
   
    messages = [
    {
        "role": "system",
        "content": '''
        You are a skilled translator with a specific task. You will be given a list of English sentences. For each and every sentence in the list:
        1) Translate it into Egyptian Arabic ammiya (بالعامية المصرية).
        2) If there's an English term, name, or phrase within a sentence, it MUST remain in the EXACT same position as in the original English sentence. Do not rearrange any words around it. 
        3) Ensure the translated list is of the SAME length as the input list.
        4) Provide a summary of the entire content in Egyptian Arabic ammiya (بالعامية المصرية), also MAINTAINING the original order of the content.
        5) Important: Ensure that your response is in a valid and well-formed JSON format. Do not skip any sentences and make sure there are no missing or extra commas, brackets, or other characters that can break the JSON format.

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
    "name": "translation_list_with_example_v6",  # Incremented version for clarity
    "description": "Translate EACH item from the given list of English content to Egyptian Arabic ammiya (بالعامية المصرية). The returned translation list MUST have the same length as the input list, ensuring a direct 1:1 correspondence. If a translation for a specific item isn't possible, the original item should be returned in its place. **Important: Ensure that the response is in a valid and well-formed JSON format.**",
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
                "type": "string",
                "description": "A concise summary of the entire content in Egyptian Arabic ammiya (بالعامية المصرية)."
            }
        },
        "required": ["translation", "summary"]
    }
}

]

,

        function_call={"name": "translation_list_with_example_v6"},
    )

    reply_content = response.choices[0].message
    response_options = reply_content.to_dict()['function_call']['arguments']
    tokens = response.usage
    output = {'response': response_options, 'tokens': tokens}
    return output




import json

import json

def safe_loads(s, max_attempts=10):
    """Try to safely load a JSON string, attempting fixes for common errors."""
    for attempt in range(max_attempts):
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            print(f"Error at attempt {attempt + 1}: {e}")
            error_pos = e.pos

            if "Expecting value" in str(e):
                # Add a null value and try again
                s = s[:error_pos] + 'null' + s[error_pos:]
            elif "Extra data" in str(e):
                # truncate the string up to the error position
                s = s[:error_pos]
            elif "Expecting property name enclosed in double quotes" in str(e):
                s = s[:error_pos] + '"' + s[error_pos:]
            elif "Expecting ':' delimiter" in str(e):
                s = s[:error_pos] + ':' + s[error_pos:]
            elif "Expecting ',' delimiter" in str(e):
                s = s[:error_pos] + ',' + s[error_pos:]
            else:
                # For any other errors, just remove the problematic character and continue
                s = s[:error_pos] + s[error_pos + 1:]
        except Exception as e:
            print(f"Unhandled exception: {e}")
            break

    print("Max attempts reached or unhandled exception occurred. Returning an empty dictionary.")
    return {}


def threaded_translation(index, sublist, output_dict):
    translation_output = get_gpt_translation(str(sublist))

    response = translation_output['response']
    response=response.replace('\'',"\"")
    response_data = safe_loads(response)


    translated_data = response_data.get('translation', [])
    summary_data = response_data.get('summary', "")

    # Append empty strings to translated_data until its length matches sublist
    while len(translated_data) < len(sublist):
        translated_data.append("")

    output_dict[index] = (translated_data, summary_data)


def translate_vtt_content(original_text):
    timeline, text = extract_text_from_vtt(original_text)
    sublist_size = 10
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
    
        # Check if all elements in the list are empty strings
        all_empty = all(item == '' for item in output_dict[i][0])
        # Check if the length of the list is less than 10
        length_less_than_10 = len(output_dict[i][0]) < 10 and len(output_dict[i][0]) != len(sublist)

        if all_empty or length_less_than_10:
            translated_data, summary_data = sublists[i],' '

        else:
            # If the condition is not met, break out of the while loop
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
    for i in range(len(arabicsummary_list)):
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








def get_grouped_list_and_indices(arabicsummary_list,lasttimes,translted_list):
    """
    Given a list of strings, this function groups the list elements into subgroups with up to 10 elements each. 
    It also provides the index from the original list for the last element in each of the subgroups.
    
    Parameters:
    - arabicsummary_list: list of strings.
    
    Returns:
    - grouped_list: list of string groups where each group contains up to 10 elements from the original list.
    - last_element_indices: list of integers indicating the index of the last element in each group in the original list.
    """
    
    # Determine the size for each group, aiming for 10 elements or fewer per group
    group_size = 10 if len(arabicsummary_list) % 10 == 0 else len(arabicsummary_list) % 10 - 1

    # Adjust for even size. If the group size is an odd number, subtract 1 to make it even
    if group_size % 2 == 1:
        group_size -= 1

    # Create the grouped_list by joining elements of arabicsummary_list. 
    # Each group will contain 'group_size' number of elements, separated by space.
   # Grouping the arabicsummary_list by the specified group_size
    grouped_summaries = [' '.join(arabicsummary_list[i:i+group_size]) for i in range(0, len(arabicsummary_list), group_size)]

    # Picking the lasttimes for the last element of each merged group
    grouped_lasttimes = [lasttimes[min(i+group_size-1, len(arabicsummary_list)-1)] for i in range(0, len(arabicsummary_list), group_size)]

    translted_list_new = [translted_list[i:i+group_size] for i in range(0, len(translted_list), group_size)]    

    return grouped_summaries, grouped_lasttimes
