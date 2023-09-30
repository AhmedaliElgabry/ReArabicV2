import os
import moviepy.editor as mp
import openai

openai.api_key = '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'





content="""
WEBVTT

00:00:00.000 --> 00:00:03.600
Ring Ring

00:00:30.000 --> 00:00:31.600
You know it all

00:00:31.600 --> 00:00:34.800
In the end, I'll be left out

00:00:34.800 --> 00:00:36.400
I'm warning to myself

00:00:36.400 --> 00:00:38.000
You can't do that

00:00:38.000 --> 00:00:39.400
Every minute, every second, my heart

00:00:39.400 --> 00:00:40.800
It controls my heart

00:00:40.800 --> 00:00:42.400
Out of control

00:00:42.400 --> 00:00:45.600
The spot, spot, spotlight that watches over me

00:00:45.600 --> 00:00:48.800
The more it shines, the more I'm drawn into the darkness

00:00:48.800 --> 00:00:52.000
I can see the end, but I know it's not right

00:00:52.000 --> 00:00:54.000
I can't stop me, can't stop me

00:00:54.000 --> 00:00:55.200
No

00:00:55.200 --> 00:00:58.400
This red, red, red line placed in front of me

00:00:58.400 --> 00:01:01.800
You and I are already on opposite sides

00:01:01.800 --> 00:01:04.800
I want to feel the thrilling highlight

00:01:04.800 --> 00:01:06.800
I can't stop me, can't stop me

00:01:06.800 --> 00:01:07.800
No

00:01:07.800 --> 00:01:11.000
This red, red, red line placed in front of me

00:01:11.000 --> 00:01:14.200
This red, red, red line placed in front of me

00:01:14.200 --> 00:01:17.600
This red, red, red line placed in front of me

00:01:17.600 --> 00:01:19.600
I can't stop me, can't stop me

00:01:19.600 --> 00:01:22.800
Close your eyes, cool

00:01:22.800 --> 00:01:26.000
Just once, no rules

00:01:26.000 --> 00:01:30.000
Pretend you don't know, lights off tonight

00:01:30.000 --> 00:01:34.000
I can't hold it in, I'm losing myself

00:01:34.000 --> 00:01:35.600
The guy is turning back

00:01:35.600 --> 00:01:38.800
It's impossible, I'm getting more and more late

00:01:38.800 --> 00:01:42.000
It's too thrilling, I want to close my eyes

00:01:42.000 --> 00:01:44.800
I don't think I can go back anymore

00:01:44.800 --> 00:01:46.400
Out of control

00:01:46.400 --> 00:01:49.600
The spot, spot, spotlight that watches over me

00:01:49.600 --> 00:01:52.800
The more it shines, the more I'm drawn into the darkness

00:01:52.800 --> 00:01:56.000
I can see the light, I know it's not right

00:01:56.000 --> 00:01:58.000
I can't stop me, can't stop me

00:01:58.000 --> 00:01:59.200
No

00:01:59.200 --> 00:02:02.400
This red, red, red line placed in front of me

00:02:02.400 --> 00:02:05.800
You and I are already on opposite sides

00:02:05.800 --> 00:02:08.800
I want to feel the thrilling highlight

00:02:08.800 --> 00:02:10.800
I can't stop me, can't stop me

00:02:10.800 --> 00:02:11.800
No

00:02:11.800 --> 00:02:15.000
Risky, risky, wiki, wiki, this is an emergency

00:02:15.000 --> 00:02:16.600
Help me, help me, somebody stop me

00:02:16.600 --> 00:02:18.800
Cause I know I can't stop me

00:02:18.800 --> 00:02:22.000
You know the answer, but you're still going

00:02:22.000 --> 00:02:25.000
I don't want to be like this, I guess there's someone else inside me

00:02:25.000 --> 00:02:31.000
I want it, but I can't want it

00:02:31.000 --> 00:02:33.600
I don't want this to end

00:02:33.600 --> 00:02:37.600
I can't stop me, can't stop me, can't stop me

00:02:37.600 --> 00:02:40.800
The spot, spot, spotlight that watches over me

00:02:40.800 --> 00:02:44.000
The more it shines, the more I'm drawn into the darkness

00:02:44.000 --> 00:02:47.200
I can see the light, I know it's not right

00:02:47.200 --> 00:02:49.200
I can't stop me, can't stop me

00:02:49.200 --> 00:02:50.400
No

00:02:50.400 --> 00:02:53.600
This red, red, red line placed in front of me

00:02:53.600 --> 00:02:57.000
You and I are already on opposite sides

00:02:57.000 --> 00:03:00.000
I want to feel the thrilling highlight

00:03:00.000 --> 00:03:02.000
I can't stop me, can't stop me

00:03:02.000 --> 00:03:03.000
No

00:03:03.000 --> 00:03:07.000
Risky, risky, wiki, wiki, this is an emergency

00:03:07.000 --> 00:03:10.000
Help me, help me, somebody stop me

00:03:10.000 --> 00:03:12.000
Cause I know I can't stop me

00:03:12.000 --> 00:03:13.000
No

00:03:13.000 --> 00:03:15.000
I can't stop me, can't stop me

00:03:15.000 --> 00:03:16.000
No

00:03:21.000 --> 00:03:23.000
I can't stop me, can't stop me

00:03:23.000 --> 00:03:24.000
No

00:03:24.000 --> 00:03:26.000
I can't stop me, can't stop me

00:03:26.000 --> 00:03:27.000
No

"""
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






from pytube import YouTube

link = r"https://www.youtube.com/watch?v=XHmVwOvcemw"

from pytube import YouTube

# Prompt the user for the video URL
yt = YouTube(link)

# Get the highest resolution stream of the video
ys = yt.streams.get_highest_resolution()

# Download the video in the current working directory
print("Downloading...")
ys.download()
print("Download completed!")


