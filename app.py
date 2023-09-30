from flask import Flask, render_template, request, redirect, jsonify, url_for
from moviepy.editor import VideoFileClip
import UsedFunctions
import json 
from pydub import AudioSegment
import os 
from moviepy.editor import *



app = Flask(__name__)
filtered_segments = []


video_path = None  # Global variable to store the path of the uploaded video
durations=[]  # Global variable to store the durations of the inserted video
lasttimes=[] # Global variable to store the insert time for each inserted video 
videoName=''
imagePaths=[]
totaltime=0

# Helper function to convert HH:MM:SS.sss to seconds
def time_to_seconds(time_str):
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)








def get_video_vtt_segments(content):

    global durations, lasttimes, imagePaths
    
    (translation_list, arabicsummary_list, lasttimes) = UsedFunctions.translate_vtt_content(content)
    print('finised the first step (gpt translation)')
   
    
    all_vtt_content, durations = UsedFunctions.process_translation_list(translation_list, arabicsummary_list, lasttimes)
    print('finised the second step (gpt vtt for the summary)')
    
    
    slides = UsedFunctions.generate_slides(arabicsummary_list)
    print('finised the third step (generating summaries)')
    
    imagePaths = UsedFunctions.save_html_to_img(slides)
    print('finised the forth step (saving images)')

    segments = all_vtt_content.strip().split('\n\n')


    return segments





@app.route('/')
def index():
    """
    Serve the main page of the application.
    """
    return render_template('main.html')
   

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Handles the upload of a video file or processing a video URL.
    
    Expects:
    - Form data with a key 'videofile' containing the uploaded video file (if provided).
    - Form data with a key 'videoUrl' containing the video URL (if provided).
    - The video file is expected to be in MP4 format.

    Returns:
    - If valid, saves the video and redirects to a secondary page (assuming 'model.html').
    - If the file type is invalid or the video URL is not valid, returns a 400 error response.
    """
    global video_path, videoName
    use_premium = request.form.get('usePremium')
    print(use_premium)
    if use_premium == 'on':
        UsedFunctions.model = "gpt-4"
        # The checkbox is checked
        # Perform actions for premium features
    else:
        UsedFunctions.model = "gpt-3.5-turbo-32k"
    
    print('used model is', UsedFunctions.model)

    uploaded_file = request.files.get('videofile')
    video_url = request.form.get('videoUrl')

    if uploaded_file and uploaded_file.filename.endswith('.mp4'):
        video_path = 'videos/' + uploaded_file.filename
        uploaded_file.save(video_path)
        videoName=uploaded_file.filename
        return render_template('model.html')
    elif video_url:
        # Process the video URL
        video_path,videoName = UsedFunctions.download_video(video_url)
        if video_path:
            return render_template('model.html')  # You can adjust this
        else:
            return 'Invalid video URL', 400
    else:
        return 'Invalid file type or video URL', 400


@app.route('/whisper', methods=['POST'])
def process_with_whisper():
    """
    Extracts audio from an uploaded video and transcribes it using the Whisper model.
    After transcription, the text is paraphrased using the GPT model.

    Returns:
    - The transcribed and paraphrased text.
    - An error response if no video has been uploaded or if there are any issues in the process.
    """
    global lasttimes,durations,video_path,totaltime
    if not video_path:
        return 'No video uploaded.', 400
    
    start=time.time()
    video_path = video_path.replace("'", '"')
    print('this is the video path',video_path)
    audio_path = UsedFunctions.extract_audio(video_path)
    print('finised audio extractions')
    original_text = UsedFunctions.whisperModel(audio_path)
    print('finised whisper extractions',original_text)
     # get video duration 
    segments=get_video_vtt_segments(original_text)
    end=time.time()
    totaltime=(end-start)/60
    print(f'the gpt part took {(end-start)/60} m ')


    return render_template('transcription.html', original_text=segments)


import time

@app.route('/vtt', methods=['POST'])
def vttfile():
    """
    Processes a VTT (Web Video Text Tracks) file uploaded by the user.
    The content of the VTT file is then paraphrased using the GPT model.

    Returns:
    - The paraphrased content from the VTT file.
    - An error response if the uploaded file is not a valid VTT file.
    """
    global lasttimes,durations,video_path,totaltime
    start=time.time()
    vttfile = request.files['vttfile']

    if vttfile.filename != '' and vttfile.filename.endswith('.vtt'):
        content = vttfile.read().decode('utf-8')

        # get video duration 
        segments=get_video_vtt_segments(content)
        end=time.time()
        totaltime=(end-start)/60
        print(f'the gpt part took {(end-start)/60} m ')
        return render_template('transcription.html', original_text=segments)
    else:
        return 'Invalid file type', 400
    
   




import os


@app.route('/recording_combined', methods=['POST'])
def recording_combined():

    global lasttimes, durations , video_path,imagePaths,totaltime
    segments = request.form['segments']
    num_lines = int(request.form['numLines'])
    mode = request.form['recordingMode']
    lasttimesfloats=[time_to_seconds(i) for i in lasttimes]
    durationsfloats=[time_to_seconds(i) for i in durations]
    start=time.time()
    UsedFunctions.insert_images_into_video_ffmpeg(video_path, imagePaths, lasttimesfloats, durationsfloats)
    finalpath=r"output_video.mp4"
    # premparing the segments and the lines
    end=time.time()
    totaltime+=(end-start)/60

    print(f'the insertion part took {(end-start)/60} m ')
    segmentsList=segments.split('||')
    num_linesInt=int(num_lines)
    start=time.time()

    videosPaths=UsedFunctions.prepare_video_segments(finalpath,segmentsList,num_linesInt)
    videosPaths='||'.join(videosPaths)
    end=time.time()
    totaltime+=(end-start)/60

    print(f'the cutting part took {(end-start)/60} m ')
    # delete the images 
    for file in imagePaths:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted: {file}")
        else:
            print(f"{file} not found!")

    if mode == 'mode1':
        return render_template('recoredingmode1.html', segments=segments, num_lines=num_lines,videosPaths=videosPaths)
    elif mode == 'mode2':
        return render_template('recording_page.html', segments=segments, num_lines=num_lines,videosPaths=videosPaths)








@app.route('/save-final-audio', methods=['POST'])
def save_audio():
    """
    Processes the audio recording from the user.
    The recorded audio is then integrated into the original video, replacing the video's audio.

    Expects:
    - Form data with a key 'audio' containing the recorded audio file.

    Returns:
    - A new video where the audio is replaced with the user's recording.
    - An error response if no audio file is provided or if any issue arises during the process.
    """
    try:
        global video_path,durations,lasttimes

        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"success": False, "message": "No audio file provided in request"}), 400

        audio = AudioSegment.from_file(audio_file, format="webm")
        audio.export("outputaudios/output1.wav", format="wav")

        output_filename = UsedFunctions.replace_audio_in_video_ffmpeg(r"output_video.mp4", r'outputaudios/output1.wav', r'static\videos')
        video_url = url_for('static', filename=f'videos/reencoded_output_voice.mp4')

            # Check if the file exists
        if os.path.exists(r"videos\my_custom_video_name.mp4"):
            os.remove(r"videos\my_custom_video_name.mp4")
        else:
            print(r"videos\my_custom_video_name.mp4")
        return render_template('listen.html', path=video_url)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500 








@app.route('/save-final-audio_mode_one', methods=['POST'])
def save_audio_mode_one():
    
    """
    Processes the audio recording from the user.
    The recorded audio is then integrated into the original video, replacing the video's audio.

    Expects:
    - Form data with a key 'audio' containing the recorded audio file.

    Returns:
    - A new video where the audio is replaced with the user's recording.
    - An error response if no audio file is provided or if any issue arises during the process.
    """
    global totaltime
    try:

        expectedDuration = json.loads(request.form.get('list1'))
        accualeDuration = json.loads(request.form.get('list2'))
        print(expectedDuration,accualeDuration)

        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"success": False, "message": "No audio file provided in request"}), 400

        audio = AudioSegment.from_file(audio_file, format="webm")
        audio.export("outputaudios/output1.wav", format="wav")

         
        start=time.time()
        UsedFunctions.split_and_adjust_video_ffmpeg(r"output_video.mp4", accualeDuration, expectedDuration, r"static\videos\final_output.mp4")
        UsedFunctions.replace_audio_in_video_option_ffmpeg(r"static\videos\final_output.mp4", r'outputaudios/output1.wav', r'static\videos')
        end=time.time()
        print(f'the adjust part took {(end-start)/60} m ')
        totaltime+=(end-start)/60

        video_url = url_for('static', filename=f'videos/final_output_voice.mp4')
        print(video_url)
        print(f'this is the total time {totaltime}')
            # Check if the file exists
        if os.path.exists(r"videos\my_custom_video_name.mp4"):
            os.remove(r"videos\my_custom_video_name.mp4")
        else:
            print(r"videos\my_custom_video_name.mp4")
        
        return render_template('listen.html', path=video_url)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

