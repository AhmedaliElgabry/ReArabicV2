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
seo='' # Global variable to store the seo of the uploaded vidoe 
timeline='' # Global variable to store the time line of the uploaded video
videoName=''



# Helper function to convert HH:MM:SS.sss to seconds
def time_to_seconds(time_str):
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

# Create an upload directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

folder_path = "slidesImages"
all_files = os.listdir(folder_path)





def getTranslationInfor(original_text):

    output= UsedFunctions.get_gpt_translation(original_text)

    response=output['response']
    print('this is the response ',response)

    # Parse the preprocessed JSON string.
    response_data = json.loads(response,strict=False)

    # Extract the 'response' string and then parse it.

    # Now you can access the nested keys
    translation = response_data.get('translation'," ")
    seo = response_data.get('SEOkeywords'," ")
    timeline = response_data.get('time_line'," ")
    englishsummary=response_data.get('English_summar'," ")
    arabicsummary=response_data.get('Arabic_summary'," ")
    return (translation, seo,timeline,englishsummary,arabicsummary)



def get_video_vtt_segments(content):


    global seo,timeline,durations,lasttimes
    
    (translation, seo_output,timeline_output,englishsummary,arabicsummary)=getTranslationInfor(content)
    UsedFunctions.save_html_to_img(englishsummary)

    seo+=seo_output
    timeline+=timeline_output
    
    (translationmodeloutput,durations_output,lasttimes_output,vttsummary)=UsedFunctions.createVtt(arabicsummary,translation)
    durations.append(durations_output)
    lasttimes.append(lasttimes_output)

    UsedFunctions.generateExcelReport(content,translation,seo,timeline,englishsummary,arabicsummary,vttsummary,"Modeldata.xlsx")
    segments = translationmodeloutput.strip().split('\n\n')

def getDescriptionInfo(translation):

    modelTwoOutput=UsedFunctions.get_gpt_discriptions(translation)
    modelTwoResonse=modelTwoOutput.get('response')

    modelTwoResponseData=json.loads(modelTwoResonse,strict=False)

    translationmodeloutput=modelTwoResponseData.get('finalVtt'," ")
    lasttimes=modelTwoResponseData.get('starttime'," ")

    durations=modelTwoResponseData.get('durations'," ")

    print(translationmodeloutput, lasttimes, durations)
    durations =[UsedFunctions.timestamp_to_seconds(ts) for ts in durations]
    seconds_list = [UsedFunctions.timestamp_to_seconds(ts) for ts in lasttimes]






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
        print('video path is ', video_path)
        print('video name ',videoName)
        if video_path:
            return render_template('model.html')  # You can adjust this
        else:
            return 'Invalid video URL', 400
    else:
        return 'Invalid file type or video URL', 400

import math

@app.route('/whisper', methods=['POST'])
def process_with_whisper():
    """
    Extracts audio from an uploaded video and transcribes it using the Whisper model.
    After transcription, the text is paraphrased using the GPT model.

    Returns:
    - The transcribed and paraphrased text.
    - An error response if no video has been uploaded or if there are any issues in the process.
    """
    global video_path,  durations,lasttimes,seo,timeline,videoName
    if not video_path:
        return 'No video uploaded.', 400
    
    video_path = video_path.replace("'", '"')
    print('this is the video path',video_path)
    audio_path = UsedFunctions.extract_audio(video_path)
    print('finised audio extractions')
    original_text = UsedFunctions.whisperModel(audio_path)
    print('finised whisper extractions',original_text)
    (translation, seo,timeline,englishsummary,arabicsummary)=getTranslationInfor(original_text)
    print('finised translation extractions',translation)
    UsedFunctions.save_html_to_img(englishsummary)

    print('finised discriptions extractions',lasttimes, durations)
    print('finished ',seo,timeline)

    (translationmodeloutput,durations,lasttimes)=UsedFunctions.createVtt(arabicsummary,translation)
        
    segments = translationmodeloutput.strip().split('\n\n')

    return render_template('transcription.html', original_text=segments)




@app.route('/vtt', methods=['POST'])
def vttfile():
    """
    Processes a VTT (Web Video Text Tracks) file uploaded by the user.
    The content of the VTT file is then paraphrased using the GPT model.

    Returns:
    - The paraphrased content from the VTT file.
    - An error response if the uploaded file is not a valid VTT file.
    """
    global seo,timeline,lasttimes,durations
    global seo,timeline,lasttimes,durations,video_path
    vttfile = request.files['vttfile']
    if vttfile.filename != '' and vttfile.filename.endswith('.vtt'):
        content = vttfile.read().decode('utf-8')
      
        (translation, seo,timeline,englishsummary,arabicsummary)=getTranslationInfor(content)
        UsedFunctions.save_html_to_img(englishsummary)
        
        (translationmodeloutput,durations,lasttimes)=UsedFunctions.createVtt(arabicsummary,translation)
        
        segments = translationmodeloutput.strip().split('\n\n')
        # get video duration 
        duration=UsedFunctions.get_video_duration(video_path)
        if duration<=300:
            segments=get_video_vtt_segments(content)    
        else:
            spitedcontent=UsedFunctions.split_webvtt(content)
            for i in spitedcontent:
 
        return render_template('transcription.html', original_text=segments)
    else:
        return 'Invalid file type', 400


import os


@app.route('/recording_combined', methods=['POST'])
def recording_combined():

    global lasttimes, durations , video_path
    segments = request.form['segments']
    num_lines = int(request.form['numLines'])
    mode = request.form['recordingMode']
    lasttimesfloats=[time_to_seconds(i) for i in lasttimes]
    durationsfloats=[time_to_seconds(i) for i in durations]
    image_files=UsedFunctions.get_image_files_in_folder(r"slidesImages")
    UsedFunctions.insert_images_into_video(video_path, image_files, lasttimesfloats, durationsfloats)
    finalpath=r"output_video.mp4"
    # premparing the segments and the lines
    segmentsList=segments.split('||')
    num_linesInt=int(num_lines)




    videosPaths=UsedFunctions.prepare_video_segments(finalpath,segmentsList,num_linesInt)
    videosPaths='||'.join(videosPaths)


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
        global video_path,durations,lasttimes,seo,timeline

        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"success": False, "message": "No audio file provided in request"}), 400

        audio = AudioSegment.from_file(audio_file, format="webm")
        audio.export("outputaudios/output1.wav", format="wav")

        output_filename = UsedFunctions.replace_audio_in_video(r"output_video.mp4", r'outputaudios/output1.wav', r'static\videos')
        video_url = url_for('static', filename=f'videos/{output_filename}')

            # Check if the file exists
        if os.path.exists(r"videos\my_custom_video_name.mp4"):
            os.remove(r"videos\my_custom_video_name.mp4")
        else:
            print(r"videos\my_custom_video_name.mp4")
        return render_template('listen.html', path=video_url,SEOkeywords=seo,time_line=timeline)

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

    try:
        global video_path,durations,lasttimes,timeline,seo

        list1 = json.loads(request.form.get('list1'))
        list2 = json.loads(request.form.get('list2'))
        print(list1,list2)

        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"success": False, "message": "No audio file provided in request"}), 400

        audio = AudioSegment.from_file(audio_file, format="webm")
        audio.export("outputaudios/output1.wav", format="wav")

        output_filename = UsedFunctions.replace_audio_in_video_option_one(r"output_video.mp4", r'outputaudios/output1.wav', r'static\videos')
        
        UsedFunctions.split_and_adjust_video_moviepy(output_filename, list2, list1)

        video_url = url_for('static', filename=f'videos/final_output.mp4')
            # Check if the file exists
        if os.path.exists(r"videos\my_custom_video_name.mp4"):
            os.remove(r"videos\my_custom_video_name.mp4")
        else:
            print(r"videos\my_custom_video_name.mp4")
        
        return render_template('listen.html', path=video_url,SEOkeywords=seo,time_line=timeline)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

