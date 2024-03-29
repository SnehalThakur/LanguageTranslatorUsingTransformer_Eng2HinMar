def bytes_to_mp3(audio_bytes, output_file_path):
    audio_segment = AudioSegment.from_file(BytesIO(audio_bytes))
    audio_segment.export(output_file_path, format="mp3")