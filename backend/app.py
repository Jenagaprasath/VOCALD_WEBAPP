from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime, timezone, timedelta
from voice_processor import process_audio_file
import time

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

IST = timezone(timedelta(hours=5, minutes=30))
UPLOAD_FOLDER = 'uploads'
DB_PATH = 'vocald.db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_db_connection(max_retries=3):
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False, isolation_level='DEFERRED')
            conn.execute('PRAGMA journal_mode=WAL')
            return conn
        except sqlite3.OperationalError as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            raise e

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS recordings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL, filepath TEXT NOT NULL,
        upload_date TEXT NOT NULL, processed BOOLEAN DEFAULT 0)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS speakers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recording_id INTEGER NOT NULL, speaker_index INTEGER NOT NULL,
        name TEXT NOT NULL, confidence REAL NOT NULL, voice_profile_id INTEGER,
        FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS voice_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, embedding BLOB NOT NULL,
        first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
        total_recordings INTEGER DEFAULT 1)''')
    conn.commit()
    conn.close()

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    files = request.files.getlist('files')
    uploaded = []
    for file in files:
        if file.filename == '':
            continue
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        upload_date = datetime.now(IST).isoformat()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO recordings (filename, filepath, upload_date, processed) VALUES (?, ?, ?, ?)',
            (file.filename, filepath, upload_date, False))
        recording_id = cursor.lastrowid
        conn.commit()
        conn.close()
        print(f"\n🎙️ Analyzing: {file.filename}")
        speakers = process_audio_file(filepath, file.filename)
        conn = get_db_connection()
        cursor = conn.cursor()
        for speaker in speakers:
            cursor.execute('''INSERT INTO speakers (recording_id, speaker_index, name, confidence, voice_profile_id)
                VALUES (?, ?, ?, ?, ?)''',
                (recording_id, speaker['speaker_index'], speaker['name'], speaker['confidence'], speaker.get('voice_profile_id')))
        cursor.execute('UPDATE recordings SET processed = ? WHERE id = ?', (True, recording_id))
        conn.commit()
        conn.close()
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"⚠️ Could not delete file: {e}")
        uploaded.append({'id': recording_id, 'filename': file.filename, 'speakers': len(speakers)})
    return jsonify({'message': 'Files uploaded and analyzed successfully', 'files': uploaded}), 201

@app.route('/api/recordings', methods=['GET'])
def get_recordings():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''SELECT r.id, r.filename, r.upload_date, COUNT(s.id) as total_speakers
        FROM recordings r LEFT JOIN speakers s ON r.id = s.recording_id
        GROUP BY r.id ORDER BY r.upload_date DESC''')
    recordings = [{'id': r[0], 'filename': r[1], 'upload_date': r[2], 'total_speakers': r[3]} for r in cursor.fetchall()]
    conn.close()
    return jsonify({'recordings': recordings})

@app.route('/api/recordings/<int:recording_id>', methods=['GET'])
def get_recording_details(recording_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM recordings WHERE id = ?', (recording_id,))
    recording = cursor.fetchone()
    if not recording:
        conn.close()
        return jsonify({'error': 'Recording not found'}), 404
    cursor.execute('SELECT speaker_index, name, confidence, voice_profile_id FROM speakers WHERE recording_id = ? ORDER BY speaker_index', (recording_id,))
    speakers = [{'speaker_index': r[0], 'name': r[1], 'confidence': r[2], 'voice_profile_id': r[3]} for r in cursor.fetchall()]
    conn.close()
    return jsonify({'id': recording[0], 'filename': recording[1], 'upload_date': recording[3], 'processed': bool(recording[4]), 'speakers': speakers})

@app.route('/api/recordings/<int:recording_id>/speakers/<int:speaker_index>', methods=['PUT'])
def update_speaker_name(recording_id, speaker_index):
    data = request.json
    new_name = data.get('name')
    if not new_name:
        return jsonify({'error': 'Name is required'}), 400
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT voice_profile_id FROM speakers WHERE recording_id = ? AND speaker_index = ?', (recording_id, speaker_index))
    result = cursor.fetchone()
    if result and result[0]:
        cursor.execute('UPDATE voice_profiles SET name = ? WHERE id = ?', (new_name, result[0]))
        cursor.execute('UPDATE speakers SET name = ? WHERE voice_profile_id = ?', (new_name, result[0]))
    else:
        cursor.execute('UPDATE speakers SET name = ? WHERE recording_id = ? AND speaker_index = ?', (new_name, recording_id, speaker_index))
    conn.commit()
    conn.close()
    return get_recording_details(recording_id)

@app.route('/api/recordings/<int:recording_id>', methods=['DELETE'])
def delete_recording(recording_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT filepath FROM recordings WHERE id = ?', (recording_id,))
    result = cursor.fetchone()
    if result and os.path.exists(result[0]):
        os.remove(result[0])
    cursor.execute('DELETE FROM speakers WHERE recording_id = ?', (recording_id,))
    cursor.execute('DELETE FROM recordings WHERE id = ?', (recording_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Recording deleted successfully'})

@app.route('/api/voice-profiles', methods=['GET'])
def get_voice_profiles():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, first_seen, last_seen, total_recordings FROM voice_profiles ORDER BY total_recordings DESC')
    profiles = [{'id': r[0], 'name': r[1], 'first_seen': r[2], 'last_seen': r[3], 'total_recordings': r[4]} for r in cursor.fetchall()]
    conn.close()
    return jsonify({'profiles': profiles})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, port=port, host='0.0.0.0')
