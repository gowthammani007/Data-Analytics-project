import tkinter as tk
import pyaudio
import threading
import time
import queue
import math
import wave
import pyautogui
import pandas as pd

# ----------------- NEW IMPORTS FOR AWS TRANSCRIBE -----------------
import asyncio
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# ------------------- CONFIGURATION PARAMETERS -------------------
AUDIO_RATE = 16000
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 100
SILENCE_DURATION = 1.0
SILENCE_CHUNKS = int((AUDIO_RATE / CHUNK_SIZE) * SILENCE_DURATION)

# --------------------- READ YOUR DATABASE -----------------------
df = pd.read_excel("invoice words.xlsx")
DATABASE = df.iloc[:, 0].tolist()

# ------------------------------------------------------------------
# 1) AWS Transcribe Handler & Logic
# ------------------------------------------------------------------

class MyEventHandler(TranscriptResultStreamHandler):
    """
    Custom handler that captures ONLY final transcripts from AWS Transcribe.
    """
    def __init__(self, stream):
        super().__init__(stream)
        self.final_transcript = []

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if result.is_partial:
                continue
            for alt in result.alternatives:
                self.final_transcript.append(alt.transcript)

async def transcribe_wav_file(filename):
    client = TranscribeStreamingClient(region="us-east-1")
    transcribe_stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )

    handler = MyEventHandler(transcribe_stream.output_stream)

    async def read_file_in_chunks():
        with wave.open(filename, 'rb') as wf:
            while True:
                data = wf.readframes(CHUNK_SIZE)
                if not data:
                    break
                yield data

    async def send_audio_chunks():
        async for chunk in read_file_in_chunks():
            await transcribe_stream.input_stream.send_audio_event(audio_chunk=chunk)
        await transcribe_stream.input_stream.end_stream()

    await asyncio.gather(send_audio_chunks(), handler.handle_events())
    return " ".join(handler.final_transcript).strip()

def run_stt_on_file(filename):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(transcribe_wav_file(filename))
    finally:
        loop.close()
    return result

# ----------------------- AUDIO THREAD --------------------------- 
class AudioCaptureThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.chunk_queue = queue.Queue()
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

    def run(self):
        while self.running:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.chunk_queue.put(data)
            except:
                continue

    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

# ---------------------- AUDIO PROCESSING ------------------------
def chunk_energy(chunk):
    samples = []
    for i in range(0, len(chunk), 2):
        val = int.from_bytes(chunk[i:i+2], byteorder='little', signed=True)
        samples.append(val)
    sum_sqr = sum(s*s for s in samples)
    return math.sqrt(sum_sqr / len(samples))

def is_silence(chunk):
    return chunk_energy(chunk) < SILENCE_THRESHOLD

def write_wav(filename, data, sample_rate=AUDIO_RATE):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)

def detect_utterances(chunk_queue, on_utterance_ready_callback):
    utterance_data = bytearray()
    recording = False
    silence_count = 0
    pre_speech_chunk = None

    while True:
        try:
            data = chunk_queue.get(timeout=0.5)
        except:
            # OPTIONAL: small sleep to avoid busy waiting
            time.sleep(0.01)
            continue

        if data is None:
            # A signal to break out
            break

        silent = is_silence(data)
        if not recording:
            if not silent:
                recording = True
                if pre_speech_chunk:
                    utterance_data.extend(pre_speech_chunk)
                utterance_data.extend(data)
                silence_count = 0
            else:
                pre_speech_chunk = data
        else:
            if silent:
                silence_count += 1
            else:
                silence_count = 0
            utterance_data.extend(data)

            if silence_count >= SILENCE_CHUNKS:
                filename = f"data_{int(time.time())}.wav"
                write_wav(filename, utterance_data)
                recognized_text = run_stt_on_file(filename)
                on_utterance_ready_callback(recognized_text)
                utterance_data = bytearray()
                recording = False
                silence_count = 0

# ---------------------- TEXT CONVERSION -------------------------
number_map = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9"
}

def word_to_digit(text):
    """
    Convert spelled-out digits ("one", "two") -> numeric ("1", "2").
    """
    words = text.split()
    converted = []
    for w in words:
        if w in number_map:
            converted.append(number_map[w])
        else:
            converted.append(w)
    return " ".join(converted)

def parse_digits_for_ordinals(txt):
    """
    E.g., "621" => "sixth one" if desired.
    This is just a placeholder example logic. Adapt as needed.
    """
    fallback_map = {
        "521": "fifth one",
        "621": "sixth one",
        "721": "seventh one",
        "821": "eighth one",
        "921": "ninth one",
        "51":  "fifth one",
        "61":  "sixth one",
        "71":  "seventh one",
        # ...
    }
    if txt in fallback_map:
        return fallback_map[txt]
    return txt

def fix_ordinal_ones(text):
    """
    E.g., "1st 1" => "first one", etc.
    """
    replacements = {
        "1st 1": "first one",
        "2nd 1": "second one",
        "3rd 1": "third one",
        "4th 1": "fourth one",
        "5th 1": "fifth one",
        "6th 1": "sixth one",
        "7th 1": "seventh one",
        "8th 1": "eighth one",
        "9th 1": "ninth one",
        "10th 1": "tenth one",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def replace_first_second_ones(text):
    """
    "first one" => "first", "second one" => "second", etc.
    So the user can say "first one" to pick the first match,
    which becomes just "first" for selection logic.
    """
    replacements = {
        "first one":   "first",
        "second one":  "second",
        "third one":   "third",
        "fourth one":  "fourth",
        "fifth one":   "fifth",
        "sixth one":   "sixth",
        "seventh one": "seventh",
        "eighth one":  "eighth",
        "ninth one":   "ninth",
        "tenth one":   "tenth",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# ---------------------- MAIN APPLICATION ------------------------
class VoiceDictationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Recognition App")
        self.root.geometry("600x600")

        self.listening = False

        # recognized_chunks: list of each recognized text chunk
        # recognized_text_1: the concatenation of all chunks
        self.recognized_chunks = []
        self.recognized_text_1 = ""

        # Stores multiple potential matches for the last recognized chunk
        self.current_matched_words = []

        # -------------------- GUI SETUP ---------------------
        self.text_box_1 = tk.Text(root, height=12, width=70)
        self.text_box_1.pack(pady=10)
        self.text_box_1.config(state=tk.DISABLED)

        self.start_stop_button = tk.Button(
            root, text="Start", command=self.toggle_mic,
            bg="green", fg="white", width=15, height=2
        )
        self.start_stop_button.pack(pady=10)

        self.status_label = tk.Label(root, text="Status: Ready",
                                     bg="black", fg="white",
                                     width=20, height=2)
        self.status_label.pack(side="bottom", pady=10)

        self.text_box_1.tag_configure("bold", font=("Helvetica", 12, "bold"))

        self.audio_thread = None
        self.chunk_queue = None

        # -------------------- COLUMN TRACKING ---------------------
        # If you have exactly 2 columns, set num_columns=2
        # If you have more columns, set it accordingly.
        self.current_column = 0
        self.num_columns = 3   # <--- ADJUST THIS AS NEEDED

    # ------------------- LISTENING LOGIC ------------------
    def toggle_mic(self):
        if not self.listening:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        self.listening = True
        self.start_stop_button.config(text="Stop", bg="red")
        self.status_label.config(text="Status: Listening")
        self.chunk_queue = queue.Queue()
        self.audio_thread = AudioCaptureThread()
        self.audio_thread.start()

        self.listening_thread = threading.Thread(
            target=detect_utterances,
            args=(self.audio_thread.chunk_queue, self.on_utterance_ready)
        )
        self.listening_thread.start()

    def stop_listening(self):
        self.listening = False
        self.start_stop_button.config(text="Start", bg="green")
        self.status_label.config(text="Status: Ready")

        if self.audio_thread:
            self.audio_thread.stop()
        if self.chunk_queue:
            # Send None to break out of detect_utterances loop
            self.chunk_queue.put(None)

    # ------------------- CALLBACKS ------------------------
    def on_utterance_ready(self, recognized_text):
        self.root.after(0, self.process_recognized_text, recognized_text)

    def process_recognized_text(self, recognized_text):
        """
        Handle a newly recognized phrase from AWS Transcribe.
        """
        # 1) Clean up trailing punctuation, to lower
        recognized_text = recognized_text.rstrip(".,!?").strip().lower()
        if not recognized_text:
            return

        # 2) Fix ordinal anomalies, parse digits, etc.
        recognized_text = fix_ordinal_ones(recognized_text)
        recognized_text = parse_digits_for_ordinals(recognized_text)
        recognized_text = replace_first_second_ones(recognized_text)  # "first one" -> "first"
        recognized_text = word_to_digit(recognized_text)              # "one" -> "1" (for normal text)

        # 3) Check for special commands (clear, next, etc.)
        if self.handle_special_commands(recognized_text):
            return

        # ---------------------------------------------------
        # 4) If user said "first", "second", "third", etc.,
        #    interpret it as a selection from current_matched_words
        # ---------------------------------------------------
        selection_keywords = [
            "first", "second", "third", "fourth", "fifth",
            "sixth", "seventh", "eighth", "ninth", "tenth", "last"
        ]
        words = recognized_text.split()
        is_selection_command = any(k in words for k in selection_keywords)

        if is_selection_command and self.current_matched_words:
            self.select_matched_word(recognized_text)
            return

        # ---------------------------------------------------
        # 5) Otherwise, treat the recognized_text as normal text chunk
        # ---------------------------------------------------
        self.recognized_chunks.append(recognized_text)

        # Join recognized text with spaces for better readability
        self.recognized_text_1 = " ".join(self.recognized_chunks)

        # Type the new chunk to the external cursor
        pyautogui.typewrite(recognized_text)

        # Update the Tkinter display
        self.update_text_box(self.text_box_1, self.recognized_text_1)

        # 6) Look for matches in DATABASE
        matched_words = [
            word for word in DATABASE
            if isinstance(word, str) and recognized_text in word.lower()
        ]
        if matched_words:
            self.current_matched_words = matched_words

            # If exactly one match, auto-replace
            if len(matched_words) == 1:
                selected_word = matched_words[0]
                # remove the chunk we just typed
                backspaces = '\b' * len(recognized_text)
                pyautogui.typewrite(backspaces)

                # type the matched word
                pyautogui.typewrite(selected_word)

                # update recognized_chunks
                self.recognized_chunks[-1] = selected_word
                self.recognized_text_1 = " ".join(self.recognized_chunks)
                self.update_text_box(self.text_box_1, self.recognized_text_1)
            else:
                # multiple matches: list them
                matches_text = "\n".join(
                    f"{i+1}) {w}" for i, w in enumerate(matched_words)
                )
                self.text_box_1.config(state=tk.NORMAL)
                self.text_box_1.insert(tk.END, "\n" + matches_text + "\n")
                self.text_box_1.config(state=tk.DISABLED)
        else:
            self.current_matched_words = []

    # ----------------------------------------------------------------
    # Handle Special Commands
    # ----------------------------------------------------------------
    def handle_special_commands(self, recognized_text):
        """
        Check if recognized_text is "clear", "clear all", "next", etc.
        and handle them. Return True if command was handled.
        """
        # 1) Clear All
        if recognized_text == "clear all":
            total_len = sum(len(chunk) for chunk in self.recognized_chunks)
            pyautogui.typewrite('\b' * total_len)
            self.recognized_chunks = []
            self.recognized_text_1 = ""
            self.update_text_box(self.text_box_1, "")
            return True

        # 2) Clear (remove the last chunk)
        if recognized_text == "clear":
            if self.recognized_chunks:
                last_chunk = self.recognized_chunks.pop()
                pyautogui.typewrite('\b' * len(last_chunk))
            self.recognized_text_1 = " ".join(self.recognized_chunks)
            self.update_text_box(self.text_box_1, self.recognized_text_1)
            return True

        # 3) Delete Row (example from your code)
        if recognized_text == "delete row":
            self.delete_row_contents()
            return True

        # 4) Move to next column or next row, based on num_columns
        if recognized_text == "next":
        # 1) Clear the Tkinter display first
            self.clear_text_box()
            # Also reset recognized_chunks so the text doesn't stay
            self.recognized_chunks = []
            self.recognized_text_1 = ""

            # 2) Now do the column logic to move in Excel
            self.current_column += 1
            if self.current_column < self.num_columns:
                pyautogui.press('tab')
            else:
                pyautogui.press('down')
                for _ in range(self.num_columns - 1):
                    pyautogui.press('left')
                self.current_column = 0

            return True

        # 5) Go back one column (shift+tab)
        if recognized_text == "go back":
            if self.current_column > 0:
                self.current_column -= 1
                pyautogui.hotkey('shift', 'tab')
            else:
                # If already in column 0, you might move up a row
                # and to the last column. This is optional, example:
                pyautogui.press('up')
                # Then tab right num_columns-1 times or 'end'?
                # Up to you how you handle "go back" from first column of a row
            return True

        # 6) Single backspace
        if recognized_text == "backspace":
            pyautogui.press('backspace')
            if self.recognized_chunks:
                if self.recognized_chunks[-1]:
                    self.recognized_chunks[-1] = self.recognized_chunks[-1][:-1]
                    if not self.recognized_chunks[-1]:
                        self.recognized_chunks.pop()
            self.recognized_text_1 = " ".join(self.recognized_chunks)
            self.update_text_box(self.text_box_1, self.recognized_text_1)
            return True

        return False

    # ----------------------------------------------------------------
    # Selection Logic (User says "first", "second", etc.)
    # ----------------------------------------------------------------
    def select_matched_word(self, command):
        """
        e.g. "first", "second", "third", "last"
        => replace the last chunk with the corresponding matched word.
        """
        ordinal_map = {
            "first":   0,  "second":  1, "third":   2, "fourth":  3,
            "fifth":   4,  "sixth":   5, "seventh": 6, "eighth":  7,
            "ninth":   8,  "tenth":   9
        }

        index = None
        words = command.split()
        for w in words:
            if w in ordinal_map:
                index = ordinal_map[w]
                break
            if w == "last":
                index = len(self.current_matched_words) - 1
                break

        if index is not None and 0 <= index < len(self.current_matched_words):
            selected_word = self.current_matched_words[index]

            # remove the last typed chunk from the external cursor
            if self.recognized_chunks:
                last_chunk = self.recognized_chunks[-1]
                pyautogui.typewrite('\b' * len(last_chunk))

                # replace last chunk with selected_word
                self.recognized_chunks[-1] = selected_word

            # type the selected matched word
            pyautogui.typewrite(selected_word)

            # update recognized_text_1 and display
            self.recognized_text_1 = " ".join(self.recognized_chunks)
            self.update_text_box(self.text_box_1, self.recognized_text_1)
        else:
            print(f"Invalid selection: {command}")

    # ------------------- HELPER METHODS --------------------
    def clear_text_box(self):
        self.text_box_1.config(state=tk.NORMAL)
        self.text_box_1.delete("1.0", tk.END)
        self.text_box_1.config(state=tk.DISABLED)

    def reset_variables(self):
        self.recognized_chunks = []
        self.recognized_text_1 = ""
        self.current_matched_words = []

    def update_text_box(self, text_box, text):
        text_box.config(state=tk.NORMAL)
        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, text, "bold")
        text_box.config(state=tk.DISABLED)

    def delete_row_contents(self):
        """
        Delete the cell(s) based on self.current_column.

        - If current_column == 0 (1st column): no shift-left, just delete the current cell.
        - If current_column == 1 (2nd column): press shift-left once, delete, then left once.
        - If current_column == 2 (3rd column): press shift-left twice, delete, then left twice.
        - ... and so forth.

        This logic assumes columns are zero-indexed.
        """
        # Press ESC to ensure we are not actively editing a cell.
        pyautogui.press('esc')

        # 1) Press shift+left self.current_column times
        for _ in range(self.current_column):
            pyautogui.hotkey('shift', 'left')

        # 2) Press 'delete' to delete selected cells
        pyautogui.press('delete')

        # 3) Press left arrow self.current_column times
        #    to return the cursor to the leftmost selected column (optional).
        for _ in range(self.current_column):
            pyautogui.press('left')


# --------------------- MAIN LAUNCH -----------------------
if __name__ == "__main__":
    root = tk.Tk()

    # Optional: handle window closing event to stop audio gracefully
    def on_closing():
        app.stop_listening()
        root.destroy()

    app = VoiceDictationApp(root)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
