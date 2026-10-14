import asyncio
import os
import glob
import shutil
import subprocess

SCRIPT_LINES = [
    (
        "Hi judges. This is CardioTriage 3D. When doctors check heart disease,"
        " they usually look at spreadsheets of numbers. But numbers do not show"
        " where a heart blockage is happening. We built this tool to turn"
        " patient data into an interactive 3D heart model that helps doctors"
        " spot blocked arteries and plan treatment."
    ),
    (
        "Here on the main screen, we can load a patient's medical details like"
        " age, blood pressure, ECG, and lab tests. Right away, you see our 3D"
        " heart model. The three main heart arteries light up in different"
        " colors. If an artery is healthy, it stays green. If it is at risk,"
        " it turns amber or red, showing the tight spot."
    ),
    (
        "Doctors can look at the heart from different angles, just like they"
        " do during hospital checks, and click on any artery to see why our"
        " model flagged it."
    ),
    (
        "Using our control panel on the left, a doctor can lock the patient's"
        " current state and test what happens if they improve their health."
        " For example, if we lower blood pressure and cholesterol, watch how"
        " the risk scores drop instantly, and the 3D heart updates live on"
        " screen."
    ),
    (
        "We wanted to make sure this system is genuinely reliable. We trained"
        " our models using real hospital data from the UCI repository, making"
        " sure we never leaked the final answers into our training inputs."
        " CardioTriage 3D bridges the gap between raw data and clear 3D"
        " medical visualization. Thank you!"
    ),
]

FULL_TEXT = " ".join(SCRIPT_LINES)
AUDIO_PATH = "narration.mp3"
WAV_PATH = "narration.wav"
VIDEOS_DIR = "videos_dir"
FINAL_OUTPUT = "demo_submission.mp4"
DOWNLOADS_OUTPUT = os.path.expanduser("~/Downloads/demo_submission.mp4")


def get_media_duration(file_path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        return float(out)
    except Exception:
        return 65.0


def generate_audio_sync():
    print("[*] Generating English narration audio...")
    # First attempt: Windows SAPI high-clarity voice
    escaped_text = FULL_TEXT.replace("'", "''")
    ps_cmd = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('Microsoft David Desktop'); "
        f"$s.Rate = 0; "
        f"$s.SetOutputToWaveFile('{WAV_PATH}'); "
        f"$s.Speak('{escaped_text}'); "
        f"$s.Dispose();"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
    
    # Convert WAV to MP3 using ffmpeg
    subprocess.run([
        "ffmpeg", "-y", "-i", WAV_PATH,
        "-c:a", "libmp3lame", "-q:a", "2",
        AUDIO_PATH
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    duration = get_media_duration(AUDIO_PATH)
    print(f"[+] Audio narration generated successfully: {AUDIO_PATH} (Duration: {duration:.2f}s)")
    return duration


async def record_browser_actions(target_duration):
    from playwright.async_api import async_playwright
    print(f"[*] Launching Playwright to record app actions (Target duration: ~{target_duration:.1f}s)...")
    
    if os.path.exists(VIDEOS_DIR):
        shutil.rmtree(VIDEOS_DIR)
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=VIDEOS_DIR,
            record_video_size={"width": 1920, "height": 1080}
        )
        page = await context.new_page()

        print("  -> Loading http://localhost:8000...")
        await page.goto("http://localhost:8000", wait_until="networkidle")
        await page.wait_for_timeout(4000)

        # Introduction: 3D Heart & Preset Patient #7
        print("  -> Scene 1: Triple-Vessel Patient Overview & 3D Heart Model...")
        await page.wait_for_timeout(10000)

        # Scene 2: Camera Views (LAO, RAO, AP)
        print("  -> Scene 2: Switching Cath-Lab Projections (LAO / RAO / AP)...")
        try:
            await page.click("#btn-cam-lao")
            await page.wait_for_timeout(5000)
            await page.click("#btn-cam-rao")
            await page.wait_for_timeout(5000)
            await page.click("#btn-cam-ap")
            await page.wait_for_timeout(4000)
        except Exception as e:
            print(f"    [!] Cam switch note: {e}")

        # Scene 3: Inspect Vessel Attributions (LCX, RCA, LAD)
        print("  -> Scene 3: Inspecting Artery Blockage Probabilities & SHAP...")
        try:
            await page.click("#card-lcx")
            await page.wait_for_timeout(4500)
            await page.click("#card-rca")
            await page.wait_for_timeout(4500)
            await page.click("#card-lad")
            await page.wait_for_timeout(4500)
        except Exception as e:
            print(f"    [!] Vessel click note: {e}")

        # Scene 4: Simulate Risk Control (Counterfactuals)
        print("  -> Scene 4: Simulating Risk Control & Counterfactual Deltas...")
        try:
            await page.click("#btn-simulate-lifestyle")
            await page.wait_for_timeout(9000)
        except Exception as e:
            print(f"    [!] Simulate click note: {e}")

        # Scene 5: 5-Fold Cross-Validation Audit Modal
        print("  -> Scene 5: Demonstrating 5-Fold CV Audit Modal & Zero Target Leakage...")
        try:
            await page.click("#btn-open-metrics-modal")
            await page.wait_for_timeout(8000)
            await page.click("#btn-modal-dismiss")
            await page.wait_for_timeout(3000)
        except Exception as e:
            print(f"    [!] Audit modal click note: {e}")

        # Final buffer
        await page.wait_for_timeout(4000)

        await context.close()
        await browser.close()
    
    print("[+] Browser action recording complete.")


def merge_media():
    print("[*] Merging browser video with narration audio via FFmpeg...")
    webm_files = glob.glob(os.path.join(VIDEOS_DIR, "*.webm"))
    if not webm_files:
        raise RuntimeError("No recorded video file found in videos_dir/.")
    recorded_webm = webm_files[0]
    
    video_dur = get_media_duration(recorded_webm)
    audio_dur = get_media_duration(AUDIO_PATH)
    print(f"  -> Video Duration: {video_dur:.2f}s | Audio Duration: {audio_dur:.2f}s")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", recorded_webm,
        "-i", AUDIO_PATH,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        FINAL_OUTPUT
    ]
    subprocess.run(cmd, check=True)
    print(f"[+] Final submission video created successfully: {FINAL_OUTPUT}")

    # Copy to Downloads for easy playback
    try:
        shutil.copyfile(FINAL_OUTPUT, DOWNLOADS_OUTPUT)
        print(f"[+] Video also copied to Downloads: {DOWNLOADS_OUTPUT}")
    except Exception as e:
        print(f"  [!] Note copying to Downloads: {e}")


if __name__ == "__main__":
    audio_duration = generate_audio_sync()
    asyncio.run(record_browser_actions(audio_duration))
    merge_media()

