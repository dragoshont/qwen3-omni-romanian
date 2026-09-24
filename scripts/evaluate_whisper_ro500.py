import gc
import json
import os
import sys
import torch
import soundfile as sf
import librosa
import jiwer
from transformers import pipeline

sys.stdout.reconfigure(encoding="utf-8")

def evaluate_whisper():
    print("=== RUNNING WHISPER LARGE-V3-TURBO GATE ON RO500 SYNTHESIZED AUDIO ===")
    
    test_files = [
        ("outputs/romanian_mtp_trained_eval/ro_sample_01_ro500_mtp.wav", "De asemenea, autoritățile locale au închis școala pentru restul zilei."),
        ("outputs/romanian_mtp_trained_eval/ro_sample_02_ro500_mtp.wav", "Numai adevărul vă poate face liberi, trebuie să cunoașteți adevărul."),
        ("outputs/romanian_mtp_trained_eval/ro_sample_03_ro500_mtp.wav", "Totodată, doamna a cerut să vorbească cu cineva responsabil de situație."),
        ("outputs/romanian_mtp_trained_eval/ro_sample_051_ro500_mtp.wav", "Alte surse vorbesc de un milion de bolnavi aflați în tratament."),
        ("outputs/romanian_mtp_trained_eval/ro_sample_0151_ro500_mtp.wav", "În urma unor procese, judecătorii au decis că suma trebuie să fie trecută pe factură.")
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=device,
        dtype=torch.float16
    )
    
    results = []
    total_wer = 0.0
    total_cer = 0.0
    
    for wav_path, ref in test_files:
        audio_np, sr = sf.read(wav_path)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 16000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        audio_input = {"raw": audio_np, "sampling_rate": 16000}
        res = asr(audio_input, generate_kwargs={"language": "ro", "task": "transcribe"})
        hyp = res["text"].strip()
        
        w = jiwer.wer(ref, hyp)
        c = jiwer.cer(ref, hyp)
        
        total_wer += w
        total_cer += c
        
        results.append({
            "wav_path": wav_path,
            "ref": ref,
            "hyp": hyp,
            "wer": round(w, 4),
            "cer": round(c, 4)
        })
        
        print(f"\nSample: {os.path.basename(wav_path)}")
        print(f"  Reference:    \"{ref}\"")
        print(f"  Recognized:   \"{hyp}\"")
        print(f"  WER: {w*100:.1f}% | CER: {c*100:.1f}%")
        
    avg_wer = total_wer / len(test_files)
    avg_cer = total_cer / len(test_files)
    
    print("\n" + "="*50)
    print("WHISPER BENCHMARK SUMMARY (RO500 TRAINED MODEL):")
    print(f"  Average WER: {avg_wer*100:.2f}%")
    print(f"  Average CER: {avg_cer*100:.2f}%")
    print(f"  Intelligibility Gate (<10% WER): {'PASS' if avg_wer < 0.10 else 'REVIEW'}")
    print("="*50)
    
    with open("reports/ro500_whisper_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"avg_wer": round(avg_wer, 4), "avg_cer": round(avg_cer, 4), "samples": results}, f, indent=2)
        
    del asr
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    evaluate_whisper()
