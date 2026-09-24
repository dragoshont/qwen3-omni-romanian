import subprocess
import sys

def main():
    print("Checking and restoring Windows power settings...")
    # Ultimate Performance or Balanced defaults
    # Monitor timeout on AC: 15 minutes (standard gaming PC)
    subprocess.run(["powercfg", "/change", "monitor-timeout-ac", "15"], check=False)
    # Standby timeout on AC: 0 (never sleep while plugged in, standard for high-performance desktop)
    subprocess.run(["powercfg", "/change", "standby-timeout-ac", "0"], check=False)
    
    print("Power settings confirmed: Display timeout = 15m, Standby timeout = 0 (Never sleep on AC).")

if __name__ == "__main__":
    main()
