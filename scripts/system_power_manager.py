import ctypes
import os
import subprocess
import sys
import time

# Windows Execution States
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040

HWND_BROADCAST = 0xFFFF
WM_SYSCOMMAND = 0x0112
SC_MONITORPOWER = 0xF170
MONITOR_OFF = 2
MONITOR_ON = -1

def keep_system_awake():
    """Keep system CPU and GPU running without sleep, but allow display to turn off."""
    res = ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    )
    print(f"SetThreadExecutionState result: {hex(res)} (System and AwayMode active, Display sleep allowed)")
    return res

def turn_off_display():
    """Turn off the physical monitor immediately to save power and darkness."""
    print("Sending SC_MONITORPOWER signal to turn off display...")
    ctypes.windll.user32.SendMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, MONITOR_OFF)

def set_ultimate_performance():
    """Ensure Ultimate Performance power scheme is active during AI training."""
    cmd = "powercfg /setactive 83f84283-f6f0-4ee7-a06a-6adb6c97a9cb"
    subprocess.run(cmd, shell=True, check=True)
    print("Power scheme set to Ultimate Performance.")

def revert_power_settings():
    """Revert back to Balanced power scheme and restore normal execution state."""
    cmd = "powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e"
    subprocess.run(cmd, shell=True, check=True)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("Reverted to Balanced power scheme and standard execution state.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "revert":
        revert_power_settings()
    elif len(sys.argv) > 1 and sys.argv[1] == "off_display":
        keep_system_awake()
        set_ultimate_performance()
        turn_off_display()
    else:
        keep_system_awake()
        set_ultimate_performance()
